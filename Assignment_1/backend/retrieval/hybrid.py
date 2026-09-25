"""Teach and run Qdrant-native hybrid retrieval for MediBot.

A medical question can need two kinds of matching at the same time:

* Dense retrieval understands meaning. For example, it can connect
  ``blood sugar treatment`` with a chunk about ``diabetes management``.
* BM25 sparse retrieval rewards exact words. It is useful when a nurse needs
  the literal terms ``IV cannula``, ``paediatric``, and ``5kg``.

For each document chunk, this module stores both vector types in Qdrant. For a
question, it asks Qdrant to search both vector types and combine their ranked
lists with Reciprocal Rank Fusion (RRF). Qdrant performs that fusion internally
in one request; Python does not merge two result lists itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.ingestion.chunker import chunk_document
from backend.retrieval.filters import build_role_filter


# Local Qdrant storage persists vectors between program runs.
QDRANT_DATA_PATH = Path(__file__).resolve().parents[3] / "qdrant_data"
# A new name prevents an old dense-only collection schema from being reused.
COLLECTION_NAME = "medibot_hybrid_collection"
# Qdrant uses these names to distinguish the two vectors on each point.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
# FastEmbed models used to create each representation of the same text.
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
# Retrieve a broad candidate set from each search before RRF returns the best 3.
DEFAULT_CANDIDATE_LIMIT = 10


def get_qdrant_client() -> QdrantClient:
    """Open local Qdrant and create the hybrid collection on its first use.

    A Qdrant *collection* is like a table in a database. Each stored chunk is a
    point in that table. This function defines two named vector fields on every
    point:

    - ``dense``: 384 decimal numbers describing the chunk's semantic meaning.
      Cosine distance lets Qdrant find vectors pointing in a similar direction.
    - ``bm25``: a sparse list of only important token positions and weights.
      This retains exact medical terminology without storing many zero values.

    Returns:
        A client connected to the on-disk Qdrant database.
    """
    QDRANT_DATA_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DATA_PATH))

    # The schema is created once. Later calls simply reopen the same collection.
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=384, distance=Distance.COSINE),
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
    return client


def _as_qdrant_sparse_vector(embedding) -> SparseVector:
    """Convert FastEmbed's BM25 output into the object Qdrant can store/query.

    FastEmbed returns two equal-length arrays. ``indices`` identifies vocabulary
    tokens and ``values`` says how important each token is. For example, a rare
    exact term may receive a useful weight while unimportant words are omitted.
    Qdrant's ``SparseVector`` stores those arrays without creating a huge vector
    containing mostly zeroes.
    """
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def _embed_chunks(texts: list[str]) -> Iterator[tuple[list[float], SparseVector]]:
    """Create one dense vector and one BM25 vector for every chunk text.

    Args:
        texts: The contextualised chunk text from Milestone 1. It includes the
            parent heading, which gives both retrievers useful context.

    Yields:
        Pairs in the same order as ``texts``: first the dense vector, then the
        BM25 sparse vector. That ordering is important because both vectors must
        be attached to the same Qdrant point.
    """
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    # Each model sees the identical text, but represents it in a different way.
    dense_embeddings = dense_model.embed(texts)
    sparse_embeddings = sparse_model.embed(texts)

    # ``strict=True`` catches an unexpected mismatch instead of silently pairing
    # one chunk with another chunk's BM25 vector.
    for dense, sparse in zip(dense_embeddings, sparse_embeddings, strict=True):
        yield dense.tolist(), _as_qdrant_sparse_vector(sparse)


def ingest_documents(document_path: str | Path, collection: str) -> None:
    """Chunk one PDF and store its dense and BM25 forms in Qdrant.

    Args:
        pdf_path: PDF to read through the Milestone 1 structural chunker.
        collection: MediBot collection label, such as ``clinical``. It is kept
            in the payload with the source and access metadata.

    The stored Qdrant point has one payload and two vectors. Indexing both
    vectors now is essential: a hybrid search cannot use BM25 later if it was
    never stored alongside the dense vector.
    """
    print(f"Starting hybrid ingestion for {document_path} into '{collection}'...")

    # Milestone 1 returns structured text plus metadata for every PDF chunk.
    chunks = list(chunk_document(document_path, collection))
    if not chunks:
        print("No chunks found.")
        return

    print("Generating dense and BM25 vectors...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed_chunks(texts)

    # One PointStruct represents one chunk. Its named-vector dictionary puts
    # dense and BM25 representations of that exact chunk in the same point.
    points = [
        PointStruct(
            # Stable UUIDs prevent chunks from different documents overwriting each other.
            id=str(uuid5(NAMESPACE_URL, "|".join((str(Path(document_path).resolve()), collection, str(index), chunk["section_title"], chunk["text"])))),
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload=chunk,
        )
        for index, (chunk, (dense_vector, sparse_vector)) in enumerate(
            zip(chunks, embeddings, strict=True)
        )
    ]

    client = get_qdrant_client()
    try:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    finally:
        # Close the local client cleanly after the one-time write operation.
        client.close()
    print(f"Successfully indexed {len(points)} hybrid chunks.")


def indexed_chunk_count() -> int:
    """Return how many chunks are already stored in the hybrid collection.

    This is a small status check, not an ingestion step. The command-line
    application uses it to give a helpful message when someone tries to search
    before running the one-time ``ingest`` command.
    """
    client = get_qdrant_client()
    try:
        result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )
        return result.count
    finally:
        client.close()

def list_ingested_documents() -> list[dict]:
    """Return each indexed source document and its stored chunk count."""
    client = get_qdrant_client()
    document_counts: dict[tuple[str, str], int] = {}
    offset = None

    try:
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                collection = str(payload.get("collection", "unknown"))
                source = str(payload.get("source_document", "unknown"))
                key = (collection, source)
                document_counts[key] = document_counts.get(key, 0) + 1

            if offset is None:
                break
    finally:
        client.close()

    return [
        {"collection": collection, "source_document": source, "chunks": chunks}
        for (collection, source), chunks in sorted(document_counts.items())
    ]

def hybrid_search(
    query: str,
    role: str,
    top_k: int = 3,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
):
    """Return one Qdrant-fused result list for a user's medical question.

    Args:
        query: The natural-language question to retrieve information for.
        role: Authenticated staff role used by Qdrant's payload filter.
        top_k: Number of final, fused chunks to return. The default is three.
        candidate_limit: Number of candidates each retriever contributes before
            fusion. Ten gives RRF more choices than the final answer needs.

    How RRF works:
        Dense search and BM25 first make separate ranked candidate lists. RRF
        gives a chunk credit for appearing near the top of *either* list, then
        adds its credits. A chunk ranked well by both methods normally rises;
        a chunk found only by exact keyword matching can still be retained.
        RRF uses rank positions rather than trying to compare incompatible dense
        similarity scores with BM25 scores directly.

    Returns:
        Qdrant's single, already-fused list of top ``top_k`` scored points.
    """
    if top_k < 1 or candidate_limit < top_k:
        raise ValueError(
            "candidate_limit must be greater than or equal to top_k and both must be positive."
        )

    # The same Qdrant-side security rule protects both candidate searches.
    role_filter = build_role_filter(role)

    # Turn the same question into the two representations stored at index time.
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    dense_query = next(dense_model.embed([query])).tolist()
    sparse_query = _as_qdrant_sparse_vector(next(sparse_model.embed([query])))

    # ``prefetch`` defines both candidate searches. ``Fusion.RRF`` tells Qdrant
    # to fuse them server-side in this single query_points request.
    client = get_qdrant_client()
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_query, using=DENSE_VECTOR_NAME, filter=role_filter, limit=candidate_limit),
                Prefetch(query=sparse_query, using=SPARSE_VECTOR_NAME, filter=role_filter, limit=candidate_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return response.points
    finally:
        # The returned points are already in memory, so it is safe to close.
        client.close()



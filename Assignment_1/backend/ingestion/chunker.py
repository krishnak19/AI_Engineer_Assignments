"""Create hierarchical, structure-aware chunks and their MediBot metadata.

The flow in this module is:

``PDF -> DoclingDocument -> HybridChunker -> chunk dictionary -> console preview``

Each yielded dictionary is intentionally storage-ready. A later milestone may
embed its ``text`` and store the remaining fields as vector-store metadata, but
this milestone only creates and prints those objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import sys

from docling.chunking import HybridChunker

from backend.ingestion.loader import load_document


ACCESS_ROLES_BY_COLLECTION = {
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}
"""The security contract expressed as collection-to-role mappings.

When a chunk is created, it receives the roles for its collection as
``access_roles``. Later, Qdrant will use this metadata to retrieve only chunks
a staff role may access. We create it now so access rules travel with content.
"""


def _chunk_type(chunk) -> str:
    """Map Docling's element labels into MediBot's four chunk types.

    Docling records source elements in ``chunk.meta.doc_items``. Their labels
    produce a small, consistent schema: ``table`` for a table, ``code`` for a
    code block, ``heading`` for a standalone title/heading, and ``text`` for
    ordinary paragraph or list content. A table takes precedence because its
    identity matters even when it appears under a heading.
    """
    labels = {getattr(item, "label", "") for item in chunk.meta.doc_items}
    if "table" in labels:
        return "table"
    if "code" in labels:
        return "code"
    if "section_header" in labels or "title" in labels:
        return "heading"
    return "text"


def _section_title(chunk) -> str:
    """Return the parent-heading path Docling associated with a chunk.

    Nested content may have several headings. Joining them with `` > `` creates
    a readable path, such as ``Diabetes > Pharmacological management``.
    ``Document`` is a safe fallback for content before the first heading.
    """
    headings = getattr(chunk.meta, "headings", []) or []
    return " > ".join(headings) if headings else "Document"


def chunk_document(document_path: str | Path, collection: str) -> Iterator[dict]:
    """Yield structure-aware chunks with the metadata required by MediBot.

    Args:
        pdf_path: Path to the one PDF to process.
        collection: One of the five MediBot collections. It determines the
            ``collection`` and ``access_roles`` metadata values.

    Yields:
        Dictionaries shaped like::

            {
                "text": "Heading followed by the chunk content",
                "source_document": "original-file-name.pdf",
                "collection": "clinical",
                "access_roles": ["doctor", "admin"],
                "section_title": "Parent heading",
                "chunk_type": "text | table | heading | code",
            }

    ``HybridChunker`` follows Docling's recognised hierarchy (document ->
    section -> subsection -> paragraph/table). Only if an element is too large
    does it apply a token-aware split. This avoids a fixed-size splitter
    separating a table from its labels or an instruction from its heading.

    ``contextualize`` adds the parent heading to the chunk text. This matters
    for future embeddings: ``500 mg twice daily`` is ambiguous alone, while
    ``Diabetes treatment\n500 mg twice daily`` retains its meaning.
    """
    if collection not in ACCESS_ROLES_BY_COLLECTION:
        raise ValueError(f"Unknown collection: {collection}")

    path = Path(document_path)
    # Step 1: load a structural document, not flattened PDF text.
    document = load_document(path)
    # Step 2: split using Docling's hierarchy-aware, token-aware chunker.
    chunker = HybridChunker()

    for chunk in chunker.chunk(document):
        # Step 3: derive context text and metadata from the Docling chunk.
        section_title = _section_title(chunk)
        text = chunker.contextualize(chunk)
        # ``yield`` streams one dictionary at a time, which makes a later
        # ingestion loop memory-friendly.
        yield {
            "text": text,
            "source_document": path.name,
            "collection": collection,
            "access_roles": ACCESS_ROLES_BY_COLLECTION[collection],
            "section_title": section_title,
            "chunk_type": _chunk_type(chunk),
        }


def chunk_pdf(pdf_path: str | Path, collection: str) -> Iterator[dict]:
    """Backward-compatible alias for PDF callers."""
    return chunk_document(pdf_path, collection)


def print_chunk_preview(pdf_path: str | Path, collection: str, limit: int = 5) -> None:
    """Print a small, readable sample for Milestone 1 verification.

    Args:
        pdf_path: PDF passed through :func:`chunk_pdf`.
        collection: Collection name used to create metadata.
        limit: Maximum chunks to print; five is readable while often showing
            more than one type of document element.

    The function prints ``text`` separately because that will later be embedded,
    whereas metadata will later support access filters and source citations.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for number, chunk in enumerate(chunk_document(pdf_path, collection), start=1):
        if number > limit:
            break
        metadata = {key: value for key, value in chunk.items() if key != "text"}
        print(f"\n--- Chunk {number} ---")
        print("Metadata:", metadata)
        print("Text:")
        print(chunk["text"])


if __name__ == "__main__":
    # This runs only for ``python -m backend.ingestion.chunker``. Importing the
    # functions from another file will not parse a PDF or print output.
    project_root = Path(__file__).resolve().parents[2]
    print_chunk_preview(
        project_root / "data" / "clinical" / "treatment_protocols.pdf",
        collection="clinical",
    )

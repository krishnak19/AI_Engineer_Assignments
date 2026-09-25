"""Cross-encoder reranking for RBAC-filtered hybrid search candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sentence_transformers import CrossEncoder

from backend.retrieval.hybrid import DEFAULT_CANDIDATE_LIMIT, hybrid_search

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKED_LIMIT = 3


@dataclass(frozen=True)
class RerankedResult:
    """Keep a candidate and its scores before and after reranking."""

    point: Any
    original_position: int
    original_score: float
    rerank_score: float


def rerank_candidates(
    query: str,
    candidates: Sequence[Any],
    top_k: int = DEFAULT_RERANKED_LIMIT,
    model: Any | None = None,
) -> list[RerankedResult]:
    """Read each question/chunk pair together and retain the best chunks.

    Tests can supply ``model`` to avoid downloading the real pretrained model.
    Application code leaves it unset and uses the configured cross-encoder.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not candidates:
        return []

    cross_encoder = model or CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    pairs = [
        (query, str((candidate.payload or {}).get("text", "")))
        for candidate in candidates
    ]
    scores = cross_encoder.predict(pairs)
    if len(scores) != len(candidates):
        raise ValueError("The cross-encoder returned an unexpected number of scores.")

    scored = [
        RerankedResult(
            point=candidate,
            original_position=position,
            original_score=float(candidate.score),
            rerank_score=float(score),
        )
        for position, (candidate, score) in enumerate(
            zip(candidates, scores, strict=True), start=1
        )
    ]
    scored.sort(key=lambda result: result.rerank_score, reverse=True)
    return scored[:top_k]


def retrieve_and_rerank(
    query: str,
    role: str,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    top_k: int = DEFAULT_RERANKED_LIMIT,
    model: Any | None = None,
) -> list[RerankedResult]:
    """Retrieve ten authorised candidates, then retain the best three.

    RBAC stays inside ``hybrid_search``. Forbidden chunks are excluded by
    Qdrant before RRF, so they can never enter this reranking step.
    """
    if candidate_limit < top_k:
        raise ValueError("candidate_limit must be greater than or equal to top_k.")
    candidates = hybrid_search(
        query, role=role, top_k=candidate_limit, candidate_limit=candidate_limit
    )
    return rerank_candidates(query, candidates, top_k=top_k, model=model)
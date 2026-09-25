"""Focused tests for Milestone 5 cross-encoder reranking."""

from types import SimpleNamespace

import pytest

from backend.retrieval import reranker


def _point(text: str, score: float):
    """Create a small Qdrant-like point without using a real database."""
    return SimpleNamespace(payload={"text": text}, score=score)


class FakeCrossEncoder:
    """Return predictable scores so tests check our ordering logic only."""

    def __init__(self, scores):
        self.scores = scores
        self.seen_pairs = None

    def predict(self, pairs):
        self.seen_pairs = pairs
        return self.scores


def test_reranker_reorders_candidates_and_keeps_original_details():
    """Cross-encoder scores control final order while RRF details survive."""
    candidates = [_point("first", 0.9), _point("second", 0.8), _point("third", 0.7)]
    model = FakeCrossEncoder([0.1, 0.95, 0.4])

    results = reranker.rerank_candidates("question", candidates, top_k=2, model=model)

    assert [item.point.payload["text"] for item in results] == ["second", "third"]
    assert [item.original_position for item in results] == [2, 3]
    assert results[0].original_score == pytest.approx(0.8)
    assert results[0].rerank_score == pytest.approx(0.95)
    assert model.seen_pairs == [
        ("question", "first"),
        ("question", "second"),
        ("question", "third"),
    ]


def test_retrieve_and_rerank_requests_ten_authorised_candidates(monkeypatch):
    """The orchestration asks hybrid retrieval for ten, then retains three."""
    captured = {}
    candidates = [_point(f"chunk {index}", 1 / index) for index in range(1, 11)]

    def fake_hybrid_search(query, role, top_k, candidate_limit):
        captured.update(
            query=query, role=role, top_k=top_k, candidate_limit=candidate_limit
        )
        return candidates

    monkeypatch.setattr(reranker, "hybrid_search", fake_hybrid_search)
    results = reranker.retrieve_and_rerank(
        "care question", "nurse", model=FakeCrossEncoder(list(range(10)))
    )

    assert captured == {
        "query": "care question", "role": "nurse",
        "top_k": 10, "candidate_limit": 10,
    }
    assert len(results) == 3
    assert [item.point.payload["text"] for item in results] == [
        "chunk 10", "chunk 9", "chunk 8"
    ]


def test_empty_candidates_do_not_load_a_model():
    """An empty retrieval result should return immediately and cheaply."""
    assert reranker.rerank_candidates("question", []) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k):
    """At least one final result must be requested."""
    with pytest.raises(ValueError, match="positive"):
        reranker.rerank_candidates("question", [_point("chunk", 1.0)], top_k=top_k)
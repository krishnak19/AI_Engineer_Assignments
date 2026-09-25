"""Focused tests for grounded document answering with Groq."""

from types import SimpleNamespace

import pytest

from backend.rag.document_rag import generate_document_answer


def _result(number: int):
    """Create a small reranked-result shape used by the answer function."""
    point = SimpleNamespace(
        payload={
            "source_document": f"document-{number}.pdf",
            "section_title": f"Section {number}",
            "text": f"Evidence text {number}",
        }
    )
    return SimpleNamespace(point=point)


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="Grounded answer [Source 2]")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_answer_uses_exactly_the_three_visible_chunks():
    """All and only final chunks are included with stable source labels."""
    client = FakeGroq()
    results = [_result(1), _result(2), _result(3)]

    answer = generate_document_answer("What is supported?", results, client=client)

    assert answer == "Grounded answer [Source 2]"
    request = client.chat.completions.request
    assert request["temperature"] == 0
    assert request["max_completion_tokens"] == 512
    prompt = request["messages"][1]["content"]
    assert "What is supported?" in prompt
    for number in range(1, 4):
        assert f"[Source {number}]" in prompt
        assert f"Evidence text {number}" in prompt


def test_empty_results_return_insufficient_evidence_without_calling_groq():
    """No evidence means no API request and no invented answer."""
    assert generate_document_answer("Unknown?", []) == (
        "I could not find that in the provided documents."
    )


def test_more_than_three_chunks_are_rejected():
    """The LLM boundary cannot silently receive unseen extra evidence."""
    with pytest.raises(ValueError, match="at most three"):
        generate_document_answer(
            "Question", [_result(1), _result(2), _result(3), _result(4)],
            client=FakeGroq(),
        )
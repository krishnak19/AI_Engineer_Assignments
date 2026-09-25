"""Focused tests for the Milestone 6 SQL RAG pipeline."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.rag.sql_rag import clean_sql, execute_read_only_sql, sql_rag_chain

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "mediassist.db"


class FakeCompletions:
    """Return queued LLM messages and remember both request prompts."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=next(self.responses))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    """Provide the small part of the Groq interface used by SQL RAG."""

    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


@pytest.mark.parametrize(
    ("question", "sql", "expected_column"),
    [
        (
            "How many claims are in each status?",
            "SELECT status, COUNT(*) AS claim_count FROM claims GROUP BY status",
            "claim_count",
        ),
        (
            "What is the total claimed amount by department?",
            "SELECT department, SUM(claimed_amount) AS total FROM claims GROUP BY department",
            "total",
        ),
        (
            "Which equipment categories have open tickets?",
            "SELECT category, COUNT(*) AS open_count FROM maintenance_tickets "
            "WHERE status != 'resolved' GROUP BY category",
            "open_count",
        ),
        (
            "What is the average approved amount by insurer?",
            "SELECT insurer, AVG(approved_amount) AS average FROM claims GROUP BY insurer",
            "average",
        ),
    ],
)
def test_four_analytical_questions_complete_both_llm_steps(
    question, sql, expected_column
):
    """Four questions execute SQL and pass real rows to answer generation."""
    client = FakeGroq([f"```sql\n{sql};\n```", "A concise analytical answer."])

    answer = sql_rag_chain(
        question, role="billing_executive", client=client, database_path=DATABASE_PATH
    )

    assert answer == "A concise analytical answer."
    requests = client.chat.completions.requests
    assert len(requests) == 2
    assert "claims(" in requests[0]["messages"][1]["content"]
    final_prompt = requests[1]["messages"][1]["content"]
    assert expected_column in final_prompt
    assert "Rows: [" in final_prompt


@pytest.mark.parametrize("role", ["doctor", "nurse", "technician"])
def test_unauthorised_roles_are_blocked_before_calling_the_llm(role):
    """Clinical and equipment roles cannot enter the SQL analytics pipeline."""
    client = FakeGroq([])
    with pytest.raises(PermissionError, match="billing_executive and admin"):
        sql_rag_chain("Count all claims", role=role, client=client)
    assert client.chat.completions.requests == []


def test_admin_role_is_authorised():
    """Administrators can use the same SQL analytics chain."""
    client = FakeGroq(["SELECT COUNT(*) AS total FROM claims", "There are 85 claims."])
    assert sql_rag_chain("Count claims", role="admin", client=client) == (
        "There are 85 claims."
    )


def test_clean_sql_removes_explanation_and_markdown_fences():
    """Typical decorated LLM output becomes one executable statement."""
    raw = "Here is the query:\n```sql\nSELECT COUNT(*) FROM claims;\n```"
    assert clean_sql(raw) == "SELECT COUNT(*) FROM claims"


@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DELETE FROM claims",
        "SELECT * FROM claims; DROP TABLE claims",
        "SELECT * FROM sqlite_master",
    ],
)
def test_unsafe_or_unknown_table_sql_is_rejected(unsafe_sql):
    """Generated SQL cannot write, stack statements, or read other tables."""
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        execute_read_only_sql(unsafe_sql, DATABASE_PATH)

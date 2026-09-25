"""Answer authorised analytical questions using MediAssist's SQLite data."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from groq import Groq

from backend.config import GROQ_MODEL, get_groq_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "db" / "mediassist.db"
SQL_RAG_ROLES = frozenset({"billing_executive", "admin"})
ALLOWED_TABLES = frozenset({"claims", "maintenance_tickets"})
MAX_RESULT_ROWS = 100

SQL_SYSTEM_PROMPT = """You translate questions into one safe SQLite query.
Return SQL only, with no Markdown or explanation. Use only SELECT or WITH and
only the tables and columns in the supplied schema. Never modify the database.
Use SQLite syntax. Unless the question asks for grouped results, prefer one
aggregate row. Limit non-aggregate detail results to 100 rows."""

ANSWER_SYSTEM_PROMPT = """You are MediBot's data analyst. Answer the question
using only the supplied SQL result. Do not add facts that are absent from the
result. Explain the result clearly and concisely. If the result is empty, say
that no matching records were found."""

_START_OF_SQL = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|ANALYZE)\b",
    re.IGNORECASE,
)


def _database_uri(database_path: Path) -> str:
    """Build a SQLite URI that prevents writes to the database file."""
    return f"file:{database_path.resolve().as_posix()}?mode=ro"


def _read_schema(database_path: Path) -> str:
    """Describe allowed tables and columns for the SQL-generating LLM."""
    if not database_path.is_file():
        raise FileNotFoundError(f"MediAssist database not found: {database_path}")
    descriptions = []
    with sqlite3.connect(_database_uri(database_path), uri=True) as connection:
        for table in sorted(ALLOWED_TABLES):
            columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            if not columns:
                raise RuntimeError(f"Expected table is missing: {table}")
            column_text = ", ".join(f"{row[1]} {row[2]}" for row in columns)
            descriptions.append(f"{table}({column_text})")
    return "\n".join(descriptions)


def clean_sql(raw_output: str) -> str:
    """Extract one SQL statement from fences or brief LLM explanation text."""
    if not raw_output or not raw_output.strip():
        raise ValueError("The LLM returned empty SQL.")
    text = raw_output.strip().replace("```sql", "").replace("```SQL", "")
    text = text.replace("```", "").strip()
    start = _START_OF_SQL.search(text)
    if not start:
        raise ValueError("The LLM response did not contain a SELECT query.")
    sql = text[start.start():].strip()
    if ";" in sql:
        statement, remainder = sql.split(";", 1)
        if remainder.strip():
            raise ValueError("Only one SQL statement is allowed.")
        sql = statement.strip()
    return sql


def validate_sql(sql: str) -> None:
    """Reject non-read-only, multi-statement, commented, or dangerous SQL."""
    if not _START_OF_SQL.match(sql.lstrip()):
        raise ValueError("SQL must start with SELECT or WITH.")
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("SQL comments are not allowed.")
    if _FORBIDDEN_SQL.search(sql):
        raise ValueError("Only read-only SQL is allowed.")


def _read_only_authorizer(
    action: int,
    argument1: str | None,
    _argument2: str | None,
    _database_name: str | None,
    _trigger_name: str | None,
) -> int:
    """Allow SQLite reads only from the two Milestone 6 data tables."""
    if action == sqlite3.SQLITE_READ and argument1 not in ALLOWED_TABLES:
        return sqlite3.SQLITE_DENY
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", -1),
    }
    return sqlite3.SQLITE_OK if action in allowed_actions else sqlite3.SQLITE_DENY


def execute_read_only_sql(
    sql: str, database_path: Path = DATABASE_PATH
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute validated SQL read-only and return column names plus rows."""
    validate_sql(sql)
    with sqlite3.connect(_database_uri(database_path), uri=True) as connection:
        connection.set_authorizer(_read_only_authorizer)
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
    if len(rows) > MAX_RESULT_ROWS:
        raise ValueError(f"SQL result exceeds the {MAX_RESULT_ROWS}-row safety limit.")
    return columns, rows


def _completion(client: Any, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Call Groq deterministically and return a non-empty text response."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content.strip()


def sql_rag_chain(
    question: str,
    *,
    role: str,
    client: Any | None = None,
    database_path: Path = DATABASE_PATH,
) -> str:
    """Run question -> cleaned SQL -> SQLite -> natural-language answer.

    The role is required as a keyword so callers cannot accidentally bypass the
    Milestone 6 access rule. Optional arguments keep tests deterministic.
    """
    if role not in SQL_RAG_ROLES:
        raise PermissionError(
            "SQL analytics is available only to billing_executive and admin roles."
        )
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    groq_client = client or Groq(api_key=get_groq_api_key())
    schema = _read_schema(database_path)
    raw_sql = _completion(
        groq_client,
        SQL_SYSTEM_PROMPT,
        f"Database schema:\n{schema}\n\nQuestion:\n{question.strip()}",
        max_tokens=300,
    )
    sql = clean_sql(raw_sql)
    columns, rows = execute_read_only_sql(sql, database_path)
    result_text = f"Columns: {columns}\nRows: {rows}"
    return _completion(
        groq_client,
        ANSWER_SYSTEM_PROMPT,
        f"Question:\n{question.strip()}\n\nSQL used:\n{sql}\n\nSQL result:\n{result_text}",
        max_tokens=512,
    )

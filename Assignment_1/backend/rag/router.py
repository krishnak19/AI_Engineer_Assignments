'''Classify chat questions and route them to the correct existing RAG path.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from backend.rag.document_rag import generate_document_answer
from backend.rag.sql_rag import SQL_RAG_ROLES, sql_rag_chain
from backend.retrieval.reranker import retrieve_and_rerank

SQL_SUBJECT_TERMS = (
    'claim', 'claims', 'claimed amount', 'approved amount', 'insurer',
    'maintenance ticket', 'maintenance tickets', 'open ticket', 'resolved ticket',
)
ANALYTICAL_TERMS = (
    'how many', 'count', 'total', 'average', 'sum', 'group by', 'breakdown',
    'highest', 'lowest', 'most', 'least', 'per department', 'by department',
    'by insurer', 'by status', 'by category',
)


@dataclass(frozen=True)
class ChatResult:
    '''A transport-neutral answer returned to the FastAPI layer.'''

    answer: str
    retrieval_type: str
    sources: list[dict[str, str]]


def classify_question(question: str) -> str:
    '''Classify analytical database questions using transparent keyword rules.'''
    normalized = ' '.join(question.lower().split())
    has_subject = any(term in normalized for term in SQL_SUBJECT_TERMS)
    has_analysis = any(term in normalized for term in ANALYTICAL_TERMS)
    return 'sql' if has_subject and has_analysis else 'document'


def _serialize_sources(results: Sequence[Any]) -> list[dict[str, str]]:
    '''Convert final reranked chunks into safe source metadata.'''
    sources = []
    for number, result in enumerate(results, start=1):
        payload = result.point.payload or {}
        sources.append({
            'label': f'Source {number}',
            'source_document': str(payload.get('source_document', 'Unknown')),
            'section_title': str(payload.get('section_title', 'Unknown')),
            'collection': str(payload.get('collection', 'Unknown')),
        })
    return sources


def route_question(
    question: str,
    role: str,
    *,
    retrieve: Callable[..., Sequence[Any]] = retrieve_and_rerank,
    document_answer: Callable[..., str] = generate_document_answer,
    sql_answer: Callable[..., str] = sql_rag_chain,
) -> ChatResult:
    '''Run SQL RAG or document RAG after classification and role enforcement.'''
    route_type = classify_question(question)
    if route_type == 'sql':
        if role not in SQL_RAG_ROLES:
            raise PermissionError(
                'SQL analytics is available only to billing_executive and admin roles.'
            )
        return ChatResult(sql_answer(question, role=role), 'sql_rag', [])

    results = list(retrieve(question, role=role))
    return ChatResult(
        document_answer(question, results),
        'hybrid_rag',
        _serialize_sources(results),
    )

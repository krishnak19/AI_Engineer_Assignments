"""Command-line entry point for the current MediBot retrieval application.

Keep this file small: it decides which application command to run, while the
specialist modules contain the actual ingestion and retrieval work. As MediBot
grows, future commands can be added here without making ``hybrid.py`` depend on
command-line details.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from backend.auth import authenticate_user, create_access_token, get_current_user
from backend.config import MEDIBOT_FRONTEND_ORIGIN
from backend.rag.document_rag import generate_document_answer
from backend.rag.router import route_question
from backend.retrieval.filters import SUPPORTED_ROLES

from backend.retrieval.hybrid import (
    DEFAULT_CANDIDATE_LIMIT,
    indexed_chunk_count,
    ingest_documents,
    list_ingested_documents,
)
from backend.retrieval.reranker import DEFAULT_RERANKED_LIMIT, retrieve_and_rerank

ROLE_COLLECTIONS = {
    'doctor': ['general', 'clinical'],
    'nurse': ['general', 'nursing'],
    'billing_executive': ['general', 'billing'],
    'technician': ['general', 'equipment'],
    'admin': ['general', 'clinical', 'nursing', 'billing', 'equipment'],
}

app = FastAPI(title='MediBot API', version='0.7.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[MEDIBOT_FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Authorization', 'Content-Type'],
)


class LoginRequest(BaseModel):
    '''Credentials submitted by one of the five demo staff accounts.'''

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    role: str


class ChatRequest(BaseModel):
    '''One non-empty natural-language question for the authenticated role.'''

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.get('/health')
def health() -> dict[str, str]:
    '''Report that the API process is ready to receive requests.'''
    return {'status': 'ok'}


@app.post('/login')
def login(request: LoginRequest) -> dict[str, str | int]:
    '''Validate demo credentials and return a signed bearer token.'''
    if not authenticate_user(request.username, request.password, request.role):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username, password, or role.',
        )
    return {
        'access_token': create_access_token(request.username, request.role),
        'token_type': 'bearer',
        'expires_in': 3600,
    }


@app.get('/collections/{role}')
def collections(
    role: str,
    current_user: dict[str, str] = Depends(get_current_user),
) -> dict[str, object]:
    '''Return only the collections belonging to the authenticated role.'''
    if role not in ROLE_COLLECTIONS:
        raise HTTPException(status_code=404, detail='Unknown role.')
    if current_user['role'] != role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You may view collections only for your authenticated role.',
        )
    return {'role': role, 'collections': ROLE_COLLECTIONS[role]}


@app.post('/chat')
def chat(
    request: ChatRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> dict[str, object]:
    '''Classify and answer a question using only the authenticated role.'''
    try:
        result = route_question(request.question, current_user['role'])
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {
        'answer': result.answer,
        'retrieval_type': result.retrieval_type,
        'sources': result.sources,
        'role': current_user['role'],
    }


def print_search_results(results) -> None:
    """Display final chunks and their before-and-after ranking information."""
    for position, result in enumerate(results, start=1):
        point = result.point
        print(
            f"--- Reranked result {position} | "
            f"original RRF position {result.original_position} | "
            f"RRF score {result.original_score:.4f} | "
            f"cross-encoder score {result.rerank_score:.4f} ---"
        )
        print(
            "Metadata: "
            f"Roles={point.payload.get('access_roles', [])} | "
            f"Section={point.payload.get('section_title', 'Unknown')}"
        )
        preview = str(point.payload.get("text", ""))
        # Some medical symbols are unavailable in older Windows code pages.
        encoding = sys.stdout.encoding or "utf-8"
        safe_preview = preview.encode(encoding, errors="replace").decode(encoding)
        print(f"Text:\n{safe_preview}...\n")


def build_cli_parser() -> ArgumentParser:
    """Define the explicit ``ingest`` and ``search`` application commands.

    Keeping these commands separate is intentional. ``ingest`` performs the
    expensive PDF chunking and embedding once. ``search`` reads the vectors that
    Qdrant has already stored, so normal questions never trigger ingestion.
    """
    project_root = Path(__file__).resolve().parents[1]
    default_pdf = project_root / "data" / "clinical" / "treatment_protocols.pdf"

    parser = ArgumentParser(
        description="MediBot's small, local hybrid-retrieval application."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser(
        "ingest", help="Create and persist vectors for one PDF or Markdown file."
    )
    ingest_parser.add_argument(
        "--document", "--pdf",
        dest="document",
        type=Path,
        default=default_pdf,
        help=f"PDF or Markdown file to index (default: {default_pdf})",
    )
    ingest_parser.add_argument(
        "--collection",
        default="clinical",
        help="Metadata collection label (default: clinical)",
    )

    commands.add_parser(
        "ingested-documents",
        help="List source documents currently stored in the vector database.",
    )
    search_parser = commands.add_parser(
        "search", help="Search already-indexed vectors; never runs ingestion."
    )
    search_parser.add_argument("query", help="Medical question to search for.")
    search_parser.add_argument(
        "--role", required=True, choices=SUPPORTED_ROLES,
        help="Authenticated staff role used to filter results inside Qdrant.",
    )
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_RERANKED_LIMIT, choices=range(1, 4), help="Final reranked results to show, maximum 3 (default: 3)"
    )
    search_parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Dense and BM25 candidates before RRF (default: 10)",
    )
    return parser


def main() -> None:
    """Run the command requested by the person using MediBot.

    The only branch that writes vectors is ``ingest``. The ``search`` branch
    first confirms that persisted chunks exist, then sends the query to the
    hybrid retriever and prints its fused results.
    """
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_documents(args.document, args.collection)
        return

    if args.command == "ingested-documents":
        documents = list_ingested_documents()
        if not documents:
            print("No documents have been ingested.")
            return

        print(f"{'Collection':<14} {'Document':<42} Chunks")
        print(f"{'-' * 12:<14} {'-' * 40:<42} {'-' * 6}")
        for document in documents:
            print(
                f"{document['collection']:<14} "
                f"{document['source_document']:<42} "
                f"{document['chunks']}"
            )
        return
    chunk_count = indexed_chunk_count()
    if chunk_count == 0:
        parser.error(
            "No indexed chunks found. Run `python -m backend.main ingest` once first."
        )

    print(f"Searching {chunk_count} indexed chunks as role '{args.role}' for: '{args.query}'\n")
    results = retrieve_and_rerank(
        args.query,
        role=args.role,
        top_k=args.top_k,
        candidate_limit=args.candidates,
    )
    print_search_results(results)
    print("--- Grounded document answer (Groq) ---")
    print(generate_document_answer(args.query, results))


if __name__ == "__main__":
    main()




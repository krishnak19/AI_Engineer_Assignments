"""Turn a PDF into a structure-aware Docling document.

This module is the first stage of MediBot's future ingestion pipeline. It does
not create embeddings or store anything in a database. Its single job is to
let Docling recognise the document's structure (for example, headings,
paragraphs, and tables) before the chunker works with it.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".md"}

def load_document(document_path: str | Path):
    """Convert one PDF into Docling's structured document representation.

    Args:
        pdf_path: The path to one local ``.pdf`` file. It can be a string or a
            :class:`pathlib.Path` object.

    Returns:
        A ``DoclingDocument``. Unlike plain extracted text, this object keeps
        headings, paragraphs, and tables distinct. This lets the chunker keep a
        table with its heading instead of splitting both into unrelated text.

    Raises:
        ValueError: If the supplied path is not a PDF.
        FileNotFoundError: If the PDF does not exist at the supplied path.

    Chunking is deliberately kept in ``chunker.py``. Separating loading from
    chunking makes each stage easy to test and understand independently.
    """
    path = Path(document_path)
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
        raise ValueError(f"Expected one of ({supported}), got: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    # DocumentConverter is Docling's structural PDF parser. It returns a
    # document model rather than one flattened string of text.
    converter = DocumentConverter()
    return converter.convert(path).document


def load_pdf(pdf_path: str | Path):
    """Backward-compatible alias for PDF callers."""
    return load_document(pdf_path)

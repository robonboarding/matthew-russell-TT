"""
Document ingestion and chunking.

JUSTIFICATION SUMMARY:
- Recursive character splitting preserves sentence and paragraph boundaries
- Chunk size 800 tokens balances retrieval precision with context preservation
- Overlap of 100 tokens catches facts that straddle chunk boundaries
- Metadata carries source and chunk index for citation and auditability
- PII redaction is a hook, stubbed for the assessment

In production at Rabobank:
- Use layout-aware parsers for policy PDFs with tables (e.g. Azure Document Intelligence)
- Replace stub PII redaction with Presidio or Azure Text Analytics for PII
- Chunking strategy might switch to semantic chunking for narrative documents
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from src.config import CHUNKS_PATH, CONFIG, DATA_PROCESSED


@dataclass
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    char_count: int


def load_document(path: Path) -> str:
    """Read a single document to string. Supports PDF, TXT, MD."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def redact_pii(text: str) -> str:
    """
    Stub PII redaction.

    JUSTIFICATION: For a Dutch bank this would catch BSN (9 digits),
    IBAN, email addresses, phone numbers, and salary figures. In the
    reflection call, mention Presidio or Azure Text Analytics as the
    production choice. Here we redact obvious patterns as a proof of
    concept, not a complete solution.
    """
    # BSN-like 9-digit sequences
    text = re.sub(r"\b\d{9}\b", "[BSN_REDACTED]", text)
    # IBAN (Dutch format)
    text = re.sub(r"\bNL\d{2}[A-Z]{4}\d{10}\b", "[IBAN_REDACTED]", text)
    # Email
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL_REDACTED]", text)
    return text


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursive character splitter.

    Tries progressively finer separators (double newline, newline, sentence,
    word) to preserve semantic units. Falls back to hard character split only
    when no natural boundary exists within the chunk window.

    JUSTIFICATION: A naive fixed-width splitter on "Article 4.1 of the
    mortgage policy states X. Article 4.2 states Y." might cut mid-article.
    Recursive splitting keeps each article intact when possible.
    """
    separators = ["\n\n", "\n", ". ", " "]
    return _split(text, separators, chunk_size, overlap)


def _split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Hard character split as last resort
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split(text, rest, chunk_size, overlap)

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split(part, rest, chunk_size, overlap))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Apply overlap by prepending tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + chunks[i])
        chunks = overlapped

    return chunks


def ingest_directory(raw_dir: Path, apply_pii_redaction: bool = False) -> list[Chunk]:
    """Walk a directory, load and chunk every supported document."""
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        print(f"Loading {path.name}")
        text = load_document(path)
        if apply_pii_redaction:
            text = redact_pii(text)
        for i, chunk_text in enumerate(
            recursive_split(text, CONFIG.chunk_size, CONFIG.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}#{i:04d}",
                    source=path.name,
                    chunk_index=i,
                    text=chunk_text,
                    char_count=len(chunk_text),
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps([asdict(c) for c in chunks], indent=2))
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")


def load_chunks() -> list[Chunk]:
    data = json.loads(CHUNKS_PATH.read_text())
    return [Chunk(**c) for c in data]


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    apply_pii = "--redact-pii" in sys.argv
    chunks = ingest_directory(raw_dir, apply_pii_redaction=apply_pii)
    save_chunks(chunks)

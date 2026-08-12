"""
context_nexus.ingestion
=======================
Multi-source document and code ingestion parser.
Extracts text chunks and structural relationship edges for:
  - Text & Markdown (.txt, .md)
  - PDF documents (.pdf via pypdf)
  - HTML content (.html, .htm)
  - Source code files (Python, JS, TS, Go, Rust)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_nexus.hybrid_search import fast_chunk_text


@dataclass
class IngestedDocument:
    """Document representation produced by the ingestion pipeline."""

    doc_id: str
    file_path: str
    file_type: str
    raw_content: str
    chunks: list[str] = field(default_factory=list)
    import_edges: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentIngestor:
    """Parser for multi-source files (PDF, Markdown, HTML, source code)."""

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_file(self, file_path: str | Path) -> IngestedDocument:
        """Parse *file_path* and extract chunks and import/dependency edges."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        doc_id = path.name

        if ext == ".pdf":
            content = self._parse_pdf(path)
            file_type = "pdf"
        elif ext in (".html", ".htm"):
            content = self._parse_html(path)
            file_type = "html"
        else:
            content = path.read_text(encoding="utf-8", errors="replace")
            file_type = ext.lstrip(".") or "text"

        chunks = fast_chunk_text(content, self.chunk_size, self.overlap)
        edges = self._extract_code_edges(content, file_type, doc_id)

        return IngestedDocument(
            doc_id=doc_id,
            file_path=str(path.resolve()),
            file_type=file_type,
            raw_content=content,
            chunks=chunks,
            import_edges=edges,
        )

    def ingest_text(self, text: str, doc_id: str = "doc_inline") -> IngestedDocument:
        """Ingest raw string content."""
        chunks = fast_chunk_text(text, self.chunk_size, self.overlap)
        return IngestedDocument(
            doc_id=doc_id,
            file_path="inline",
            file_type="inline",
            raw_content=text,
            chunks=chunks,
        )

    def _parse_pdf(self, path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages)
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")

    def _parse_html(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Strip HTML tags via regex
        clean = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", clean).strip()

    def _extract_code_edges(self, content: str, file_type: str, doc_id: str) -> list[tuple[str, str]]:
        """Extract import/dependency edges for source code files."""
        edges: list[tuple[str, str]] = []
        if file_type == "py" or file_type == "python":
            imports = re.findall(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", content, re.MULTILINE)
            for imp in imports:
                edges.append((doc_id, imp.split(".")[0]))
        elif file_type in ("js", "ts", "javascript", "typescript"):
            imports = re.findall(r"from\s+['\"]([^'\"]+)['\"]", content)
            for imp in imports:
                edges.append((doc_id, imp))
        elif file_type == "go":
            imports = re.findall(r"import\s+[\"\']([^\"\']+)[\"\']", content)
            for imp in imports:
                edges.append((doc_id, imp.split("/")[-1]))
        elif file_type == "rs" or file_type == "rust":
            imports = re.findall(r"use\s+([A-Za-z0-9_]+)", content)
            for imp in imports:
                edges.append((doc_id, imp))
        return edges

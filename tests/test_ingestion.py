"""
tests/test_ingestion.py
========================
Unit tests for context_nexus.ingestion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from context_nexus.ingestion import DocumentIngestor


class TestDocumentIngestor:

    def test_ingest_text(self) -> None:
        ingestor = DocumentIngestor(chunk_size=10, overlap=2)
        doc = ingestor.ingest_text("one two three four five six seven eight nine ten eleven twelve", doc_id="d1")
        assert doc.doc_id == "d1"
        assert len(doc.chunks) >= 2

    def test_ingest_code_extracts_import_edges(self) -> None:
        ingestor = DocumentIngestor()
        code = "import psycopg2\nimport os\nfrom math import sqrt"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            f_path = f.name

        doc = ingestor.ingest_file(f_path)
        assert len(doc.import_edges) >= 2
        Path(f_path).unlink(missing_ok=True)

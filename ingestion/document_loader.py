# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
ingestion/document_loader.py

Reads a document from the local filesystem (or an http(s) URL), extracts
its text, and splits it into overlapping chunks.

No cloud SDK or credentials are needed. If your documents live in a cloud
bucket, download them first (or pass a public / pre-signed https URL).

This is only used by the Summarization workflow. The Tool-Use workflow
doesn't load documents — it loads benchmark task definitions instead (see
workflows/tool_use/tasks.py). That's expected: not every workflow needs
every piece of infrastructure, and that's fine.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse


@dataclass
class Chunk:
    """One piece of a document, after splitting on word boundaries."""
    text: str
    index: int
    source: str


class DocumentLoader:
    def __init__(self, timeout: float = 60.0):
        # `timeout` only applies to http(s) sources.
        self.timeout = timeout

    def load(self, source: str, chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
        """Read `source` (a local path or http(s) URL), extract its text, and return it as chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        name, raw_bytes = self._read(source)
        text = self._extract_text(name, raw_bytes)
        return self._split_into_chunks(text, chunk_size, overlap, source=source)

    def _read(self, source: str):
        """Return (file_name, raw_bytes) for a local path or http(s) URL."""
        scheme = urlparse(source).scheme.lower()

        if scheme in ("http", "https"):
            import requests
            response = requests.get(source, timeout=self.timeout)
            response.raise_for_status()
            name = Path(urlparse(source).path).name or "document.txt"
            return name, response.content

        if scheme in ("gs", "s3", "az", "abfs"):
            raise ValueError(
                f"Cloud storage URIs ({scheme}://) are no longer supported. "
                "Download the file locally and pass its path to --file, "
                "or pass a public/pre-signed https:// URL instead."
            )

        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        return path.name, path.read_bytes()

    def _extract_text(self, file_name: str, raw: bytes) -> str:
        extension = file_name.rsplit(".", 1)[-1].lower()
        if extension == "pdf":
            return self._extract_pdf_text(raw)
        elif extension == "docx":
            return self._extract_docx_text(raw)
        else:
            # txt, csv, and anything else: treat as plain text
            return raw.decode("utf-8", errors="replace")

    def _extract_pdf_text(self, raw: bytes) -> str:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    def _extract_docx_text(self, raw: bytes) -> str:
        import docx
        doc = docx.Document(io.BytesIO(raw))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _split_into_chunks(self, text: str, size: int, overlap: int, source: str) -> List[Chunk]:
        """
        Split `text` into chunks of `size` words, each chunk overlapping
        the previous one by `overlap` words (so context isn't lost at
        chunk boundaries).
        """
        words = text.split()
        chunks = []
        start = 0
        index = 0

        while start < len(words):
            end = min(start + size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, index=index, source=source))
            start += size - overlap
            index += 1

        return chunks

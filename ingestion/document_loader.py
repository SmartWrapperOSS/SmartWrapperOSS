# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
ingestion/document_loader.py

Downloads a file from Google Cloud Storage, extracts its text, and splits
it into overlapping chunks.

This is only used by the Summarization workflow. The Tool-Use workflow
doesn't load documents — it loads benchmark task definitions instead (see
workflows/tool_use/tasks.py). That's expected: not every workflow needs
every piece of infrastructure, and that's fine.
"""

import io
import os
from dataclasses import dataclass
from typing import List

from google.cloud import storage


@dataclass
class Chunk:
    """One piece of a document, after splitting on word boundaries."""
    text: str
    index: int
    source: str


class DocumentLoader:
    def __init__(self, credentials_path: str = None):
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = storage.Client()

    def load(self, gcs_uri: str, chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
        """Download `gcs_uri`, extract its text, and return it as chunks."""
        bucket_name, blob_name = self._parse_uri(gcs_uri)
        raw_bytes = self._download(bucket_name, blob_name)
        text = self._extract_text(blob_name, raw_bytes)
        return self._split_into_chunks(text, chunk_size, overlap, source=gcs_uri)

    def _parse_uri(self, uri: str):
        assert uri.startswith("gs://"), "URI must start with gs://"
        bucket_name, blob_name = uri[5:].split("/", 1)
        return bucket_name, blob_name

    def _download(self, bucket_name: str, blob_name: str) -> bytes:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()

    def _extract_text(self, blob_name: str, raw: bytes) -> str:
        extension = blob_name.rsplit(".", 1)[-1].lower()
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

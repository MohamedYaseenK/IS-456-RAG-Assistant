"""
ingest.py
---------
Load the IS 456 PDF, clean text, and chunk it using LangChain's
RecursiveCharacterTextSplitter (per-page, to preserve page_no metadata).
"""

import re
import json
import argparse

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Collapse excessive whitespace/newlines from OCR'd text into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_all_pages(pdf_path: str) -> list[dict]:
    """Open the PDF and return cleaned text per page."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({
            "page_no": i + 1,
            "text": clean_text(page.get_text())
        })
    doc.close()
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Split each page's text into overlapping chunks, preserving page_no metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0
    for p in pages:
        splits = splitter.split_text(p["text"])
        for s in splits:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "page_no": p["page_no"],
                "text": s.strip()
            })
            chunk_id += 1
    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)


def run_ingestion(pdf_path: str, output_path: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Full ingestion pipeline: load -> chunk -> save. Returns the chunks."""
    pages = load_all_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    save_chunks(chunks, output_path)
    print(f"Ingested {len(pages)} pages -> {len(chunks)} chunks -> {output_path}")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/raw.pdf")
    parser.add_argument("--output", default="data/processed/chunks.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    run_ingestion(args.input, args.output, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
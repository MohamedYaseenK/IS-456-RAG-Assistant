"""
ingest.py
---------
Responsible for turning data/raw/raw.pdf into clean, clause-aware chunks
ready for embedding.

Pipeline:
    load_pdf()        -> raw text per page
    clean_text()       -> strip OCR noise, normalize whitespace
    detect_clauses()    -> find clause boundaries (e.g. "26.5.3.2", "8.2.4")
    build_chunks()      -> group text into chunks WITH metadata
    save_chunks()       -> persist to data/processed/chunks.json

Run standalone:
    python -m src.ingest --input data/raw/raw.pdf --output data/processed/chunks.json
"""

import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List

import fitz  # PyMuPDF


@dataclass
class Chunk:
    """A single retrievable unit of the document."""
    chunk_id: str          # e.g. "chunk_0042"
    clause_no: str          # e.g. "26.5.3.2"  (empty string if not detected)
    section_title: str      # nearest heading above this clause, if any
    page_no: int            # 1-indexed PDF page number
    text: str                # the actual chunk text sent to the embedder


def load_pdf(pdf_path: str) -> List[dict]:
    """
    Open the PDF and extract raw text per page.

    Return format: [{"page_no": 1, "text": "..."}, {"page_no": 2, "text": "..."}, ...]

    TODO:
        - open pdf_path with fitz.open()
        - iterate pages, call page.get_text()
        - IMPORTANT: page.get_text() vs page.get_text("blocks") -
          try both and see which preserves clause-number line breaks better.
          IS 456 clause numbers usually sit at the start of a line
          (e.g. "26.5.3.2 If a change in direction..."), so whichever
          extraction mode keeps that structure intact is what you want.
        - return list of dicts as described above
    """
    raise NotImplementedError


def clean_text(text: str) -> str:
    """
    Basic OCR noise cleanup. Do NOT over-clean - some noise is harmless,
    aggressive regex can silently delete real content (numbers, symbols).

    Things worth handling (inspect the raw text first, don't guess):
        - Repeated whitespace / newlines -> single space
        - Hyphenated line-break words e.g. "reinforce-\\nment" -> "reinforcement"
          (careful: this is risky, OCR text is already messy - test on samples)
        - Stray page-header/footer noise, e.g. repeating "IS 456 : 2000" on
          every page - decide whether to strip this or keep it (keeping it
          is often safer / less destructive than a bad regex)

    TODO: implement, and print before/after on 2-3 sample pages to sanity check.
    """
    raise NotImplementedError


# Clause numbers in IS 456 look like: 5, 5.1, 5.1.2, 26.5.3.2, 8.2.4.2 etc.
# They appear at the START of a line, followed by a space and then text
# (e.g. "26.5.3.2 If a change in direction...").
# Build this regex yourself by inspecting 5-10 real examples in raw.pdf first.
CLAUSE_PATTERN = re.compile(r"TODO_WRITE_THIS_REGEX_YOURSELF")


def detect_clauses(page_text: str) -> List[dict]:
    """
    Given the text of ONE page, find all clause-number matches and
    split the text into (clause_no, clause_text) segments.

    Think about:
        - What happens to text BEFORE the first clause match on a page?
          (it likely belongs to the previous page's last clause - how will
          you stitch that across page boundaries in build_chunks()?)
        - False positives: table row numbers, dates, or amendment numbers
          (e.g. "AMENDMENT NO. 2") might match your regex by accident.
          How will you filter those out?

    Return: [{"clause_no": "26.5.3.2", "text": "If a change..."}, ...]
    (clause_no can be "" for text with no detected clause heading)

    TODO: implement using CLAUSE_PATTERN.finditer()
    """
    raise NotImplementedError


def build_chunks(pages: List[dict], max_chunk_chars: int = 1200) -> List[Chunk]:
    """
    Turn per-page clause segments into final Chunk objects.

    Key design decisions YOU need to make (this is the core of the assignment):
        1. Should one chunk = one clause, even if some clauses are only
           1 line long (e.g. "26.5.1 General")? Or should short adjacent
           clauses be merged up to max_chunk_chars?
        2. What do you do with clauses that are LONGER than max_chunk_chars
           (some clauses in IS 456, e.g. 26.5.3.2, are long)? Split them?
           On what boundary (sentence? sub-clause?) if so?
        3. How do you track section_title? (e.g. everything under
           "SECTION 5 STRUCTURAL DESIGN (LIMIT STATE METHOD)" should carry
           that as metadata until the next SECTION heading appears)

    Return a list of Chunk objects, each with a unique chunk_id.

    TODO: implement
    """
    raise NotImplementedError


def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """
    Serialize chunks to JSON (list of dicts) at output_path.
    TODO: implement (hint: [asdict(c) for c in chunks], json.dump)
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/raw.pdf")
    parser.add_argument("--output", default="data/processed/chunks.json")
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    args = parser.parse_args()

    pages = load_pdf(args.input)
    pages = [{"page_no": p["page_no"], "text": clean_text(p["text"])} for p in pages]
    chunks = build_chunks(pages, max_chunk_chars=args.max_chunk_chars)
    save_chunks(chunks, args.output)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
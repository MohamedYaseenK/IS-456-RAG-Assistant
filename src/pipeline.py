"""
pipeline.py
-----------
Ties ingestion, retrieval, and generation together into one callable RAG pipeline.
"""

import json
import os

from src.ingest import run_ingestion
from src.retrieve import Retriever
from src.generate import Generator


class RAGPipeline:
    def __init__(self, chunks_path: str = "data/processed/chunks.json", chroma_path: str = "data/processed/chroma_db"):
        self.chunks_path = chunks_path
        self.retriever = Retriever(chroma_path=chroma_path)
        self.generator = Generator()

    def build(self, pdf_path: str = "data/raw/raw.pdf") -> None:
        """Run ingestion + indexing from scratch. Call this once, or when the source PDF changes."""
        if not os.path.exists(self.chunks_path):
            chunks = run_ingestion(pdf_path, self.chunks_path)
        else:
            with open(self.chunks_path) as f:
                chunks = json.load(f)
        self.retriever.build_index(chunks)

    def ask(self, query: str, top_k: int = 5) -> dict:
        """Run the full RAG flow for a single query."""
        retrieved_chunks = self.retriever.search(query, top_k=top_k)
        answer = self.generator.generate(query, retrieved_chunks)
        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_chunks
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.build()
    result = pipeline.ask("What is the minimum grade of concrete for severe exposure condition?")
    print(result["answer"])
"""
retrieve.py
-----------
Embed chunks with a local sentence-transformers model, store in ChromaDB,
and expose a search() function for nearest-neighbor retrieval.
"""

import json
import argparse

import chromadb
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Retriever:
    def __init__(self, chroma_path: str = "data/processed/chroma_db", collection_name: str = "is456_chunks"):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def build_index(self, chunks: list[dict]) -> None:
        """Embed all chunks and add them to the ChromaDB collection."""
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)

        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"page_no": c["page_no"]} for c in chunks]
        )
        print(f"Indexed {self.collection.count()} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant chunks for a query."""
        query_embedding = self.model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [
            {"text": doc, "page_no": meta["page_no"], "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/chunks.json")
    parser.add_argument("--chroma-path", default="data/processed/chroma_db")
    args = parser.parse_args()

    with open(args.chunks) as f:
        chunks = json.load(f)

    retriever = Retriever(chroma_path=args.chroma_path)
    retriever.build_index(chunks)


if __name__ == "__main__":
    main()
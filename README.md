# IS 456 RAG Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about **IS 456:2000**, the Indian Standard code for Plain and Reinforced Concrete — built end-to-end on a fully free tech stack as a portfolio project.

**Live demo:** _[(https://is456rag.streamlit.app)]_

---

## Overview

This project retrieves relevant clauses from IS 456:2000 and generates grounded, page-cited answers using an LLM. It was built to explore practical RAG engineering problems — OCR-noisy source documents, chunking strategy tradeoffs, retrieval evaluation, and free-tier LLM constraints — rather than to be a polished production tool.

**Example query:**
> "What is the minimum grade of concrete for severe exposure condition?"

> The minimum grade of concrete for severe exposure condition is **M20** for plain concrete and **M30** for reinforced concrete, as per Table 5. *(Page 20)*

---

## Architecture

```
raw.pdf (OCR'd scan)
      │
      ▼
 ingest.py      PyMuPDF text extraction → clean → RecursiveCharacterTextSplitter
      │         (800 char chunks, 150 overlap, per-page metadata)
      ▼
 retrieve.py    BAAI/bge-small-en-v1.5 embeddings → ChromaDB (cosine similarity)
      │
      ▼
 generate.py    Groq (Llama 3.1 8B) primary → Gemini fallback
      │         Prompt grounds answer strictly in retrieved context + cites page numbers
      ▼
 pipeline.py    Orchestrates ingest → retrieve → generate
      │
      ▼
 app.py         Streamlit chat UI, session-limited (10 queries/session)
```

| Component | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, reliable text + layout extraction |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits on natural boundaries before hard-cutting; per-page to preserve metadata |
| Embeddings | `BAAI/bge-small-en-v1.5` | Free, local, CPU-friendly, strong retrieval quality for size |
| Vector store | ChromaDB (persistent, local) | Free, zero-config, no external service |
| LLM | Groq (Llama 3.1 8B Instant), Gemini 2.0 Flash fallback | Free tiers, fast inference, automatic fallback on quota/errors |
| Frontend | Streamlit | Fast to build, good for demo/portfolio |

---

## Setup

```bash
git clone <this-repo>
cd is456-rag-chatbot
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):
```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key   # optional, used as fallback
```

Build the index and run the app:
```bash
python -m src.pipeline     # one-time: ingest PDF + build vector index
streamlit run app.py
```

---

## Evaluation

Retrieval and generation quality are evaluated with **RAGAS** against a hand-built set of IS 456 question–answer pairs (`data/eval/qa_test_set.json`), covering:
- Prose/definitional questions (e.g. clause explanations)
- Numeric/table lookups (e.g. Table 5 exposure conditions)
- Multi-clause reasoning questions

| Metric | Score |
|---|---|
| Faithfulness | _TBD_ |
| Answer Relevancy | _TBD_ |
| Context Precision | _TBD_ |
| Context Recall | _TBD_ |

_(Run `python scripts/run_eval.py` to regenerate these numbers.)_

---

## Known Limitations

- **OCR noise:** the source PDF is an older scan; OCR introduces digit/letter misreads (e.g. "5" → "1") and merged words in some sections. This is a source data limitation, not a pipeline bug.
- **Table retrieval is weaker than prose retrieval.** Tables (e.g. Table 3, 5, 19, 27) lose row/column structure during plain-text extraction, and fixed-size chunking doesn't respect table boundaries. Measured impact is documented in the evaluation results above. A table-aware extraction pass (e.g. via `pdfplumber`/`camelot`) is a planned v2 improvement.
- **Free-tier LLM quota:** the deployed demo runs on free API tiers and is intentionally session-limited (10 queries/session) to preserve availability. This is a conscious cost tradeoff, not an oversight — see `app.py`.
- **Clause-level citation** (e.g. "per Clause 26.5.1.2") is not currently implemented; citations are at the page-number level. Clause-aware chunking was attempted and shelved after diminishing returns on OCR'd table-of-contents parsing — documented as a deliberate scope decision.

---

## Project Structure

```
├── data/
│   ├── raw/              # source PDF
│   ├── processed/        # chunks.json, chroma_db/
│   └── eval/             # RAGAS question-answer test set
├── src/
│   ├── ingest.py         # PDF parsing + chunking
│   ├── retrieve.py       # embeddings + vector search
│   ├── generate.py       # LLM prompt + call (with fallback)
│   └── pipeline.py       # orchestration
├── scripts/
│   └── run_eval.py       # RAGAS evaluation runner
├── app.py                # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Table-aware chunk extraction
- [ ] Clause-number citation (not just page number)
- [ ] Hybrid search (BM25 + dense) with reranking
- [ ] FastAPI backend + Docker deployment
- [ ] MLflow experiment tracking for chunking/retrieval parameter sweeps

---

## Disclaimer

This is a portfolio/learning project, not a certified engineering reference. Always consult the official IS 456:2000 standard and a qualified structural engineer for actual design work.

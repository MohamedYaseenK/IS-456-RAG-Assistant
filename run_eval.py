"""
run_eval.py
-----------
Runs the RAG pipeline against data/eval/qa_test_set.json and scores it
with RAGAS (faithfulness, answer relevancy, context precision, context recall).

Usage:
    python scripts/run_eval.py
"""

import json
import os
import pandas as pd
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.pipeline import RAGPipeline


def load_eval_set(path: str = "data/eval/qa_test_set.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_on_eval_set(pipeline: RAGPipeline, eval_set: list[dict]) -> dict:
    """
    Run every eval question through the RAG pipeline and collect the
    fields RAGAS needs: question, generated answer, retrieved contexts,
    and the ground truth answer.
    """
    questions, answers, contexts, ground_truths, categories = [], [], [], [], []

    for item in eval_set:
        result = pipeline.ask(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([src["text"] for src in result["sources"]])  # RAGAS wants a list of context strings per question
        ground_truths.append(item["ground_truth"])
        categories.append(item["category"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "category": categories,  # kept for our own breakdown, not used by RAGAS itself
    }




def main():
    eval_set = load_eval_set()[:5]  # test with 5 questions first
    print(f"Loaded {len(eval_set)} eval questions")

    pipeline = RAGPipeline()
    pipeline.build()

    data = run_pipeline_on_eval_set(pipeline, eval_set)
    categories = data.pop("category")

    dataset = Dataset.from_dict(data)

    # RAGAS needs its OWN judge LLM + embeddings - point both at free options
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    from ragas.run_config import RunConfig

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(
            max_workers=1,      # one request at a time - slowest but safest against TPM limits
            timeout=180,
        ),
    )

    df = result.to_pandas()
    df["category"] = categories

    print("\n=== Overall RAGAS Scores ===")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    print("\n=== Scores by Category ===")
    print(df.groupby("category")[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    df.to_csv("data/eval/ragas_results.csv", index=False)
    print("\nSaved detailed results to data/eval/ragas_results.csv")

if __name__ == "__main__":
    main()
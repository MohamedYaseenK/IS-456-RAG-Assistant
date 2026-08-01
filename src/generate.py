"""
generate.py
-----------
Build the RAG prompt and call the LLM (Groq primary, Gemini fallback).
"""

import os
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

try:
    import google.generativeai as genai
    _gemini_available = True
except ImportError:
    _gemini_available = False


SYSTEM_PROMPT_TEMPLATE = """You are an assistant answering questions about IS 456:2000, the Indian Standard code for Plain and Reinforced Concrete.

Use ONLY the context below to answer the question. If the context doesn't contain enough information to answer, say so clearly - do not make up information.

Always cite the page number(s) your answer is based on.

Context:
{context}

Question: {query}

Answer:"""


class Generator:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.gemini_model = None
        if _gemini_available and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    @staticmethod
    def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Page {c['page_no']}]\n{c['text']}"
            for c in retrieved_chunks
        )
        return SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)

    def _call_groq(self, prompt: str, model_name: str = "llama-3.1-8b-instant") -> str:
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        response = self.gemini_model.generate_content(prompt, generation_config={"temperature": 0.1})
        return response.text

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """Generate an answer, trying Groq first, falling back to Gemini on failure."""
        prompt = self.build_prompt(query, retrieved_chunks)
        try:
            return self._call_groq(prompt)
        except Exception as e:
            print(f"Groq failed ({e}), trying Gemini fallback...")
            if self.gemini_model is not None:
                return self._call_gemini(prompt)
            raise
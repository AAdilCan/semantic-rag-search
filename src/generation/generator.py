"""
RAG answer generator using a local Ollama LLM.

Pipeline:
  1. Caller retrieves top-k relevant chunks via Retriever
  2. This module builds a RAG prompt with context + question
  3. Calls local Ollama API (free, no key required)
  4. Returns GenerationResult with answer and cited sources

Requires: `ollama pull llama3.2` (or `ollama pull mistral`) before first run.
Assumes Ollama is running at http://localhost:11434.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import List

from src.retrieval.retriever import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

_PROMPT_TEMPLATE = textwrap.dedent("""
You are a research assistant answering questions about scientific papers.
Use ONLY the context below to answer the question. If the context does not
contain enough information, say so explicitly — do not hallucinate.

Context:
{context}

Question: {question}

Answer (cite paper titles in brackets when quoting or paraphrasing):
""").strip()


@dataclass
class GenerationResult:
    """Structured output from the RAG generator."""

    question: str
    answer: str
    sources: List[str]       # paper titles used as context, deduplicated
    model: str
    n_context_chunks: int
    raw_context: str = field(repr=False, default="")


class RAGGenerator:
    """
    Retrieval-augmented answer generator backed by a local Ollama model.

    Args:
        model: Ollama model tag (e.g. 'llama3.2', 'mistral').
        host: Ollama server base URL.
        max_context_chunks: Maximum retrieved chunks included in the prompt.
        temperature: Sampling temperature — low values reduce hallucination.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        max_context_chunks: int = 5,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.host = host
        self.max_context_chunks = max_context_chunks
        self.temperature = temperature
        self._client = self._build_client()

    def _build_client(self):
        try:
            import ollama
            return ollama.Client(host=self.host)
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Run: pip install ollama"
            ) from e

    def generate(
        self,
        question: str,
        context_results: List[SearchResult],
    ) -> GenerationResult:
        """
        Generate an answer grounded in retrieved chunks.

        Args:
            question: The user's research question.
            context_results: Retrieved SearchResult objects (RAG context).

        Returns:
            GenerationResult with answer text and paper citations.
        """
        chunks = context_results[: self.max_context_chunks]
        context_str = self._build_context(chunks)
        prompt = _PROMPT_TEMPLATE.format(context=context_str, question=question)

        logger.info(
            "Calling Ollama model=%s with %d context chunks", self.model, len(chunks)
        )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            answer = response.message.content.strip()
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise RuntimeError(
                f"Ollama generation failed. Is 'ollama serve' running at {self.host}?\n"
                f"Run: ollama pull {self.model}"
            ) from e

        sources = list(dict.fromkeys(r.title for r in chunks))  # ordered, deduplicated

        return GenerationResult(
            question=question,
            answer=answer,
            sources=sources,
            model=self.model,
            n_context_chunks=len(chunks),
            raw_context=context_str,
        )

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        """Format retrieved chunks into a numbered context block for the prompt."""
        lines = []
        for i, r in enumerate(results, start=1):
            header = f"[{i}] {r.title}"
            if r.authors:
                first_authors = ", ".join(r.authors[:2])
                if len(r.authors) > 2:
                    first_authors += " et al."
                header += f" — {first_authors}"
            lines.append(header)
            lines.append(r.text)
            lines.append("")
        return "\n".join(lines).strip()

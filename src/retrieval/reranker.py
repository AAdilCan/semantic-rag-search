"""
Cross-encoder reranking for improved retrieval precision.

Uses a bi-encoder for initial recall (fast), then cross-encoder for
scoring query+passage pairs jointly (slower, more accurate).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — free, ~80MB, no API key.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.retrieval.retriever import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank SearchResult objects using a cross-encoder model.

    The cross-encoder reads (query, passage) pairs jointly, producing a
    relevance score that's more accurate than bi-encoder cosine similarity
    at the cost of O(n) forward passes per query.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER) -> None:
        logger.info("Loading cross-encoder: %s", model_name)
        self._model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank results by cross-encoder relevance score.

        Args:
            query: The search query string.
            results: Initial bi-encoder retrieval results.
            top_n: Return only top_n after reranking (None = all).

        Returns:
            New list of SearchResult sorted by cross-encoder score, with
            rank fields updated to reflect the new ordering.
        """
        if not results:
            return results

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        reranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        output = []
        for rank, (score, result) in enumerate(reranked, start=1):
            output.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    title=result.title,
                    paper_id=result.paper_id,
                    authors=result.authors,
                    url=result.url,
                    rank=rank,
                    metadata=result.metadata,
                )
            )

        return output[:top_n] if top_n is not None else output

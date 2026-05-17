"""
Retrieval and generation evaluation.

Retrieval metrics:
- Hit@k: fraction of queries where the relevant chunk appears in top-k
- MRR: Mean Reciprocal Rank — rewards finding the answer at rank 1 vs. rank 10

Generation metrics:
- ROUGE-L: longest common subsequence F1 between generated answer and reference text

Synthetic QA generation:
- Selects random chunks as "ground truth" context, forms keyword-based questions.
  No LLM needed for question generation — good enough to measure retrieval recall.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """A synthetic question-answer pair anchored to a specific chunk."""

    question: str
    answer_chunk_id: str
    answer_text: str
    paper_title: str


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation results."""

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    n_queries: int

    def to_dict(self) -> dict:
        return {
            "hit@1": round(self.hit_at_1, 4),
            "hit@3": round(self.hit_at_3, 4),
            "hit@5": round(self.hit_at_5, 4),
            "hit@10": round(self.hit_at_10, 4),
            "mrr": round(self.mrr, 4),
            "n_queries": self.n_queries,
        }


@dataclass
class GenerationMetrics:
    """Generation quality results."""

    rouge_l_f1: float
    n_evaluated: int

    def to_dict(self) -> dict:
        return {
            "rouge_l_f1": round(self.rouge_l_f1, 4),
            "n_evaluated": self.n_evaluated,
        }


def generate_synthetic_qa(
    chunks_jsonl: Path,
    n_pairs: int = 100,
    seed: int = 42,
    min_words: int = 30,
) -> List[QAPair]:
    """
    Build synthetic QA pairs from the chunk corpus.

    Treats each sampled chunk's text as the reference answer and generates a
    keyword-based question from its first 15 words. The chunk_id serves as
    the ground-truth retrieval target.

    Args:
        chunks_jsonl: Path to JSONL file of Chunk objects.
        n_pairs: Number of QA pairs to generate.
        seed: Random seed for reproducibility.
        min_words: Minimum chunk word count to include.
    """
    rng = random.Random(seed)

    eligible = []
    with open(chunks_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("word_count", 0) >= min_words:
                    eligible.append(data)

    if len(eligible) < n_pairs:
        logger.warning(
            "Only %d eligible chunks for %d pairs — using all", len(eligible), n_pairs
        )
        selected = eligible
    else:
        selected = rng.sample(eligible, n_pairs)

    pairs = []
    for ch in selected:
        question = _chunk_to_question(ch["text"], ch["title"])
        pairs.append(
            QAPair(
                question=question,
                answer_chunk_id=ch["chunk_id"],
                answer_text=ch["text"],
                paper_title=ch["title"],
            )
        )
    return pairs


def _chunk_to_question(text: str, title: str) -> str:
    """
    Heuristic question generation from a chunk.

    Extracts the leading topic phrase and wraps it in a question template.
    Lightweight but sufficient for retrieval eval where ground truth is known.
    """
    words = text.split()[:15]
    snippet = " ".join(words).rstrip(".,;:")
    return f"What does the paper '{title}' discuss regarding: {snippet}?"


def evaluate_retrieval(
    retriever,
    qa_pairs: List[QAPair],
    top_k: int = 10,
) -> RetrievalMetrics:
    """
    Run retrieval evaluation over synthetic QA pairs.

    For each pair, checks whether the ground-truth chunk_id appears in
    the top-k retrieved results and at what rank.

    Args:
        retriever: A Retriever instance with a search() method.
        qa_pairs: QA pairs with known answer_chunk_id.
        top_k: Retrieval depth (how many results to fetch per query).

    Returns:
        RetrievalMetrics with Hit@k at four thresholds and MRR.
    """
    hits: dict[int, list[bool]] = {1: [], 3: [], 5: [], 10: []}
    reciprocal_ranks: List[float] = []

    for pair in qa_pairs:
        results = retriever.search(pair.question, top_k=top_k)
        result_ids = [r.chunk_id for r in results]

        rank = None
        for i, cid in enumerate(result_ids, start=1):
            if cid == pair.answer_chunk_id:
                rank = i
                break

        for k in [1, 3, 5, 10]:
            hits[k].append(rank is not None and rank <= k)
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    n = len(qa_pairs)
    return RetrievalMetrics(
        hit_at_1=float(np.mean(hits[1])),
        hit_at_3=float(np.mean(hits[3])),
        hit_at_5=float(np.mean(hits[5])),
        hit_at_10=float(np.mean(hits[10])),
        mrr=float(np.mean(reciprocal_ranks)),
        n_queries=n,
    )


def evaluate_generation(
    generator,
    retriever,
    qa_pairs: List[QAPair],
    top_k: int = 5,
    max_pairs: int = 20,
) -> GenerationMetrics:
    """
    Evaluate RAG generation quality using ROUGE-L against chunk reference text.

    Args:
        generator: A RAGGenerator instance.
        retriever: A Retriever instance for context retrieval.
        qa_pairs: QA pairs with reference answer_text.
        top_k: Context chunks to retrieve per question.
        max_pairs: Cap on pairs to evaluate (LLM calls are slow locally).

    Returns:
        GenerationMetrics with mean ROUGE-L F1 over evaluated pairs.
    """
    from rouge_score import rouge_scorer as rs_module

    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    pairs_to_eval = qa_pairs[:max_pairs]
    rouge_scores = []

    for pair in pairs_to_eval:
        try:
            context = retriever.search(pair.question, top_k=top_k)
            result = generator.generate(pair.question, context)
            score = scorer.score(pair.answer_text, result.answer)
            rouge_scores.append(score["rougeL"].fmeasure)
        except Exception as e:
            logger.warning("Generation failed for '%s': %s", pair.question[:60], e)

    return GenerationMetrics(
        rouge_l_f1=float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        n_evaluated=len(rouge_scores),
    )

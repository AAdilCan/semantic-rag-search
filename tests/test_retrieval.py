"""Tests for the retrieval module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embeddings.index import IndexManifest, VectorIndex
from src.preprocessing.chunker import Chunk
from src.retrieval.retriever import Retriever, SearchResult


def _make_manifest(strategy: str = "sentence") -> IndexManifest:
    return IndexManifest(
        index_type="FlatIP",
        n_vectors=10,
        dimension=384,
        model_name="all-MiniLM-L6-v2",
        strategy=strategy,
        build_time_sec=0.001,
        n_lists=None,
        n_probe=None,
    )


def _make_mock_index(chunk_ids: list[str], scores: list[float] | None = None) -> VectorIndex:
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(chunk_ids))]
    mock = MagicMock(spec=VectorIndex)
    mock.search.return_value = [list(zip(chunk_ids, scores))]
    mock.manifest = _make_manifest()
    return mock


def _make_chunk(chunk_id: str, text: str = "Sample text about transformer models.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id="paper-001",
        title="Test Paper on Transformers",
        text=text,
        chunk_index=0,
        strategy="sentence",
        word_count=len(text.split()),
        metadata={
            "authors": ["Author A", "Author B"],
            "url": "https://arxiv.org/abs/test",
            "categories": ["cs.CL"],
            "published": "2023-01-01",
        },
    )


def _make_retriever(chunk_ids: list[str], scores: list[float] | None = None) -> Retriever:
    mock_index = _make_mock_index(chunk_ids, scores)
    chunk_store = {cid: _make_chunk(cid) for cid in chunk_ids}
    mock_encoder = MagicMock()
    mock_encoder.encode_queries.return_value = np.zeros((1, 384), dtype=np.float32)
    return Retriever(mock_index, chunk_store, mock_encoder)


# --- Basic retrieval ---

def test_search_returns_correct_count():
    retriever = _make_retriever(["c1", "c2", "c3"])
    results = retriever.search("transformer attention", top_k=3)
    assert len(results) == 3


def test_search_results_are_search_result_objects():
    retriever = _make_retriever(["c1", "c2"])
    results = retriever.search("query", top_k=2)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_scores_are_descending():
    scores = [0.9, 0.7, 0.5]
    retriever = _make_retriever(["c1", "c2", "c3"], scores)
    results = retriever.search("query", top_k=3)
    result_scores = [r.score for r in results]
    assert result_scores == sorted(result_scores, reverse=True)


def test_search_ranks_start_at_one():
    retriever = _make_retriever(["c1", "c2", "c3"])
    results = retriever.search("query", top_k=3)
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3


# --- Edge cases ---

def test_missing_chunk_in_store_is_skipped():
    mock_index = _make_mock_index(["c1", "c2", "c3"])
    chunk_store = {"c1": _make_chunk("c1"), "c3": _make_chunk("c3")}  # c2 missing
    mock_encoder = MagicMock()
    mock_encoder.encode_queries.return_value = np.zeros((1, 384), dtype=np.float32)

    retriever = Retriever(mock_index, chunk_store, mock_encoder)
    results = retriever.search("query", top_k=3)

    assert len(results) == 2
    assert all(r.chunk_id != "c2" for r in results)


def test_empty_store_returns_no_results():
    mock_index = _make_mock_index(["c1"])
    mock_encoder = MagicMock()
    mock_encoder.encode_queries.return_value = np.zeros((1, 384), dtype=np.float32)

    retriever = Retriever(mock_index, {}, mock_encoder)
    results = retriever.search("query")
    assert results == []


def test_metadata_fields_populated():
    retriever = _make_retriever(["c1"])
    results = retriever.search("query", top_k=1)

    r = results[0]
    assert r.paper_id == "paper-001"
    assert r.title == "Test Paper on Transformers"
    assert isinstance(r.authors, list)
    assert r.url.startswith("https://")


# --- Evaluator helpers ---

def test_generate_synthetic_qa_returns_pairs(tmp_path):
    import json
    from src.evaluation.evaluator import generate_synthetic_qa

    chunks_file = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "chunk_id": f"paper-001-sentence-{i}",
            "paper_id": "paper-001",
            "title": "Test Paper",
            "text": " ".join(["word"] * 40),
            "chunk_index": i,
            "strategy": "sentence",
            "word_count": 40,
            "metadata": {"authors": [], "url": "", "categories": [], "published": "2023-01-01"},
        }
        for i in range(20)
    ]
    with open(chunks_file, "w") as f:
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")

    pairs = generate_synthetic_qa(chunks_file, n_pairs=10)
    assert len(pairs) == 10
    assert all(p.answer_chunk_id.startswith("paper-001") for p in pairs)


def test_evaluate_retrieval_hit_at_k():
    from src.evaluation.evaluator import QAPair, evaluate_retrieval

    pair = QAPair(
        question="test question",
        answer_chunk_id="c1",
        answer_text="test answer",
        paper_title="Test",
    )

    retriever = _make_retriever(["c1", "c2", "c3"])
    metrics = evaluate_retrieval(retriever, [pair], top_k=10)

    assert metrics.hit_at_1 == 1.0  # c1 is at rank 1
    assert metrics.mrr == 1.0
    assert metrics.n_queries == 1

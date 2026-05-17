"""Tests for the chunking module."""

import pytest

from src.corpus.downloader import Paper
from src.preprocessing.chunker import (
    ChunkerConfig,
    ChunkStrategy,
    chunk_paper,
    fixed_window_chunks,
    paragraph_chunks,
    sentence_chunks,
)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        paper_id="test-001",
        title="Deep Learning for NLP",
        abstract=(
            "Deep learning has transformed natural language processing over the last decade. "
            "Transformer architectures enable parallel processing of sequences efficiently. "
            "Pre-training on large corpora transfers well to downstream tasks. "
            "Fine-tuning requires only a small number of labeled examples. "
            "We survey these developments and identify open research directions."
        ),
        authors=["Test Author"],
        categories=["cs.CL"],
        published="2023-01-01",
        url="https://arxiv.org/abs/test-001",
    )


def test_sentence_chunks_produces_output(paper):
    cfg = ChunkerConfig(max_sentences_per_chunk=2)
    chunks = list(sentence_chunks(paper, cfg))
    assert len(chunks) > 0


def test_sentence_chunk_ids_are_unique(paper):
    chunks = list(sentence_chunks(paper, ChunkerConfig()))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_window_produces_output(paper):
    cfg = ChunkerConfig(window_words=20, stride_words=10)
    chunks = list(fixed_window_chunks(paper, cfg))
    assert len(chunks) > 0


def test_fixed_window_overlap(paper):
    """Consecutive chunks should share words when stride < window."""
    cfg = ChunkerConfig(window_words=30, stride_words=15)
    chunks = list(fixed_window_chunks(paper, cfg))
    if len(chunks) >= 2:
        words_a = set(chunks[0].text.split())
        words_b = set(chunks[1].text.split())
        assert len(words_a & words_b) > 0


def test_paragraph_chunks_produces_output(paper):
    chunks = list(paragraph_chunks(paper, ChunkerConfig()))
    assert len(chunks) >= 1


def test_chunk_strategy_dispatch_all_strategies(paper):
    for strategy in ChunkStrategy:
        chunks = chunk_paper(paper, strategy)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1, f"No chunks for strategy {strategy}"


def test_chunk_metadata_preserved(paper):
    chunks = chunk_paper(paper, ChunkStrategy.SENTENCE)
    for chunk in chunks:
        assert chunk.paper_id == paper.paper_id
        assert chunk.title == paper.title
        assert chunk.word_count > 0
        assert isinstance(chunk.metadata, dict)


def test_empty_abstract_does_not_crash():
    short_paper = Paper(
        paper_id="empty",
        title="Title",
        abstract="",
        authors=[],
        categories=[],
        published="2023-01-01",
        url="",
    )
    for strategy in ChunkStrategy:
        result = chunk_paper(short_paper, strategy)
        assert isinstance(result, list)


def test_min_chunk_words_filter(paper):
    cfg = ChunkerConfig(min_chunk_words=10_000)  # impossibly high
    chunks = chunk_paper(paper, ChunkStrategy.FIXED_WINDOW, cfg)
    assert len(chunks) == 0


def test_chunk_id_format(paper):
    chunks = chunk_paper(paper, ChunkStrategy.SENTENCE)
    for chunk in chunks:
        assert chunk.paper_id in chunk.chunk_id
        assert "sentence" in chunk.chunk_id


def test_word_count_matches_text(paper):
    chunks = chunk_paper(paper, ChunkStrategy.FIXED_WINDOW)
    for chunk in chunks:
        actual = len(chunk.text.split())
        assert abs(actual - chunk.word_count) <= 1  # allow off-by-one from metadata

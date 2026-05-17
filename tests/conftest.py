"""Shared pytest fixtures."""

import pytest

from src.corpus.downloader import Paper


@pytest.fixture
def sample_papers() -> list[Paper]:
    return [
        Paper(
            paper_id="2301.00001",
            title="Attention is all you need revisited",
            abstract=(
                "We revisit the transformer architecture and propose several improvements. "
                "Our model achieves state-of-the-art results on multiple benchmarks. "
                "Extensive experiments validate our design choices. "
                "The key innovation is a modified multi-head attention mechanism."
            ),
            authors=["Alice Smith", "Bob Jones"],
            categories=["cs.CL", "cs.LG"],
            published="2023-01-01",
            url="https://arxiv.org/abs/2301.00001",
        ),
        Paper(
            paper_id="2301.00002",
            title="Self-supervised learning for NLP",
            abstract=(
                "Self-supervised learning has revolutionized natural language processing. "
                "We survey recent advances and identify open challenges in the field. "
                "We also propose a new framework for evaluating self-supervised models. "
                "Our evaluation covers both semantic and syntactic tasks."
            ),
            authors=["Carol White"],
            categories=["cs.CL"],
            published="2023-01-02",
            url="https://arxiv.org/abs/2301.00002",
        ),
    ]

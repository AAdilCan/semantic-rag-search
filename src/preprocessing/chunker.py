"""
Chunking strategies for RAG document preparation.

Three strategies:
- sentence: split on sentence boundaries (regex-based, no NLTK dep)
- fixed_window: sliding window over words with configurable size + overlap
- paragraph: split on blank-line boundaries

All strategies produce Chunk objects that carry paper metadata forward.
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from src.corpus.downloader import Paper
from src.preprocessing.normalize import normalize_text, word_count


class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    FIXED_WINDOW = "fixed_window"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    chunk_id: str
    paper_id: str
    title: str
    text: str
    chunk_index: int
    strategy: str
    word_count: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


@dataclass
class ChunkerConfig:
    """Configuration for all chunking strategies."""

    # Sentence chunker
    min_sentence_words: int = 5
    max_sentences_per_chunk: int = 4

    # Fixed-window chunker
    window_words: int = 100
    stride_words: int = 50  # overlap = window - stride

    # Paragraph chunker
    min_paragraph_words: int = 10
    max_paragraph_words: int = 400

    # Shared: drop chunks shorter than this
    min_chunk_words: int = 5


# Sentence boundary regex: end punctuation followed by space + capital, or newline
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\"])|(?<=[.!?])\n")


def _split_sentences(text: str) -> list[str]:
    """Naive but dependency-free sentence splitter."""
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_chunks(
    paper: Paper,
    cfg: ChunkerConfig,
) -> Iterator[Chunk]:
    """
    Group consecutive sentences into chunks of up to cfg.max_sentences_per_chunk.

    Short sentences (< min_sentence_words) are merged into the next one to avoid
    single-sentence fragments that are too short to embed meaningfully.
    """
    text = normalize_text(f"{paper.title}. {paper.abstract}")
    sentences = _split_sentences(text)

    buffer: list[str] = []
    chunk_idx = 0

    for sent in sentences:
        if word_count(sent) < cfg.min_sentence_words:
            # Too short — merge into buffer rather than standalone
            if buffer:
                buffer[-1] = buffer[-1] + " " + sent
            else:
                buffer.append(sent)
            continue

        buffer.append(sent)

        if len(buffer) >= cfg.max_sentences_per_chunk:
            chunk_text = " ".join(buffer)
            wc = word_count(chunk_text)
            if wc >= cfg.min_chunk_words:
                yield _make_chunk(paper, chunk_text, chunk_idx, ChunkStrategy.SENTENCE, wc)
                chunk_idx += 1
            buffer = []

    # Flush remaining
    if buffer:
        chunk_text = " ".join(buffer)
        wc = word_count(chunk_text)
        if wc >= cfg.min_chunk_words:
            yield _make_chunk(paper, chunk_text, chunk_idx, ChunkStrategy.SENTENCE, wc)


def fixed_window_chunks(
    paper: Paper,
    cfg: ChunkerConfig,
) -> Iterator[Chunk]:
    """
    Sliding window over words with stride < window to create overlap.

    For arXiv abstracts (~150 words), this produces 2-4 chunks per paper
    with the default 100-word window and 50-word stride.
    """
    text = normalize_text(f"{paper.title}. {paper.abstract}")
    words = text.split()

    if len(words) < cfg.min_chunk_words:
        return

    chunk_idx = 0
    start = 0

    while start < len(words):
        end = min(start + cfg.window_words, len(words))
        chunk_words = words[start:end]
        wc = len(chunk_words)

        if wc >= cfg.min_chunk_words:
            chunk_text = " ".join(chunk_words)
            yield _make_chunk(paper, chunk_text, chunk_idx, ChunkStrategy.FIXED_WINDOW, wc)
            chunk_idx += 1

        if end == len(words):
            break
        start += cfg.stride_words


def paragraph_chunks(
    paper: Paper,
    cfg: ChunkerConfig,
) -> Iterator[Chunk]:
    """
    Split on blank lines (paragraph boundaries).

    For most arXiv abstracts the abstract itself is one paragraph; this strategy
    therefore prepends the title as its own chunk and treats the full abstract
    as a second chunk (unless it contains embedded paragraph breaks).
    """
    # Title is its own micro-chunk for title-based retrieval
    title_wc = word_count(paper.title)
    if title_wc >= cfg.min_chunk_words:
        yield _make_chunk(paper, paper.title, 0, ChunkStrategy.PARAGRAPH, title_wc)

    abstract_normalized = normalize_text(paper.abstract)
    raw_paras = re.split(r"\n\n+", abstract_normalized)
    chunk_idx = 1

    for para in raw_paras:
        para = para.strip()
        wc = word_count(para)

        if wc < cfg.min_paragraph_words:
            continue

        if wc > cfg.max_paragraph_words:
            # Paragraph is very long — split at word boundary
            words = para.split()
            for i in range(0, len(words), cfg.max_paragraph_words):
                sub = " ".join(words[i : i + cfg.max_paragraph_words])
                sub_wc = word_count(sub)
                if sub_wc >= cfg.min_chunk_words:
                    yield _make_chunk(paper, sub, chunk_idx, ChunkStrategy.PARAGRAPH, sub_wc)
                    chunk_idx += 1
        else:
            yield _make_chunk(paper, para, chunk_idx, ChunkStrategy.PARAGRAPH, wc)
            chunk_idx += 1


def chunk_paper(
    paper: Paper,
    strategy: ChunkStrategy,
    cfg: ChunkerConfig | None = None,
) -> list[Chunk]:
    """Dispatch to the appropriate chunker."""
    if cfg is None:
        cfg = ChunkerConfig()

    dispatch = {
        ChunkStrategy.SENTENCE: sentence_chunks,
        ChunkStrategy.FIXED_WINDOW: fixed_window_chunks,
        ChunkStrategy.PARAGRAPH: paragraph_chunks,
    }
    return list(dispatch[strategy](paper, cfg))


def _make_chunk(
    paper: Paper,
    text: str,
    chunk_idx: int,
    strategy: ChunkStrategy,
    wc: int,
) -> Chunk:
    chunk_id = f"{paper.paper_id}-{strategy.value}-{chunk_idx}"
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper.paper_id,
        title=paper.title,
        text=text,
        chunk_index=chunk_idx,
        strategy=strategy.value,
        word_count=wc,
        metadata={
            "authors": paper.authors[:3],  # store first 3 authors to keep metadata lean
            "categories": paper.categories,
            "published": paper.published,
            "url": paper.url,
        },
    )

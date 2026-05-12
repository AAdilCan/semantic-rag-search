"""
Document loading pipeline: JSONL corpus → normalized + chunked documents.

Usage:
    python -m src.preprocessing.loader \
        --input data/raw/cs_CL_papers.jsonl \
        --strategy sentence \
        --output data/processed/chunks_sentence.jsonl

    python -m src.preprocessing.loader --input data/raw/cs_CL_papers.jsonl --all-strategies
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
from tqdm import tqdm

from src.corpus.downloader import load_jsonl, Paper
from src.preprocessing.chunker import (
    Chunk,
    ChunkerConfig,
    ChunkStrategy,
    chunk_paper,
)
from src.preprocessing.normalize import normalize_paper

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Orchestrates corpus loading → normalization → chunking → serialization.

    Args:
        cfg: Chunker configuration (uses defaults if None).
    """

    def __init__(self, cfg: ChunkerConfig | None = None) -> None:
        self.cfg = cfg or ChunkerConfig()

    def load_and_chunk(
        self,
        input_path: Path,
        strategy: ChunkStrategy,
    ) -> list[Chunk]:
        """
        Load papers from a JSONL file, normalize, and chunk them.

        Returns a flat list of Chunk objects.
        """
        papers = load_jsonl(input_path)
        logger.info("Loaded %d papers from %s", len(papers), input_path)

        chunks: list[Chunk] = []
        skipped = 0

        for paper in tqdm(papers, desc=f"Chunking [{strategy.value}]"):
            normalized = normalize_paper(paper)
            paper_chunks = chunk_paper(normalized, strategy, self.cfg)
            if not paper_chunks:
                skipped += 1
                continue
            chunks.extend(paper_chunks)

        logger.info(
            "Produced %d chunks from %d papers (skipped %d)",
            len(chunks),
            len(papers),
            skipped,
        )
        return chunks

    def save_chunks(self, chunks: list[Chunk], output_path: Path) -> None:
        """Serialize chunks to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        logger.info("Saved %d chunks to %s", len(chunks), output_path)

    def load_chunks(self, path: Path) -> list[Chunk]:
        """Deserialize chunks from JSONL."""
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    chunks.append(Chunk(**data))
        return chunks

    def chunk_stats(self, chunks: list[Chunk]) -> dict:
        """Compute summary statistics over a list of chunks."""
        if not chunks:
            return {}

        wcs = [c.word_count for c in chunks]
        by_strategy: dict[str, int] = {}
        for c in chunks:
            by_strategy[c.strategy] = by_strategy.get(c.strategy, 0) + 1

        papers = {c.paper_id for c in chunks}

        return {
            "total_chunks": len(chunks),
            "unique_papers": len(papers),
            "avg_words_per_chunk": round(sum(wcs) / len(wcs), 1),
            "min_words": min(wcs),
            "max_words": max(wcs),
            "chunks_per_paper": round(len(chunks) / len(papers), 2),
            "by_strategy": by_strategy,
        }


def _print_stats(stats: dict) -> None:
    print("\n--- Chunk Statistics ---")
    print(f"  Total chunks:        {stats['total_chunks']}")
    print(f"  Unique papers:       {stats['unique_papers']}")
    print(f"  Chunks per paper:    {stats['chunks_per_paper']}")
    print(f"  Words per chunk:     avg={stats['avg_words_per_chunk']}  "
          f"min={stats['min_words']}  max={stats['max_words']}")
    if stats.get("by_strategy"):
        print(f"  By strategy:         {stats['by_strategy']}")
    print()


@click.command()
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input JSONL corpus file.",
)
@click.option(
    "--output", "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output JSONL path for chunks. Auto-named if omitted.",
)
@click.option(
    "--strategy",
    default="sentence",
    type=click.Choice([s.value for s in ChunkStrategy]),
    show_default=True,
    help="Chunking strategy.",
)
@click.option(
    "--all-strategies",
    is_flag=True,
    default=False,
    help="Run all three strategies and save each to data/processed/.",
)
@click.option("--window-words", default=100, show_default=True)
@click.option("--stride-words", default=50, show_default=True)
@click.option("--max-sentences", default=4, show_default=True)
def main(
    input_path: Path,
    output_path: Path | None,
    strategy: str,
    all_strategies: bool,
    window_words: int,
    stride_words: int,
    max_sentences: int,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ChunkerConfig(
        window_words=window_words,
        stride_words=stride_words,
        max_sentences_per_chunk=max_sentences,
    )
    loader = DocumentLoader(cfg)

    strategies = list(ChunkStrategy) if all_strategies else [ChunkStrategy(strategy)]

    for strat in strategies:
        if output_path and not all_strategies:
            out = output_path
        else:
            stem = input_path.stem
            out = Path(f"data/processed/{stem}_{strat.value}.jsonl")

        if out.exists():
            logger.info("Output %s already exists, loading for stats…", out)
            chunks = loader.load_chunks(out)
        else:
            chunks = loader.load_and_chunk(input_path, strat)
            loader.save_chunks(chunks, out)

        stats = loader.chunk_stats(chunks)
        _print_stats(stats)


if __name__ == "__main__":
    main()

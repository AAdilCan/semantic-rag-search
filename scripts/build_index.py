"""
Build FAISS vector indices from pre-chunked corpus.

Usage:
    python scripts/build_index.py --strategy fixed_window
    python scripts/build_index.py --strategy sentence --compare
    python scripts/build_index.py --all-strategies
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Prevent tokenizers from spawning extra processes on macOS (causes SIGSEGV with torch+faiss)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.chunker import Chunk
from src.embeddings.encoder import ChunkEncoder, EncoderConfig, EmbeddingResult, save_embeddings
from src.embeddings.index import build_flat_index, build_ivf_index, compare_index_types, IndexConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

DATA_DIR = Path("data/processed")
INDEX_DIR = Path("data/indices")
EMBED_DIR = Path("data/embeddings")
REPORTS_DIR = Path("reports")

STRATEGIES = ["fixed_window", "sentence", "paragraph"]


def load_chunks(strategy: str) -> list[Chunk]:
    path = DATA_DIR / f"cs_CL_papers_{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}. Run the preprocessing step first.")

    chunks = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    logger.info("Loaded %d chunks (%s strategy)", len(chunks), strategy)
    return chunks


def build_and_save(
    strategy: str,
    model_name: str,
    batch_size: int,
    run_compare: bool,
) -> None:
    chunks = load_chunks(strategy)

    encoder = ChunkEncoder(EncoderConfig(model_name=model_name, batch_size=batch_size))
    result: EmbeddingResult = encoder.encode(chunks)

    # Persist raw embeddings for reuse
    emb_out = EMBED_DIR / strategy
    save_embeddings(result, emb_out)

    # Build FlatIP index (always — ground truth)
    flat = build_flat_index(result, strategy)
    flat_dir = INDEX_DIR / strategy / "flat"
    flat.save(flat_dir)

    # Build IVFFlat index
    ivf_cfg = IndexConfig(n_lists=max(4, int(np.sqrt(result.n_chunks))), n_probe=8)
    ivf = build_ivf_index(result, strategy, ivf_cfg)
    ivf_dir = INDEX_DIR / strategy / "ivf"
    ivf.save(ivf_dir)

    if run_compare:
        _run_comparison(result, strategy, encoder)


def _run_comparison(result: EmbeddingResult, strategy: str, encoder: ChunkEncoder) -> None:
    # Sample 50 test queries from chunk texts to benchmark recall/latency
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(result.chunk_ids), size=min(50, len(result.chunk_ids)), replace=False)
    sample_texts = [result.chunk_ids[i] for i in sample_idx]  # use ids as proxy keys

    # Re-encode a random selection of chunk texts as queries
    from src.preprocessing.chunker import Chunk
    import json

    path = DATA_DIR / f"cs_CL_papers_{strategy}.jsonl"
    all_chunks = []
    with path.open() as f:
        for line in f:
            all_chunks.append(json.loads(line))

    query_texts = [all_chunks[i]["text"] for i in sample_idx]
    query_vecs = encoder.encode_queries(query_texts)

    comparison = compare_index_types(result, strategy, query_vecs, top_k=10)

    report = {
        "strategy": strategy,
        "model": result.model_name,
        "n_chunks": result.n_chunks,
        "dimension": result.dimension,
        "flat_latency_ms": comparison["flat_latency_ms"],
        "ivf_latency_ms": comparison["ivf_latency_ms"],
        "ivf_recall_at_10": comparison["ivf_recall_at_k"],
        "n_test_queries": comparison["n_queries"],
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"index_comparison_{strategy}.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        "Comparison (%s): FlatIP %.1fms/q | IVFFlat %.1fms/q | Recall@10=%.3f",
        strategy,
        comparison["flat_latency_ms"],
        comparison["ivf_latency_ms"],
        comparison["ivf_recall_at_k"],
    )
    logger.info("Report saved to %s", report_path)


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(STRATEGIES),
    default="fixed_window",
    show_default=True,
    help="Chunking strategy whose index to build.",
)
@click.option(
    "--all-strategies",
    is_flag=True,
    default=False,
    help="Build indices for all chunking strategies.",
)
@click.option(
    "--model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="Sentence-transformers model name.",
)
@click.option("--batch-size", default=64, show_default=True)
@click.option(
    "--compare",
    is_flag=True,
    default=False,
    help="Benchmark FlatIP vs IVFFlat and save comparison report.",
)
def main(
    strategy: str,
    all_strategies: bool,
    model: str,
    batch_size: int,
    compare: bool,
) -> None:
    """Encode chunks and build FAISS vector indices."""
    strategies = STRATEGIES if all_strategies else [strategy]

    for s in strategies:
        logger.info("=== Building index for strategy: %s ===", s)
        build_and_save(s, model, batch_size, compare)

    logger.info("Done. Indices written to %s", INDEX_DIR)


if __name__ == "__main__":
    main()

"""
FAISS index construction and serialization.

Supports two index types with quantitative comparison:
- FlatIP: exact inner-product search (cosine with L2-normalized vectors)
- IVFFlat: approximate search with inverted file, faster at scale

Index is saved alongside a JSON manifest for reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.embeddings.encoder import EmbeddingResult

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    FLAT_IP = "FlatIP"
    IVF_FLAT = "IVFFlat"


@dataclass
class IndexConfig:
    index_type: IndexType = IndexType.FLAT_IP
    # IVFFlat only
    n_lists: int = 32       # number of Voronoi cells; rule of thumb: sqrt(n_vectors)
    n_probe: int = 8        # cells to scan at query time; higher → better recall, slower


@dataclass
class IndexManifest:
    index_type: str
    n_vectors: int
    dimension: int
    model_name: str
    strategy: str           # chunking strategy used to build this index
    build_time_sec: float
    n_lists: int | None
    n_probe: int | None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        return cls(**json.loads(path.read_text()))


class VectorIndex:
    """Wraps a FAISS index with search and persistence helpers."""

    def __init__(
        self,
        index: faiss.Index,
        chunk_ids: List[str],
        manifest: IndexManifest,
    ) -> None:
        self._index = index
        self.chunk_ids = chunk_ids
        self.manifest = manifest

    @property
    def n_vectors(self) -> int:
        return self._index.ntotal

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Search for top_k nearest chunks for each query vector.

        Returns a list of (chunk_id, score) tuples per query.
        Score is inner-product / cosine similarity when vectors are L2-normalized.
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        query_vectors = query_vectors.astype(np.float32)
        scores, indices = self._index.search(query_vectors, top_k)

        results = []
        for q_scores, q_indices in zip(scores, indices):
            hits = []
            for score, idx in zip(q_scores, q_indices):
                if idx == -1:
                    continue
                hits.append((self.chunk_ids[idx], float(score)))
            results.append(hits)
        return results

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(output_dir / "index.faiss"))

        ids_path = output_dir / "chunk_ids.txt"
        ids_path.write_text("\n".join(self.chunk_ids))

        self.manifest.save(output_dir / "manifest.json")

        logger.info(
            "Saved %s index (%d vectors, dim=%d) to %s",
            self.manifest.index_type,
            self.n_vectors,
            self.manifest.dimension,
            output_dir,
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "index.faiss"))
        chunk_ids = (index_dir / "chunk_ids.txt").read_text().splitlines()
        manifest = IndexManifest.load(index_dir / "manifest.json")
        return cls(index, chunk_ids, manifest)


def build_flat_index(
    result: EmbeddingResult,
    strategy: str,
) -> VectorIndex:
    """Exact inner-product index (fast for small corpora, gold-standard recall)."""
    dim = result.dimension
    t0 = time.perf_counter()

    index = faiss.IndexFlatIP(dim)
    index.add(result.embeddings)

    elapsed = time.perf_counter() - t0
    logger.info("Built FlatIP index: %d vectors in %.2fs", result.n_chunks, elapsed)

    manifest = IndexManifest(
        index_type=IndexType.FLAT_IP.value,
        n_vectors=result.n_chunks,
        dimension=dim,
        model_name=result.model_name,
        strategy=strategy,
        build_time_sec=elapsed,
        n_lists=None,
        n_probe=None,
    )
    return VectorIndex(index, result.chunk_ids, manifest)


def build_ivf_index(
    result: EmbeddingResult,
    strategy: str,
    cfg: IndexConfig | None = None,
) -> VectorIndex:
    """
    Approximate IVFFlat index.

    Clusters vectors into n_lists Voronoi cells during training,
    then searches only n_probe cells at query time.
    Faster than FlatIP at scale; small recall drop if n_probe is too low.
    """
    if cfg is None:
        cfg = IndexConfig(index_type=IndexType.IVF_FLAT)

    dim = result.dimension
    n_vectors = result.n_chunks

    # n_lists should be <= n_vectors; cap at sqrt heuristic
    n_lists = min(cfg.n_lists, max(1, int(np.sqrt(n_vectors))))
    t0 = time.perf_counter()

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)

    logger.info("Training IVFFlat index: n_lists=%d, n_vectors=%d", n_lists, n_vectors)
    index.train(result.embeddings)
    index.add(result.embeddings)
    index.nprobe = cfg.n_probe

    elapsed = time.perf_counter() - t0
    logger.info("Built IVFFlat index: %d vectors in %.2fs", n_vectors, elapsed)

    manifest = IndexManifest(
        index_type=IndexType.IVF_FLAT.value,
        n_vectors=n_vectors,
        dimension=dim,
        model_name=result.model_name,
        strategy=strategy,
        build_time_sec=elapsed,
        n_lists=n_lists,
        n_probe=cfg.n_probe,
    )
    return VectorIndex(index, result.chunk_ids, manifest)


def compare_index_types(
    result: EmbeddingResult,
    strategy: str,
    test_queries: np.ndarray,
    top_k: int = 10,
) -> dict:
    """
    Build both index types and report recall@k + latency.

    Uses FlatIP as ground truth for recall measurement.
    """
    flat = build_flat_index(result, strategy)
    ivf = build_ivf_index(result, strategy)

    # Latency benchmark
    n_queries = len(test_queries)

    t0 = time.perf_counter()
    flat_results = flat.search(test_queries, top_k)
    flat_latency = (time.perf_counter() - t0) / n_queries * 1000  # ms per query

    t0 = time.perf_counter()
    ivf_results = ivf.search(test_queries, top_k)
    ivf_latency = (time.perf_counter() - t0) / n_queries * 1000

    # Recall@k: fraction of FlatIP top-k results that appear in IVF top-k
    recalls = []
    for flat_hits, ivf_hits in zip(flat_results, ivf_results):
        flat_ids = {h[0] for h in flat_hits}
        ivf_ids = {h[0] for h in ivf_hits}
        if flat_ids:
            recalls.append(len(flat_ids & ivf_ids) / len(flat_ids))
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "flat_latency_ms": round(flat_latency, 3),
        "ivf_latency_ms": round(ivf_latency, 3),
        "ivf_recall_at_k": round(mean_recall, 4),
        "top_k": top_k,
        "n_queries": n_queries,
        "flat_index": flat,
        "ivf_index": ivf,
    }

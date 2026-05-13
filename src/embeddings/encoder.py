"""
Batch embedding encoder using sentence-transformers.

Wraps SentenceTransformer with batching, progress reporting,
and L2-normalized output ready for cosine similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EncoderConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 64
    normalize: bool = True  # L2-normalize → dot-product == cosine similarity
    show_progress: bool = True
    device: str = "cpu"


@dataclass
class EmbeddingResult:
    """Embeddings paired with their source chunk IDs."""

    chunk_ids: List[str]
    embeddings: np.ndarray  # shape (n_chunks, dim)
    model_name: str
    dimension: int

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)


class ChunkEncoder:
    """Encode Chunk objects to dense vectors with sentence-transformers."""

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        self.cfg = cfg or EncoderConfig()
        logger.info("Loading model %s on %s", self.cfg.model_name, self.cfg.device)
        self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)

    @property
    def dimension(self) -> int:
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            return self._model.get_sentence_embedding_dimension()

    def encode(self, chunks: List[Chunk]) -> EmbeddingResult:
        """Encode a list of chunks, returning L2-normalized embeddings."""
        if not chunks:
            raise ValueError("chunks list is empty")

        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        logger.info(
            "Encoding %d chunks (batch_size=%d, model=%s)",
            len(texts),
            self.cfg.batch_size,
            self.cfg.model_name,
        )

        embeddings = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=self.cfg.show_progress,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
        )

        return EmbeddingResult(
            chunk_ids=chunk_ids,
            embeddings=embeddings.astype(np.float32),
            model_name=self.cfg.model_name,
            dimension=self.dimension,
        )

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode query strings for retrieval. Returns float32 array (n, dim)."""
        embeddings = self._model.encode(
            queries,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> "ChunkEncoder":
        cfg = EncoderConfig(model_name=model_name, device=device)
        return cls(cfg)


def save_embeddings(result: EmbeddingResult, output_dir: Path) -> dict[str, Path]:
    """Persist embeddings and chunk_ids to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "chunk_ids.txt"

    np.save(str(emb_path), result.embeddings)
    ids_path.write_text("\n".join(result.chunk_ids))

    logger.info("Saved %d embeddings (dim=%d) to %s", result.n_chunks, result.dimension, output_dir)
    return {"embeddings": emb_path, "chunk_ids": ids_path}


def load_embeddings(output_dir: Path) -> EmbeddingResult:
    """Load previously saved embeddings from output_dir."""
    output_dir = Path(output_dir)
    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "chunk_ids.txt"

    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"No embeddings found in {output_dir}")

    embeddings = np.load(str(emb_path))
    chunk_ids = ids_path.read_text().splitlines()

    return EmbeddingResult(
        chunk_ids=chunk_ids,
        embeddings=embeddings.astype(np.float32),
        model_name="unknown",
        dimension=embeddings.shape[1],
    )

"""
Retrieval module: query encoding → FAISS search → metadata hydration.

Loads a VectorIndex and a chunk store, then returns ranked SearchResult
objects that carry both the similarity score and the original chunk text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.embeddings.encoder import ChunkEncoder
from src.embeddings.index import VectorIndex
from src.preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval hit with score and chunk provenance."""

    chunk_id: str
    text: str
    score: float
    title: str
    paper_id: str
    authors: List[str]
    url: str
    rank: int = 0
    metadata: dict = field(default_factory=dict)


class Retriever:
    """
    Wraps a VectorIndex with query encoding and metadata hydration.

    Args:
        index: Loaded VectorIndex (FAISS-backed).
        chunk_store: Dict mapping chunk_id → Chunk for metadata lookup.
        encoder: ChunkEncoder for query encoding.
    """

    def __init__(
        self,
        index: VectorIndex,
        chunk_store: dict[str, Chunk],
        encoder: ChunkEncoder,
    ) -> None:
        self._index = index
        self._store = chunk_store
        self._encoder = encoder

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Retrieve top_k chunks most similar to query.

        Returns SearchResult objects sorted by score descending.
        """
        query_vec = self._encoder.encode_queries([query])
        raw_hits = self._index.search(query_vec, top_k=top_k)[0]

        results = []
        for rank, (chunk_id, score) in enumerate(raw_hits, start=1):
            chunk = self._store.get(chunk_id)
            if chunk is None:
                logger.warning("chunk_id %s not found in store", chunk_id)
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=score,
                    title=chunk.title,
                    paper_id=chunk.paper_id,
                    authors=chunk.metadata.get("authors", []),
                    url=chunk.metadata.get("url", ""),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return results

    @classmethod
    def from_disk(
        cls,
        index_dir: Path,
        chunks_jsonl: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "Retriever":
        """
        Load index + chunk store from disk paths.

        Args:
            index_dir: Path to a saved VectorIndex directory.
            chunks_jsonl: Path to the JSONL file with Chunk objects.
            model_name: sentence-transformers model for query encoding.
            device: torch device string.
        """
        index = VectorIndex.load(index_dir)
        chunk_store = _load_chunk_store(chunks_jsonl)
        encoder = ChunkEncoder.load(model_name=model_name, device=device)
        logger.info(
            "Loaded retriever: %d chunks, index=%s, model=%s",
            len(chunk_store),
            index.manifest.index_type,
            model_name,
        )
        return cls(index, chunk_store, encoder)


def _load_chunk_store(path: Path) -> dict[str, Chunk]:
    """Load JSONL chunks into a dict keyed by chunk_id."""
    store: dict[str, Chunk] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                chunk = Chunk(**data)
                store[chunk.chunk_id] = chunk
    logger.info("Loaded %d chunks from %s", len(store), path)
    return store

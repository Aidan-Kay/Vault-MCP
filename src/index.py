"""The resident index: vector matrix, chunk metadata, and a BM25 model.

No database. At this corpus size a rebuild costs seconds, and a DB would add
migrations, staleness and a failure mode for no benefit.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import chunker, vault
from .config import settings
from .embedder import Embedder

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Deliberately naive: splitting 'nomic-embed-text' into three tokens beats
    keeping it whole, because a query for one part should still match."""
    return _TOKEN_RE.findall(text.lower())


def _bm25_document(chunk: dict) -> list[str]:
    # Title and breadcrumb are indexed alongside the body so a query naming the
    # note matches even when the body never repeats the name.
    return tokenize(f"{chunk['title']} {chunk['breadcrumb']} {chunk['text']}")


@dataclass(slots=True)
class VaultIndex:
    matrix: np.ndarray  # (N, 768) float32, C-contiguous, L2-normalised
    chunks: list[dict]  # parallel to matrix rows
    bm25: BM25Okapi | None = None
    build_seconds: float = 0.0
    note_count: int = 0
    built_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.bm25 is None and self.chunks:
            self.bm25 = BM25Okapi([_bm25_document(c) for c in self.chunks])

    @property
    def size(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        megabytes = self.matrix.nbytes / 1_048_576
        return (
            f"{self.size} chunks from {self.note_count} notes, "
            f"matrix {self.matrix.shape} ({megabytes:.1f} MB), "
            f"built in {self.build_seconds:.1f}s"
        )

    @classmethod
    def empty(cls) -> VaultIndex:
        return cls(matrix=np.zeros((0, settings.embed_dims), dtype=np.float32), chunks=[])

    @classmethod
    async def build(cls, embedder: Embedder) -> VaultIndex:
        started = time.perf_counter()
        notes = vault.walk_notes()

        chunks: list[dict] = []
        for path in notes:
            try:
                chunks.extend(chunker.chunk_note(path))
            except Exception:
                # One malformed note must not take down the whole index.
                log.exception("skipping unchunkable note %s", path)

        matrix = await embedder.embed([c["embed_text"] for c in chunks])
        index = cls(
            matrix=np.ascontiguousarray(matrix),
            chunks=chunks,
            build_seconds=time.perf_counter() - started,
            note_count=len(notes),
        )
        log.info("index built: %s", index.summary())
        return index

    async def replace_note(self, embedder: Embedder, path: Path) -> VaultIndex:
        """Return a new index with one note's rows swapped out.

        Rebuilding the array beats in-place slot management: np.delete plus
        np.vstack over a few megabytes is sub-millisecond, and it keeps chunks
        trivially parallel to matrix with no tombstones or index drift. BM25 has
        to be rebuilt wholesale regardless - rank_bm25 has no incremental update.
        """
        # resolve() is non-strict, so this still yields the right key for a
        # file that has already been deleted.
        rel = vault.relpath(path)

        new_chunks: list[dict] = []
        if path.exists():
            try:
                new_chunks = chunker.chunk_note(path)
            except FileNotFoundError:
                new_chunks = []  # deleted between the event firing and the read
            except Exception:
                log.exception("re-chunk failed for %s; dropping its rows", rel)
                new_chunks = []

        keep = [i for i, chunk in enumerate(self.chunks) if chunk["path"] != rel]
        kept_matrix = self.matrix[keep] if keep else np.zeros(
            (0, settings.embed_dims), dtype=np.float32
        )
        kept_chunks = [self.chunks[i] for i in keep]

        if new_chunks:
            new_matrix = await embedder.embed([c["embed_text"] for c in new_chunks])
            matrix = np.vstack([kept_matrix, new_matrix])
        else:
            matrix = kept_matrix

        note_paths = {c["path"] for c in kept_chunks} | ({rel} if new_chunks else set())
        return VaultIndex(
            matrix=np.ascontiguousarray(matrix),
            chunks=kept_chunks + new_chunks,
            note_count=len(note_paths),
            build_seconds=self.build_seconds,
        )

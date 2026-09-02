"""Batched embedding against Ollama, with L2 normalisation on receipt."""

from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np

from .config import settings

log = logging.getLogger(__name__)

QUERY_PREFIX = "search_query: "


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity collapses to a dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)  # a zero vector must not produce NaN
    return (matrix / norms).astype(np.float32, copy=False)


class Embedder:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._dims: int | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, inputs: list[str]) -> list[list[float]]:
        payload = {"model": settings.embed_model, "input": inputs}
        url = f"{settings.ollama_url}/embeddings"

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()["data"]
                # OpenAI-compatible responses carry an index; do not trust order.
                data.sort(key=lambda row: row.get("index", 0))
                return [row["embedding"] for row in data]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    # The first call after an idle period can race a model load
                    # when OLLAMA_KEEP_ALIVE is unset.
                    log.warning("embed batch failed (%s), retrying once", exc)
                    await asyncio.sleep(2.0)
        raise RuntimeError(f"embedding failed after retry: {last_error}") from last_error

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed pre-scaffolded texts. Returns an (N, dims) normalised float32 array."""
        if not texts:
            return np.zeros((0, settings.embed_dims), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch_size):
            batch = texts[start : start + settings.embed_batch_size]
            embeddings = await self._post(batch)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs"
                )
            if self._dims is None:
                self._dims = len(embeddings[0])
                if self._dims != settings.embed_dims:
                    # A silent dimension change would corrupt the whole index.
                    raise RuntimeError(
                        f"{settings.embed_model} returned {self._dims} dims, "
                        f"expected {settings.embed_dims}"
                    )
                log.info("embedding model %s: %d dims", settings.embed_model, self._dims)
            vectors.extend(embeddings)

        return l2_normalise(np.asarray(vectors, dtype=np.float32))

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with the asymmetric prefix nomic requires."""
        matrix = await self.embed([QUERY_PREFIX + query])
        return matrix[0]

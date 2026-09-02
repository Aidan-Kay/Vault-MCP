"""Environment-derived settings, resolved once at import.

Every knob the container has is here. Nothing else reads os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    vault_path: Path
    api_key: str
    allowed_hosts: tuple[str, ...]
    ollama_url: str
    embed_model: str
    exclude_dirs: frozenset[str]
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_min_tokens: int
    search_default_k: int
    embed_batch_size: int
    embed_dims: int
    watch_debounce_seconds: float
    host: str
    port: int


def load() -> Settings:
    api_key = os.environ.get("VAULT_INDEX_API_KEY", "").strip()
    if not api_key:
        # Fail closed. An empty key must never be read as "auth disabled" for a
        # service that serves finances, insurance and addresses as plain text.
        raise RuntimeError("VAULT_INDEX_API_KEY is unset - refusing to start")

    vault_path = Path(os.environ.get("VAULT_PATH", "/vault")).resolve()
    if not vault_path.is_dir():
        raise RuntimeError(f"VAULT_PATH {vault_path} is not a directory")

    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434/v1").rstrip("/")

    return Settings(
        vault_path=vault_path,
        api_key=api_key,
        allowed_hosts=_csv("MCP_ALLOWED_HOSTS", "vault-index:8080,127.0.0.1:8090"),
        ollama_url=ollama_url,
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        exclude_dirs=frozenset(_csv("EXCLUDE_DIRS", "Workflows,Reports,.obsidian")),
        chunk_target_tokens=_int("CHUNK_TARGET_TOKENS", 400),
        chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 60),
        chunk_min_tokens=_int("CHUNK_MIN_TOKENS", 120),
        search_default_k=_int("SEARCH_DEFAULT_K", 5),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 64),
        embed_dims=_int("EMBED_DIMS", 768),
        watch_debounce_seconds=_float("WATCH_DEBOUNCE_SECONDS", 2.0),
        host=os.environ.get("BIND_HOST", "0.0.0.0"),
        port=_int("BIND_PORT", 8080),
    )


settings = load()

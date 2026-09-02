"""MCP server: tool surface, auth, transport security, startup.

Reads only. Every write stays on obsidian-local-rest-api so Obsidian's own
cache and link graph remain coherent.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from . import search as search_module
from . import vault
from .config import settings
from .embedder import Embedder
from .index import VaultIndex
from .watcher import VaultWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("vault-index")


# --------------------------------------------------------------------------
# Mutable server state
#
# _index is rebound, never mutated. Rebinding is atomic under the GIL, and
# readers capture the reference once per request, so an in-flight search always
# completes against a consistent snapshot.
# --------------------------------------------------------------------------

_index: VaultIndex = VaultIndex.empty()
_embedder: Embedder | None = None
_build_started: float = 0.0
_build_error: str | None = None
_ready = False
_reindex_lock = asyncio.Lock()


def _index_status() -> str:
    if _build_error:
        return f"The vault index failed to build: {_build_error}"
    elapsed = time.monotonic() - _build_started if _build_started else 0
    return (
        f"The vault index is still building ({elapsed:.0f}s elapsed). "
        "Retry shortly, or use vault_list / vault_read, which do not need it."
    )


async def _build_index() -> None:
    global _index, _build_error, _ready
    assert _embedder is not None
    try:
        _index = await VaultIndex.build(_embedder)
        _ready = True
    except Exception as exc:
        _build_error = str(exc)
        log.exception("index build failed")


async def _reindex(path: Path) -> None:
    global _index
    assert _embedder is not None
    async with _reindex_lock:  # serialise rebuilds; each reads the live index
        started = time.perf_counter()
        _index = await _index.replace_note(_embedder, path)
        log.info(
            "reindex %s -> %d chunks in %.0f ms",
            vault.relpath(path),
            _index.size,
            (time.perf_counter() - started) * 1000,
        )


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
    global _embedder, _build_started

    log.info("vault=%s exclude=%s", vault.ROOT, sorted(settings.exclude_dirs))
    _embedder = Embedder()
    _build_started = time.monotonic()

    # Built in the background so vault_read / vault_list / vault_map serve
    # immediately and do not depend on Ollama being up.
    build_task = asyncio.create_task(_build_index(), name="index-build")
    watcher = VaultWatcher(_reindex)

    async def _watch_when_ready() -> None:
        await build_task
        if _ready:
            await watcher.start()

    watch_task = asyncio.create_task(_watch_when_ready(), name="watch-start")

    try:
        yield
    finally:
        for task in (watch_task, build_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await watcher.stop()
        if _embedder is not None:
            await _embedder.aclose()


mcp = MCPServer(
    "vault-index",
    instructions=(
        "Read-only semantic and keyword search over the Obsidian vault. "
        "Prefer vault_search to locate information, then vault_read with "
        "section= to pull only the heading you need. Writes are not available "
        "here - use the Obsidian write tools."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool()
async def vault_search(query: str, k: int | None = None) -> str:
    """Search the vault for a topic, question, or exact term.

    Hybrid semantic + keyword search. Returns ranked excerpts with their source
    path and heading, suitable for citing directly.

    Args:
        query: Natural language question or exact term (a model number, reg
            plate or policy reference all work).
        k: Number of excerpts to return. Defaults to 5.
    """
    index = _index  # snapshot
    if not _ready:
        return _index_status()
    assert _embedder is not None
    limit = max(1, min(k or settings.search_default_k, 20))
    results = await search_module.search(index, _embedder, query, limit)
    return search_module.format_results(query, results)


@mcp.tool()
def vault_read(path: str, section: str | None = None) -> str:
    """Read a note from the vault.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md".
        section: Optional heading name. Returns only that heading's content,
            down to the next heading of equal or shallower depth. Use this
            instead of reading whole notes.
    """
    return vault.read_note(path, section)


@mcp.tool()
def vault_list(path: str = "") -> str:
    """List the contents of a vault directory.

    Args:
        path: Vault-relative directory. Defaults to the vault root.
    """
    entries = vault.list_dir(path)
    if not entries:
        return f"{path or '/'} is empty."
    lines = [f"{len(entries)} entr(ies) in {path or '/'}:"]
    for entry in entries:
        if entry["type"] == "dir":
            lines.append(f"  {entry['name']}/")
        else:
            lines.append(f"  {entry['name']}  ({entry['size']} B, {entry['modified']})")
    return "\n".join(lines)


@mcp.tool()
def vault_map(path: str) -> str:
    """Show a note's frontmatter and heading structure without its content.

    Use this to decide which section to request from vault_read.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md".
    """
    parsed = vault.parse_note(path)
    lines = [f"# {parsed['path']}", "", "## Frontmatter"]
    if parsed["frontmatter"]:
        for key, value in parsed["frontmatter"].items():
            lines.append(f"- {key}: {json.dumps(value, default=str, ensure_ascii=False)}")
    else:
        lines.append("- (none)")
    lines += ["", "## Headings"]
    if parsed["headings"]:
        for heading in parsed["headings"]:
            lines.append(f"{'  ' * (heading['depth'] - 1)}- {heading['text']}  (line {heading['line']})")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# ASGI app: transport security, then auth
# --------------------------------------------------------------------------

app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=list(settings.allowed_hosts),
        allowed_origins=["*"],  # no browser origin - MCP clients only
    ),
)


class BearerAuth:
    """Static shared secret on a private network.

    Non-HTTP scopes pass through untouched so the lifespan still runs and starts
    the session manager.
    """

    def __init__(self, inner, key: str) -> None:
        self.inner = inner
        self.expected = b"Bearer " + key.encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        supplied = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(supplied, self.expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"www-authenticate", b'Bearer realm="vault-index"'),
                        (b"content-type", b"text/plain; charset=utf-8"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await self.inner(scope, receive, send)


app = BearerAuth(app, settings.api_key)


def main() -> None:
    log.info(
        "serving MCP on %s:%d/mcp (allowed hosts: %s)",
        settings.host,
        settings.port,
        ", ".join(settings.allowed_hosts),
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()

"""MCP server: tool surface, REST surface, auth, transport security, startup.

Two interfaces over one implementation. Lyra speaks MCP; n8n's HTTP Request
nodes speak plain REST and cannot easily build a JSON-RPC envelope, so /vault/*
mirrors the shape obsidian-local-rest-api used. Both call src.operations, so the
resolver and every convention are handled once.
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
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from . import operations
from . import search as search_module
from . import target as target_mod
from . import vault
from .config import settings
from .embedder import Embedder
from .index import VaultIndex
from .watcher import VaultWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("vault-mcp")


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
    "vault-mcp",
    instructions=(
        "Semantic and keyword search, reading and writing over the Obsidian "
        "vault. Prefer vault_search to locate information, then vault_read with "
        "section= to pull only the heading you need. To edit, call vault_map "
        "first and patch the '::' path it gives you - a bare heading name works "
        "whenever it is unique, and the error tells you what to prepend when it "
        "is not. Writes bump the note's timestamp for you; updating index.md is "
        "still yours to do. Never probe for a note's existence before writing - "
        "the write tools take the missing case as an argument "
        "(vault_append's create_if_missing, vault_write's overwrite), so a read "
        "or list first only buys a round trip."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _do(fn, *args, **kwargs) -> str:
    """Run a vault operation, surfacing its error message to the model.

    The MCP runtime masks an arbitrary exception as a bare "Error executing tool
    <name>" and only lets a ToolError's message through. Every VaultError here is
    written to be acted on - the ambiguity error lists the exact paths to retry
    with - so masking it would throw away the entire point of the resolver.
    """
    try:
        return fn(*args, **kwargs)
    except vault.VaultError as exc:
        raise ToolError(str(exc)) from exc



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
    return _do(vault.read_note, path, section)


@mcp.tool()
def vault_list(path: str = "") -> str:
    """List the contents of a vault directory.

    Args:
        path: Vault-relative directory. Defaults to the vault root.
    """
    entries = _do(vault.list_dir, path)
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
    parsed = _do(vault.parse_note, path)
    lines = [f"# {parsed['path']}", "", "## Frontmatter"]
    if parsed["frontmatter"]:
        for key, value in parsed["frontmatter"].items():
            lines.append(f"- {key}: {json.dumps(value, default=str, ensure_ascii=False)}")
    else:
        lines.append("- (none)")
    lines += ["", "## Headings", "", "Patch targets. A trailing segment on its own works when it is"]
    lines += ["unique in this note; prepend ancestors with '::' when it is not.", ""]
    text = _do(lambda p: vault.read_text(vault.safe_resolve(p)), path)
    outline = target_mod.outline(text)
    lines += [f"- {entry}" for entry in outline] or ["- (none)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Write tools
#
# Thin wrappers: every one delegates to src.operations, which the REST routes
# below call too. Nothing here holds logic of its own.
# --------------------------------------------------------------------------


@mcp.tool()
def vault_patch(
    path: str,
    target: str,
    operation: str = "replace",
    content: str = "",
    target_scope: str = "content",
) -> str:
    """Edit one section of a note, addressed by its heading.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md".
        target: Heading to act on. A bare name works whenever it is unique in
            the note; otherwise join ancestors with "::", as
            "Cottage Pie::Mash::Method". Call vault_map to see the paths. If
            the target is ambiguous the error lists exactly which paths to
            choose between - re-call with one of them.
        operation: "replace", "prepend" or "append".
        content: The markdown to write.
        target_scope: "content" (default, the section body), "marker" (the
            heading line only) or "markerAndContent" (both).
    """
    return _do(operations.patch, path, target, operation, content, target_scope)


@mcp.tool()
def vault_append(path: str, content: str, create_if_missing: bool = False) -> str:
    """Append a block to the end of a note, creating it if the path is absent.

    Do not read or list first to find out whether the note is there - pass
    create_if_missing=True and this one call covers both cases. A probe
    beforehand costs a whole round trip to answer a question this tool already
    takes as an argument.

    Args:
        path: Vault-relative path.
        content: The markdown to append.
        create_if_missing: Create the note instead of failing when the path is
            absent. A note created this way is written from `content` verbatim -
            no frontmatter and no timestamp are added - so include frontmatter
            in `content` when the note should carry it. Appending to a note that
            already exists bumps its timestamp as usual.
    """
    return _do(operations.append, path, content, create_if_missing)


@mcp.tool()
def vault_write(path: str, content: str, overwrite: bool = False) -> str:
    """Create a note, or replace one wholesale.

    Include frontmatter: type, title, description, tags, timestamp. Prefer
    vault_patch for editing part of an existing note, and vault_append when you
    only want to add to the end - neither needs the note looked up first.

    Args:
        path: Vault-relative path. Parent directories are created as needed.
        content: The complete note.
        overwrite: Required to replace an existing note. Without it an existing
            path is an error, so a create can never silently clobber. Set it
            from your intent rather than from a lookup: pass it when you mean
            "create or replace", leave it off when the note must be new.
    """
    return _do(operations.write, path, content, overwrite)


@mcp.tool()
def vault_set_frontmatter(path: str, key: str, value: str | list | None = None, delete: bool = False) -> str:
    """Set or remove one frontmatter field, leaving the rest of the block alone.

    Args:
        path: Vault-relative path.
        key: Field name, e.g. "description" or "tags". A field that does not
            exist yet is inserted in the order Conventions mandates.
        value: The value. A list for "tags"; for "expires", a list of
            {"date": "YYYY-MM-DD", "what": "..."} entries - pass every entry,
            as the whole block is replaced and a dropped date stops being
            checked silently.
        delete: Remove the field instead of setting it.
    """
    return _do(operations.set_frontmatter, path, key, value, delete)


@mcp.tool()
def vault_delete(path: str) -> str:
    """Delete a note. There is no trash; the vault's git history is the undo.

    Args:
        path: Vault-relative path.
    """
    return _do(operations.delete, path)


@mcp.tool()
def vault_move(source: str, destination: str, update_links: bool = True) -> str:
    """Move or rename a note and repoint every link to it.

    Moving notes changes the vault's structure, so confirm with the user first.
    Update index.md afterwards.

    Args:
        source: Current vault-relative path.
        destination: New vault-relative path. Parent directories are created.
        update_links: Rewrite internal links pointing at the old path.
    """
    return _do(operations.move, source, destination, update_links)


# --------------------------------------------------------------------------
# ASGI app: transport security, then auth
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# REST surface
#
# n8n's HTTP Request nodes send a raw markdown body to a path-shaped URL. Route
# shapes mirror obsidian-local-rest-api so migrating a node is a find-and-replace
# on the URL and the auth header, not a rewrite into JSON-RPC.
# --------------------------------------------------------------------------


async def _body(request: Request) -> str:
    return (await request.body()).decode("utf-8")


async def vault_endpoint(request: Request) -> PlainTextResponse:
    path = request.path_params["path"]
    method = request.method

    try:
        if method == "GET":
            section = request.query_params.get("section")
            return PlainTextResponse(vault.read_note(path, section))
        if method == "PUT":
            return PlainTextResponse(
                operations.write(path, await _body(request), overwrite=True)
            )
        if method == "POST":
            return PlainTextResponse(
                operations.append(path, await _body(request), create_if_missing=True)
            )
        if method == "PATCH":
            heading = request.headers.get("target")
            if not heading:
                raise vault.VaultError("PATCH needs a Target header naming the heading")
            return PlainTextResponse(
                operations.patch(
                    path,
                    heading,
                    request.headers.get("operation", "replace"),
                    await _body(request),
                    request.headers.get("target-scope", "content"),
                )
            )
        if method == "DELETE":
            return PlainTextResponse(operations.delete(path))
    except vault.VaultError as exc:
        # 400, not 500: every one of these is the caller's path or target, and
        # the message is written to be actionable rather than diagnostic.
        return PlainTextResponse(str(exc), status_code=400)

    return PlainTextResponse(f"{method} not supported on /vault", status_code=405)


rest_app = Starlette(
    routes=[
        Route(
            "/vault/{path:path}",
            vault_endpoint,
            methods=["GET", "PUT", "POST", "PATCH", "DELETE"],
        )
    ]
)


class VaultRoutes:
    """Serve /vault/* from the REST app, everything else from the MCP app.

    A wrapper rather than a parent Starlette app so the MCP app keeps owning the
    lifespan that starts the index, the watcher and the session manager. Only
    http scopes are diverted; lifespan and everything else pass straight through.
    """

    def __init__(self, mcp_app, rest) -> None:
        self.mcp_app = mcp_app
        self.rest = rest

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/vault"):
            return await self.rest(scope, receive, send)
        return await self.mcp_app(scope, receive, send)


app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=list(settings.allowed_hosts),
        allowed_origins=["*"],  # no browser origin - MCP clients only
    ),
)

app = VaultRoutes(app, rest_app)


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
                        (b"www-authenticate", b'Bearer realm="vault-mcp"'),
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
        "serving MCP on %s:%d/mcp and REST on %s:%d/vault/<path> (allowed hosts: %s)",
        settings.host,
        settings.port,
        settings.host,
        settings.port,
        ", ".join(settings.allowed_hosts),
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()

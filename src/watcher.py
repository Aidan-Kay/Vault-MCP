"""Filesystem watching with per-path debounce.

inotify was verified to work on this vault before this phase was committed to:
it sits on a fuseblk (NTFS-3G) mount, where delivery is not guaranteed. A probe
watching from inside a container through a :ro bind mount received CREATE,
MODIFY, CLOSE_WRITE and DELETE for writes made from the host and from the
obsidian container's own separate mount of the same directory.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import vault
from .config import settings

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Runs on watchdog's thread. Does nothing but hand paths to the loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Path]) -> None:
        self._loop = loop
        self._queue = queue

    def _submit(self, raw_path: str | bytes) -> None:
        path = Path(raw_path.decode() if isinstance(raw_path, bytes) else raw_path)
        if path.suffix.lower() != ".md":
            return
        try:
            relative = path.resolve().relative_to(vault.ROOT)
        except ValueError:
            return
        if vault.is_index_excluded(relative):
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path)  # source: drop its rows
            self._submit(event.dest_path)  # destination: index it


class VaultWatcher:
    def __init__(self, on_change: Callable[[Path], Awaitable[None]]) -> None:
        self._on_change = on_change
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._observer: Observer | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._observer = Observer()
        self._observer.schedule(_Handler(loop, self._queue), str(vault.ROOT), recursive=True)
        self._observer.start()
        self._task = asyncio.create_task(self._drain(), name="vault-watch")
        log.info("watching %s (debounce %.1fs)", vault.ROOT, settings.watch_debounce_seconds)

    async def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _drain(self) -> None:
        """Coalesce bursts. Obsidian and Samba both emit several write events
        for one logical save; re-embedding on each would hammer Ollama."""
        pending: dict[Path, float] = {}
        debounce = settings.watch_debounce_seconds

        while True:
            timeout = debounce if pending else None
            try:
                path = await asyncio.wait_for(self._queue.get(), timeout)
                pending[path] = time.monotonic()
            except asyncio.TimeoutError:
                pass

            now = time.monotonic()
            settled = [p for p, seen in pending.items() if now - seen >= debounce]
            for path in settled:
                del pending[path]
                try:
                    await self._on_change(path)
                except Exception:
                    log.exception("reindex failed for %s", path)

"""Filesystem access to the vault.

Security-critical: every path argument reaching this module originated in an
LLM tool call and is untrusted. All access funnels through safe_resolve().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from .config import settings


class VaultError(ValueError):
    """Raised for a rejected path or an unreadable note."""


# Resolved once. If VAULT_PATH is itself a symlink, this is the real target,
# which is what every containment check below compares against.
ROOT = settings.vault_path.resolve()

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


@dataclass(frozen=True, slots=True)
class Heading:
    depth: int
    text: str
    line: int  # 1-based, relative to the whole file


def _is_excluded(rel: Path) -> bool:
    """True if any component is an excluded or hidden directory.

    Hidden components are excluded unconditionally: .obsidian, .trash and .git
    are tool state, never curated knowledge, and there is no case for either
    indexing or serving them.
    """
    for part in rel.parts:
        if part in settings.exclude_dirs or part.startswith("."):
            return True
    return False


def safe_resolve(rel_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a vault-relative path, or raise.

    Rejects traversal, absolute escapes, symlinks pointing outside the vault,
    and anything under an excluded directory. Never falls back to a default.
    """
    if "\x00" in rel_path:
        raise VaultError("path contains a null byte")

    # A leading slash is treated as vault-root-relative rather than rejected -
    # models write "/Pets/Levi.md" often enough that failing it is pure
    # friction. Containment is still enforced below, so this is not a shortcut.
    cleaned = rel_path.strip().lstrip("/")

    candidate = (ROOT / cleaned).resolve()

    if candidate != ROOT and not candidate.is_relative_to(ROOT):
        raise VaultError(f"path escapes the vault: {rel_path!r}")

    rel = candidate.relative_to(ROOT)
    if _is_excluded(rel):
        raise VaultError(f"path is in an excluded directory: {rel.as_posix()!r}")

    if must_exist and not candidate.exists():
        raise VaultError(f"no such path in the vault: {rel.as_posix()!r}")

    return candidate


def relpath(path: Path) -> str:
    """Vault-relative POSIX path, for display and chunk metadata."""
    return path.resolve().relative_to(ROOT).as_posix()


def frontmatter_span(text: str) -> int:
    """Number of leading lines occupied by the YAML frontmatter block."""
    match = _FRONTMATTER_RE.match(text)
    return match.group(0).count("\n") if match else 0


def iter_headings(text: str) -> list[Heading]:
    """ATX headings, skipping frontmatter and fenced code blocks.

    Fence tracking matters: the vault is full of bash blocks whose comments
    start with '#', and every one of them would otherwise parse as a heading.
    """
    lines = text.splitlines()
    start = frontmatter_span(text)
    headings: list[Heading] = []
    fence: str | None = None

    for offset, line in enumerate(lines[start:], start=start):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            headings.append(
                Heading(
                    depth=len(heading_match.group(1)),
                    text=heading_match.group(2).strip(),
                    line=offset + 1,
                )
            )
    return headings


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except IsADirectoryError as exc:
        raise VaultError(f"{relpath(path)!r} is a directory, not a note") from exc
    except OSError as exc:
        raise VaultError(f"cannot read {relpath(path)!r}: {exc}") from exc


def extract_section(text: str, section: str) -> str:
    """Return one heading's content: the heading line through to the next
    heading of equal or shallower depth."""
    headings = iter_headings(text)
    wanted = section.strip().lstrip("#").strip().casefold()

    start_heading = next((h for h in headings if h.text.casefold() == wanted), None)
    if start_heading is None:
        available = ", ".join(h.text for h in headings) or "(none)"
        raise VaultError(f"no heading {section!r} in this note. Headings: {available}")

    end_line = None
    for heading in headings:
        if heading.line > start_heading.line and heading.depth <= start_heading.depth:
            end_line = heading.line
            break

    lines = text.splitlines()
    body = lines[start_heading.line - 1 : (end_line - 1) if end_line else None]
    return "\n".join(body).rstrip() + "\n"


def read_note(rel_path: str, section: str | None = None) -> str:
    path = safe_resolve(rel_path)
    if path.is_dir():
        raise VaultError(f"{relpath(path)!r} is a directory - use vault_list")
    text = read_text(path)
    return extract_section(text, section) if section else text


def list_dir(rel_path: str = "") -> list[dict]:
    path = safe_resolve(rel_path or ".")
    if not path.is_dir():
        raise VaultError(f"{relpath(path)!r} is not a directory")

    entries: list[dict] = []
    for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            rel = child.resolve().relative_to(ROOT)
        except ValueError:
            continue  # symlink out of the vault
        if _is_excluded(rel):
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": rel.as_posix(),
                "type": "dir" if child.is_dir() else "file",
                "size": None if child.is_dir() else stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        )
    return entries


def parse_note(rel_path: str) -> dict:
    path = safe_resolve(rel_path)
    text = read_text(path)
    try:
        post = frontmatter.loads(text)
        meta = dict(post.metadata)
    except Exception:
        meta = {}  # malformed YAML must not make a note unreadable
    return {
        "path": relpath(path),
        "frontmatter": meta,
        "headings": [
            {"depth": h.depth, "text": h.text, "line": h.line} for h in iter_headings(text)
        ],
    }


def walk_notes() -> list[Path]:
    """Every indexable markdown file, exclusions applied."""
    notes: list[Path] = []
    for path in sorted(ROOT.rglob("*.md")):
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if _is_excluded(rel) or not path.is_file():
            continue
        notes.append(path)
    return notes

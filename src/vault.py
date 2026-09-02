"""Filesystem access to the vault, reads and writes.

Security-critical: every path argument reaching this module originated in an
LLM tool call and is untrusted. All access funnels through safe_resolve(), which
is now the only barrier between that string and destructive writes to the share -
the :ro mount that used to back it up is gone.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


class VaultError(ValueError):
    """Raised for a rejected path or an unreadable note."""


# Resolved once. If VAULT_PATH is itself a symlink, this is the real target,
# which is what every containment check below compares against.
ROOT = settings.vault_path.resolve()

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


@dataclass(frozen=True, slots=True)
class Heading:
    depth: int
    text: str
    line: int  # 1-based, relative to the whole file


def _is_hidden(rel: Path) -> bool:
    """True if any component is a dotted directory - .git, .obsidian, .trash."""
    return any(part.startswith(".") for part in rel.parts)


def is_protected(rel: Path) -> bool:
    """True if this path must never be written.

    This is the *only* thing standing between Lyra and `rm -rf .git`, now that
    EXCLUDE_DIRS has been narrowed to indexing. Reads are unrestricted; writes
    go through here.
    """
    return _is_hidden(rel)


def is_index_excluded(rel: Path) -> bool:
    """True if this path is kept out of the vector index and BM25.

    Workflows/ and Reports/ are machine-generated series - noise in search, but
    ordinary notes to read and write. That distinction is the whole reason this
    is separate from is_protected().
    """
    return _is_hidden(rel) or any(part in settings.exclude_dirs for part in rel.parts)


def _reject_symlinks(raw: Path) -> None:
    """Refuse any path with a symlink component.

    Must be given the *unresolved* path: resolve() follows symlinks by
    definition, so walking its output would never find one. `..` is removed
    lexically first, with no filesystem access, so this cannot itself traverse.

    The vault contains no symlinks and is not going to. Rejecting them outright
    is cheaper than reasoning about the window between resolve() and os.replace,
    and it fails loudly if one ever appears. They *are* creatable on this
    fuseblk mount - verified - so this is not a theoretical rule.
    """
    lexical = Path(os.path.normpath(raw))
    try:
        parts = lexical.relative_to(ROOT).parts
    except ValueError:
        return  # outside the vault; containment has already rejected it

    current = ROOT
    for part in parts:
        current = current / part
        if current.is_symlink():  # lstat, does not follow
            rel = current.relative_to(ROOT).as_posix()
            raise VaultError(f"path component is a symlink, which is not allowed: {rel!r}")
        if not current.exists():
            return  # nothing beyond this can exist either


def safe_resolve(rel_path: str, *, must_exist: bool = True, writing: bool = False) -> Path:
    """Resolve a vault-relative path, or raise.

    Rejects traversal, absolute escapes and symlinks. When writing=True it also
    rejects protected paths and anything that is not a .md file. Never falls
    back to a default.
    """
    if "\x00" in rel_path:
        raise VaultError("path contains a null byte")

    # A leading slash is treated as vault-root-relative rather than rejected -
    # models write "/Pets/Levi.md" often enough that failing it is pure
    # friction. Containment is still enforced below, so this is not a shortcut.
    cleaned = rel_path.strip().lstrip("/")

    raw = ROOT / cleaned
    candidate = raw.resolve()

    if candidate != ROOT and not candidate.is_relative_to(ROOT):
        raise VaultError(f"path escapes the vault: {rel_path!r}")

    rel = candidate.relative_to(ROOT)
    _reject_symlinks(raw)

    if writing:
        if is_protected(rel):
            raise VaultError(f"path is protected and cannot be written: {rel.as_posix()!r}")
        if candidate.suffix.lower() != ".md":
            raise VaultError(f"only .md files may be written, got: {rel.as_posix()!r}")

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
        if _is_hidden(rel):
            continue  # tidiness, not safety - reads are unrestricted
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
    # Imported here, not at module scope: reading YAML metadata is the only
    # thing in this module that needs it. The write path is deliberately
    # surgical and never round-trips a note through a parser, so the resolver
    # and its tests must not drag the dependency in.
    import frontmatter

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
        if is_index_excluded(rel) or not path.is_file():
            continue
        notes.append(path)
    return notes


# --------------------------------------------------------------------------
# Write primitives
#
# Nothing below round-trips a note through a parser. Every operation is a
# surgical edit on the line list, because the vault's own checkers
# (.scripts/check_frontmatter.py, check_vault_hygiene.py) reject exactly the
# formatting a naive yaml.dump or frontmatter.dumps would produce.
# --------------------------------------------------------------------------

NEW_FILE_MODE = 0o644

# OKF v0.1 spec order, per Meta/Conventions.md. A key that does not exist yet is
# inserted at its ordained position, not appended - appending 'description' to
# the end of the block is a convention violation the checker will not catch.
FIELD_ORDER = (
    "type",
    "title",
    "description",
    "tags",
    "timestamp",
    "expires",
    "expires_reason",
)

_FM_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")


def utc_now() -> str:
    """The vault's timestamp format: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_body(text: str) -> str:
    """LF line endings and exactly one trailing newline.

    check_vault_hygiene.py treats mixed endings as an error and a missing final
    newline as a warning, so this is not cosmetic.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip("\n") + "\n"


def atomic_write(path: Path, text: str) -> None:
    """Replace a note's contents in one step, preserving its mode.

    The temporary file is created in the same directory because os.replace is
    only atomic within a filesystem. A partial write on the fuseblk mount would
    leave a corrupt note that the watcher indexes immediately.

    mkstemp creates 0600 and os.replace keeps the *new* inode's mode, so without
    the chmod every note Lyra touches would silently change mode. Nothing breaks
    if it does - Samba forces uid 1000 for every accessor - but the vault has a
    settled mix of 644 and 777 and there is no reason to churn it.
    """
    text = normalise_body(text)
    mode = (path.stat().st_mode & 0o777) if path.exists() else NEW_FILE_MODE
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".vault-index-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        # newline="" defeats universal-newline translation; the text is already LF.
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


# --------------------------------------------------------------------------
# Frontmatter: surgical, never round-tripped
# --------------------------------------------------------------------------


def frontmatter_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Half-open [start, end) line range of the frontmatter *content*.

    Excludes both '---' fences. None if the note has no frontmatter block.
    """
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return 1, i
    return None


def _key_span(lines: list[str], start: int, end: int, key: str) -> tuple[int, int] | None:
    """Half-open line range occupied by one frontmatter key, value included.

    A key's value runs until the next top-level key or the closing fence, which
    is what makes this work for a block value like the `expires` sequence rather
    than only for `key: value` lines.
    """
    for i in range(start, end):
        match = _FM_KEY_RE.match(lines[i])
        if match and match.group(1) == key:
            j = i + 1
            while j < end and not _FM_KEY_RE.match(lines[j]):
                j += 1
            return i, j
    return None


def _render_value(key: str, value) -> list[str]:
    """Render one frontmatter key, in the shape Conventions mandates for it.

    Three shapes, chosen explicitly rather than by a general YAML dumper:
    inline flow for `tags`, a block sequence for `expires`, scalar for the rest.
    yaml.dump would sort the keys and render tags as a block list, both of which
    check_frontmatter.py reports as violations.
    """
    if isinstance(value, (list, tuple)):
        if key == "tags":
            joined = ", ".join(str(item) for item in value)
            return [f"{key}: [{joined}]"]
        out = [f"{key}:"]
        for entry in value:
            if isinstance(entry, dict):
                # 'date' first, per the expires schema; everything else follows.
                keys = [k for k in ("date", "what") if k in entry]
                keys += [k for k in entry if k not in keys]
                first, *rest = keys
                out.append(f"  - {first}: {entry[first]}")
                out += [f"    {k}: {entry[k]}" for k in rest]
            else:
                out.append(f"  - {entry}")
        return out
    return [f"{key}: {value}"]


def _insert_at(lines: list[str], start: int, end: int, key: str) -> int:
    """Line index at which a new key belongs, honouring FIELD_ORDER."""
    if key not in FIELD_ORDER:
        return end
    rank = FIELD_ORDER.index(key)
    for i in range(start, end):
        match = _FM_KEY_RE.match(lines[i])
        if not match:
            continue
        existing = match.group(1)
        if existing in FIELD_ORDER and FIELD_ORDER.index(existing) > rank:
            return i
    return end


def set_frontmatter(text: str, key: str, value=None, *, delete: bool = False) -> str:
    """Set or remove one frontmatter key, leaving every other byte untouched."""
    lines = normalise_body(text).split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # split() leaves a trailing empty from the final newline

    bounds = frontmatter_bounds(lines)
    if bounds is None:
        raise VaultError("note has no frontmatter block to edit")
    start, end = bounds

    span = _key_span(lines, start, end, key)

    if delete:
        if span is None:
            return normalise_body("\n".join(lines))
        lines[span[0] : span[1]] = []
        return normalise_body("\n".join(lines))

    rendered = _render_value(key, value)
    if span is None:
        at = _insert_at(lines, start, end, key)
        lines[at:at] = rendered
    else:
        lines[span[0] : span[1]] = rendered
    return normalise_body("\n".join(lines))


def bump_timestamp(text: str) -> str:
    """Set `timestamp` to now. Applied to every write, so a convention this
    mechanical never depends on the model remembering it."""
    return set_frontmatter(text, "timestamp", utc_now())

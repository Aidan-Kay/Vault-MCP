"""Markdown -> chunk records.

A chunk is a heading-scoped span of one note, plus enough scaffolding that its
embedding still identifies what the fragment is about.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from . import vault
from .config import settings

# Characters per token. A tokeniser dependency is not warranted: the target is
# soft, and nomic truncates at 8192 regardless.
CHARS_PER_TOKEN = 3.6

_PARAGRAPH_RE = re.compile(r"\n{2,}")
_SECTION_HEADING_MAX_DEPTH = 3  # '#' to '###'; deeper headings stay inline


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """(char offset, paragraph) pairs. Offsets let sub-chunks keep line numbers."""
    parts: list[tuple[int, str]] = []
    pos = 0
    for match in _PARAGRAPH_RE.finditer(text):
        parts.append((pos, text[pos : match.start()]))
        pos = match.end()
    parts.append((pos, text[pos:]))
    return [(offset, body) for offset, body in parts if body.strip()]


def _split_oversized(text: str, start_line: int) -> list[tuple[int, str]]:
    """Split at paragraph boundaries, carrying overlap into each next chunk."""
    target_chars = int(settings.chunk_target_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(settings.chunk_overlap_tokens * CHARS_PER_TOKEN)
    min_chars = int(settings.chunk_min_tokens * CHARS_PER_TOKEN)

    out: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_offset: int | None = None
    carried = ""

    def flush() -> None:
        nonlocal buffer, buffer_offset, carried
        if not buffer:
            return
        body = "\n\n".join(buffer)
        line = start_line + text[: buffer_offset or 0].count("\n")
        out.append((line, (carried + body) if carried else body))
        # Snap the overlap to a whitespace boundary so a chunk never opens
        # mid-word.
        tail = body[-overlap_chars:] if overlap_chars else ""
        if tail and len(tail) < len(body):
            space = tail.find(" ")
            tail = tail[space + 1 :] if space != -1 else tail
        carried = (tail + "\n\n") if tail.strip() else ""
        buffer, buffer_offset = [], None

    for offset, paragraph in _paragraphs(text):
        if buffer_offset is None:
            buffer_offset = offset
        held = sum(len(p) + 2 for p in buffer)
        # Only break if what is already held stands on its own. Without this, a
        # heading line followed by an unsplittable table (no blank lines, so one
        # paragraph) is emitted as a chunk containing nothing but the heading.
        if buffer and held + len(paragraph) > target_chars and held >= min_chars:
            flush()
            buffer_offset = offset
        buffer.append(paragraph)
    flush()

    # A trailing remainder below the threshold belongs to the piece before it,
    # not in the index on its own.
    if len(out) > 1 and len(out[-1][1]) < min_chars:
        line, tail = out.pop()
        head_line, head_text = out[-1]
        out[-1] = (head_line, f"{head_text}\n\n{tail}")
    return out


def _sections(text: str) -> list[tuple[str, int, str]]:
    """(breadcrumb, start line, body) for each heading-scoped span."""
    lines = text.splitlines()
    headings = [h for h in vault.iter_headings(text) if h.depth <= _SECTION_HEADING_MAX_DEPTH]
    body_start = vault.frontmatter_span(text)

    spans: list[tuple[str, int, str]] = []

    first_heading_line = headings[0].line if headings else len(lines) + 1
    preamble = "\n".join(lines[body_start : first_heading_line - 1]).strip()
    if preamble:
        spans.append(("", body_start + 1, preamble))

    stack: list[tuple[int, str]] = []
    for position, heading in enumerate(headings):
        while stack and stack[-1][0] >= heading.depth:
            stack.pop()
        breadcrumb = " > ".join([*(t for _, t in stack), heading.text])
        stack.append((heading.depth, heading.text))

        end = headings[position + 1].line - 1 if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[heading.line - 1 : end]).strip()
        if body:
            spans.append((breadcrumb, heading.line, body))
    return spans


def _merge_small(spans: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    """Fold sub-threshold sections into the following sibling.

    A bare '## Related' with four links is not a retrievable unit; on its own it
    competes with real content for a top-k slot while carrying no answer.
    """
    merged: list[tuple[str, int, str]] = []
    pending: tuple[str, int, str] | None = None

    for breadcrumb, line, body in spans:
        if pending is not None:
            breadcrumb, line, body = pending[0], pending[1], f"{pending[2]}\n\n{body}"
            pending = None
        if estimate_tokens(body) < settings.chunk_min_tokens:
            pending = (breadcrumb, line, body)
            continue
        merged.append((breadcrumb, line, body))

    if pending is not None:
        if merged:  # trailing runt: attach to the previous chunk instead
            last = merged[-1]
            merged[-1] = (last[0], last[1], f"{last[2]}\n\n{pending[2]}")
        else:
            merged.append(pending)
    return merged


def build_embed_text(title: str, description: str, breadcrumb: str, text: str) -> str:
    """Scaffold a chunk for embedding. Never returned to the model.

    The 'search_document: ' prefix is mandatory - nomic-embed-text is
    asymmetric and silently loses recall without it.
    """
    header = title
    if description:
        header = f"{title} - {description}" if title else description
    lines = [f"search_document: {header}".rstrip()]
    if breadcrumb:
        lines.append(breadcrumb)
    return "\n".join(lines) + "\n\n" + text


def chunk_note(path: Path) -> list[dict]:
    text = vault.read_text(path)
    try:
        meta = dict(frontmatter.loads(text).metadata)
    except Exception:
        meta = {}

    rel = vault.relpath(path)
    title = str(meta.get("title") or Path(rel).stem)
    description = str(meta.get("description") or "")

    records: list[dict] = []
    for breadcrumb, line, body in _merge_small(_sections(text)):
        pieces = (
            _split_oversized(body, line)
            if estimate_tokens(body) > settings.chunk_target_tokens
            else [(line, body)]
        )
        for piece_line, piece_text in pieces:
            records.append(
                {
                    "path": rel,
                    "title": title,
                    "description": description,
                    "breadcrumb": breadcrumb,
                    "text": piece_text,
                    "embed_text": build_embed_text(title, description, breadcrumb, piece_text),
                    "line": piece_line,
                }
            )
    return records

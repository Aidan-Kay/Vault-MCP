"""Section-level edits: the operations vault_patch and vault_append are built on.

Separate from vault.py because these are the only functions that need the
resolver, and separate from the tool layer because both the MCP tools and the
REST routes call them.
"""

from __future__ import annotations

from . import target as target_mod
from .vault import VaultError, normalise_body

OPERATIONS = ("replace", "prepend", "append")


def _strip_edges(lines: list[str]) -> list[str]:
    """Drop leading and trailing blank lines from a region."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def patch_section(
    text: str,
    heading_target: str,
    operation: str,
    content: str,
    *,
    target_scope: str = "content",
    note: str = "this note",
) -> tuple[str, str]:
    """Apply one operation to one resolved heading.

    Returns (new_text, resolved_path) - the caller reports the resolved full
    '::' path back to the model so a successful call is self-evidencing.
    """
    if operation not in OPERATIONS:
        raise VaultError(f"unknown operation {operation!r} - use one of {', '.join(OPERATIONS)}")

    text = normalise_body(text)
    found = target_mod.resolve(text, heading_target, note=note)
    start, end = target_mod.section_bounds(text, found, target_scope)

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    addition = _strip_edges(normalise_body(content).split("\n"))
    region = lines[start:end]
    followed = end < len(lines)

    if target_scope == "content":
        body = _strip_edges(region)
        if operation == "replace":
            merged = addition
        elif operation == "append":
            merged = body + [""] + addition if body else addition
        else:
            merged = addition + [""] + body if body else addition
        # Exactly one blank line after the heading, and exactly one before the
        # next heading. Without the second, every append widens the gap by a
        # line - the plugin gets this wrong and it shows in notes it has edited.
        new_region = [""] + merged + ([""] if followed else [])
    else:
        if operation == "replace":
            new_region = addition
        elif operation == "append":
            new_region = region + [""] + addition
        else:
            new_region = addition + [""] + region

    lines[start:end] = new_region
    return normalise_body("\n".join(lines)), found.display


def append_to_note(text: str, content: str) -> str:
    """Append a block to the end of a note, with exactly one blank line before it."""
    body = _strip_edges(normalise_body(text).split("\n"))
    addition = _strip_edges(normalise_body(content).split("\n"))
    if not addition:
        return normalise_body("\n".join(body))
    merged = body + [""] + addition if body else addition
    return normalise_body("\n".join(merged))

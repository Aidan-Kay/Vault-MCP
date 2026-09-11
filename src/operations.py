"""File-level vault operations, shared by the MCP tools and the REST routes.

Everything here resolves a path, reads, edits in memory, and replaces the file
atomically. Both surfaces call these, so the resolver, the conventions and every
trap are handled exactly once regardless of which one the caller used.

Each function returns the confirmation string the caller reports back, naming
the resolved heading path where there was one. A successful call is meant to be
self-evidencing: the model can see it hit the section it meant.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

from . import edit
from . import vault
from .vault import VaultError

_LINK_TARGET = re.compile(r"\]\(([^)]+)\)")

# "appendd" is not a word. Reported back to the model verbatim, so it matters.
_PAST_TENSE = {"replace": "replaced", "append": "appended", "prepend": "prepended"}


def _timestamped(text: str) -> tuple[str, str]:
    """Bump `timestamp`, or say why it was not bumped.

    A note written without frontmatter has nothing to bump. That is reported
    rather than silently skipped - a missing timestamp is a convention breach
    the model should see immediately, not discover in a later checker run.
    """
    lines = vault.normalise_body(text).split("\n")
    if vault.frontmatter_bounds(lines) is None:
        return text, " (no frontmatter, so timestamp not bumped)"
    return vault.bump_timestamp(text), ""


def patch(
    path: str,
    target: str,
    operation: str,
    content: str,
    target_scope: str = "content",
) -> str:
    """Replace, prepend to, or append to one heading's section."""
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    text = vault.read_text(resolved)
    updated, heading = edit.patch_section(
        text, target, operation, content, target_scope=target_scope, note=rel
    )
    updated, note = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"{_PAST_TENSE[operation]} {heading!r} in {rel}{note}"


def append(path: str, content: str, create_if_missing: bool = False) -> str:
    """Append a block to the end of a note."""
    resolved = vault.safe_resolve(path, must_exist=not create_if_missing, writing=True)
    rel = vault.relpath(resolved)

    if not resolved.exists():
        # The same tail as write(). Nothing is invented here either - frontmatter
        # still has to arrive in `content` - but a note created down this branch
        # now reaches disk under the rules every other write obeys. It used to be
        # the one write path that skipped _timestamped, which meant the branch a
        # model is told to prefer was also the only one that quietly broke the
        # convention, and said "no frontmatter added" even when it was given some.
        updated, note = _timestamped(content)
        vault.atomic_write(resolved, updated)
        return f"created {rel} with the supplied content{note}"

    updated = edit.append_to_note(vault.read_text(resolved), content)
    updated, note = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"appended to {rel}{note}"


def write(path: str, content: str, overwrite: bool = False) -> str:
    """Create a note, or replace one wholesale.

    The only guard in a system with no safety net: a create that silently
    clobbers is indistinguishable from a create that worked.
    """
    resolved = vault.safe_resolve(path, must_exist=False, writing=True)
    rel = vault.relpath(resolved)
    existed = resolved.exists()

    if existed and not overwrite:
        raise VaultError(
            f"{rel} already exists. Pass overwrite=true to replace it, or use "
            "vault_patch to edit one section."
        )

    updated, note = _timestamped(content)
    vault.atomic_write(resolved, updated)
    return f"{'overwrote' if existed else 'created'} {rel}{note}"


def set_frontmatter(path: str, key: str, value=None, delete: bool = False) -> str:
    """Set or remove one frontmatter key, leaving every other byte untouched."""
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    text = vault.read_text(resolved)

    updated = vault.set_frontmatter(text, key, value, delete=delete)
    if key != "timestamp":
        updated, _ = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"{'removed' if delete else 'set'} {key!r} in {rel}"


def delete(path: str) -> str:
    """Delete a note. There is no trash - the vault's git history is the undo."""
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    if resolved.is_dir():
        raise VaultError(f"{rel} is a directory; only notes can be deleted")
    resolved.unlink()
    return f"deleted {rel}"


def _link_forms(rel: str) -> set[str]:
    """Every way this vault writes a link to one note.

    Both encoded and unencoded, both vault-root-absolute and bare. The unencoded
    forms are broken links by convention, but they exist and a move must not
    leave them pointing at nothing.
    """
    encoded = urllib.parse.quote(rel)
    return {rel, encoded, f"/{rel}", f"/{encoded}"}


def _rewrite_links(source_rel: str, dest_rel: str) -> int:
    """Repoint every internal link from source to dest. Returns the note count.

    Obsidian's fileManager.renameFile did this for free; this is the one place
    the migration genuinely loses something, so it is deliberately conservative:
    only link targets inside `](...)` are touched, never prose, and the
    replacement is always written in the encoded root-absolute form the
    conventions require.
    """
    stale = _link_forms(source_rel)
    replacement = "/" + urllib.parse.quote(dest_rel)
    touched = 0

    for note in vault.walk_all_notes():
        text = vault.read_text(note)

        def swap(match: re.Match) -> str:
            target = match.group(1)
            anchor = ""
            if "#" in target:
                target, _, fragment = target.partition("#")
                anchor = "#" + fragment
            return f"]({replacement}{anchor})" if target in stale else match.group(0)

        updated = _LINK_TARGET.sub(swap, text)
        if updated != text:
            # No timestamp bump: repointing a link is a mechanical consequence of
            # someone else's move, not an edit to this note's content.
            vault.atomic_write(note, updated)
            touched += 1

    return touched


def move(source: str, destination: str, update_links: bool = True) -> str:
    """Move or rename a note, optionally repointing every link to it."""
    src = vault.safe_resolve(source, writing=True)
    dest = vault.safe_resolve(destination, must_exist=False, writing=True)

    if dest.exists():
        raise VaultError(f"{vault.relpath(dest)} already exists")
    if src.is_dir():
        raise VaultError(f"{vault.relpath(src)} is a directory; only notes can be moved")

    source_rel = vault.relpath(src)
    dest_rel = dest.resolve().relative_to(vault.ROOT).as_posix()

    text = vault.read_text(src)
    updated, note = _timestamped(text)

    dest.parent.mkdir(parents=True, exist_ok=True)
    vault.atomic_write(dest, updated)
    src.unlink()

    message = f"moved {source_rel} to {dest_rel}{note}"
    if update_links:
        touched = _rewrite_links(source_rel, dest_rel)
        message += f"; repointed links in {touched} note(s)"
    else:
        message += "; links NOT updated"
    return message + ". Update index.md to match."

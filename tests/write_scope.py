"""Write scoping: the guard that keeps a confined agent inside its one note.

The incident this exists for: an agent told in prose to "carry nothing out"
replaced a whole section of the vault's root index.md while revising an
unrelated note. A sentence in a prompt is not a guard, so the last case here is
that exact shape - scoped to one note, aim a patch at index.md, and assert both
that it is refused and that index.md is byte-identical afterwards.

Unlike the other runners this one needs a *writable* vault, so it points
VAULT_PATH at a temp tree before importing src. That has to happen before the
first import of src.config, which resolves settings once at import time.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Before `from src import ...`, and deliberately an assignment rather than
# setdefault: tests/__init__ has already pointed this at the real vault, and
# this runner must not write there.
_VAULT = Path(tempfile.mkdtemp(prefix="vault-scope-"))
os.environ["VAULT_PATH"] = str(_VAULT)

from src import operations, vault  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def refused(name: str, fn) -> None:
    """The call must raise VaultError, and must not have written anything."""
    try:
        fn()
    except vault.VaultError:
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}\n    expected VaultError, got {type(exc).__name__}: {exc}")
        return
    FAILURES.append(f"{name}\n    expected VaultError, but the call succeeded")


def allowed(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}\n    expected success, got {type(exc).__name__}: {exc}")


def report() -> int:
    """1 if anything failed, having said what; 0 if not. Called on every exit."""
    if not FAILURES:
        return 0
    print(f"{len(FAILURES)} failure(s):\n")
    for failure in FAILURES:
        print(failure + "\n")
    return 1


def seed() -> None:
    (_VAULT / "Sub").mkdir(parents=True, exist_ok=True)
    (_VAULT / "index.md").write_text(
        "---\ntitle: Index\n---\n\n# Inbox\n\n- [One](Sub/One.md) - first\n- [Two](Sub/Two.md) - second\n",
        encoding="utf-8",
    )
    (_VAULT / "Proposal.md").write_text(
        "---\ntitle: Proposal\n---\n\n### Proposed Actions\n\n1. Do the original thing\n",
        encoding="utf-8",
    )
    (_VAULT / "Sub" / "One.md").write_text("---\ntitle: One\n---\n\nbody\n", encoding="utf-8")


def main() -> int:
    seed()

    # --- unscoped: everything still works exactly as before -----------------
    allowed("unscoped write", lambda: operations.write("Sub/New.md", "---\ntitle: New\n---\n\nx\n"))
    allowed("unscoped append", lambda: operations.append("index.md", "\ntrailing\n"))
    check("unscoped scope is None", vault.current_write_scope(), None)

    # --- scoped to one note --------------------------------------------------
    with vault.write_scope("Proposal.md"):
        check("scope reported", vault.current_write_scope(), "Proposal.md")

        allowed(
            "in-scope patch",
            lambda: operations.patch(
                "Proposal.md", "Proposed Actions", "replace", "\n1. Do the revised thing\n"
            ),
        )
        allowed("in-scope frontmatter", lambda: operations.set_frontmatter("Proposal.md", "rev", 1))

        refused("out-of-scope append", lambda: operations.append("index.md", "\nsneaky\n"))
        refused("out-of-scope write", lambda: operations.write("Sub/Two.md", "x", overwrite=True))
        refused("out-of-scope delete", lambda: operations.delete("Sub/One.md"))
        refused(
            "out-of-scope frontmatter",
            lambda: operations.set_frontmatter("index.md", "title", "hijacked"),
        )

        # Reads stay open. The agent revising a proposal still has to read the
        # conventions and whatever the note refers to.
        allowed("in-scope read of another note", lambda: vault.read_note("index.md"))

        # Moving touches every note that links to the source, through a path
        # that never sees safe_resolve, so it is refused rather than narrowed.
        refused("move under a scope", lambda: operations.move("Proposal.md", "Proposal2.md"))

    check("scope cleared on exit", vault.current_write_scope(), None)
    allowed("writes work again after the scope", lambda: operations.append("index.md", "\nafter\n"))

    # --- scoped to a directory ----------------------------------------------
    with vault.write_scope("Sub"):
        allowed("in-scope directory write", lambda: operations.append("Sub/One.md", "\nmore\n"))
        refused("sibling of the directory", lambda: operations.append("index.md", "\nno\n"))
        # A prefix must not match a sibling that merely starts with the same text.
        refused("prefix is not a substring", lambda: operations.write("Subterfuge.md", "x"))

    # --- the incident, reproduced -------------------------------------------
    before = (_VAULT / "index.md").read_bytes()
    with vault.write_scope("Proposal.md"):
        refused(
            "the index.md overwrite that started all this",
            lambda: operations.patch("index.md", "Inbox", "replace", "\n- one bullet\n"),
        )
    check("index.md untouched", (_VAULT / "index.md").read_bytes(), before)

    # --- the transport carries the scope, and only on the scoped path -------
    #
    # Imported here rather than at the top because src.server pulls in the mcp
    # package, which is present in the container and not on every dev box. The
    # checks above are the guard itself and run anywhere; these are the wiring
    # that hands it a scope, and they are shouted about rather than skipped
    # quietly when they cannot run.
    try:
        from src import server
    except ModuleNotFoundError as exc:
        print(f"!! transport checks NOT RUN - {exc}. Run this in the container.")
        # 2, not 0: "the guard passed and the wiring was never checked" is not a
        # pass, and must not read like one in CI or in a scrollback.
        return report() or 2

    seen: list[tuple[str, str | None]] = []

    async def inner(scope, receive, send):  # noqa: ARG001
        seen.append((scope.get("path", ""), vault.current_write_scope()))

    wrapped = server.ScopedWrites(inner)

    async def call(path: str) -> None:
        await wrapped({"type": "http", "path": path, "raw_path": path.encode()}, None, None)

    asyncio.run(call("/mcp"))
    check("plain /mcp is unscoped", seen[-1], ("/mcp", None))

    asyncio.run(call("/mcp/only/Workflows/Approvals/abc.md"))
    check(
        "scoped path sets the scope and rewrites the path",
        seen[-1],
        ("/mcp", "Workflows/Approvals/abc.md"),
    )

    asyncio.run(call("/mcp/only/Workflows%2FApprovals%2Fabc.md"))
    check("percent-encoded scope is decoded", seen[-1], ("/mcp", "Workflows/Approvals/abc.md"))

    asyncio.run(call("/mcp/only/"))
    check("empty scope falls through unscoped", seen[-1], ("/mcp/only/", None))

    asyncio.run(call("/vault/Some/Note.md"))
    check("the REST surface is untouched", seen[-1], ("/vault/Some/Note.md", None))

    # The scope must not outlive its request, or one caller's confinement
    # becomes another caller's.
    check("scope does not leak past the request", vault.current_write_scope(), None)

    if report():
        return 1
    print("write_scope: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)

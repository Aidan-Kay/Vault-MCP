"""Forgiving heading-target resolution.

The plugin this replaces keys every heading by its exact full ancestor path and
does a single property lookup, so a target that is not already the complete path
from the H1 down matches nothing. 86% of this vault's notes are wrapped in a
single H1, which makes almost every useful target a two-or-three segment path
the model has to guess up front.

Here both sides are normalised for matching only, and the target's segments are
matched against the *trailing* segments of each heading's ancestor path. A leaf
works whenever it is unique; when it is not, prepending ancestors always narrows
it, and the error says which ancestors to prepend. Stored heading text is never
rewritten - a heading with an em dash keeps its em dash on disk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .vault import Heading, VaultError, iter_headings

SEPARATOR = "::"

# Every dash-like codepoint folds to a plain hyphen: the model types '-' where
# the vault has an em dash in 241 headings. Soft hyphen is deleted, not folded -
# it is invisible in the source and must not become a matchable character.
_SOFT_HYPHEN = "­"
_DASHES = dict.fromkeys(
    map(ord, "‐‑‒–—―−⁃－"), "-"
)

_LEADING_HASHES = re.compile(r"^#{1,6}\s*")
_CLOSING_HASHES = re.compile(r"\s+#+\s*$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"\*\*|__|[`*_]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Fold a heading or a target segment to its match key.

    Matching only. Never write the result back to a note.
    """
    s = unicodedata.normalize("NFKC", text).strip()
    s = _LEADING_HASHES.sub("", s)
    s = _CLOSING_HASHES.sub("", s)
    s = _LINK.sub(r"\1", s)  # before markup, so [**a**](x) folds to 'a'
    s = _MARKUP.sub("", s)
    s = s.replace(_SOFT_HYPHEN, "")
    s = s.translate(_DASHES)
    return _WHITESPACE.sub(" ", s).strip().casefold()


@dataclass(frozen=True, slots=True)
class Candidate:
    """One heading, with the full ancestor path that addresses it."""

    heading: Heading
    path: tuple[str, ...]  # original text, root-first

    @property
    def display(self) -> str:
        return SEPARATOR.join(self.path)

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(normalise(part) for part in self.path)


def candidates(text: str) -> list[Candidate]:
    """Every heading in a note, each with its full ancestor path.

    iter_headings is already fence-aware, so bash comments starting with '#'
    are not mistaken for headings here either.
    """
    out: list[Candidate] = []
    stack: list[Heading] = []
    for heading in iter_headings(text):
        while stack and stack[-1].depth >= heading.depth:
            stack.pop()
        stack.append(heading)
        out.append(Candidate(heading=heading, path=tuple(h.text for h in stack)))
    return out


def split_target(target: str) -> list[str]:
    parts = [part.strip() for part in target.split(SEPARATOR)]
    parts = [part for part in parts if part]
    if not parts:
        raise VaultError("empty heading target")
    return parts


def resolve(text: str, target: str, *, note: str = "this note") -> Candidate:
    """Resolve a '::'-separated target to exactly one heading, or raise.

    Zero matches lists every heading in the note; more than one lists only the
    colliding paths. Both are returned as full '::' paths, because a path that
    can be pasted straight back as the next target is the difference between a
    retry that succeeds and a retry loop that gives up.
    """
    all_candidates = candidates(text)
    wanted = tuple(normalise(part) for part in split_target(target))

    matches = [c for c in all_candidates if c.key[-len(wanted) :] == wanted]

    # A heading whose whole path is a trailing sub-path of a deeper one would
    # otherwise be unaddressable: in Home/Fixtures/Boiler.md the H1 'Boiler' is
    # a suffix of 'Boiler::Boiler', so no target could ever name it. An exact
    # full-path match therefore beats a suffix match. Two headings sharing the
    # *same* full path stay ambiguous, which is the honest answer - the plugin
    # silently patches the last one.
    if len(matches) > 1:
        exact = [c for c in matches if len(c.key) == len(wanted)]
        if len(exact) == 1:
            return exact[0]
        matches = exact or matches

    if len(matches) == 1:
        return matches[0]

    if not matches:
        available = "\n    ".join(c.display for c in all_candidates) or "(none)"
        raise VaultError(
            f"no heading {target!r} in {note}. Headings:\n    {available}"
        )

    colliding = "\n    ".join(c.display for c in matches)
    raise VaultError(
        f"{target!r} is ambiguous in {note} - {len(matches)} headings match:\n"
        f"    {colliding}\n"
        "  Re-call with the full path to disambiguate."
    )


_SCOPES = ("content", "marker", "markerAndContent")


def section_bounds(text: str, found: Candidate, scope: str = "content") -> tuple[int, int]:
    """Half-open [start, end) line range for a resolved target, 0-based.

    'content'          after the heading line to the next heading of equal or
                       shallower depth
    'marker'           the heading line only
    'markerAndContent' both
    """
    if scope not in _SCOPES:
        raise VaultError(f"unknown target_scope {scope!r} - use one of {', '.join(_SCOPES)}")

    marker = found.heading.line - 1  # Heading.line is 1-based
    end = len(text.splitlines())
    for candidate in candidates(text):
        if candidate.heading.line > found.heading.line and candidate.heading.depth <= found.heading.depth:
            end = candidate.heading.line - 1
            break

    if scope == "marker":
        return marker, marker + 1
    if scope == "content":
        return marker + 1, end
    return marker, end


def outline(text: str) -> list[str]:
    """Full '::' paths for every heading, for vault_map.

    An indented tree cannot be pasted back as a target; these can.
    """
    return [c.display for c in candidates(text)]

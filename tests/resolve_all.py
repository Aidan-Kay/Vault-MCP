"""Every heading must resolve to itself by its own full path - unless the note
genuinely contains two headings with the same full path, in which case the only
correct answer is an ambiguity error.

Deliberately not gated on a literal count: the vault grows, and a hardcoded
total turns a passing suite into a stale one. The assertion is "all of them".
"""

from __future__ import annotations

import collections
import sys

from src import target, vault


def main() -> int:
    checked = failed = expected_ambiguous = 0

    for path in sorted(vault.ROOT.rglob("*.md")):
        rel = path.relative_to(vault.ROOT)
        if vault.is_index_excluded(rel) or not path.is_file():
            continue
        note = rel.as_posix()
        text = vault.read_text(path)
        cands = target.candidates(text)

        # Ground truth: a full path is unresolvable only when the note repeats it.
        duplicated = {
            key for key, count in collections.Counter(c.key for c in cands).items() if count > 1
        }

        for candidate in cands:
            checked += 1
            try:
                found = target.resolve(text, candidate.display, note=note)
            except vault.VaultError as exc:
                if candidate.key in duplicated and "is ambiguous" in str(exc):
                    expected_ambiguous += 1
                else:
                    failed += 1
                    print(f"  FAIL {note} :: {candidate.display}")
                    print(f"    {str(exc).splitlines()[0]}")
                continue
            if candidate.key in duplicated:
                failed += 1
                print(f"  SILENT PICK {note} :: {candidate.display} - duplicate resolved anyway")
            elif found.heading.line != candidate.heading.line:
                failed += 1
                print(
                    f"  WRONG {note} :: {candidate.display} -> "
                    f"line {found.heading.line}, expected {candidate.heading.line}"
                )

    resolved = checked - failed - expected_ambiguous
    print(f"\nresolve_all: {resolved} / {checked} headings resolve by full path")
    print(f"             {expected_ambiguous} correctly refused as duplicate full paths")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

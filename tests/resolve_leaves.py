"""Leaf-only resolution, which is what the model actually types.

Two properties, both stated independently of how resolve() is implemented so
the test is not a restatement of the code:

1. A leaf that occurs once in a note always resolves to it. This is the whole
   point of the migration - the plugin cannot do it at all.
2. A leaf that occurs more than once is either refused as ambiguous, or resolves
   to the one heading whose *entire* path is that leaf. It is never silently
   pointed at one of several equally good candidates, which is the failure mode
   that makes the current stack unsafe rather than merely annoying.

The repeats are recomputed here rather than asserted against a number written
down on a past date.
"""

from __future__ import annotations

import collections
import sys

from src import target, vault


def main() -> int:
    unique_ok = repeated_refused = repeated_exact = failed = 0

    for path in sorted(vault.ROOT.rglob("*.md")):
        rel = path.relative_to(vault.ROOT)
        if vault.is_index_excluded(rel) or not path.is_file():
            continue
        note = rel.as_posix()
        text = vault.read_text(path)
        cands = target.candidates(text)
        counts = collections.Counter(c.key[-1] for c in cands)

        for leaf, count in counts.items():
            matching = [c for c in cands if c.key[-1] == leaf]
            display = matching[0].path[-1]
            try:
                found = target.resolve(text, display, note=note)
            except vault.VaultError as exc:
                if count > 1 and "is ambiguous" in str(exc):
                    repeated_refused += 1
                else:
                    failed += 1
                    print(f"  FAIL {note} :: {display}")
                    print(f"    {str(exc).splitlines()[0]}")
                continue

            if count == 1:
                if found.heading.line == matching[0].heading.line:
                    unique_ok += 1
                else:
                    failed += 1
                    print(f"  WRONG {note} :: {display} resolved to the wrong heading")
            elif len(found.path) == 1:
                # The one heading whose whole path is this leaf - addressable
                # precisely because nothing else can name it.
                repeated_exact += 1
            else:
                failed += 1
                print(f"  SILENT PICK {note} :: {display} -> {found.display}")

    print(f"\nresolve_leaves: {unique_ok} unique leaves resolve")
    print(f"                {repeated_refused} repeated leaves correctly refused")
    print(f"                {repeated_exact} repeated leaves resolved by exact whole-path match")
    print(f"                {failed} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

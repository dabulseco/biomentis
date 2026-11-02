"""Tests for the rubric model and its markdown file format.

Run with:
    python -m biomentis.agent.tests.test_rubric
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


CANON = (
    "# Rubric\n\n"
    "## Deliverables\n"
    "- [D1] A table of primer sequences\n"
    "- [D2] A figure of coverage\n\n"
    "## Constraints\n"
    "- [C1] Use only 2×250 bp reads\n\n"
    "## Core lessons\n"
    "- [L1] Report units for every number\n"
)


def test_round_trip() -> None:
    from biomentis.gate.rubric import format_rubric_markdown, parse_rubric_markdown

    r = parse_rubric_markdown(CANON)
    _check("canonical text parses", r is not None)
    assert r is not None
    _check("items and kinds", [(i.id, i.kind) for i in r.items] ==
           [("D1", "deliverable"), ("D2", "deliverable"), ("C1", "constraint"), ("L1", "lesson")],
           str(r.items))
    _check("default source", r.source == "uploaded" and r.prompt_sha == "")
    _check("round trip is exact", format_rubric_markdown(r) == CANON, format_rubric_markdown(r))
    headless = CANON.replace("# Rubric\n\n", "")
    r2 = parse_rubric_markdown(headless, source="converted", prompt_sha="abc")
    _check("heading is optional; source/sha kwargs", r2 is not None and r2.source == "converted"
           and r2.prompt_sha == "abc")


def test_non_standard() -> None:
    from biomentis.gate.rubric import parse_rubric_markdown

    _check("prose is None", parse_rubric_markdown("Please make a good report.") is None)
    _check("empty is None", parse_rubric_markdown("") is None)
    _check("missing section is None",
           parse_rubric_markdown("## Deliverables\n- [D1] x\n\n## Constraints\n- [C1] y\n") is None)
    _check("unknown section is None",
           parse_rubric_markdown(CANON + "\n## Extras\n- [D9] z\n") is None)
    _check("prefix/section mismatch is None",
           parse_rubric_markdown(CANON.replace("[C1]", "[D7]")) is None)
    _check("section order matters",
           parse_rubric_markdown("## Constraints\n- [C1] a\n\n## Deliverables\n- [D1] b\n\n"
                                 "## Core lessons\n- [L1] c\n") is None)


def test_duplicates_and_frozen() -> None:
    from biomentis.gate.rubric import parse_rubric_markdown

    try:
        parse_rubric_markdown(CANON.replace("[D2]", "[D1]"))
        _check("duplicate IDs raise ValueError", False, "no error")
    except ValueError:
        _check("duplicate IDs raise ValueError", True)
    r = parse_rubric_markdown(CANON)
    assert r is not None
    for obj, label in ((r.items[0], "item"), (r, "rubric")):
        try:
            if label == "item":
                obj.text = "x"
            else:
                obj.source = "x"
            _check(f"{label} is frozen", False, "assignment worked")
        except dataclasses.FrozenInstanceError:
            _check(f"{label} is frozen", True)


def test_save_and_sha() -> None:
    from biomentis.gate.rubric import parse_rubric_markdown, prompt_sha, save_rubric
    import hashlib

    _check("prompt_sha strips", prompt_sha("  hi \n") == hashlib.sha256(b"hi").hexdigest())
    sha = prompt_sha("hi")
    r = parse_rubric_markdown(CANON, prompt_sha=sha)
    assert r is not None
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "new" / "sub"
        p = save_rubric(r, folder)
        _check("creates folder, returns path", p == folder / f"rubric-{sha[:8]}.md" and p.exists(), str(p))
        _check("file holds canonical text", p.read_text(encoding="utf-8") == CANON)


def run_all() -> int:
    for test in (test_round_trip, test_non_standard, test_duplicates_and_frozen, test_save_and_sha):
        print(f"\n{test.__name__}")
        try:
            test()
        except Exception as e:  # noqa: BLE001
            _check(f"{test.__name__} ran", False, repr(e))
    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())

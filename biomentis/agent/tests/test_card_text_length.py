"""Tests for how long teaching-card text is shortened before a student sees it.

Regression origin (session of 2026-11-01): scientific cards in instructor /
tutor mode showed "What: ... The scientific question this step actually
answers is whether t\n\n[…truncated for prompt size…]". The prompt-input
helper `_truncate` was also applied to the card fields the model wrote, and the
scientific "what" cap (700 chars) was below what the prompt's "2-3 sentences"
routinely produce.

What these tests pin down:

  1.  A card field never shows the "truncated for prompt size" marker
  2.  An over-long field is cut after a whole sentence, not mid-word
  3.  A field with no sentence break falls back to a word boundary + "…"
  4.  A typical 3-sentence scientific "what" (~750 chars) is kept whole
  5.  Short text is untouched; prompt-side `_truncate` still marks its cuts

Run with:
    python -m biomentis.agent.tests.test_card_text_length
"""

from __future__ import annotations

import sys

from biomentis.agent.tutor.instruction import (
    MODE_SCIENTIFIC,
    _MAX_WHAT_CHARS,
    _clip_display,
    _coerce_section,
    _truncate,
)

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []
_MARKER = "truncated for prompt size"

# The card from the bug report, filled out to three full sentences.
_SCI_WHAT = (
    "This step surveys the available protocol corpus to ask whether an established, "
    "validated procedure already exists for measuring the boiling-point elevation of "
    "aqueous solutions — the thermophysical measurement at the heart of the requested lab. "
    "The returned entries are all molecular-biology methods (viral vector production, "
    "purification, titration, immunoassays), so the search establishes a negative: no "
    "inherited method exists for equilibrating a liquid's vapor pressure against the "
    "atmosphere and reading the temperature at which the two become equal. The scientific "
    "question this step actually answers is whether the lab must be designed from first "
    "principles rather than adapted from an existing validated protocol."
)


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def test_no_prompt_marker_on_cards() -> None:
    print("\n[1] card fields never carry the prompt-size marker")
    long = "Word " * 2000
    section = _coerce_section({"what": long, "why": long, "impact": long}, MODE_SCIENTIFIC)
    for field in ("what", "why", "impact"):
        text = getattr(section, field)
        _check(f"{field} has no marker", _MARKER not in text, text[-80:])
    blob = _coerce_section(long, MODE_SCIENTIFIC)
    _check("string-blob section has no marker", _MARKER not in blob.what, blob.what[-80:])


def test_cut_at_sentence() -> None:
    print("\n[2] over-long text is cut after a whole sentence")
    text = "First sentence is here. Second one follows it. Third sentence runs on and on and on."
    out = _clip_display(text, 60)
    _check("ends at a sentence", out == "First sentence is here. Second one follows it.", repr(out))


def test_word_boundary_fallback() -> None:
    print("\n[3] no sentence break → word boundary + ellipsis")
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    out = _clip_display(text, 30)
    _check("ends with ellipsis", out.endswith("…"), repr(out))
    _check("within cap", len(out) <= 30, str(len(out)))
    _check("no partial word", out[:-1].split()[-1] in text.split(), repr(out))


def test_typical_scientific_what_kept_whole() -> None:
    print("\n[4] a 3-sentence scientific 'what' survives intact")
    _check("example is over the old 700 cap", len(_SCI_WHAT) > 700, str(len(_SCI_WHAT)))
    section = _coerce_section({"what": _SCI_WHAT}, MODE_SCIENTIFIC)
    _check("kept whole", section.what == _SCI_WHAT, section.what[-80:])
    _check("cap is above the example", _MAX_WHAT_CHARS[MODE_SCIENTIFIC] >= len(_SCI_WHAT))


def test_short_and_prompt_side() -> None:
    print("\n[5] short text untouched; prompt-side truncation unchanged")
    _check("short text untouched", _clip_display("  Hi there.  ", 100) == "Hi there.")
    _check("empty stays empty", _clip_display("", 10) == "")
    _check("prompt _truncate still marks", _MARKER in _truncate("x" * 500, 200))


def run_all() -> int:
    print("=" * 60)
    print("Card text length tests")
    print("=" * 60)
    test_no_prompt_marker_on_cards()
    test_cut_at_sentence()
    test_word_boundary_fallback()
    test_typical_scientific_what_kept_whole()
    test_short_and_prompt_side()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())

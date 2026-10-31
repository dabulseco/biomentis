"""Tests for the completion gate's types and its arithmetic check.

Metagenomics Run 3's final answer printed "24+3+4+2+2+2+4 = 39 ✓" and
"48+5+4+2+2+2+4 = 65 ✓": both sums are wrong (41 and 67), and only the
second-model review caught it. The gate checks such sums itself.

What these tests pin down:

  1.  Finding fills a stable key from check + quote; GateResult.blocking
      leaves out accepted keys and warnings; passed means no blocking finding
  2.  The two false sums in Run 3 give two blocking findings whose fixes hold
      41 and 67
  3.  Correct sums and corrections (boiling point Run 1, metagenomics Run 1)
      give none
  4.  Numbers that are not claims ("2×250 bp", "1.6×10¹⁷ copies",
      "$1,015/field sample", "k = 0.0375 °C mmHg⁻¹") give none
  5.  Results rounded to their stated precision pass; a wrong one is flagged;
      "≈" counts as "="
  6.  Expressions with unsupported operators (√, ², implicit multiplication)
      are skipped, not flagged
  7.  evaluate() parses numbers, operators and named variables, and returns
      None for anything it cannot parse

Run with:
    python -m biomentis.agent.tests.test_gate_arithmetic
"""

from __future__ import annotations

import os
import sys

# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "robustness")


def _fixture(name: str) -> str:
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return f.read()


def test_types():
    from biomentis.gate import Finding, GateResult

    a = Finding(check="arithmetic", severity="blocking", quote="1+1 = 3", issue="wrong")
    b = Finding(check="arithmetic", severity="blocking", quote="1+1 = 3", issue="other text")
    w = Finding(check="figure", severity="warning", quote="Fig 2", issue="no title")
    _check("key filled from check + quote", len(a.key) == 10 and a.key == b.key, a.key)
    _check("explicit key kept", Finding("claim", "warning", "q", "i", key="k1").key == "k1")
    g = GateResult(round=0, findings=[a, w], reviewer_state="ok")
    _check("blocking excludes warnings", g.blocking == [a], repr(g.blocking))
    _check("not passed with a blocking finding", not g.passed)
    g.accepted.add(a.key)
    _check("accepted key resolves the finding", g.blocking == [] and g.passed, repr(g.blocking))


def test_run3_false_sums():
    from biomentis.gate.arithmetic import check_arithmetic

    found = check_arithmetic(_fixture("meta_run3_library_sum.txt"))
    _check("two findings in Run 3's sums", len(found) == 2, repr(found))
    _check("all blocking arithmetic", all(f.check == "arithmetic" and f.severity == "blocking" for f in found),
           repr(found))
    fixes = " | ".join(f.fix for f in found)
    _check("fixes hold 41 and 67", "41" in fixes and "67" in fixes, fixes)
    quotes = [f.quote for f in found]
    _check("quotes are the claims", quotes == ["24+3+4+2+2+2+4 = 39", "48+5+4+2+2+2+4 = 65"], repr(quotes))


def test_correct_sums_pass():
    from biomentis.gate.arithmetic import check_arithmetic

    found = check_arithmetic(_fixture("bp_run1_correct_sums.txt"))
    _check("boiling-point corrections and the 33 sum pass", found == [], repr(found))
    found = check_arithmetic("3 + 4 + 5 + 6 + 1 + 5 + 4 + 2 + 3 = 33")
    _check("= 33 sum passes", found == [], repr(found))


def test_not_claims():
    from biomentis.gate.arithmetic import check_arithmetic

    for text in ("2×250 bp", "1.6×10¹⁷ copies", "$1,015/field sample", "k = 0.0375 °C mmHg⁻¹",
                 "n = 3", "Pilot total = 39 libraries (24 field)", "a = 3, 4+5 = 9",
                 'expected = 393 if p["x"] else 432', "if a == 2+2:",
                 # Verbatim from metagenomics Run 3: a read layout and a scaled result.
                 "- Overlap at 2×250 = 127 bp (insert) vs 247 bp for V4; both merge",
                 "- n=3 sediment: 41 × 60,231 = 2.47 M raw reads"):
        found = check_arithmetic(text)
        _check(f"no finding: {text!r}", found == [], repr(found))


def test_rounding():
    from biomentis.gate.arithmetic import check_arithmetic

    _check("rounded quotient passes", check_arithmetic("0.92/(0.512 × 0.9993) = 1.80") == [])
    found = check_arithmetic("0.92/(0.512 × 0.9993) = 1.70")
    _check("wrong quotient flagged", len(found) == 1 and "1.80" in found[0].fix, repr(found))
    _check("≈ passes when correct", check_arithmetic("1.06/2 ≈ 0.53") == [])
    _check("≈ flagged when wrong", len(check_arithmetic("1.06/2 ≈ 0.63")) == 1)
    found = check_arithmetic("(1-0.84) = 16")
    _check("fix keeps the value's digits", len(found) == 1 and found[0].fix.endswith("= 0.16"), repr(found))
    _check("bold and sentence text around a claim", len(check_arithmetic("**24+3 = 28**")) == 1
           and len(check_arithmetic("Total. 4+5 = 10.")) == 1)
    _check("spaced product still checked", check_arithmetic("2 × 250 = 500") == []
           and len(check_arithmetic("2 × 250 = 400")) == 1)
    _check("thousands separators and $", check_arithmetic("$1,015 × 24 = $24,360") == [])
    _check("chained claims checked", len(check_arithmetic("2+2 = 4 = 5")) == 0
           and len(check_arithmetic("1+1 = 2, 2 × 3 = 7")) == 1)


def test_unsupported_skipped():
    from biomentis.gate.arithmetic import check_arithmetic

    for text in ("√(0.0593² + 0.0296²) = 0.0663", "√(0.0593² + 0.0296²) = 0.9",
                 "100.577 + 0.0375(760-749.8) = 200", "ln(2)/0.5 = 9", "log(100)/2 = 1",
                 "2×3 = 6×10¹⁷", "45,000 raw x 0.95 x 0.80 = 20,007"):
        found = check_arithmetic(text)
        _check(f"skipped: {text!r}", found == [], repr(found))


def test_evaluate():
    from biomentis.gate.arithmetic import evaluate

    _check("sum", evaluate("24+3+4+2+2+2+4") == 41)
    _check("precedence and parens", evaluate("2 + 3 × (4 - 1)") == 11)
    _check("x and · multiply", evaluate("2x250") == 500 and evaluate("2·3") == 6)
    _check("unary minus", evaluate("-3 + 5") == 2)
    _check("variables", evaluate("total_mass / n", {"total_mass": 10.0, "n": 4.0}) == 2.5)
    _check("unknown name", evaluate("a + 1", {}) is None and evaluate("a + 1") is None)
    _check("implicit multiplication", evaluate("0.0375(760-749.82)") is None)
    _check("division by zero", evaluate("1/0") is None)
    _check("junk", evaluate("√2") is None and evaluate("") is None and evaluate("2 +") is None)


def run_all() -> int:
    for test in (test_types, test_run3_false_sums, test_correct_sums_pass, test_not_claims,
                 test_rounding, test_unsupported_skipped, test_evaluate):
        print(f"\n{test.__name__}")
        try:
            test()
        except Exception as e:
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())

"""Tests for the gate's figure checks: cited but not drawn, stale, garbled title.

Run with:
    python -m biomentis.agent.tests.test_gate_figures
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _fig(title: str, code: str = "plt.plot([1]); plt.show()") -> list:
    return [
        AIMessage(content=f"<execute>{code}</execute>"),
        AIMessage(content=f"<observation>Plot captured.\nFigure '{title}': x = day</observation>"),
    ]


def _ev(msgs):
    from biomentis.run_quality import run_evidence
    return run_evidence([HumanMessage(content="Cost the workflow.")] + msgs)


STALE_TITLE = "Fig 12a — First-year outlay $158.00; reagents are only $77.00"


def test_missing() -> None:
    print("\n[a] cited but not drawn")
    from biomentis.gate.figures import check_figures

    msgs = []
    for i in range(1, 14):
        msgs += _fig(f"Fig {i} — trace")
    fs = check_figures("See Fig 14, Fig 15.", _ev(msgs))
    _check("two blocking findings", len(fs) == 2 and all(f.severity == "blocking" and f.check == "figure" for f in fs), str(fs))
    _check("they name 14 and 15", "14" in fs[0].issue and "15" in fs[1].issue, str(fs))


def test_stale() -> None:
    print("\n[b-e] stale figures")
    from biomentis.gate.figures import check_figures

    first = _fig(STALE_TITLE) + [AIMessage(content="<solution>Outlay $158</solution>")]
    revised = first + [AIMessage(content="<solution>Outlay $414</solution>")]
    fs = check_figures("Outlay is $414.", _ev(revised))
    _check("b: one blocking stale finding", len(fs) == 1 and fs[0].severity == "blocking" and "158" in fs[0].issue, str(fs))
    redrawn = first + _fig("Fig 12a — First-year outlay $414.00") + [AIMessage(content="<solution>Outlay $414</solution>")]
    _check("c: redrawn figure gives no finding", check_figures("Outlay is $414.", _ev(redrawn)) == [], str(check_figures("Outlay is $414.", _ev(redrawn))))
    small = _fig("Fig 3 — 3 traces") + [AIMessage(content="<solution>a</solution>"), AIMessage(content="<solution>b</solution>")]
    _check("e: small numbers are not stale", check_figures("b", _ev(small)) == [], "")
    single = _fig(STALE_TITLE) + [AIMessage(content="<solution>Outlay $414</solution>")]
    _check("no earlier candidate: nothing stale", check_figures("Outlay is $414.", _ev(single)) == [], "")
    kept = _fig(STALE_TITLE) + [AIMessage(content="<solution>x</solution>"), AIMessage(content="<solution>y</solution>")]
    _check("numbers still in the answer are not stale", check_figures("Outlay $158.00, reagents $77", _ev(kept)) == [], "")
    pair = [AIMessage(content="<solution>x</solution>"), AIMessage(content="<solution>y</solution>")]
    r = check_figures("Outlay is $414.", _ev(_fig("Fig 5 — outlay $414.37") + pair))
    _check("rounding to the answer's precision is not stale", r == [], str(r))
    r = check_figures("Nothing numeric.", _ev(_fig("Fig 2 — 2024 field season, 1,000 reads") + pair))
    _check("years and sample sizes are not stale", r == [], str(r))
    r = check_figures("Outlay is $414.", _ev(_fig(STALE_TITLE) + pair))
    _check("$158 vs $414 still blocks", len(r) == 1 and r[0].severity == "blocking", str(r))


def test_garbled() -> None:
    print("\n[d] garbled title")
    from biomentis.gate.figures import check_figures

    code = 'ax.set_title(f"outlay ${total:.2f}; reagents ${r:.2f}")'
    fs = check_figures("ok", _ev(_fig("Fig 1 — outlay", code)))
    _check("one warning", len(fs) == 1 and fs[0].severity == "warning", str(fs))
    ok = 'ax.set_title(f"outlay \\${total:.2f}")'
    _check("escaped dollar is fine", check_figures("ok", _ev(_fig("Fig 1 — outlay", ok))) == [], "")


def test_evidence_fields() -> None:
    print("\n[f] evidence fields")
    ev = _ev(_fig("Fig 1 — a") + [AIMessage(content="<solution>x</solution>")])
    _check("figure_titles kept", len(ev.figure_titles) == 1, str(ev.figure_titles))
    r = ev.figure_records[0]
    _check("record has step and code", r.step == 2 and "plt.plot" in r.code, str(r))


def run_all() -> int:
    for t in (test_missing, test_stale, test_garbled, test_evidence_fields):
        try:
            t()
        except Exception as e:
            _check(f"{t.__name__} ran", False, repr(e))
    print(f"\nResults: {len(_PASSED)} passed, {len(_FAILED)} failed")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())

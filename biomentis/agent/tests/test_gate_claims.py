"""Tests for the gate's claim checks: unsourced "verified", unlabelled simulated data, evidence.

Run with:
    python -m biomentis.agent.tests.test_gate_claims
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage

_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _ev(msgs):
    from biomentis.run_quality import run_evidence
    return run_evidence([HumanMessage(content="Design a metagenomics analysis.")] + msgs)


def main() -> int:
    from biomentis.gate.claims import check_evidence, check_simulated_labels, check_verified_claims

    sentence = "FAPROTAX 1.2.12 (current release, Louca lab download page — verified)."
    answer = "## Tools\nWe use QIIME2. " + sentence + " Then we cluster."
    fs = check_verified_claims(answer, "<observation>no version here</observation>")
    _check("unsourced version is blocking", len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "claim", str(fs))
    _check("quote is the exact sentence", bool(fs) and fs[0].quote == sentence and fs[0].quote in answer, str(fs))
    _check("version in observation passes", check_verified_claims(answer, "FAPROTAX_1.2.12.zip listed") == [])
    _check("URL passes", check_verified_claims(sentence.replace("download page", "https://x.org/f"), "") == [])
    _check("verified without version passes", check_verified_claims("24+3+... = 41 ✓ verified by addition", "") == [])

    code = ["import numpy as np\nx = np.random.normal(size=10)\nprint(x)"]
    bad = "## Results\n\nSample data table (class pool; barometer 748 mmHg; correction applied to all)\n\n| a | b |\n"
    fs = check_simulated_labels(bad, code)
    _check("unlabelled sample table is blocking", len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "claim", str(fs))
    _check("quote is the caption line", bool(fs) and fs[0].quote.startswith("Sample data table (class pool"), str(fs))
    good = "## Results\n\nSample data table — simulated for illustration\n\n| a | b |\n"
    _check("labelled in heading passes", check_simulated_labels(good, code) == [])
    later = "Sample data table\n\nNote: these values are synthetic.\n"
    _check("label in next 300 chars passes", check_simulated_labels(later, code) == [])
    _check("no random code: no finding", check_simulated_labels(bad, ["print(1)"]) == [])

    lit_calc_fig = [
        AIMessage(content="<execute>print(query_pubmed('x'))</execute>"),
        AIMessage(content="<observation>Title: a paper</observation>"),
        AIMessage(content="<execute>import matplotlib.pyplot as plt\nx = 3 * 4\nplt.plot([x]); plt.show()</execute>"),
        AIMessage(content="<observation>Plot captured.\nFigure 'T': x = day</observation>"),
    ]
    _check("complete run: no evidence finding", [f for f in check_evidence(_ev(lit_calc_fig)) if f.severity == "blocking"] == [])
    thin = [
        AIMessage(content="<execute>print(2 + 2)</execute>"),
        AIMessage(content="<observation>4</observation>"),
    ]
    fs = check_evidence(_ev(thin))
    _check("no literature tool: blocking evidence finding",
           any(f.check == "evidence" and f.severity == "blocking" and "literature" in f.key for f in fs), str(fs))

    print(f"\n{len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

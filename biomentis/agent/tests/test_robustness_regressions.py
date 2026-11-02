"""Regression suite from the run comparisons (spec section 8.1).

Each case runs the real checker on text copied verbatim from the exported
BioMentis runs (metagenomics Runs 1-3, boiling-point Runs 1-2); the snippets are
in fixtures/robustness/. The only fake is the LLM that extracts quantities: it
returns claims whose quotes are the verbatim snippets.

Run with:
    python -m biomentis.agent.tests.test_robustness_regressions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

FIXTURES = Path(__file__).parent / "fixtures" / "robustness"

_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def _line(name: str, containing: str) -> str:
    return next(line for line in _lines(name) if containing in line)


def _llm(*items: dict):
    """A fake LLM that always extracts ``items`` (quantity claims with verbatim quotes)."""
    reply = json.dumps({"quantities": list(items)})
    return SimpleNamespace(invoke=lambda msgs: AIMessage(content=reply))


def _item(name, value, quote, condition="", derivation="", unit=""):
    return {"name": name, "value": value, "unit": unit, "quote": quote,
            "condition": condition, "derivation": derivation}


def _fig(title: str, code: str = "plt.plot([1]); plt.show()") -> list:
    return [
        AIMessage(content=f"<execute>{code}</execute>"),
        AIMessage(content=f"<observation>Plot captured.\nFigure '{title}': x = day</observation>"),
    ]


def _ev(msgs):
    from biomentis.run_quality import run_evidence
    return run_evidence([HumanMessage(content="Design the lab.")] + msgs)


def case_false_sum() -> None:
    print("\n[1] false sum '= 39' (metagenomics Run 3; the sum is 41)")
    from biomentis.gate.arithmetic import check_arithmetic

    text = (FIXTURES / "meta_run3_library_sum.txt").read_text(encoding="utf-8")
    fs = check_arithmetic(text)
    bad = [f for f in fs if "= 39" in f.quote]
    _check("blocking arithmetic finding on the 39", len(bad) == 1 and bad[0].severity == "blocking"
           and bad[0].check == "arithmetic", str(fs))
    _check("it names the right total, 41", bool(bad) and "41" in bad[0].issue + bad[0].fix, str(bad))
    # The same export line has a second slip nobody had listed: 48+5+4+2+2+2+4 is 67, not 65.
    other = [f for f in fs if "= 65" in f.quote]
    _check("the neighbouring '= 65' (really 67) is caught too", len(other) == 1 and "67" in other[0].fix, str(fs))

    ok = (FIXTURES / "bp_run1_correct_sums.txt").read_text(encoding="utf-8")
    _check("correct sums, units and LaTeX (boiling-point Run 1) give no finding", check_arithmetic(ok) == [], str(check_arithmetic(ok)))


def case_nacl_values() -> None:
    print("\n[2] NaCl i given as 1.80 / 1.84 / 1.86 (boiling-point Run 2)")
    from biomentis.gate.quantities import check_consistency, extract_quantities

    lines = _lines("bp_run2_nacl_i.txt")
    answer = "\n".join(lines)
    q180, q184, q186 = (_line("bp_run2_nacl_i.txt", s) for s in ("= 1.80", "is 1.84", "(1.86/2.00)"))
    llm = _llm(
        _item("nacl_i_app", 1.80, q180, "1.00 mol/kg"),
        _item("nacl_i_app", 1.84, q184, "1.00 mol/kg"),
        _item("nacl_i_app", 1.86, q186, "1.00 mol/kg"),
    )
    claims = extract_quantities(answer, llm)
    fs = check_consistency(claims)
    _check("three claims extracted", len(claims) == 3, str(claims))
    _check("one blocking quantity finding", len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "quantity", str(fs))
    _check("it lists all three values", bool(fs) and all(v in fs[0].issue for v in ("1.8", "1.84", "1.86")), str(fs))
    _check("its quote is verbatim text from the answer", bool(fs) and fs[0].quote in answer, str(fs))


def case_n_per_solute() -> None:
    print("\n[3] 'n = 8 per solute' when the design gives one group x 2 trials")
    from biomentis.gate.quantities import check_derivations, extract_quantities

    answer = "\n".join(_lines("bp_run2_n_per_solute.txt"))
    trials = _line("bp_run2_n_per_solute.txt", "2 solution trials per group")
    llm = _llm(
        _item("groups_per_solute", 1, "one solute each"),
        _item("trials_per_group", 2, "2 solution trials per group (within-group)"),
        _item("values_per_solute_n", 8, "8 values per solute at class level", derivation="groups_per_solute * trials_per_group"),
    )
    claims = extract_quantities(answer, llm)
    fs = check_derivations(claims)
    _check("blocking derivation finding", len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "quantity", str(fs))
    _check("it says 2, not 8", bool(fs) and "= 2" in fs[0].issue and fs[0].fix == "2", str(fs))
    _check("its quote is in the answer", bool(fs) and fs[0].quote in trials, str(fs))


def case_missing_figures() -> None:
    print("\n[4] Figs 14-16 cited but never drawn (boiling-point Run 1)")
    from biomentis.gate.figures import check_figures

    answer = "\n".join(_lines("bp_run1_figs_cited.txt"))
    msgs = []
    for i in range(1, 14):
        msgs += _fig(f"Fig {i} — trace")
    fs = check_figures(answer, _ev(msgs))
    named = " ".join(f.issue for f in fs)
    _check("three blocking figure findings", len(fs) == 3 and all(f.severity == "blocking" and f.check == "figure" for f in fs), str(fs))
    _check("they name 14, 15 and 16", all(n in named for n in ("14", "15", "16")), named)
    drawn_all = msgs + _fig("Fig 14 — Pitzer") + _fig("Fig 15 — baseline") + _fig("Fig 16 — cost")
    _check("drawing them clears the findings", check_figures(answer, _ev(drawn_all)) == [], str(check_figures(answer, _ev(drawn_all))))


def case_stale_figure() -> None:
    print("\n[5] stale figure: $158 drawn, text revised to $414 (boiling-point Run 1)")
    from biomentis.gate.figures import check_figures

    drawn, first, revised = _lines("bp_run1_stale_cost.txt")
    msgs = _fig(drawn) + [AIMessage(content=f"<solution>{first}</solution>"),
                          AIMessage(content=f"<solution>{revised}</solution>")]
    fs = check_figures(revised, _ev(msgs))
    _check("one blocking stale finding naming 158", len(fs) == 1 and fs[0].severity == "blocking"
           and fs[0].check == "figure" and "158" in fs[0].issue, str(fs))
    redrawn = msgs[:3] + _fig("Fig 12a — First-year outlay $414.00; reagents are only $77.00") + msgs[3:]
    _check("redrawing the figure clears it", check_figures(revised, _ev(redrawn)) == [], str(check_figures(revised, _ev(redrawn))))


def case_unsourced_verified() -> None:
    print("\n[6] 'FAPROTAX 1.2.12 verified' with no source (metagenomics Run 3)")
    from biomentis.gate.claims import check_verified_claims

    sentence = (FIXTURES / "meta_run3_faprotax_verified.txt").read_text(encoding="utf-8").strip().lstrip("- ")
    answer = "## Tools\n" + sentence + "\n"
    fs = check_verified_claims(answer, "<observation>qiime2 2024.5 installed</observation>")
    _check("blocking claim finding quoting the sentence", len(fs) == 1 and fs[0].severity == "blocking"
           and fs[0].check == "claim" and fs[0].quote == sentence, str(fs))
    _check("a fetched page that shows 1.2.12 is a source",
           check_verified_claims(answer, "<observation>FAPROTAX 1.2.12 (complete package)</observation>") == [], "")


def case_budget_x4() -> None:
    print("\n[7] budget regression: correct class quantities replaced by x4 (boiling-point Run 2)")
    from biomentis.gate.quantities import check_regressions, extract_quantities, output_numbers

    glucose, total = (_line("bp_run2_budget_x4.txt", s) for s in ("36.04 g", "159 g"))
    draft1 = extract_quantities(glucose, _llm(_item("glucose_class_g", 36.04, "36.04 g", "4 groups, one solute each", unit="g")))
    draft2 = extract_quantities(total, _llm(_item("glucose_class_g", 159, "159 g", "4 groups, one solute each", unit="g")))
    fs = check_regressions(draft1, draft2, set())
    _check("blocking regression finding", len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "regression", str(fs))
    _check("it quotes 36.04 and 159", bool(fs) and "36.04" in fs[0].issue and "159" in fs[0].issue, str(fs))

    printed = [AIMessage(content="<observation>glucose per class: 159.0 g</observation>")]
    _check("a code step that printed 159 justifies the change",
           check_regressions(draft1, draft2, output_numbers([HumanMessage(content="t")] + printed)) == [], "")


def case_unlabelled_sample_data() -> None:
    print("\n[8] sample data not labelled as simulated (boiling-point Run 2)")
    from biomentis.gate.claims import check_simulated_labels

    caption = _lines("bp_run2_sample_data.txt")[0]
    answer = f"## Data handling\n\n{caption}\n\n| Trial | Water (g) | T_plateau (°C) |\n"
    # The export's code typed its values in by hand; the closest verbatim case is a table next to random-number code.
    code = ["import numpy as np\nT = 100.0 + np.random.normal(0, 0.03, 4)\nprint(T)"]
    fs = check_simulated_labels(answer, code)
    _check("blocking claim finding on the caption", len(fs) == 1 and fs[0].severity == "blocking"
           and fs[0].check == "claim" and fs[0].quote.startswith("Sample data table"), str(fs))
    labelled = answer.replace(caption, caption + "\n\nThese values are simulated for illustration.")
    _check("labelling it clears the finding", check_simulated_labels(labelled, code) == [], "")


def case_label_fix() -> None:
    print("\n[9] 'revised after self-critique #3' when the reviewer sent it back (metagenomics Run 3)")
    from biomentis.run_quality import CRITIC_FEEDBACK_PREFIX, GATE_FEEDBACK_PREFIX, REVIEW_FEEDBACK_PREFIX
    from biomentis.ui_core import revision_source, revision_title, stream_agent_events

    old = (FIXTURES / "meta_run3_selfcritique_label.txt").read_text(encoding="utf-8").strip()
    script = [
        AIMessage(content="<solution>draft 1</solution>"),
        HumanMessage(content=CRITIC_FEEDBACK_PREFIX + " Check it."),
        AIMessage(content="<solution>draft 2</solution>"),
        HumanMessage(content=CRITIC_FEEDBACK_PREFIX + " Check it again."),
        AIMessage(content="<solution>draft 3</solution>"),
        HumanMessage(content=REVIEW_FEEDBACK_PREFIX + " and found problems."),
        AIMessage(content="<solution>draft 4</solution>"),
        HumanMessage(content=GATE_FEEDBACK_PREFIX + ": arithmetic."),
        AIMessage(content="<solution>draft 5</solution>"),
    ]
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for msg in script:
            history.append(msg)
            yield {"messages": list(history)}

    agent = SimpleNamespace(app=SimpleNamespace(stream=_stream), use_tool_retriever=False, path="/tmp", user_task="")
    titles = [e.title for e in stream_agent_events(agent, "Plan it.", [], [HumanMessage(content="hi")], thread_id=1) if e.type == "solution"]
    _check("two self-critique rounds, not three", sum("self-critique" in t for t in titles) == 2, str(titles))
    _check("the old 'self-critique #3' title is gone", old not in titles and not any("#" in t for t in titles), str(titles))
    _check("the reviewer round says independent review", titles[3:4] == ["✅ Answer (round 3: revised after independent review)"], str(titles))
    _check("the check round says failed checks", titles[4:5] == ["✅ Answer (round 4: revised after failed checks)"], str(titles))
    _check("helpers agree", revision_source([HumanMessage(content=REVIEW_FEEDBACK_PREFIX + "x")]) == "reviewer"
           and revision_title(3, "reviewer") == "✅ Answer (round 3: revised after independent review)", "")


def main() -> int:
    for case in (case_false_sum, case_nacl_values, case_n_per_solute, case_missing_figures, case_stale_figure,
                 case_unsourced_verified, case_budget_x4, case_unlabelled_sample_data, case_label_fix):
        try:
            case()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _check(f"{case.__name__} ran", False, repr(exc))
    print(f"\n{len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

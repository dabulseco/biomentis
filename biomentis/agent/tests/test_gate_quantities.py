"""Tests for the gate's quantity checks: extraction, consistency, derivations, regressions, code output.

Run with:
    python -m biomentis.agent.tests.test_gate_quantities
"""

from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _q(name, value, quote="", condition="", derivation="", unit=""):
    from biomentis.gate.quantities import QuantityClaim
    return QuantityClaim(name=name, value=value, unit=unit, quote=quote or f"{name} {value}",
                         condition=condition, derivation=derivation)


def _fake(replies):
    """A fake LLM returning each reply in turn, recording the messages it got."""
    calls = []

    def invoke(msgs):
        calls.append(msgs)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        return AIMessage(content=reply(msgs) if callable(reply) else reply)
    return SimpleNamespace(invoke=invoke, calls=calls)


def main() -> int:
    from biomentis.gate.quantities import (
        EXTRACT_SYSTEM_PROMPT, check_against_outputs, check_consistency, check_derivations,
        check_regressions, extract_quantities, output_numbers,
    )

    # Consistency
    same = [_q("nacl_i_app", v, f"i = {v}", "1.00 mol/kg") for v in (1.80, 1.84, 1.86)]
    fs = check_consistency(same)
    _check("same name and condition, values differ: blocking",
           len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "quantity", str(fs))
    _check("blocking quotes an answer claim", bool(fs) and fs[0].quote in {c.quote for c in same}, str(fs))
    fs = check_consistency([_q("nacl_i_app", 1.78, condition="0.50 mol/kg"),
                            _q("nacl_i_app", 1.87, condition="1.00 mol/kg")])
    _check("different conditions: warning only",
           len(fs) == 1 and fs[0].severity == "warning" and fs[0].check == "quantity", str(fs))
    fs = check_consistency([_q("n_samples", 12), _q("n_samples", 14)])
    _check("empty condition: warning", len(fs) == 1 and fs[0].severity == "warning", str(fs))
    _check("within 0.5%: no finding",
           check_consistency([_q("x_total", 1000, condition="c"), _q("x_total", 1004, condition="c")]) == [])

    # Derivations
    fs = check_derivations([
        _q("values_per_solute", 8, "8 values per solute", derivation="groups_per_solute * trials_per_group"),
        _q("groups_per_solute", 1), _q("trials_per_group", 2),
    ])
    _check("wrong derivation: blocking, fix 2",
           len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "quantity" and fs[0].fix == "2"
           and fs[0].quote == "8 values per solute", str(fs))
    _check("right derivation: none", check_derivations([
        _q("values_per_solute", 2, derivation="groups_per_solute * trials_per_group"),
        _q("groups_per_solute", 1), _q("trials_per_group", 2)]) == [])
    _check("missing name: skipped", check_derivations([
        _q("values_per_solute", 8, derivation="groups_per_solute * trials_per_group"),
        _q("groups_per_solute", 1)]) == [])
    _check("unsupported expression: skipped", check_derivations([
        _q("y", 8, derivation="a ^ 2"), _q("a", 4)]) == [])

    # Regressions
    prev = [_q("glucose_class_g", 40, "40 g glucose")]
    new = [_q("glucose_class_g", 159, "159 g glucose")]
    fs = check_regressions(prev, new, {3.0, 41.0})
    _check("unexplained change: blocking regression",
           len(fs) == 1 and fs[0].severity == "blocking" and fs[0].check == "regression"
           and "40" in fs[0].issue and "159" in fs[0].issue and fs[0].quote == "159 g glucose", str(fs))
    _check("recomputed value: none", check_regressions(prev, new, {159.0}) == [])
    _check("unchanged: none", check_regressions(prev, [_q("glucose_class_g", 40.1)], set()) == [])

    # Code-output cross-check
    fs = check_against_outputs([_q("libraries_total", 39)], {41.0})
    _check("total not in outputs: 1 warning",
           len(fs) == 1 and fs[0].severity == "warning" and fs[0].check == "quantity"
           and "not found in any code output" in fs[0].issue, str(fs))
    _check("total in outputs: none", check_against_outputs([_q("libraries_total", 41)], {41.0}) == [])
    _check("plain name not checked", check_against_outputs([_q("temperature_c", 37)], {41.0}) == [])
    _check("derived claim checked",
           len(check_against_outputs([_q("v", 8, derivation="a * b")], {41.0})) == 1)

    # output_numbers: observations of the current run only
    msgs = [
        HumanMessage(content="old question"),
        AIMessage(content="<observation>999</observation>"),
        HumanMessage(content="Design the libraries."),
        AIMessage(content="<execute>print(24 + 17)</execute>"),
        AIMessage(content="<observation>total 41, cost $1,015.50, rate 1.6e-3, delta -2</observation>"),
        HumanMessage(content="<observation>i = 1.87</observation>"),
        AIMessage(content="I think it is 77."),
    ]
    nums = output_numbers(msgs)
    _check("observation numbers collected",
           {41.0, 1015.5, 0.0016, -2.0, 1.87} <= nums, str(nums))
    _check("earlier run and non-observation text excluded", 999.0 not in nums and 77.0 not in nums
           and 24.0 not in nums, str(nums))

    # Extraction: prompt, fences + thinking text, de-duplication
    for phrase in ("snake_case", "condition", "derivation", "quantities"):
        _check(f"prompt mentions {phrase}", phrase in EXTRACT_SYSTEM_PROMPT)
    item = {"name": "libraries_total", "value": 41, "unit": "libraries", "quote": "41 libraries"}
    reply = ("Thinking {about it}...\n```json\n" + json.dumps({"quantities": [item, item,
             {"name": "cost_total", "value": "1,015", "unit": "USD", "quote": "$1,015", "condition": ""},
             {"name": "bad", "value": "lots"}]}) + "\n```")
    llm = _fake([reply])
    claims = extract_quantities("## Design\n41 libraries cost $1,015.", llm)
    _check("one call for a short answer", len(llm.calls) == 1, str(len(llm.calls)))
    _check("system + human messages", isinstance(llm.calls[0][0], SystemMessage)
           and llm.calls[0][0].content == EXTRACT_SYSTEM_PROMPT and isinstance(llm.calls[0][1], HumanMessage))
    _check("fenced JSON after thinking parsed, duplicate merged, bad value skipped",
           [(c.name, c.value) for c in claims] == [("libraries_total", 41.0), ("cost_total", 1015.0)], str(claims))

    # Unparsable reply: retried once, then skipped with a note
    llm = _fake(["not json", json.dumps({"quantities": [item]})])
    claims = extract_quantities("## A\n41 libraries", llm)
    _check("unparsable reply retried once", len(llm.calls) == 2 and len(claims) == 1, f"{len(llm.calls)} {claims}")
    llm = _fake(["not json"])
    _check("twice unparsable: skipped", extract_quantities("## A\n41", llm) == [] and len(llm.calls) == 2)

    # Chunking: a 30,000-character answer with 5 sections (Review Focus 5)
    sections = []
    for k in range(1, 6):
        head = f"## Section {k}\nmarker_s{k} = {k * 10} units\n\n"
        sections.append(head + ("filler text line.\n" * 400)[: 6000 - len(head)])
    answer = "".join(sections)
    _check("answer is 30,000 characters", len(answer) == 30000, str(len(answer)))

    def by_marker(msgs):
        found = re.findall(r"marker_s(\d) = (\d+)", msgs[1].content)
        return json.dumps({"quantities": [
            {"name": f"marker_s{k}", "value": float(v), "unit": "units", "quote": f"marker_s{k} = {v} units"}
            for k, v in found]})
    llm = _fake([by_marker])
    claims = extract_quantities(answer, llm)
    names = {c.name for c in claims}
    _check("chunked: LLM called at least 3 times", len(llm.calls) >= 3, str(len(llm.calls)))
    _check("chunks within the limit", all(len(m[1].content) <= 12000 for m in llm.calls))
    _check("nothing dropped", "".join(m[1].content for m in llm.calls) == answer)
    _check("first and last section claims present", {"marker_s1", "marker_s5"} <= names, str(names))
    _check("all section claims present", len(names) == 5, str(names))

    big = "## Huge\n" + "\n\n".join("p" * 3000 for _ in range(10)) + "\n" + "z" * 15000
    llm = _fake([json.dumps({"quantities": []})])
    extract_quantities(big, llm, max_chunk_chars=5000)
    _check("oversized section split on paragraphs, then hard-split, nothing dropped",
           "".join(m[1].content for m in llm.calls) == big
           and all(len(m[1].content) <= 5000 for m in llm.calls), str([len(m[1].content) for m in llm.calls]))

    print(f"\n{len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

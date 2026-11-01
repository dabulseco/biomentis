"""Tests for the gate's critique: it works from the findings and rubric and keeps verified values.

  (a) The critic's prompt holds the verbatim keep-what-passed sentence, the
      rubric IDs, the blocking finding and the passed quantity; the old
      "Critic harshly" text is gone.
  (b) The critic runs in every critique round, and the feedback message still
      starts with the gate feedback prefix, followed by the critic's reply.
  (c) A claim with a finding is not listed as passed; no rubric / no claims
      are stated plainly.

Run with:
    python -m biomentis.agent.tests.test_gate_critique
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="biomentis_gate_critique_")

from langchain_core.messages import AIMessage, HumanMessage

from biomentis.agent.tests.test_gate_graph import _EVIDENCE_STEPS, _RIGHT, _WRONG, _Scripted
from biomentis.agent.tests.test_gate_graph import _agent as gate_agent
from biomentis.gate import Finding, GateResult
from biomentis.gate.quantities import QuantityClaim
from biomentis.gate.runner import CRITIQUE_PROMPT_TEMPLATE, critique_prompt
from biomentis.run_quality import GATE_FEEDBACK_PREFIX

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []

KEEP = ("Do not change any value listed under PASSED QUANTITIES unless a code step in this round shows "
        "it is wrong; if you change one, recompute it in <execute></execute> first.")


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


class _Quantities:
    def invoke(self, messages):
        return AIMessage(content=json.dumps({"quantities": [
            {"name": "glucose_class_g", "value": 40, "unit": "g", "quote": "40 g of glucose",
             "condition": "", "derivation": ""}]}))


def test_graph_prompt() -> None:
    print("\n(a)(b) the critic's prompt and the feedback message in the graph")
    wrong = "<solution>The total is 24+3 = 28 items, using 40 g of glucose.</solution>"
    right = "<solution>The total is 24+3 = 27 items, using 40 g of glucose.</solution>"
    llm = _Scripted(_EVIDENCE_STEPS + [wrong, right])
    critic = _Scripted(["Recheck the total."])
    agent = gate_agent(llm, critic, planned=1)
    agent.gate_llm = _Quantities()
    agent.go("What is 24+3?")
    _check("the critic ran once", len(critic.requests) == 1, str(len(critic.requests)))
    prompt = critic.requests[0][-1].content if critic.requests else ""
    _check("the verbatim keep sentence", KEEP in prompt, prompt)
    for rid in ("D1", "C1", "L1"):
        _check(f"rubric item {rid}", f"[{rid}]" in prompt, prompt)
    _check("the blocking finding", "24+3 = 28" in prompt, prompt)
    _check("the passed quantity", "glucose_class_g = 40 g" in prompt, prompt)
    _check("no old harsh critic", "Critic harshly" not in prompt, prompt)
    _check("order: findings, rubric, missing work",
           prompt.index("blocking finding") < prompt.index("rubric item") < prompt.index("missing work"), prompt)
    fb = [m for m in llm.requests[3] if isinstance(m, HumanMessage)
          and m.content.startswith(GATE_FEEDBACK_PREFIX)] if len(llm.requests) > 3 else []
    _check("feedback starts with the gate prefix, then the critic",
           len(fb) == 1 and "Recheck the total." in fb[0].content, str(fb))

    llm = _Scripted(_EVIDENCE_STEPS + [wrong, wrong, right])
    critic = _Scripted(["First.", "Second."])
    agent = gate_agent(llm, critic, planned=0)
    agent.gate_llm = _Quantities()
    agent.go("What is 24+3?")
    _check("the critic runs in an unplanned critique round too", len(critic.requests) >= 1, str(len(critic.requests)))


def test_prompt_builder() -> None:
    print("\n(c) the prompt builder")
    claims = [QuantityClaim("a_total", 5.0, "", "a = 5"), QuantityClaim("b_g", 2.5, "g", "b = 2.5 g")]
    result = GateResult(0, [Finding("quantity", "blocking", "b = 2.5 g", "does not match", "recompute")], "off")
    p = critique_prompt("T", None, result, claims)
    _check("no rubric stated", "(no rubric for this run)" in p, p)
    _check("a claim without a finding is passed", "a_total = 5" in p, p)
    _check("a claim with a finding is not", "b_g = 2.5" not in p, p)
    p = critique_prompt("T", None, GateResult(0, [], "off"), None)
    _check("none", "(none)" in p, p)
    _check("template has all placeholders",
           all(k in CRITIQUE_PROMPT_TEMPLATE for k in ("{task}", "{rubric}", "{findings}", "{passed_quantities}")))


def main() -> int:
    test_graph_prompt()
    test_prompt_builder()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

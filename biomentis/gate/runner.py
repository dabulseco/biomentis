"""Run every completion check on the run's latest candidate answer.

``run_gate`` is called by the agent graph's ``gate`` node, after the candidate
``<solution>`` message has been appended, so ``run_evidence`` sees it as the
latest answer. ``gate_feedback`` is the message that sends the findings back.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage

from biomentis.gate import Finding, GateResult
from biomentis.gate.arithmetic import check_arithmetic
from biomentis.gate.claims import check_evidence, check_simulated_labels, check_verified_claims
from biomentis.gate.figures import check_figures
from biomentis.gate.quantities import (
    check_against_outputs,
    check_consistency,
    check_derivations,
    check_regressions,
    extract_quantities,
    output_numbers,
)
from biomentis.run_quality import GATE_FEEDBACK_PREFIX, current_run, run_evidence

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)


def _text(m: BaseMessage) -> str:
    return m.content if isinstance(m.content, str) else ""


def _solution_indices(run: list[BaseMessage]) -> list[int]:
    return [i for i, m in enumerate(run) if isinstance(m, AIMessage) and _SOLUTION_RE.search(_text(m))]


def _review_findings(agent, messages: list[BaseMessage], result: GateResult) -> tuple[list[Finding], str]:
    """The reviewer's findings and state ("ok" | "failed"). Task 12 fills this in."""
    return [], "ok"


def run_gate(agent, messages: list[BaseMessage], round_number: int, rubric=None) -> GateResult:
    run = current_run(messages)
    solutions = _solution_indices(run)
    answer = _SOLUTION_RE.search(_text(run[solutions[-1]])).group(1) if solutions else ""
    ev = run_evidence(messages)
    observations = [_text(m) for m in run if _text(m).lstrip().startswith("<observation>")]
    code = [match.group(1) for m in run if (match := _EXECUTE_RE.search(_text(m)))]

    findings: list[Finding] = []
    findings += check_arithmetic(answer)
    findings += check_figures(answer, ev)
    findings += check_verified_claims(answer, "\n".join(observations))
    findings += check_simulated_labels(answer, code)
    findings += check_evidence(ev)

    llm = getattr(agent, "gate_llm", None) or agent.llm
    try:
        claims = extract_quantities(answer, llm)
    except Exception as exc:  # the other checks still count
        print(f"[gate] quantity extraction failed ({type(exc).__name__}: {exc}); no quantity checks this round")
        claims = None
    if claims is not None:
        findings += check_consistency(claims)
        findings += check_derivations(claims)
        prev = getattr(agent, "_gate_prev_claims", None)
        if prev is not None and len(solutions) >= 2:
            # Numbers printed since the previous candidate justify a change.
            findings += check_regressions(prev, claims, output_numbers(run[solutions[-2] + 1:]))
        findings += check_against_outputs(claims, output_numbers(messages))
        agent._gate_prev_claims = claims

    control = getattr(agent, "round_control", None)
    result = GateResult(
        round=round_number,
        findings=findings,
        reviewer_state="off",
        accepted=set(control.accepted) if control is not None else set(),
    )
    if getattr(agent, "review_llm", None) is not None:
        review, result.reviewer_state = _review_findings(agent, messages, result)
        result.findings += review
    return result


def _describe(f: Finding) -> str:
    quote = f' "{f.quote}"' if f.quote else ""
    fix = f" Fix: {f.fix}" if f.fix else ""
    return f"- [{f.check}]{quote}: {f.issue}{fix}"


def gate_feedback(result: GateResult) -> str:
    """The message that sends the findings back: blocking first, then warnings."""
    blocking = result.blocking
    warnings = [f for f in result.findings if f.severity == "warning"]
    if blocking:
        lines = [f"{GATE_FEEDBACK_PREFIX} in your final answer. Fix each of these:"]
        lines += [_describe(f) for f in blocking]
    else:
        lines = [f"{GATE_FEEDBACK_PREFIX} in your final answer: none that block it."]
    if warnings:
        lines += ["", "Warnings (fix them if they are real problems):"]
        lines += [_describe(f) for f in warnings]
    lines += [
        "",
        "Recompute any number you change in an <execute></execute> step. Then give the complete final "
        "answer again inside <solution></solution>, keeping the parts that were right.",
    ]
    return "\n".join(lines)

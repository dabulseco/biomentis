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
from biomentis.independent_review import coverage_findings, review_findings
from biomentis.run_quality import GATE_FEEDBACK_PREFIX, current_run, run_evidence

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)


def _text(m: BaseMessage) -> str:
    return m.content if isinstance(m.content, str) else ""


def _solution_indices(run: list[BaseMessage]) -> list[int]:
    return [i for i, m in enumerate(run) if isinstance(m, AIMessage) and _SOLUTION_RE.search(_text(m))]


# The checks whose questioned numbers the reviewer should see in full.
_FOCUS_CHECKS = ("quantity", "arithmetic", "regression")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def focus_numbers(findings: list[Finding]) -> set[str]:
    """The numbers quoted by this round's quantity, arithmetic and regression findings.

    As written in the answer, plus the same number without thousands
    separators (code prints 1015, the answer says 1,015). Only numbers with
    a decimal point or at least 3 significant digits count: 12, 24 or 36
    would match nearly every step's output.
    """
    numbers: set[str] = set()
    for f in findings:
        if f.check not in _FOCUS_CHECKS:
            continue
        for n in _NUMBER_RE.findall(f.quote):
            n = n.rstrip(",")
            if "." not in n and len(n.replace(",", "").lstrip("0")) < 3:
                continue
            numbers |= {n, n.replace(",", "")}
    return numbers


def _review_findings(agent, messages: list[BaseMessage], result: GateResult,
                     rubric=None) -> tuple[list[Finding], str]:
    """The reviewer's findings and state: "ok", or "failed" with a "review failed" warning.

    The review fails when its call raised or its reply was still unusable
    after the one retry; then no reviewer-dependent check is reported.
    Rubric coverage is checked only when the run has a rubric.
    """
    review = agent.run_review(messages, rubric=rubric, focus_numbers=focus_numbers(result.findings))
    if review.error:
        return [Finding(check="review", severity="warning", quote="", issue=f"review failed: {review.error}")], "failed"
    findings = coverage_findings(review, rubric) if rubric is not None else []
    return findings + review_findings(review), "ok"


def _safe(check: str, name: str, fn, *args) -> list[Finding]:
    """Run one checker; a crash becomes a warning, so it never costs the run its answer."""
    try:
        return list(fn(*args))
    except Exception as exc:
        print(f"[gate] {name} failed ({type(exc).__name__}: {exc}); continuing without it")
        return [Finding(check=check, severity="warning", quote="",
                        issue=f"{name} failed: {type(exc).__name__}: {exc}")]


def run_gate(agent, messages: list[BaseMessage], round_number: int, rubric=None) -> GateResult:
    run = current_run(messages)
    solutions = _solution_indices(run)
    answer = _SOLUTION_RE.search(_text(run[solutions[-1]])).group(1) if solutions else ""
    ev = run_evidence(messages)
    observations = [_text(m) for m in run if _text(m).lstrip().startswith("<observation>")]
    code = [match.group(1) for m in run if (match := _EXECUTE_RE.search(_text(m)))]

    findings: list[Finding] = []
    findings += _safe("arithmetic", "check_arithmetic", check_arithmetic, answer)
    findings += _safe("figure", "check_figures", check_figures, answer, ev)
    findings += _safe("claim", "check_verified_claims", check_verified_claims, answer, "\n".join(observations))
    findings += _safe("claim", "check_simulated_labels", check_simulated_labels, answer, code)
    findings += _safe("evidence", "check_evidence", check_evidence, ev)

    llm = getattr(agent, "gate_llm", None) or agent.llm
    try:
        claims = extract_quantities(answer, llm)
    except Exception as exc:  # the other checks still count
        print(f"[gate] quantity extraction failed ({type(exc).__name__}: {exc}); no quantity checks this round")
        claims = None
    if claims is not None:
        findings += _safe("quantity", "check_consistency", check_consistency, claims)
        findings += _safe("quantity", "check_derivations", check_derivations, claims)
        prev = getattr(agent, "_gate_prev_claims", None)
        if prev is not None and len(solutions) >= 2:
            # Numbers printed since the previous candidate justify a change.
            findings += _safe("regression", "check_regressions",
                              lambda: check_regressions(prev, claims, output_numbers(run[solutions[-2] + 1:])))
        findings += _safe("quantity", "check_against_outputs",
                          lambda: check_against_outputs(claims, output_numbers(messages)))
        agent._gate_prev_claims = claims

    control = getattr(agent, "round_control", None)
    result = GateResult(
        round=round_number,
        findings=findings,
        reviewer_state="off",
        accepted=set(control.accepted) if control is not None else set(),
    )
    if getattr(agent, "review_llm", None) is not None:
        try:
            review, result.reviewer_state = _review_findings(agent, messages, result, rubric)
        except Exception as exc:  # a reviewer outage is reported, not fatal
            print(f"[gate] review failed ({type(exc).__name__}: {exc})")
            review = [Finding(check="review", severity="warning", quote="",
                              issue=f"review failed: {type(exc).__name__}: {exc}")]
            result.reviewer_state = "failed"
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
        lines = [f"{GATE_FEEDBACK_PREFIX}: none blocking. This is a planned review round; "
                 "improve the answer where the feedback below shows a real gap."]
    if warnings:
        lines += ["", "Warnings (fix them if they are real problems):"]
        lines += [_describe(f) for f in warnings]
    lines += [
        "",
        "Recompute any number you change in an <execute></execute> step. Then give the complete final "
        "answer again inside <solution></solution>, keeping the parts that were right.",
    ]
    return "\n".join(lines)

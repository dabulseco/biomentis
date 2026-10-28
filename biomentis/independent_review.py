"""Independent review of a finished answer by a second model.

The agent's self-critique runs on the agent's own model, so it shares that
model's blind spots: in a 2026-10-28 CHO-workflow run, deepseek-v4.1-flash:cloud
corrected five of its numbers in its critique round and kept a 50x scale-up
error through both drafts. The evidence check (run_quality) only confirms that
sources, calculations and figures exist, not that the answer agrees with them.

This module has a different model (the Critic model picked in the sidebar)
review the run once it has an answer: whether each change the self-critique
made was valid, whether the final answer's numbers, units and claims match the
code output and sources of the run, and whether the reason given for each
major workflow decision holds. A "needs rework" verdict sends the problems back
to the agent for one correction round, then the reviewer checks again; both
reviews are reported, and anything left is the user's call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from biomentis.run_quality import CRITIC_FEEDBACK_PREFIX, REVIEW_FEEDBACK_PREFIX, current_run

VERDICTS = ("accept", "accept_with_corrections", "needs_rework")
VERDICT_LABELS = {
    "accept": "✅ Accept",
    "accept_with_corrections": "🟡 Accept with corrections",
    "needs_rework": "🔴 Needs rework",
}

# The evidence handed to the reviewer: each step's code and output, cut to fit.
_STEP_CODE_CHARS = 1500
_STEP_OUTPUT_CHARS = 2500
_EVIDENCE_TOTAL_CHARS = 90_000

REVIEW_SYSTEM_PROMPT = """You are an independent scientific reviewer. A research agent (a different model) worked on a task by writing and running code, consulting sources, and drafting an answer; its own self-critique then revised the draft. You check the work. You cannot run code: judge from the code outputs and sources shown in EVIDENCE.

Check, in this order:
1. CHANGES: every substantive difference between FIRST_DRAFT and FINAL_ANSWER. For each, is the change valid (supported by EVIDENCE or sound domain knowledge), invalid (wrong, or made worse), or unverified (nothing in EVIDENCE settles it)?
2. PROBLEMS in FINAL_ANSWER:
   - number: a value that disagrees with what the code in EVIDENCE computed, or that is wrong for the domain
   - unit: a unit error or an inconsistent unit (e.g. rpm used where rev/s is needed, pg vs ng)
   - unsupported: a factual claim, limit or specification that no source in EVIDENCE supports and that is doubtful
   - contradiction: two parts of the answer that disagree
   - rationale: a design decision whose stated reason is wrong, missing, or does not follow
3. RATIONALE: the 3 to 8 most important decisions in FINAL_ANSWER (choices of method, order of steps, criteria, limits). Is the reason given for each sound?

Verdict:
- "needs_rework" if any problem would change a decision, a quantity someone would act on, or a safety/quality limit
- "accept_with_corrections" if the problems are minor and local
- "accept" if you found no problems

Be specific and conservative: every problem must quote the exact text from FINAL_ANSWER it concerns, and say what is wrong and what the correct value or reasoning is. Do not flag style. Do not invent problems to have something to say.

Return ONLY a JSON object, no markdown fences, with exactly these keys:
{
  "verdict": "accept" | "accept_with_corrections" | "needs_rework",
  "summary": "<2-3 sentences: overall judgment>",
  "changes": [{"change": "<what changed>", "status": "valid" | "invalid" | "unverified", "reason": "<why, citing EVIDENCE where possible>"}],
  "problems": [{"kind": "number" | "unit" | "unsupported" | "contradiction" | "rationale", "quote": "<exact text from FINAL_ANSWER>", "issue": "<what is wrong>", "fix": "<the correct value or reasoning>"}],
  "rationale": [{"decision": "<the decision>", "sound": "yes" | "no" | "unverified", "note": "<one sentence>"}]
}"""


@dataclass
class Review:
    """One reviewer verdict, or the reason there isn't one (`error`)."""

    verdict: str = ""
    summary: str = ""
    changes: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    rationale: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def needs_rework(self) -> bool:
        return not self.error and self.verdict == "needs_rework"


def _solution_body(text: str) -> str:
    match = re.search(r"<solution>(.*?)</solution>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more characters]"


def review_inputs(messages: list[BaseMessage]) -> dict:
    """The task, first draft, critiques, final answer and evidence of the latest run."""
    run = current_run(messages)
    task = run[0].content if run and isinstance(run[0].content, str) else ""
    answers = [_solution_body(m.content) for m in run if isinstance(m, AIMessage) and isinstance(m.content, str)]
    answers = [a for a in answers if a]
    critiques = [
        m.content[len(CRITIC_FEEDBACK_PREFIX) :].strip()
        for m in run
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith(CRITIC_FEEDBACK_PREFIX)
    ]
    steps: list[tuple[str, str]] = []
    code = None
    for m in run:
        content = m.content if isinstance(m.content, str) else ""
        if content.lstrip().startswith("<observation>"):
            if code is not None:
                steps.append((code, content.strip()[len("<observation>") :].removesuffix("</observation>")))
            code = None
            continue
        match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL | re.IGNORECASE)
        if match and isinstance(m, AIMessage):
            code = match.group(1)
    # Shrink each step's share when a long run would not fit.
    per_step = max(1, len(steps))
    scale = min(1.0, _EVIDENCE_TOTAL_CHARS / (per_step * (_STEP_CODE_CHARS + _STEP_OUTPUT_CHARS)))
    code_cap = max(300, int(_STEP_CODE_CHARS * scale))
    output_cap = max(500, int(_STEP_OUTPUT_CHARS * scale))
    evidence = "\n\n".join(
        f"[step {i}] CODE:\n{_clip(c, code_cap)}\n[step {i}] OUTPUT:\n{_clip(o, output_cap)}"
        for i, (c, o) in enumerate(steps, 1)
    )
    return {
        "task": task,
        "first_draft": answers[0] if answers else "",
        "final_answer": answers[-1] if answers else "",
        "critiques": critiques,
        "evidence": evidence,
    }


def build_review_prompt(inputs: dict) -> str:
    critiques = "\n\n".join(f"[round {i}]\n{c}" for i, c in enumerate(inputs["critiques"], 1)) or "(none)"
    same = inputs["first_draft"] == inputs["final_answer"]
    return "\n\n".join(
        [
            f"TASK:\n{inputs['task']}",
            f"EVIDENCE (code the agent ran and its output):\n{inputs['evidence'] or '(no code was run)'}",
            "FIRST_DRAFT:\n" + ("(same as FINAL_ANSWER: the self-critique did not change it)" if same else inputs["first_draft"]),
            f"SELF_CRITIQUE (the agent's own feedback, which led to the revision):\n{critiques}",
            f"FINAL_ANSWER:\n{inputs['final_answer']}",
        ]
    )


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _dicts(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def parse_review(text: str) -> Review:
    """A Review from the reviewer's reply; `error` set when it isn't usable."""
    data = _extract_json(text)
    if data is None:
        return Review(error="the reviewer's reply was not valid JSON")
    verdict = str(data.get("verdict", "")).strip().lower().replace(" ", "_")
    if verdict not in VERDICTS:
        return Review(error=f"the reviewer gave no valid verdict ({verdict or 'none'})")
    return Review(
        verdict=verdict,
        summary=str(data.get("summary", "")).strip(),
        changes=_dicts(data.get("changes")),
        problems=_dicts(data.get("problems")),
        rationale=_dicts(data.get("rationale")),
    )


def feedback_for_agent(review: Review, reviewer: str) -> str:
    """The correction request: the reviewer's problems, to fix and nothing else."""
    lines = [f"{REVIEW_FEEDBACK_PREFIX} ({reviewer}) and found problems that must be fixed:", ""]
    for i, p in enumerate(review.problems, 1):
        lines.append(f"{i}. [{p.get('kind', 'problem')}] \"{p.get('quote', '')}\"")
        lines.append(f"   Problem: {p.get('issue', '')}")
        if p.get("fix"):
            lines.append(f"   Suggested fix: {p['fix']}")
    invalid = [c for c in review.changes if c.get("status") == "invalid"]
    if invalid:
        lines += ["", "Changes from your first draft that the reviewer judged wrong:"]
        lines += [f"- {c.get('change', '')}: {c.get('reason', '')}" for c in invalid]
    lines += [
        "",
        "Fix only these. Where a number is involved, recompute it in code inside <execute></execute> and use "
        "the computed value. If you disagree with a finding, check it in code or against a source and say "
        "why in the answer. Then give the complete corrected final answer inside <solution></solution>.",
    ]
    return "\n".join(lines)


def format_review(review: Review, reviewer: str, after_correction: bool) -> tuple[str, str]:
    """(title, markdown) for showing one review under the answer."""
    title = "🔎 Re-review after correction" if after_correction else "🔎 Independent review"
    if review.error:
        return title, f"**Reviewer:** {reviewer}\n\nNo review: {review.error}."
    parts = [f"**Reviewer:** {reviewer} · **Verdict:** {VERDICT_LABELS[review.verdict]}"]
    if review.summary:
        parts.append(review.summary)
    if review.problems:
        rows = [
            f"- **{p.get('kind', 'problem')}** — “{p.get('quote', '')}”: {p.get('issue', '')}"
            + (f" *Fix:* {p['fix']}" if p.get("fix") else "")
            for p in review.problems
        ]
        parts.append("**Problems found**\n" + "\n".join(rows))
    if review.changes:
        marks = {"valid": "✓", "invalid": "✗", "unverified": "?"}
        rows = [
            f"- {marks.get(c.get('status', ''), '?')} {c.get('change', '')} — {c.get('reason', '')}" for c in review.changes
        ]
        parts.append("**Changes made by the self-critique**\n" + "\n".join(rows))
    if review.rationale:
        marks = {"yes": "✓", "no": "✗", "unverified": "?"}
        rows = [
            f"- {marks.get(str(r.get('sound', '')).lower(), '?')} {r.get('decision', '')} — {r.get('note', '')}"
            for r in review.rationale
        ]
        parts.append("**Workflow rationale**\n" + "\n".join(rows))
    return title, "\n\n".join(parts)

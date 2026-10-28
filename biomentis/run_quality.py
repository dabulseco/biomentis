"""What a run actually did: the evidence check before an answer, and the summary after.

Two runs of the same prompt on the same model (deepseek-v4.1-flash:cloud,
2026-10-28) differed by 19 code steps and 3 figures against 6 steps and none,
because how far the research goes was left to the model. This module makes the
difference visible and enforceable:

* `missing_evidence` lists what a research answer is missing (a literature
  source, a figure, a calculation). With the check on, `generate` sends the
  first answer back with `evidence_nudge` instead of accepting it, at most
  `MAX_EVIDENCE_ASKS` times per run.
* `format_quality_summary` is the one-line "how this answer was made" shown
  under the answer, so a thin run is obvious at a glance.

Everything is read from the run's message history, so it works the same for a
live run, a traced run, and a run with tracing off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage

from biomentis.utils import ANSWER_CONTINUE_NUDGE, NOT_DONE_NUDGE, PARSE_ERROR_NUDGE

# Tools that consult the literature or the web. Local protocols are counted
# separately: they are lab methods, not evidence for a design choice.
LITERATURE_TOOLS = (
    "query_pubmed",
    "query_arxiv",
    "query_scholar",
    "search_google",
    "keyless_web_search",
    "advanced_web_search_claude",
    "extract_url_content",
    "extract_pdf_content",
    "fetch_supplementary_info_from_doi",
)
PROTOCOL_TOOLS = ("read_local_protocol", "list_local_protocols", "search_protocols", "get_protocol_details")

# The critic's feedback starts with this (see A1.execute_self_critic).
CRITIC_FEEDBACK_PREFIX = "Wait... this is not enough to solve the task."
# The independent reviewer's findings, sent back to the agent, start with this
# (see biomentis.independent_review).
REVIEW_FEEDBACK_PREFIX = "An independent reviewer checked your final answer"

MAX_EVIDENCE_ASKS = 2
EVIDENCE_NUDGE_PREFIX = "Before this can be the final answer, the run still needs:"

_ERROR_RE = re.compile(r"^\s*(Error\b|Traceback|.*\bError querying\b)", re.IGNORECASE)


@dataclass
class RunEvidence:
    """Counts for one run, from its message history."""

    code_steps: int = 0
    calculation_steps: int = 0
    literature_tools: set[str] = field(default_factory=set)
    protocol_steps: int = 0
    figures: int = 0
    failed_steps: int = 0
    truncated_outputs: int = 0
    retries: int = 0
    unfinished_sent_back: int = 0
    critique_rounds: int = 0
    evidence_asks: int = 0
    review_corrections: int = 0


def _is_loop_message(content: str) -> bool:
    """A HumanMessage the agent loop wrote, not the user."""
    from biomentis.context_budget import WRAP_UP_INSTRUCTION

    return (
        content in (PARSE_ERROR_NUDGE, NOT_DONE_NUDGE, ANSWER_CONTINUE_NUDGE, WRAP_UP_INSTRUCTION)
        or content.startswith(EVIDENCE_NUDGE_PREFIX)
        or content.startswith(CRITIC_FEEDBACK_PREFIX)
        or content.startswith(REVIEW_FEEDBACK_PREFIX)
        or content.lstrip().startswith("<observation>")
    )


def current_run(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The messages of the latest run: from the user's last prompt on.

    The history also holds earlier chat turns, which must not count.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and not _is_loop_message(m.content):
            return messages[i:]
    return list(messages)


def run_evidence(messages: list[BaseMessage]) -> RunEvidence:
    """Count what the latest run in `messages` did."""
    ev = RunEvidence()
    pending_code: str | None = None
    for m in current_run(messages):
        content = m.content if isinstance(m.content, str) else ""
        if isinstance(m, HumanMessage):
            if content == PARSE_ERROR_NUDGE:
                ev.retries += 1
            elif content == NOT_DONE_NUDGE:
                ev.unfinished_sent_back += 1
            elif content.startswith(EVIDENCE_NUDGE_PREFIX):
                ev.evidence_asks += 1
            elif content.startswith(CRITIC_FEEDBACK_PREFIX):
                ev.critique_rounds += 1
            elif content.startswith(REVIEW_FEEDBACK_PREFIX):
                ev.review_corrections += 1
            # observations are HumanMessages on Claude; fall through for them
            if not content.lstrip().startswith("<observation>"):
                continue
        stripped = content.lstrip()
        if stripped.startswith("<observation>"):
            body = stripped[len("<observation>") :]
            failed = bool(_ERROR_RE.match(body))
            ev.failed_steps += failed
            ev.truncated_outputs += body.lstrip().startswith("The output is too long")
            ev.figures += body.count("Plot captured.")
            if pending_code is not None and not failed:
                used = {t for t in LITERATURE_TOOLS if f"{t}(" in pending_code}
                ev.literature_tools |= used
                protocols = any(f"{t}(" in pending_code for t in PROTOCOL_TOOLS)
                ev.protocol_steps += protocols
                if not used and not protocols:
                    ev.calculation_steps += 1
            pending_code = None
            continue
        match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL | re.IGNORECASE)
        if match:
            ev.code_steps += 1
            pending_code = match.group(1)
    return ev


# Each kind of evidence: (short label for the summary, request to the model).
EVIDENCE_ITEMS = {
    "literature": (
        "no literature source",
        "at least one literature or web source consulted in code (for example query_pubmed, "
        "query_arxiv or search_google), with the findings used in the answer",
    ),
    "calculation": (
        "no calculations in code",
        "the quantitative claims computed in code (rates, speeds, power, yields, concentrations, "
        "limits), with units checked, rather than stated",
    ),
    "figure": (
        "no figure",
        "at least one figure of the key comparison or trend, made with matplotlib",
    ),
}


def missing_evidence(ev: RunEvidence) -> list[str]:
    """Keys of EVIDENCE_ITEMS the run lacks; empty when it has them all."""
    missing = []
    if not ev.literature_tools:
        missing.append("literature")
    if ev.calculation_steps == 0:
        missing.append("calculation")
    if ev.figures == 0:
        missing.append("figure")
    return missing


def evidence_nudge(missing: list[str]) -> str:
    """The message that sends an answer back for the evidence it lacks."""
    items = "\n".join(f"- {EVIDENCE_ITEMS[key][1]}" for key in missing)
    return (
        f"{EVIDENCE_NUDGE_PREFIX}\n{items}\n"
        "Do these now in <execute></execute> steps. Then give the complete final answer again inside "
        "<solution></solution>, updated with what you found."
    )


def format_quality_summary(ev: RunEvidence, settings: str = "") -> str:
    """One line on how the answer was made, plus lines on what was missing.

    `settings` (model, temperatures, reviewer) is shown as its own line, so
    runs can be compared for reproducibility.
    """

    def n(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    parts = [
        n(ev.code_steps, "code step"),
        n(ev.calculation_steps, "calculation"),
        n(len(ev.literature_tools), "literature source"),
        n(ev.protocol_steps, "protocol lookup"),
        n(ev.figures, "figure"),
        n(ev.critique_rounds, "critique round"),
    ]
    issues = []
    if ev.failed_steps:
        issues.append(n(ev.failed_steps, "step") + " ended in an error")
    if ev.truncated_outputs:
        issues.append(n(ev.truncated_outputs, "output") + " cut at 10,000 characters")
    if ev.retries:
        issues.append(n(ev.retries, "reply") + " without a tag")
    if ev.unfinished_sent_back:
        issues.append(n(ev.unfinished_sent_back, "unfinished answer") + " sent back")
    if ev.evidence_asks:
        issues.append(f"asked {n(ev.evidence_asks, 'time')} for missing evidence")
    if ev.review_corrections:
        issues.append(n(ev.review_corrections, "correction round") + " from the independent review")
    lines = ["**How this answer was made:** " + " · ".join(parts)]
    if settings:
        lines.append(f"**Settings:** {settings}")
    if issues:
        lines.append("**Along the way:** " + "; ".join(issues))
    missing = missing_evidence(ev)
    if missing:
        labels = ", ".join(EVIDENCE_ITEMS[key][0] for key in missing)
        lines.append(f"**⚠️ Missing:** {labels}. Check those parts of the answer before relying on it.")
    return "\n\n".join(lines)

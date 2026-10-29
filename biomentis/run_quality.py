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
* Figures are asked for by name. One figure used to satisfy the check, so an
  HCP control-strategy run (2026-10-28) computed a clearance train, a 20-lot
  history and an ELISA coverage estimate as printed tables and plotted none
  of them. `unplotted_tables` finds such tables, and `unmatched_figure_refs`
  finds figures an answer cites that were never drawn; each is asked about
  once.

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
# Phrases in the nudge that mark which figure request it made, so each is
# made at most once per run.
UNPLOTTED_MARK = "computed results that have no figure yet"
FIGURE_REFS_MARK = "the figures the answer refers to"

# A computed table: this many consecutive printed lines with 2+ numbers each.
TABLE_MIN_ROWS = 4
MAX_TABLES_ASKED = 8
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?%?|nan\b|inf\b|--(?=\s|$)|n/a\b)")
_RULE_RE = re.compile(r"^[\s=\-_*#~.]*$")
# A figure's summary line: its title, then its axis labels (see describe_figure).
_FIGURE_LINE_RE = re.compile(r"^Figure '.*$", re.MULTILINE)
_FIGURE_NUMBER_RE = re.compile(r"\bFig(?:ure)?s?\.?\s*(\d+)", re.IGNORECASE)
# Words too common in headings and titles to show that a figure covers a table.
_COMMON_WORDS = {
    "item", "the", "and", "for", "from", "with", "per", "all", "each", "step", "steps",
    "corrected", "fixed", "final", "table", "figure", "fig", "data", "result", "results",
    "using", "vs", "versus", "of", "on", "in", "to", "by", "basis", "check", "summary",
}

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
    unplotted_asks: int = 0
    figure_ref_asks: int = 0
    # Each drawn figure's summary line: "Figure '<title>': x = <label> [...], y = ..."
    figure_titles: list[str] = field(default_factory=list)
    # Printed tables from calculation steps that drew no figure, each as
    # (heading, size), latest version of each, in run order. Size is rows
    # times numbers per row, so a wide series outranks a list of conversions.
    tables: list[tuple[str, int]] = field(default_factory=list)


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
                ev.unplotted_asks += UNPLOTTED_MARK in content
                ev.figure_ref_asks += FIGURE_REFS_MARK in content
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
            ev.figure_titles += _FIGURE_LINE_RE.findall(body)
            if pending_code is not None and not failed:
                used = {t for t in LITERATURE_TOOLS if f"{t}(" in pending_code}
                ev.literature_tools |= used
                protocols = any(f"{t}(" in pending_code for t in PROTOCOL_TOOLS)
                ev.protocol_steps += protocols
                if not used and not protocols:
                    ev.calculation_steps += 1
                    if "Plot captured." not in body:
                        for table in printed_tables(body):
                            _add_table(ev.tables, table)
            pending_code = None
            continue
        match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL | re.IGNORECASE)
        if match:
            ev.code_steps += 1
            pending_code = match.group(1)
    return ev


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9/]+", text.lower()) if w not in _COMMON_WORDS and len(w) > 2}


def _heading(lines: list[str]) -> str:
    """The heading of a table, from the lines printed before it (nearest last).

    The agent prints an upper-case title, a rule and a column header. The
    title names the table, the header its columns; both are kept.
    """
    candidates = [
        " ".join(ln.split()) for ln in lines
        if ln.strip() and not _RULE_RE.match(ln) and len(_NUMBER_RE.findall(ln)) < 2
    ][-3:]
    if not candidates:
        return ""
    nearest = candidates[-1][:80]
    for ln in reversed(candidates):
        letters = [c for c in ln if c.isalpha()]
        if len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.6 * len(letters):
            return ln[:100] if ln[:80] == nearest else f"{ln[:100]} [{nearest}]"
    return nearest


def printed_tables(body: str) -> list[tuple[str, int]]:
    """Tables in one step's printed output: (heading, size).

    A table is TABLE_MIN_ROWS or more consecutive lines with two or more
    numbers each: a series across steps, lots, levels or conditions.
    """
    lines = body.splitlines()
    counts = [0 if _RULE_RE.match(line) else len(_NUMBER_RE.findall(line)) for line in lines]
    tables, start = [], None
    for i, count in enumerate([*counts, 0]):
        numeric = count >= 2
        if numeric and start is None:
            start = i
        elif not numeric and start is not None:
            rows = i - start
            if rows >= TABLE_MIN_ROWS:
                per_row = sorted(counts[start:i])[rows // 2]
                heading = _heading(lines[max(0, start - 6) : start]) or " ".join(lines[start].split())[:100]
                tables.append((heading, rows * per_row))
            start = None
    return tables


def _add_table(tables: list[tuple[str, int]], table: tuple[str, int]) -> None:
    """Add `table`, replacing an earlier version of it (a corrected rerun)."""
    words = _words(table[0])
    for i, (heading, _) in enumerate(tables):
        old = _words(heading)
        if words and old and len(words & old) >= 0.6 * min(len(words), len(old)):
            del tables[i]
            break
    tables.append(table)


def unplotted_tables(ev: RunEvidence) -> list[tuple[str, int]]:
    """Computed tables no figure of the run seems to show: the largest
    MAX_TABLES_ASKED, in run order. Which deserve a figure is left to the
    model; a list of unit conversions is a table too.

    A figure shows a table when its title and axis labels share two words
    with the table's heading and column header (or their only word).
    """
    titles = [_words(t) for t in ev.figure_titles]
    left = []
    for heading, rows in ev.tables:
        words = _words(heading)
        if any(len(words & t) >= min(2, len(words)) for t in titles if words):
            continue
        left.append((heading, rows))
    largest = sorted(left, key=lambda t: -t[1])[:MAX_TABLES_ASKED]
    return [t for t in left if t in largest]


def unmatched_figure_refs(answer: str, ev: RunEvidence) -> list[int]:
    """Figure numbers the answer cites that no drawn figure accounts for.

    Drawn figures are matched by a number in their title ("Fig 2 ..."). An
    untitled figure may be any one of the rest, so refs are only reported
    when there are more of them than untitled figures.
    """
    answer = re.sub(r"<execute>.*?</execute>|```.*?```", "", answer or "", flags=re.DOTALL)
    refs = {int(n) for n in _FIGURE_NUMBER_RE.findall(answer)}
    if not refs:
        return []
    numbered = set()
    unnumbered = set()
    for line in ev.figure_titles:
        title = line.split("':", 1)[0]
        found = {int(n) for n in _FIGURE_NUMBER_RE.findall(title)}
        if found:
            numbered |= found
        else:
            unnumbered.add(title)
    missing = sorted(refs - numbered)
    return missing if len(missing) > len(unnumbered) else []


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
    "unplotted": (
        "computed tables without a figure",
        f"figures for these {UNPLOTTED_MARK}",
    ),
    "figure_refs": (
        "figures cited but not drawn",
        f"{FIGURE_REFS_MARK} but that were never drawn",
    ),
}


def missing_evidence(ev: RunEvidence, answer: str | None = None) -> list[str]:
    """Keys of EVIDENCE_ITEMS the run lacks; empty when it has them all.

    With `answer` (the draft being checked), also the figure requests:
    computed tables without a figure, and figures cited but not drawn. Each
    is made once per run; after that the answer may say why it has none.
    """
    missing = []
    if not ev.literature_tools:
        missing.append("literature")
    if ev.calculation_steps == 0:
        missing.append("calculation")
    if answer is None:
        if ev.figures == 0:
            missing.append("figure")
        return missing
    if not ev.unplotted_asks and unplotted_tables(ev):
        missing.append("unplotted")
    elif ev.figures == 0:
        missing.append("figure")
    if not ev.figure_ref_asks and unmatched_figure_refs(answer, ev):
        missing.append("figure_refs")
    return missing


def evidence_nudge(missing: list[str], ev: RunEvidence | None = None, answer: str = "") -> str:
    """The message that sends an answer back for the evidence it lacks."""
    lines = []
    for key in missing:
        lines.append(f"- {EVIDENCE_ITEMS[key][1]}")
        if key == "unplotted" and ev is not None:
            lines += [f"    - {heading}" for heading, _ in unplotted_tables(ev)]
            lines.append(
                "  Plot the ones that show a trend across steps, levels, lots or time, a comparison, or a "
                "distribution against a limit (pick the chart from the data's shape, see PLOTS AND "
                "STRUCTURES). Skip any that a figure would not help, such as a list of unit conversions."
            )
        if key == "figure_refs" and ev is not None:
            refs = unmatched_figure_refs(answer, ev)
            numbers = ", ".join(str(n) for n in refs)
            lines.append(
                f"    - Figure{'s' if len(refs) > 1 else ''} {numbers}. Draw each with a title that starts with its number "
                "(e.g. 'Fig 2 — ...'), or remove the reference."
            )
    items = "\n".join(lines)
    return (
        f"{EVIDENCE_NUDGE_PREFIX}\n{items}\n"
        "Do these now in <execute></execute> steps. Then give the complete final answer again inside "
        "<solution></solution>, updated with what you found."
    )


def format_quality_summary(ev: RunEvidence, settings: str = "", unsaved_figure_steps: int = 0) -> str:
    """One line on how the answer was made, plus lines on what was missing.

    `settings` (model, temperatures, reviewer) is shown as its own line, so
    runs can be compared for reproducibility. `unsaved_figure_steps` counts
    steps whose figure was drawn but never saved, so it is in neither the
    transcript nor its export (the app's own fault, not the model's).
    """

    def n(count: int, word: str) -> str:
        if count == 1:
            return f"{count} {word}"
        return f"{count} {word[:-1] + 'ies' if word.endswith('y') and word[-2] not in 'aeiou' else word + 's'}"

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
    if unsaved_figure_steps:
        lines.append(
            f"**⚠️ Figures lost:** {n(unsaved_figure_steps, 'step')} drew a figure that was not saved, "
            "so it is missing from this transcript and its export."
        )
    unplotted = unplotted_tables(ev)
    if unplotted:
        names = "; ".join(heading for heading, _ in unplotted)
        lines.append(f"**Not plotted:** {n(len(unplotted), 'computed table')} without a figure: {names}")
    missing = missing_evidence(ev)
    if missing:
        labels = ", ".join(EVIDENCE_ITEMS[key][0] for key in missing)
        lines.append(f"**⚠️ Missing:** {labels}. Check those parts of the answer before relying on it.")
    return "\n\n".join(lines)

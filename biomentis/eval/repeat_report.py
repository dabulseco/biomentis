"""The repeat-run comparison report: one prompt run K times, side by side.

``write_report`` writes ``<out_dir>/reports/<date>-repeat-<prompt_sha[:8]>.md``
(or ``…-partial.md`` while some runs are still running or paused) with:

* the settings and the prompt's SHA
* a rubric table: one row per rubric item, one column per run; ✓ when the
  run's last review marked the item "present" with a quote, ✗ when it was
  "missing", "wrong", present without a quote or not reported, – when the
  run had no (working) reviewer
* each run's status, rounds, telemetry and every gate finding
* **Disagreements**: final-answer quantities, grouped by (name, condition),
  that at least two runs report and whose largest and smallest values
  differ by more than 5%, with each run's value
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from biomentis.gate import GateResult
from biomentis.gate.rubric import Rubric
from biomentis.independent_review import _rubric_id

DISAGREEMENT_TOLERANCE = 0.05  # relative

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)


@dataclass
class RunOutcome:
    index: int  # 1-based
    run_id: str
    final_status: str
    rounds_used: int
    gate_results: list[GateResult]
    final_answer: str
    telemetry: str
    # For the comparison: rubric ID -> "✓"/"✗" from the last review, None
    # when the run had no working reviewer; the final candidate's claims.
    coverage: dict[str, str] | None = None
    claims: list = field(default_factory=list)


def solution_text(content) -> str:
    """The answer inside <solution> tags, else the whole reply."""
    text = content if isinstance(content, str) else str(content or "")
    match = _SOLUTION_RE.search(text)
    return (match.group(1) if match else text).strip()


def coverage_marks(rubric: Rubric, run_reviews: list[dict] | None) -> dict[str, str] | None:
    """✓/✗ per rubric item from the run's last usable review; None if there was none."""
    reviews = [r.get("review") for r in (run_reviews or []) if isinstance(r, dict)]
    reviews = [r for r in reviews if r is not None and not getattr(r, "error", "")]
    if not reviews:
        return None
    reported: dict[str, dict] = {}
    for entry in getattr(reviews[-1], "coverage", None) or []:
        reported.setdefault(_rubric_id(entry.get("id", "")), entry)
    marks = {}
    for item in rubric.items:
        entry = reported.get(_rubric_id(item.id)) or {}
        status = str(entry.get("status", "")).strip().lower()
        quote = str(entry.get("quote", "")).strip()
        marks[item.id] = "✓" if status == "present" and quote else "✗"
    return marks


def _num(value: float) -> str:
    return format(value, ",.6g")


def disagreements(outcomes: list[RunOutcome]) -> list[tuple[str, str, str, list[tuple[int, float]]]]:
    """(name, condition, unit, [(run index, value)]) for quantities the runs disagree on."""
    groups: dict[tuple[str, str], dict[int, tuple[float, str]]] = {}
    for o in outcomes:
        for c in o.claims or []:
            key = (str(getattr(c, "name", "")), str(getattr(c, "condition", "") or ""))
            try:
                value = float(c.value)
            except (TypeError, ValueError):
                continue
            groups.setdefault(key, {}).setdefault(o.index, (value, str(getattr(c, "unit", "") or "")))
    found = []
    for (name, condition), per_run in sorted(groups.items()):
        if len(per_run) < 2:
            continue
        values = [v for v, _ in per_run.values()]
        lo, hi = min(values), max(values)
        scale = max(abs(lo), abs(hi))
        if scale == 0 or (hi - lo) <= DISAGREEMENT_TOLERANCE * scale:
            continue
        unit = next((u for _, u in per_run.values() if u), "")
        found.append((name, condition, unit, [(i, per_run[i][0]) for i in sorted(per_run)]))
    return found


def report_path(out_dir: Path, sha: str, date: str, partial: bool = False) -> Path:
    suffix = "-partial" if partial else ""
    return Path(out_dir) / "reports" / f"{date}-repeat-{sha[:8]}{suffix}.md"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def write_report(
    prompt: str,
    rubric: Rubric,
    settings: str,
    outcomes: list[RunOutcome],
    out_dir: Path,
    *,
    partial: bool = False,
    waiting: list[str] | tuple[str, ...] = (),
    date: str | None = None,
    total_runs: int | None = None,
) -> Path:
    """Write the comparison report (see the module docstring); return its path.

    `partial` writes the ``…-partial.md`` file, listing `waiting` (the runs
    not finished yet). `date` fixes the date in the file name, so a batch's
    partial and final reports pair up even across midnight.
    """
    from biomentis.gate.rubric import prompt_sha

    sha = rubric.prompt_sha or prompt_sha(prompt)
    date = date or datetime.now().strftime("%Y-%m-%d")
    outcomes = sorted(outcomes, key=lambda o: o.index)
    total = total_runs if total_runs is not None else len(outcomes)

    lines = [f"# Repeat runs{' (partial)' if partial else ''}", ""]
    lines.append(f"- Date: {date}")
    lines.append(f"- Settings: {settings or '(not given)'}")
    lines.append(f"- Prompt SHA: `{sha}`")
    lines.append(f"- Runs finished: {len(outcomes)} of {total}")
    lines.append(f"- Rubric: {len(rubric.items)} items ({rubric.source}), file `rubric-{sha[:8]}.md`")
    lines += ["", "## Prompt", ""]
    lines += [f"> {ln}" if ln else ">" for ln in prompt.strip().splitlines()]

    if partial:
        lines += ["", "## Not finished yet", ""]
        lines += [f"- {w}" for w in waiting] or ["- (none)"]

    lines += ["", "## Rubric coverage", ""]
    if outcomes:
        lines.append("| Item | " + " | ".join(f"Run {o.index}" for o in outcomes) + " |")
        lines.append("|---|" + "---|" * len(outcomes))
        for item in rubric.items:
            cells = [("–" if o.coverage is None else o.coverage.get(item.id, "✗")) for o in outcomes]
            lines.append(f"| {item.id} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("✓ delivered (marked present with a quote) · ✗ missing, wrong or not reported · – no reviewer")
    else:
        lines.append("No run has finished yet.")
    lines.append("")
    lines += [f"- **{item.id}** ({item.kind}): {item.text}" for item in rubric.items]

    lines += ["", "## Runs"]
    for o in outcomes:
        lines += ["", f"### Run {o.index}", ""]
        lines.append(f"- Run ID: `{o.run_id}`")
        lines.append(f"- Status: {o.final_status}")
        lines.append(f"- Rounds: {o.rounds_used}")
        if o.telemetry:
            lines.append("- Telemetry:")
            lines += [f"  {ln}" for ln in o.telemetry.strip().splitlines()]
        else:
            lines.append("- Telemetry: (none)")
        findings = [(r, f) for r in o.gate_results for f in r.findings]
        lines.append(f"- Findings ({len(findings)}):")
        if not findings:
            lines.append("  - (none)")
        for r, f in findings:
            accepted = " (accepted as is)" if f.key in r.accepted else ""
            quote = f' — "{_cell(f.quote)}"' if f.quote else ""
            lines.append(f"  - round {r.round}, {f.severity}{accepted}, {f.check}: {_cell(f.issue)}{quote}")

    lines += ["", "## Disagreements", ""]
    found = disagreements(outcomes)
    if not found:
        lines.append(f"No quantity reported by two or more runs differs by more than "
                     f"{DISAGREEMENT_TOLERANCE:.0%}.")
    for name, condition, unit, values in found:
        label = f"`{name}`" + (f" ({condition})" if condition else "") + (f" [{unit}]" if unit else "")
        lines.append(f"- {label}: " + ", ".join(f"Run {i} = {_num(v)}" for i, v in values))

    path = report_path(out_dir, sha, date, partial=partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

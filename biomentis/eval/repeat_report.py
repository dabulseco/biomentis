"""The repeat-run comparison report: one prompt run K times, side by side.

``write_report`` writes ``<out_dir>/reports/<date>-repeat-<prompt_sha[:8]>.md``
(or ``…-partial.md`` while some runs are still running or paused) with:

* the settings and the prompt's SHA
* a rubric table: one row per rubric item, one column per run; ✓ when the
  run's last review marked the item "present" with a quote, ✗ when it was
  "missing", "wrong", present without a quote or not reported, – when the
  run had no (working) reviewer
* each run's status, rounds, telemetry and every gate finding
* **Summary**: runs per final status, rounds used (min / mean / max),
  quantities extracted per run, and how many disagreements of each kind
* **Disagreements**: final-answer quantities, grouped by (name, condition),
  that at least two runs report and whose largest and smallest values
  differ by more than ``DISAGREEMENT_TOLERANCE`` (5%), with each run's value.
  Each is marked a **variation** when the spread is at most ``VARIATION_MAX``
  (20%) and a **contradiction** above it. Runs with no extracted quantities
  are listed; when no quantity name appears in two runs, the report says so.

Quantities are matched by the name the extractor gave them; names are not
normalised across runs (no extra model call), so the same quantity under
two names is not compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from biomentis.gate import GateResult
from biomentis.gate.rubric import Rubric
from biomentis.independent_review import _rubric_id

# Relative spread, (largest - smallest) / largest magnitude, of one quantity across runs:
DISAGREEMENT_TOLERANCE = 0.05  # at or below: the runs agree (not listed)
VARIATION_MAX = 0.20  # above the tolerance, up to this: "variation"; above it: "contradiction"

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


def _groups(outcomes: list[RunOutcome]) -> dict[tuple[str, str], dict[int, tuple[float, str]]]:
    """(name, condition) -> {run index: (value, unit)}, the first claim of each run."""
    groups: dict[tuple[str, str], dict[int, tuple[float, str]]] = {}
    for o in outcomes:
        for c in o.claims or []:
            key = (str(getattr(c, "name", "")), str(getattr(c, "condition", "") or ""))
            try:
                value = float(c.value)
            except (TypeError, ValueError):
                continue
            groups.setdefault(key, {}).setdefault(o.index, (value, str(getattr(c, "unit", "") or "")))
    return groups


def comparable_count(outcomes: list[RunOutcome]) -> int:
    """How many (name, condition) quantities at least two runs report."""
    return sum(1 for per_run in _groups(outcomes).values() if len(per_run) >= 2)


def disagreement_kind(spread: float) -> str:
    return "variation" if spread <= VARIATION_MAX else "contradiction"


def disagreements(outcomes: list[RunOutcome]) -> list[tuple[str, str, str, list[tuple[int, float]], str]]:
    """(name, condition, unit, [(run index, value)], "variation" | "contradiction") per disagreement."""
    found = []
    for (name, condition), per_run in sorted(_groups(outcomes).items()):
        if len(per_run) < 2:
            continue
        values = [v for v, _ in per_run.values()]
        lo, hi = min(values), max(values)
        scale = max(abs(lo), abs(hi))
        if scale == 0 or (hi - lo) <= DISAGREEMENT_TOLERANCE * scale:
            continue
        unit = next((u for _, u in per_run.values() if u), "")
        found.append((name, condition, unit, [(i, per_run[i][0]) for i in sorted(per_run)],
                      disagreement_kind((hi - lo) / scale)))
    return found


_ROUNDS_IN_LABEL_RE = re.compile(r" after \d+ rounds?")


def _summary(outcomes: list[RunOutcome], found: list) -> list[str]:
    """The Summary section's lines (see the module docstring)."""
    if not outcomes:
        return ["No run has finished yet."]
    statuses: dict[str, int] = {}
    for o in outcomes:
        label = _ROUNDS_IN_LABEL_RE.sub("", o.final_status)
        statuses[label] = statuses.get(label, 0) + 1
    lines = ["- Final status: " + "; ".join(f"{label} × {n}" for label, n in statuses.items())]
    rounds = [o.rounds_used for o in outcomes]
    lines.append(f"- Rounds used: min {min(rounds)}, mean {sum(rounds) / len(rounds):.1f}, max {max(rounds)}")
    lines.append("- Quantities extracted: " + ", ".join(
        f"Run {o.index} = {len(o.claims or [])}" + (" (no quantities extracted)" if not o.claims else "")
        for o in outcomes))
    kinds = [kind for *_, kind in found]
    lines.append(f"- Disagreements: {len(found)} ({kinds.count('variation')} variation, "
                 f"{kinds.count('contradiction')} contradiction)")
    return lines


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

    found = disagreements(outcomes)
    lines += ["", "## Summary", ""]
    lines += _summary(outcomes, found)

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
    lines.append(f"Variation: the values differ by more than {DISAGREEMENT_TOLERANCE:.0%} and at most "
                 f"{VARIATION_MAX:.0%}; contradiction: by more than {VARIATION_MAX:.0%}.")
    lines.append("")
    empty = [o.index for o in outcomes if not o.claims]
    if empty:
        lines.append("Runs with no quantities extracted: " + ", ".join(f"Run {i}" for i in empty) + ".")
        lines.append("")
    if outcomes and not comparable_count(outcomes):
        lines.append("No comparable quantities across runs (names differ per run).")
    elif not found:
        lines.append(f"No quantity reported by two or more runs differs by more than "
                     f"{DISAGREEMENT_TOLERANCE:.0%}.")
    for name, condition, unit, values, kind in found:
        label = f"`{name}`" + (f" ({condition})" if condition else "") + (f" [{unit}]" if unit else "")
        lines.append(f"- **{kind}** {label}: " + ", ".join(f"Run {i} = {_num(v)}" for i, v in values))

    path = report_path(out_dir, sha, date, partial=partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

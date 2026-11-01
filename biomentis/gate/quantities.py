"""Quantity checks: the numbers an answer depends on, extracted by an LLM.

``extract_quantities`` asks a model for every quantity in the answer, one
call per chunk of ``##``/``###`` sections, so a long answer is read in full.
The checks then compare the claims with each other (consistency), with the
expressions they say they follow from (derivations), with the previous
candidate answer (regressions) and with the numbers the run's code printed.

Two values agree when they differ by at most 0.5%.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from biomentis.gate import Finding
from biomentis.gate.arithmetic import evaluate

TOLERANCE = 0.005


@dataclass
class QuantityClaim:
    name: str            # normalised by the extractor, e.g. "libraries_total_n3", "nacl_i_app_1m"
    value: float
    unit: str
    quote: str
    condition: str = ""  # e.g. "1.00 mol/kg", "n=3 design"
    derivation: str = "" # optional expression over other claim names, e.g. "groups_per_solute * trials_per_group"


EXTRACT_SYSTEM_PROMPT = """You extract numeric quantities from a research answer for automatic checking.

List every numeric quantity that a design, budget, count or result in the text depends on: sample and
library counts, replicates, volumes, masses, concentrations, costs, totals, measured or computed results.
Skip numbers that are only labels (figure numbers, years, section numbers, citations).

For each quantity give:
- "name": a stable snake_case name for what the number is, the same name every time the same quantity
  appears (e.g. "libraries_total", "nacl_i_app", "sequencing_cost_total"). End totals with "_total",
  counts with "_n" and costs with "_cost" where that fits.
- "value": the number as a plain JSON number (no units, no thousands separators).
- "unit": its unit, or "" if it has none.
- "quote": the exact text from the answer where it is stated, copied character for character.
- "condition": whenever the value depends on a condition (a concentration, temperature, design variant,
  sample group), that condition, e.g. "1.00 mol/kg" or "n=3 design"; otherwise "".
- "derivation": whenever the answer shows how the value follows from other quantities, that calculation
  as an expression over their names with + - * / and parentheses, e.g.
  "groups_per_solute * trials_per_group"; otherwise "". Every name used must also be listed.

Reply with only a JSON object, no other text:
{"quantities": [{"name": "...", "value": 0, "unit": "...", "quote": "...", "condition": "", "derivation": ""}]}
If there are none, reply {"quantities": []}."""


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def _fmt(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:.4g}"


# ---------------------------------------------------------------- extraction

_HEADING = re.compile(r"^#{2,3} ")


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Consecutive pieces joined into chunks of at most ``limit`` characters."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > limit:
            chunks.append(current)
            current = ""
        current += piece
    if current:
        chunks.append(current)
    return chunks


def _split_section(section: str, limit: int) -> list[str]:
    """A section too long for one chunk, split on blank lines, then hard-split."""
    paragraphs = [p for p in re.split(r"(?<=\n\n)", section) if p]
    pieces: list[str] = []
    for p in paragraphs:
        pieces += [p[i:i + limit] for i in range(0, len(p), limit)] if len(p) > limit else [p]
    return _pack(pieces, limit)


def chunk_answer(answer: str, max_chunk_chars: int) -> list[str]:
    """The answer in chunks of whole ``##``/``###`` sections; their concatenation is the answer."""
    sections: list[str] = []
    for line in answer.splitlines(keepends=True):
        if not sections or _HEADING.match(line):
            sections.append(line)
        else:
            sections[-1] += line
    pieces: list[str] = []
    for s in sections:
        pieces += _split_section(s, max_chunk_chars) if len(s) > max_chunk_chars else [s]
    return _pack(pieces, max_chunk_chars)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in content
                       if isinstance(b, (str, dict)))
    return str(content or "")


def _quantities_json(text: str) -> list | None:
    """The "quantities" list of the first JSON object that has one, or None."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("quantities"), list):
            return value["quantities"]
    return None


def _claim(item) -> QuantityClaim | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    raw = item.get("value")
    if not name or isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(str(raw).replace(",", "").replace("−", "-").strip())
    except ValueError:
        return None
    return QuantityClaim(
        name=name,
        value=value,
        unit=str(item.get("unit") or ""),
        quote=str(item.get("quote") or ""),
        condition=str(item.get("condition") or "").strip(),
        derivation=str(item.get("derivation") or "").strip(),
    )


def extract_quantities(answer: str, llm, max_chunk_chars: int = 12000,
                       skipped: list[int] | None = None) -> list[QuantityClaim]:
    """Every quantity in ``answer``, one ``llm`` call per chunk, merged.

    A chunk with no usable reply after a retry is left out; its number
    (1-based) is appended to `skipped`, when given, so the caller can say
    the check did not cover the whole answer.
    """
    claims: list[QuantityClaim] = []
    seen: set[tuple[str, float, str]] = set()
    chunks = chunk_answer(answer, max_chunk_chars)
    for n, chunk in enumerate(chunks, 1):
        items = None
        for _attempt in range(2):
            try:
                reply = llm.invoke([SystemMessage(content=EXTRACT_SYSTEM_PROMPT), HumanMessage(content=chunk)])
            except Exception as exc:  # a failed call counts as an unusable reply
                print(f"[gate] quantity extraction: chunk {n}/{len(chunks)} call failed: {exc}")
                continue
            items = _quantities_json(_text(getattr(reply, "content", reply)))
            if items is not None:
                break
        if items is None:
            print(f"[gate] quantity extraction: chunk {n}/{len(chunks)} skipped, no usable JSON after a retry")
            if skipped is not None:
                skipped.append(n)
            continue
        for item in items:
            c = _claim(item)
            if c is None:
                continue
            ident = (c.name, c.value, c.quote)
            if ident not in seen:
                seen.add(ident)
                claims.append(c)
    return claims


# ---------------------------------------------------------------- checks

def check_consistency(claims: list[QuantityClaim]) -> list[Finding]:
    """Blocking when one quantity under one condition has two values; a warning across conditions."""
    by_name: dict[str, list[QuantityClaim]] = {}
    for c in claims:
        by_name.setdefault(c.name, []).append(c)
    findings: list[Finding] = []
    for name, group in by_name.items():
        # Claims with no condition each stand alone: nothing says they should agree.
        conditions: dict[str, list[QuantityClaim]] = {}
        for i, c in enumerate(group):
            conditions.setdefault(c.condition or f"\0{i}", []).append(c)
        for cond, same in conditions.items():
            odd = next((c for c in same[1:] if not _close(c.value, same[0].value)), None)
            if odd is not None:
                values = ", ".join(_fmt(c.value) for c in same)
                findings.append(Finding(
                    check="quantity", severity="blocking", quote=odd.quote,
                    issue=f"{name} at {cond} is given as {values}",
                    fix=f"use one value for {name} at {cond} throughout",
                ))
        groups = list(conditions.values())
        pairs = [(a, b) for i, ga in enumerate(groups) for gb in groups[i + 1:] for a in ga for b in gb]
        odd_pair = next(((a, b) for a, b in pairs if not _close(a.value, b.value)), None)
        if odd_pair is not None:
            a, b = odd_pair
            findings.append(Finding(
                check="quantity", severity="warning", quote=b.quote,
                issue=(f"{name} is {_fmt(a.value)} ({a.condition or 'no condition stated'}) and "
                       f"{_fmt(b.value)} ({b.condition or 'no condition stated'})"),
                fix=f"state the condition each value of {name} applies to, or use one value",
            ))
    return findings


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def check_derivations(claims: list[QuantityClaim]) -> list[Finding]:
    """Blocking when a claim's value does not follow from the derivation it gives."""
    findings: list[Finding] = []
    for c in claims:
        if not c.derivation:
            continue
        names = set(_NAME.findall(c.derivation))
        variables: dict[str, float] = {}
        for name in names:
            sources = [o for o in claims if o.name == name and o is not c]
            if not sources:
                break
            # Prefer the value stated under the same condition.
            match = next((o for o in sources if o.condition == c.condition), sources[0])
            variables[name] = match.value
        else:
            computed = evaluate(c.derivation, variables)
            if computed is None or _close(computed, c.value):
                continue
            findings.append(Finding(
                check="quantity", severity="blocking", quote=c.quote,
                issue=(f"{c.name} is given as {_fmt(c.value)}, but {c.derivation} = {_fmt(computed)} "
                       f"(" + ", ".join(f"{k} = {_fmt(v)}" for k, v in sorted(variables.items())) + ")"),
                fix=_fmt(computed),
            ))
    return findings


def _in(value: float, numbers: set[float]) -> bool:
    return any(_close(value, n) for n in numbers)


def check_regressions(prev: list[QuantityClaim], new: list[QuantityClaim],
                      recomputed_values: set[float]) -> list[Finding]:
    """Blocking when a quantity changed since the previous candidate with no code output showing it."""
    def index(claims):
        by_key: dict[tuple[str, str], QuantityClaim] = {}
        count: dict[str, int] = {}
        for c in claims:
            by_key.setdefault((c.name, c.condition), c)
            count[c.name] = count.get(c.name, 0) + 1
        return by_key, count

    prev_by_key, prev_count = index(prev)
    new_by_key, new_count = index(new)
    prev_by_name = {c.name: c for c in prev}
    findings: list[Finding] = []
    for key, c in new_by_key.items():
        old = prev_by_key.get(key)
        # A quantity stated once in each answer is the same one even if its condition was reworded.
        if old is None and prev_count.get(c.name) == 1 and new_count[c.name] == 1:
            old = prev_by_name[c.name]
        if old is None or _close(old.value, c.value) or _in(c.value, recomputed_values):
            continue
        findings.append(Finding(
            check="regression", severity="blocking", quote=c.quote,
            issue=(f"{c.name} changed from {_fmt(old.value)} (\"{old.quote}\") to {_fmt(c.value)} "
                   f"(\"{c.quote}\"), and no code step since the previous answer printed {_fmt(c.value)}"),
            fix=f"recompute {c.name} in code, or restore {_fmt(old.value)} if the change was not intended",
        ))
    return findings


def check_against_outputs(claims: list[QuantityClaim], output_numbers: set[float]) -> list[Finding]:
    """A warning for a derived total, count or cost that no code step printed."""
    findings: list[Finding] = []
    for c in claims:
        if not (c.derivation or c.name.endswith(("_total", "_n", "_cost"))):
            continue
        if _in(c.value, output_numbers):
            continue
        findings.append(Finding(
            check="quantity", severity="warning", quote=c.quote,
            issue=f"{c.name} = {_fmt(c.value)} not found in any code output",
            fix=f"compute {c.name} in code and use the printed value",
        ))
    return findings


# ---------------------------------------------------------------- code output

_OUTPUT_NUMBER = re.compile(
    r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def output_numbers(messages: list[BaseMessage]) -> set[float]:
    """Every number printed in an observation (code output) of the latest run.

    Pass the messages since the previous candidate answer to get the numbers
    recomputed since then (for ``check_regressions``).
    """
    from biomentis.run_quality import current_run

    numbers: set[float] = set()
    for m in current_run(messages):
        content = m.content if isinstance(m.content, str) else ""
        if not content.lstrip().startswith("<observation>"):
            continue
        for text in _OUTPUT_NUMBER.findall(content):
            try:
                numbers.add(float(text.replace(",", "").replace("−", "-")))
            except ValueError:
                pass
    return numbers

"""Figure integrity: cited but not drawn, stale after a revision, garbled titles."""

from __future__ import annotations

import re

from biomentis.gate import Finding
from biomentis.run_quality import _FIGURE_NUMBER_RE, FigureRecord, RunEvidence, unmatched_figure_refs

_AMOUNT_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|(?<![\w.])\d[\d,]*(?:\.\d+)?")
_TITLE_CALL_RE = re.compile(r"\b(?:set_title|suptitle|title)\s*\(")
# '$' that is not escaped and is followed by a digit or an f-string field.
_BARE_DOLLAR_RE = re.compile(r"(?<!\\)\$\s?[\d{]")


def _title(line: str) -> str:
    m = re.match(r"Figure '(.*)':", line)
    return m.group(1) if m else line


def _value(token: str) -> float:
    return float(re.sub(r"[^\d.]", "", token).rstrip(".") or "nan")


def _decimals(token: str) -> int:
    digits = re.sub(r"[^\d.]", "", token)
    return len(digits.split(".", 1)[1]) if "." in digits else 0


def _matches(title_tok: str, answer_tok: str) -> bool:
    """Equal to the coarser stated precision, or within 0.5% relative."""
    a, b = _value(title_tok), _value(answer_tok)
    if a != a or b != b:
        return False
    half_step = 0.5 * 10 ** -min(_decimals(title_tok), _decimals(answer_tok))
    return abs(a - b) <= max(half_step, 0.005 * max(abs(a), abs(b)))


def _is_year(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100


def _sig_figs(token: str) -> int:
    digits = re.sub(r"[^\d.]", "", token)
    if "." in digits:
        return len(digits.replace(".", "").lstrip("0"))
    return len(digits.lstrip("0").rstrip("0"))


def _claimed_numbers(title: str) -> list[str]:
    """Numbers in a title that a reader would take as results."""
    title = _FIGURE_NUMBER_RE.sub("", title)
    out = []
    for tok in _AMOUNT_RE.findall(title):
        tok = tok.strip()
        if tok[0] == "$" or tok[-1] == "%" or (_sig_figs(tok) >= 3 and not _is_year(tok)):
            out.append(tok)
    return out


def _answer_numbers(answer: str) -> list[str]:
    return _AMOUNT_RE.findall(_FIGURE_NUMBER_RE.sub("", answer))


def _identity(rec: FigureRecord) -> str:
    title = _title(rec.title_line)
    nums = _FIGURE_NUMBER_RE.findall(title)
    return f"fig{nums[0]}" if nums else re.sub(r"[\d\W]+", " ", title).strip().lower()


def _title_call_args(code: str) -> list[str]:
    out = []
    for m in _TITLE_CALL_RE.finditer(code):
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[i], 0)
            i += 1
        out.append(code[m.end() : i - 1])
    return out


def check_figures(answer: str, ev: RunEvidence) -> list[Finding]:
    findings: list[Finding] = []

    for n in unmatched_figure_refs(answer, ev):
        findings.append(Finding(
            check="figure", severity="blocking", quote=f"Fig {n}",
            issue=f"The answer cites Figure {n}, but no figure with that number was drawn.",
            fix=f"Draw Figure {n} (title it 'Fig {n} ...') or remove the reference.",
        ))

    if ev.last_solution_index >= 0:
        values = _answer_numbers(answer)
        for rec in ev.figure_records:
            if rec.step >= ev.last_solution_index:
                continue
            ident = _identity(rec)
            if any(r.step > rec.step and _identity(r) == ident for r in ev.figure_records):
                continue  # redrawn since
            title = _title(rec.title_line)
            absent = [
                t for t in _claimed_numbers(title)
                if not any(_matches(t, v) for v in values)
            ]
            if absent:
                findings.append(Finding(
                    check="figure", severity="blocking", quote=title,
                    issue=(f"This figure was drawn before the answer was revised and shows {', '.join(absent)}, "
                           "which the final answer does not contain, so it is probably out of date."),
                    fix="Redraw the figure from the revised numbers.",
                ))

    seen: set[int] = set()
    for rec in ev.figure_records:
        if rec.step in seen:
            continue
        seen.add(rec.step)
        for arg in _title_call_args(rec.code):
            if _BARE_DOLLAR_RE.search(arg):
                findings.append(Finding(
                    check="figure", severity="warning", quote=arg.strip()[:120],
                    issue="A plot title has an unescaped '$' before a number; matplotlib reads text between '$' signs as math, which garbles the title.",
                    fix="Escape each dollar sign as \\$ in titles and labels.",
                ))
                break
    return findings

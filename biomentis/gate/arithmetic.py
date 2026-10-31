"""Arithmetic check: recompute every ``expr = number`` the answer states.

A claim is an expression of numbers and ``+ - × x * / ( ) ·`` followed by
``=`` (or ``≈``) and a number. Anything else is skipped, never flagged:
expressions glued to words or symbols (``√(…)``, ``ln(…)``, ``x²``),
implicit multiplication (``0.0375(760-749.82)``), a bare number on the left
(``n = 3``), a layout such as ``2×250`` (no spaces), and results written in
scientific notation, as a percentage or with a magnitude (``2.47 M``).

``evaluate`` is the parser, shared with the derivation check; it never uses
``eval``.
"""

from __future__ import annotations

import re

from biomentis.gate import Finding

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_TOKEN = re.compile(rf"\s*(?:(?P<num>{_NUMBER})|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+−×*/·()]))")
_MUL = {"×", "*", "·", "x"}

# The longest run of expression characters ending a piece of text. A comma
# belongs to it only as a thousands separator.
_EXPR_TAIL = re.compile(r"(?:[0-9.+\-−×x*/()·\s$]|(?<=\d),(?=\d{3}))*$")
# What may sit right before an expression that does not start with a space.
_BOUNDARY = set("`:;[{\"'|*>")
_RESULT = re.compile(rf"\s*(?:\$|\*\*|`|\\mathbf\{{|\\textbf\{{|\{{)*\s*(?P<num>[-−]?(?:{_NUMBER}))")
# A result followed by one of these is not a plain number (2×10¹⁷, 25%, 6 + 1).
_RESULT_CONTINUES = re.compile(r"\s*(?:\*(?!\*)|[-+−×/·^(%⁰¹²³⁴⁵⁶⁷⁸⁹⁻]|x\s*\d|[eE][-+]?\d|\d)")
# A result with a magnitude word is not the plain number (2.47 M reads).
_RESULT_SCALED = re.compile(r"\s*(?:[kKMGTB]|million|billion|thousand|bn|mn)\b")
# "2×250" with no spaces is a notation (read layout, plate, dimensions),
# as in "overlap at 2×250 = 127 bp", not a product.
_LAYOUT = re.compile(r"\d+[×x]\d+")
_EQUALS = re.compile(r"(?<![=!<>])[=≈](?![=>])")


def _tokens(expr: str) -> list[tuple[str, str]] | None:
    expr = expr.replace("$", "")
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        if expr[pos:].strip() == "":
            break
        m = _TOKEN.match(expr, pos)
        if not m:
            return None
        if m.group("num"):
            out.append(("num", m.group("num").replace(",", "")))
        elif m.group("name"):
            name = m.group("name")
            # "x" after an operand is multiplication: 2x250, 2 x 3.
            if name[0] == "x" and out and (out[-1][0] in ("num", "name") or out[-1][1] == ")"):
                out.append(("op", "x"))
                rest = name[1:]
                if rest:
                    if not rest.isdigit():
                        return None
                    out.append(("num", rest))
            else:
                out.append(("name", name))
        else:
            out.append(("op", "-" if m.group("op") == "−" else m.group("op")))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], variables: dict[str, float]):
        self.toks = tokens
        self.i = 0
        self.vars = variables
        self.binary_ops = 0

    def peek(self) -> str | None:
        return self.toks[self.i][1] if self.i < len(self.toks) and self.toks[self.i][0] == "op" else None

    def expr(self) -> float:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.toks[self.i][1]
            self.i += 1
            self.binary_ops += 1
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self) -> float:
        value = self.factor()
        while self.peek() in _MUL or self.peek() == "/":
            op = self.toks[self.i][1]
            self.i += 1
            self.binary_ops += 1
            rhs = self.factor()
            if op == "/":
                value = value / rhs  # ZeroDivisionError is caught by the caller
            else:
                value = value * rhs
        return value

    def factor(self) -> float:
        if self.peek() in ("+", "-"):
            sign = -1.0 if self.toks[self.i][1] == "-" else 1.0
            self.i += 1
            return sign * self.factor()
        if self.i >= len(self.toks):
            raise ValueError("unexpected end")
        kind, text = self.toks[self.i]
        self.i += 1
        if kind == "num":
            return float(text)
        if kind == "name":
            if text not in self.vars:
                raise ValueError(f"unknown name {text}")
            return float(self.vars[text])
        if text == "(":
            value = self.expr()
            if self.peek() != ")":
                raise ValueError("missing )")
            self.i += 1
            return value
        raise ValueError(f"unexpected {text}")


def _parse(expr: str, variables: dict[str, float] | None) -> tuple[float, int] | None:
    toks = _tokens(expr)
    if not toks:
        return None
    p = _Parser(toks, variables or {})
    try:
        value = p.expr()
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if p.i != len(toks):
        return None
    return value, p.binary_ops


def evaluate(expr: str, variables: dict[str, float] | None = None) -> float | None:
    """The value of ``expr``, or None if it uses anything unsupported."""
    parsed = _parse(expr, variables)
    return None if parsed is None else parsed[0]


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def _format(value: float, places: int, thousands: bool) -> str:
    """The value at the stated precision, or 4 significant digits if that hides it."""
    text = f"{value:,.{places}f}" if thousands else f"{value:.{places}f}"
    if places == 0 and value != round(value) and abs(value) < 1000:
        text = f"{value:.4g}"
    return text


def _claims_in_line(line: str):
    """Yield (quote, expression, stated result text) for each claim in a line."""
    equals = [m.start() for m in _EQUALS.finditer(line)]
    for n, eq in enumerate(equals):
        # A claim's expression is short; the cap keeps a long line linear.
        left_start = max(equals[n - 1] + 1 if n else 0, eq - 300)
        left = line[left_start:eq]
        tail = _EXPR_TAIL.search(left)
        raw = tail.group(0) if tail else ""
        if not raw.strip():
            continue
        start = left_start + tail.start()
        # An expression glued to a word or symbol is part of something bigger
        # (ln(2), √(…), x²): skip it.
        separated = raw[0].isspace() or raw.startswith(("**", ". ", ", "))
        if not separated and start > 0 and line[start - 1] not in _BOUNDARY:
            continue
        # Drop sentence punctuation and markdown bold before the expression.
        lead = len(raw) - len(raw.lstrip())
        if raw.startswith(("**", ". ", ", "), lead):
            lead += 2
            lead += len(raw[lead:]) - len(raw[lead:].lstrip())
        expr = raw[lead:].rstrip()
        # One that starts with an operator continues something before it
        # ("45,000 raw x 0.95 = …", "0.0375(760-749.8) = …"): skip it.
        if not expr or expr[0] in "*/×x·)":
            continue
        result = _RESULT.match(line, eq + 1)
        if not result:
            continue
        if _RESULT_CONTINUES.match(line, result.end()) or _RESULT_SCALED.match(line, result.end()):
            continue
        if _LAYOUT.fullmatch(expr):
            continue
        # Exact text from the start of the expression to the end of the result.
        quote = line[start + lead:result.end()]
        yield quote, expr, result.group("num")


def check_arithmetic(answer: str) -> list[Finding]:
    """A blocking finding for every stated result that does not match its expression."""
    findings: list[Finding] = []
    for line in answer.splitlines():
        for quote, expr, stated_text in _claims_in_line(line):
            parsed = _parse(expr, None)
            if parsed is None or parsed[1] == 0:
                continue
            computed = parsed[0]
            stated = float(stated_text.replace(",", "").replace("−", "-"))
            places = _decimals(stated_text)
            tolerance = max(0.5 * 10 ** -places, 0.005 * abs(computed))
            if abs(computed - stated) <= tolerance:
                continue
            value = _format(computed, places, "," in stated_text)
            findings.append(Finding(
                check="arithmetic",
                severity="blocking",
                quote=quote,
                issue=f"{expr} is {value}, not {stated_text}",
                fix=f"{expr} = {value}",
            ))
    return findings

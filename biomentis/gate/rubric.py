"""Rubric model and its markdown file format.

A rubric lists what a run's answer must deliver, the constraints it must
respect and the lessons it must carry. The canonical file format is::

    # Rubric

    ## Deliverables
    - [D1] text

    ## Constraints
    - [C1] text

    ## Core lessons
    - [L1] text

The ``# Rubric`` line (followed by a blank line) is optional on input and
always written on output. The three sections come in exactly this order, one
blank line between sections, one trailing newline. The ID prefix gives the
kind: D = deliverable, C = constraint, L = lesson, and must match its section.
IDs are unique across the file.

``parse_rubric_markdown`` returns None for anything that is not in this
format (prose, missing or unknown sections, wrong order, mismatched prefix)
and raises ValueError only for duplicate IDs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_TITLE = "# Rubric"
# (heading, kind, id prefix), in canonical order
_SECTIONS = (
    ("Deliverables", "deliverable", "D"),
    ("Constraints", "constraint", "C"),
    ("Core lessons", "lesson", "L"),
)
_ITEM_RE = re.compile(r"^- \[([DCL]\d+)\] (\S.*)$")


@dataclass(frozen=True)
class RubricItem:
    id: str
    kind: str  # "deliverable" | "constraint" | "lesson"
    text: str


@dataclass(frozen=True)
class Rubric:
    items: tuple[RubricItem, ...]
    source: str  # "uploaded" | "converted" | "agent-drafted"
    prompt_sha: str


def prompt_sha(prompt: str) -> str:
    """sha256 hex digest of the stripped prompt."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def parse_rubric_markdown(text: str, source: str = "uploaded", prompt_sha: str = "") -> Rubric | None:
    """Parse the standard format; None if the text is not in it."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if lines and lines[0] == _TITLE:
        lines = lines[1:]
    order = [h for h, _, _ in _SECTIONS]
    seen: list[str] = []
    current: tuple[str, str, str] | None = None
    items: list[RubricItem] = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("## "):
            match = [s for s in _SECTIONS if ln == f"## {s[0]}"]
            if not match:
                return None
            current = match[0]
            seen.append(current[0])
            continue
        m = _ITEM_RE.match(ln)
        if m is None or current is None or not m.group(1).startswith(current[2]):
            return None
        items.append(RubricItem(id=m.group(1), kind=current[1], text=m.group(2).strip()))
    if seen != order or not items:
        return None
    ids = [i.id for i in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate rubric IDs: {', '.join(dupes)}")
    return Rubric(items=tuple(items), source=source, prompt_sha=prompt_sha)


def format_rubric_markdown(r: Rubric) -> str:
    """Write the canonical format (see the module docstring)."""
    blocks = []
    for heading, kind, _ in _SECTIONS:
        body = "".join(f"- [{i.id}] {i.text}\n" for i in r.items if i.kind == kind)
        blocks.append(f"## {heading}\n{body}")
    return f"{_TITLE}\n\n" + "\n".join(blocks)


def save_rubric(r: Rubric, folder: Path) -> Path:
    """Write rubric-<prompt_sha[:8]>.md into folder (created if missing)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rubric-{r.prompt_sha[:8]}.md"
    path.write_text(format_rubric_markdown(r), encoding="utf-8")
    return path

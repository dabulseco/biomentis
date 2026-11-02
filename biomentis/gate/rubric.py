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


# ------------------------------------------------- drafting and converting

_FORMAT_SPEC = """Reply with only the rubric, in exactly this markdown format and nothing else:

## Deliverables
- [D1] ...
- [D2] ...

## Constraints
- [C1] ...

## Core lessons
- [L1] ...

The three sections come in this order, each with at least one item. Number the items D1, D2, ...,
C1, C2, ... and L1, L2, ... with no gaps or repeats. One item per line, one short checkable sentence each."""

DRAFT_SYSTEM_PROMPT = f"""You write the grading rubric for a research or teaching task before anyone answers it.
The rubric is frozen for the run and an answer is checked against it, so every item must be something a
reader can verify in the answer.

- Deliverables: every item the prompt explicitly asks for, one per item. Where the prompt asks for
  something per part (each phase, each sample, each step), list it for each part. Do not add
  deliverables the prompt does not ask for.
- Constraints: every stated limit (equipment, cost, time, group sizes, scope, format, safety
  flags), one per item.
- Core lessons: 3–6 principles that any correct answer must convey, whatever its specific approach.
  They are about the subject, not about formatting.

{_FORMAT_SPEC}"""

CONVERT_SYSTEM_PROMPT = f"""You convert a teacher's rubric into a standard format. The rubric is frozen for the run and
an answer is checked against it.

Keep the teacher's intent: every requirement in the rubric becomes exactly one item, in the teacher's
words where possible. Put what the answer must contain under Deliverables, the limits it must respect
under Constraints, and the principles it must convey under Core lessons. Do not add requirements of
your own; use the task prompt only to understand the rubric. If the rubric has nothing for a section,
add one item that follows from the task prompt.

{_FORMAT_SPEC}"""

RETRY_MESSAGE = (
    "That reply could not be read as a rubric. Reply again with only the rubric in the exact format: "
    "the three sections ## Deliverables, ## Constraints and ## Core lessons in that order, each item on "
    "its own line as - [D1] text, - [C1] text or - [L1] text, unique IDs, no other text."
)

_THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _reply_text(reply) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in content
                       if isinstance(b, str | dict))
    return str(content or "")


def _rubric_from_reply(text: str, source: str, sha: str) -> Rubric | None:
    """The rubric in a model's reply: thinking and code fences stripped, read from ## Deliverables on."""
    text = _THINK_RE.sub("", text)
    start = text.find("## Deliverables")
    if start < 0:
        return None
    kept: list[str] = []
    in_lessons = False
    for ln in text[start:].splitlines():
        if ln.strip().startswith("```"):
            if in_lessons:  # a closing fence ends the rubric
                break
            continue
        if ln.strip() == "## Core lessons":
            in_lessons = True
        elif in_lessons and ln.strip() and not _ITEM_RE.match(ln.rstrip()):
            break  # trailing prose after the last lesson
        kept.append(ln)
    body = "\n".join(kept)
    try:
        return parse_rubric_markdown(body, source=source, prompt_sha=sha)
    except ValueError:  # duplicate IDs: unreadable, ask again
        return None


def _ask_rubric(system: str, request: str, llm, source: str, sha: str) -> Rubric:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=system), HumanMessage(content=request)]
    for attempt in range(2):
        text = _reply_text(llm.invoke(messages))
        rubric = _rubric_from_reply(text, source, sha)
        if rubric is not None:
            return rubric
        if attempt == 0:
            messages = messages + [AIMessage(content=text or "(empty reply)"), HumanMessage(content=RETRY_MESSAGE)]
    raise ValueError(f"the model's {source} rubric could not be read after a retry")


def draft_rubric(prompt: str, llm) -> Rubric:
    """A rubric drafted from the task prompt by ``llm`` (source "agent-drafted")."""
    request = f"Task prompt:\n\n{prompt.strip()}"
    return _ask_rubric(DRAFT_SYSTEM_PROMPT, request, llm, "agent-drafted", prompt_sha(prompt))


def convert_rubric(upload: str, prompt: str, llm) -> Rubric:
    """A teacher's rubric in any form, rewritten in the standard format (source "converted")."""
    request = f"Teacher's rubric:\n\n{upload.strip()}\n\nTask prompt:\n\n{prompt.strip()}"
    return _ask_rubric(CONVERT_SYSTEM_PROMPT, request, llm, "converted", prompt_sha(prompt))


def resolve_rubric(prompt: str, upload: str | None, llm) -> Rubric:
    """The run's rubric: a canonical upload as is, any other upload converted, else drafted."""
    if upload and upload.strip():
        try:
            parsed = parse_rubric_markdown(upload, source="uploaded", prompt_sha=prompt_sha(prompt))
        except ValueError:  # duplicate IDs: let the model renumber it
            parsed = None
        return parsed if parsed is not None else convert_rubric(upload, prompt, llm)
    return draft_rubric(prompt, llm)

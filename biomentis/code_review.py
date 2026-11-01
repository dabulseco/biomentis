"""On-demand code review: ask a model to check a script and return a corrected one.

Pure core, no UI and no queue: a request goes in, a `CodeReview` comes out.
`review_code` never raises; a failed call becomes a review with `error` set.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from biomentis.agent.tutor.instruction import _extract_json
from biomentis.independent_review import _clip
from biomentis.run_journal import format_code_block

MAX_CODE_CHARS = 40_000
MAX_EARLIER_CHARS = 30_000
MAX_OUTPUT_CHARS = 8_000
MAX_PURPOSE_CHARS = 1_500

# The title the transcript gives the agent's reasoning (ui_core.stream_agent_events).
REASONING_TITLE = "🤔 Reasoning"

VERDICTS = ("ok", "run_with_fixes", "do_not_trust")
KINDS = ("bug", "mismatch", "units", "science", "portability")
SEVERITIES = ("high", "medium", "low")

SYSTEM_PROMPT = """You are a careful scientific code reviewer. A student is about to run the SCRIPT below on their own computer, outside the research agent that wrote it. Find every error that would make its results wrong or stop it from running, then give a corrected script.

Check, in this order:
1. BUGS: wrong formulas, wrong variables, wrong function (e.g. natural log where log10 is meant), off-by-one, logic errors.
2. MISMATCHES: a comment, variable name, or printed label that disagrees with what the code actually computes (e.g. a comment says "2 uL" but the code uses 20; a value printed as "h" that is in minutes). Compare the OUTPUT, if given, against the labels.
3. UNITS AND CONSTANTS: unit conversions, physical constants, magnitudes. Recompute at least one key value by hand and say whether it is physically plausible.
4. SCIENCE: assumptions or reference values that are wrong for the field.
5. PORTABILITY: anything that will not work outside the agent (agent-only imports such as biomentis.tool, files that may not exist, missing packages, names defined in earlier steps).

Be specific: quote the line, say what is wrong, and give the correct value. Do not report style issues. Do not invent problems.

Earlier steps: the agent runs its Python steps in one shared namespace, so the script may use names defined in earlier steps. When EARLIER CODE is given, use it to tell where each name comes from. Every name defined in earlier steps that the script uses must be defined in the corrected script.

Corrected script: always return corrected_code, even when the verdict is "ok" (then it is the original made self-contained). It must run on its own with standard scientific Python packages (numpy, scipy, pandas, matplotlib), or standard tools for R/Bash, in the same language as the original script. Replace agent-only helpers with those, or keep them with a clear comment that they need the app. Python must be compatible with Python 3.10+ (no 3.12-only syntax, such as reusing the same quote character inside an f-string). Do not add a header comment; one is added for you.

Return ONLY a JSON object, no markdown fences:
{
  "verdict": "ok" | "run_with_fixes" | "do_not_trust",
  "issues": [{"line": <int or null>, "kind": "bug"|"mismatch"|"units"|"science"|"portability", "severity": "high"|"medium"|"low", "quote": "<code text>", "problem": "<what is wrong>", "fix": "<what it should be>"}],
  "corrected_code": "<the complete corrected script, runnable as-is>",
  "summary": "<2-3 sentences for a student>"
}"""


@dataclass
class CodeReviewRequest:
    code: str
    language: str = "python"
    output: str = ""
    task: str = ""
    purpose: str = ""
    earlier_code: str = ""
    run_id: str | None = None
    block_index: int | None = None
    source: str = "agent"  # "agent" | "answer" | "tutor" | "repl"


@dataclass
class ReviewIssue:
    line: int | None
    kind: str
    severity: str
    quote: str
    problem: str
    fix: str


@dataclass
class CodeReview:
    verdict: str = ""
    issues: list[ReviewIssue] = field(default_factory=list)
    corrected_code: str = ""
    summary: str = ""
    model: str = ""
    error: str = ""
    syntax_error: str = ""
    created: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodeReview":
        d = d or {}
        issues = []
        for i in d.get("issues") or []:
            if isinstance(i, dict):
                issues.append(
                    ReviewIssue(
                        line=i.get("line") if isinstance(i.get("line"), int) else None,
                        kind=str(i.get("kind", "bug")),
                        severity=str(i.get("severity", "medium")),
                        quote=str(i.get("quote", "")),
                        problem=str(i.get("problem", "")),
                        fix=str(i.get("fix", "")),
                    )
                )
        return cls(
            verdict=str(d.get("verdict", "")),
            issues=issues,
            corrected_code=str(d.get("corrected_code", "")),
            summary=str(d.get("summary", "")),
            model=str(d.get("model", "")),
            error=str(d.get("error", "")),
            syntax_error=str(d.get("syntax_error", "")),
            created=str(d.get("created", "")),
        )


def review_key(req: CodeReviewRequest, model: str) -> str:
    """Stable id of a review: the same code, context and model give the same key."""
    h = hashlib.sha256()
    for part in (req.code, req.output, req.earlier_code, req.task, req.language, model):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _numbered(code: str) -> str:
    return "\n".join(f"{n:>4}: {line}" for n, line in enumerate(code.split("\n"), 1))


def build_review_messages(req: CodeReviewRequest) -> list[BaseMessage]:
    # Number the original lines first, then clip, so labels match the block the student sees.
    numbered = _clip(_numbered(req.code), MAX_CODE_CHARS)
    parts = [f"LANGUAGE: {req.language}"]
    if req.task.strip():
        parts.append(f"TASK:\n{req.task.strip()}")
    if req.purpose.strip():
        parts.append(f"WHAT THIS STEP WAS FOR:\n{req.purpose.strip()}")
    if req.earlier_code.strip():
        parts.append(f"EARLIER CODE (earlier steps of the same run):\n{_clip(req.earlier_code, MAX_EARLIER_CHARS)}")
    parts.append(f"SCRIPT (line-numbered):\n{numbered}")
    if req.output.strip():
        parts.append(f"OUTPUT of the script:\n{_clip(req.output, MAX_OUTPUT_CHARS)}")
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n\n".join(parts))]


def _norm(value, allowed: tuple, default: str) -> str:
    v = str(value or "").strip().lower().replace(" ", "_")
    return v if v in allowed else default


def parse_code_review(text: str) -> CodeReview:
    """Tolerant parse of a model reply; `error` is set when it is unreadable."""
    data = _extract_json(text or "")
    if not isinstance(data, dict):
        return CodeReview(error="The reviewer's reply was not readable JSON.")
    issues = []
    raw = data.get("issues")
    for i in raw if isinstance(raw, list) else []:
        if not isinstance(i, dict):
            continue
        line = i.get("line")
        if isinstance(line, str) and line.strip().isdigit():
            line = int(line.strip())
        issues.append(
            ReviewIssue(
                line=line if isinstance(line, int) and not isinstance(line, bool) else None,
                kind=_norm(i.get("kind"), KINDS, "bug"),
                severity=_norm(i.get("severity"), SEVERITIES, "medium"),
                quote=str(i.get("quote") or ""),
                problem=str(i.get("problem") or ""),
                fix=str(i.get("fix") or ""),
            )
        )
    corrected = data.get("corrected_code")
    return CodeReview(
        verdict=_norm(data.get("verdict"), VERDICTS, "run_with_fixes"),
        issues=issues,
        corrected_code=corrected.strip("\n") if isinstance(corrected, str) else "",
        summary=str(data.get("summary") or ""),
    )


def _reply_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return content if isinstance(content, str) else str(content or "")


def _syntax_error(code: str, language: str) -> str:
    """The syntax error in `code`, or "" when it parses (or cannot be checked)."""
    lang = (language or "").lower()
    if lang in ("python", "py"):
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} (line {e.lineno})"
        return ""
    if lang in ("bash", "sh", "shell"):
        try:
            r = subprocess.run(["bash", "-n"], input=code, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return ""  # bash unavailable: skip the check
        return f"SyntaxError: {r.stderr.strip()[:300]}" if r.returncode != 0 else ""
    return ""


def _header(model_name: str) -> str:
    return (
        f"# Generated by Code Review ({model_name}) on {date.today().isoformat()} "
        "- check before use.\n"
    )


def review_code(llm, req: CodeReviewRequest, model_name: str) -> CodeReview:
    """Review a script; retry once if the reply is unreadable, lacks a usable script or has a syntax error."""
    messages = build_review_messages(req)
    best: CodeReview | None = None
    last = CodeReview(error="No reply from the reviewer.")
    for attempt in range(2):
        try:
            text = _reply_text(llm.invoke(messages))
        except Exception as e:  # never raise
            if best is not None:
                break
            last = CodeReview(error=f"The review call failed: {e}")
            break
        review = parse_code_review(text)
        problem = ""
        if review.error:
            problem = "Your reply was not a readable JSON object. Reply with ONLY the JSON object described, with no other text."
        else:
            if review.corrected_code:
                review.syntax_error = _syntax_error(review.corrected_code, req.language)
            if not review.corrected_code:
                problem = "Your reply had no corrected_code. Always return the complete corrected script in corrected_code, even when the verdict is ok."
            elif review.syntax_error:
                problem = (
                    f"The corrected_code you returned fails its syntax check: {review.syntax_error}. "
                    "Return the whole review again with a corrected_code that parses."
                )
        if not review.error and not (best is not None and best.corrected_code and not review.corrected_code):
            best = review
        else:
            last = review
        if not problem:
            break
        if attempt == 0:
            messages = messages + [AIMessage(content=text), HumanMessage(content=problem)]
    result = best if best is not None else last
    if result.error:
        return result
    result.model = model_name
    result.created = date.today().isoformat()
    if result.corrected_code:
        result.corrected_code = _header(model_name) + result.corrected_code + "\n"
    return result


def _position(transcript: list[dict], entry: dict) -> int | None:
    """Index of `entry` itself (by identity) in the transcript, or None."""
    return next((i for i, e in enumerate(transcript) if e is entry), None)


def _earlier_code(transcript: list[dict], position: int) -> str:
    """Every code entry before `position`, in the `run_journal.code_script` block format.

    The agent's Python namespace persists across tutor segments and turns, so
    every code entry before this point counts, whichever run it came from.
    """
    before = [e for e in transcript[:position] if e.get("kind") == "code"]
    return "\n\n".join(
        format_code_block(i, e.get("language") or "python", e.get("content") or "")
        for i, e in enumerate(before, start=1)
    )


def task_for_entry(transcript: list[dict], entry: dict) -> str:
    """The run prompt an entry belongs to: the nearest user entry before it ("" if none)."""
    position = _position(transcript, entry)
    if position is None:
        return ""
    for e in reversed(transcript[:position]):
        if e.get("role") == "user":
            return e.get("content") or ""
    return ""


def requests_for_answer(transcript: list[dict], entry: dict, task: str) -> list[CodeReviewRequest]:
    """One review request per fenced code block in an answer entry.

    The earlier code is every code entry before the answer, so a block that
    reuses the run's names can be made self-contained.
    """
    blocks = extract_code_blocks(entry.get("content") or "")
    if not blocks:
        return []
    position = _position(transcript, entry)
    earlier = _earlier_code(transcript, position) if position is not None else ""
    return [
        CodeReviewRequest(code=code, language=lang, task=task, earlier_code=earlier, block_index=i, source="answer")
        for i, (lang, code) in enumerate(blocks)
    ]


def request_for_entry(transcript: list[dict], entry: dict, task: str) -> CodeReviewRequest:
    """Build the review request for one agent code entry of the transcript.

    The output is the observation that follows the entry, the purpose is the
    reasoning the agent wrote just before it, and the earlier code is all code
    entries before it in the transcript, in the `run_journal.code_script` block format.
    An entry from before entries were tagged (no run_id/block_index) still
    works; it just has no earlier code.
    """
    position = _position(transcript, entry)
    output = purpose = earlier = ""
    if position is not None:
        # Output: the observation before the next code block, if any.
        for e in transcript[position + 1 :]:
            if e.get("kind") == "observation":
                output = e.get("content") or ""
                break
            if e.get("kind") == "code":
                break
        # Purpose: the reasoning immediately before the code, if any.
        for e in reversed(transcript[:position]):
            if e.get("kind") == "code" or e.get("role") == "user":
                break
            if not e.get("kind") and e.get("title") == REASONING_TITLE:
                purpose = (e.get("content") or "")[:MAX_PURPOSE_CHARS]
                break
        earlier = _earlier_code(transcript, position)

    run_id, block_index = entry.get("journal_run_id"), entry.get("block_index")
    return CodeReviewRequest(
        code=entry.get("content") or "",
        language=entry.get("language") or "python",
        output=output,
        task=task,
        purpose=purpose,
        earlier_code=earlier,
        run_id=run_id,
        block_index=block_index,
        source="agent",
    )


_FENCE = re.compile(r"^```[ \t]*([\w+-]*)[^\n]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def extract_code_blocks(markdown: str) -> list[tuple[str, str]]:
    """Fenced code blocks of a markdown text as `(language, code)`; an unlabelled block is python."""
    return [((lang or "python").lower(), body.rstrip("\n")) for lang, body in _FENCE.findall(markdown or "")]

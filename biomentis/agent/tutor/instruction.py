"""Per-step instruction cards.

`InstructionGenerator` turns each `UIEvent` from `stream_agent_events` into
a teaching card the student can read alongside the agent's work.

A card carries TWO independently-enabled instructional modes, because
"explain this step" means two different things to a student:

  - `technical`  — the technology being carried out: the method, tool,
    library, parameters, data structures and I/O. Strictly engineering
    content; no biology.
  - `scientific` — the science: what is being investigated, why it
    matters scientifically, and how this step's result feeds the material
    being assembled to answer THIS specific query.

Each mode has its own `what` / `why` / `prerequisites` / `look_for`, kept
separate so a student reading the technical lens never has the science
mixed into it (and vice versa). `scientific` additionally has `impact`
(how the step advances the answer to the user's query). Shared across
both modes: `citations`, `bloom_target`, `dok_target`.

Either mode may be enabled alone, both together, or neither. The enabled
set is passed down from the UI (`InstructionGenerator(modes=...)` or
`generate(..., modes=...)`); only the enabled modes are requested from
the LLM, so turning one off makes the prompt smaller rather than
generating text the student never sees. With no modes enabled, no LLM
call is made at all.

Pipeline:
  1. Truncate the event's content to a bounded size (the LLM prompt has
     a context budget, and a 50k-char traceback is not useful to teach).
  2. Build a KB retrieval query from the event content (a short snippet
     is enough; we don't need a separate query-rewriter LLM call).
  3. Retrieve up to 4 KB snippets. If the KB is empty, skip retrieval
     and the card will be generated with no citations.
  4. Compose a strict-JSON prompt for the enabled modes and call the LLM.
  5. Parse the JSON, validate every field, drop invented citations,
     coerce types where safe.
  6. Cache the result by (event_type, content_hash, kb_signature, modes)
     so a noisy re-emission of the same step doesn't re-bill the LLM.

Failure modes are handled at every step:
  - LLM raises or returns non-JSON → soft-failure card with
    `_generation_failed=True` and `what` set to the event title.
  - JSON is malformed → same soft-failure path.
  - Citation refers to a source not in the retrieved set → silently
    dropped (never invent a source).

The dataclass and the engine both import this module. The engine calls
`InstructionGenerator.generate(event)`; the dataclass is returned.

`generate_roadmap` is a separate, one-time call made once per run (not
per step): it turns the agent's own first-message plan into a short
`RoadmapCard` (overview + ordered steps) shown before the first
per-step card, so the student sees where the walkthrough is headed
before diving into step-by-step detail.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


# --- Instruction modes ----------------------------------------------------
#
# The two lenses a student can be taught a step through. They are
# independent switches, not a single either/or: a student debugging a
# pipeline wants `technical` only, a student learning the biology wants
# `scientific` only, a student learning both wants both, and a student who
# just wants to watch the agent work turns both off.

MODE_TECHNICAL = "technical"
MODE_SCIENTIFIC = "scientific"
ALL_MODES: tuple[str, ...] = (MODE_TECHNICAL, MODE_SCIENTIFIC)

# Display metadata, kept next to the modes themselves so the renderer, the
# log, and the prompt can't drift apart on what a mode is called. `color`
# is a Streamlit markdown color name (see `ui_tutor._render_instruction_card`)
# — technical is blue, scientific is violet.
MODE_META: dict[str, dict[str, str]] = {
    MODE_TECHNICAL: {
        "label": "Technical details",
        "short": "Technical",
        "color": "blue",
        "icon": ":material/build:",
        "blurb": "How this step is carried out: methods, tools, parameters, data.",
    },
    MODE_SCIENTIFIC: {
        "label": "Scientific content",
        "short": "Scientific",
        "color": "violet",
        "icon": ":material/science:",
        "blurb": "The science: what is being asked, why it matters, and how it builds the answer.",
    },
}


def normalize_modes(modes: Any) -> tuple[str, ...]:
    """Coerce any mode spec to a canonical, deduped tuple in `ALL_MODES` order.

    Accepts None (-> all modes, the historical default), a single mode
    string, or any iterable of mode strings. Unknown names are dropped
    rather than raising: a stale value in `st.session_state` should
    degrade to "that mode is off", never break the run.
    """
    if modes is None:
        return ALL_MODES
    if isinstance(modes, str):
        modes = [modes]
    try:
        wanted = {str(m).strip().lower() for m in modes}
    except TypeError:
        return ()
    return tuple(m for m in ALL_MODES if m in wanted)


# --- InstructionCard dataclass --------------------------------------------


@dataclass
class CardSection:
    """One instructional lens over a step.

    The same four fields carry different content per mode — that
    separation is the point. `technical.prerequisites` are the artifacts
    and tooling the step needs in order to run; `scientific.prerequisites`
    are the concepts the student needs to follow the science. Keeping them
    in parallel structures (rather than one merged list) is what lets the
    renderer show one mode without leaking the other's content.

    `impact` is populated for `scientific` only: how this step's result
    changes the material being assembled to answer the user's query.
    """

    what: str = ""
    why: str = ""
    prerequisites: list[str] = field(default_factory=list)
    look_for: list[str] = field(default_factory=list)
    impact: str = ""  # scientific mode only

    def is_empty(self) -> bool:
        return not (
            self.what or self.why or self.impact or self.prerequisites or self.look_for
        )

    def to_dict(self) -> dict:
        d = {
            "what": self.what,
            "why": self.why,
            "prerequisites": list(self.prerequisites),
            "look_for": list(self.look_for),
        }
        if self.impact:
            d["impact"] = self.impact
        return d


@dataclass
class InstructionCard:
    """A teaching card attached to one agent event.

    Holds one `CardSection` per instructional mode plus the fields that
    are mode-independent (citations are evidence for both lenses; Bloom
    and DOK describe the step, not the lens).

    `modes` records which lenses were actually requested when this card
    was generated — the renderer uses it to tell "this mode was off" apart
    from "this mode was on but the LLM returned nothing for it".

    The legacy flat attributes (`what`, `why`, `prerequisites`,
    `look_for`) survive as read-only properties so the logger, the Critic
    digest, and the chat's `push_recent_card` keep working unchanged.
    They read the technical section first and fall back to the scientific
    one.
    """

    technical: CardSection = field(default_factory=CardSection)
    scientific: CardSection = field(default_factory=CardSection)
    citations: list[dict] = field(default_factory=list)  # [{source, page, snippet}]
    bloom_target: str = ""  # "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate" | "Create"
    dok_target: int = 0     # 1 | 2 | 3 | 4
    modes: tuple[str, ...] = ALL_MODES
    # Internal: True if the LLM call failed and we returned a soft-failure
    # card. The renderer can use this to show a "tutor unavailable" hint.
    _generation_failed: bool = field(default=False, repr=False)

    # ---- mode access ----------------------------------------------------

    def section(self, mode: str) -> CardSection:
        """Return the section for `mode` (an empty section for unknown names)."""
        return getattr(self, mode) if mode in ALL_MODES else CardSection()

    def active_sections(self) -> list[tuple[str, CardSection]]:
        """(mode, section) pairs for enabled modes that actually have content."""
        return [
            (m, self.section(m))
            for m in normalize_modes(self.modes)
            if not self.section(m).is_empty()
        ]

    def is_empty(self) -> bool:
        return not self.active_sections()

    # ---- backwards-compatible flat view ---------------------------------

    @property
    def what(self) -> str:
        return self.technical.what or self.scientific.what

    @property
    def why(self) -> str:
        return self.technical.why or self.scientific.why

    @property
    def prerequisites(self) -> list[str]:
        return list(self.technical.prerequisites) + list(self.scientific.prerequisites)

    @property
    def look_for(self) -> list[str]:
        return list(self.technical.look_for) + list(self.scientific.look_for)


# --- Bloom/DOK validation -------------------------------------------------


_BLOOM_ALLOWED = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
_DOK_ALLOWED = {1, 2, 3, 4}

# Event-type default for Bloom — used to anchor the prompt and to pick a
# sensible fallback if the LLM's response is malformed. Calibration is
# deliberately modest: most agent steps are Apply or Analyze, not Create.
_BLOOM_DEFAULTS = {
    "reasoning": "Understand",
    "code": "Apply",
    "observation": "Analyze",
    "solution": "Create",
    "summary": "Evaluate",
    "file": "Apply",
    "status": "",
    "complete": "",
}
_DOK_DEFAULTS = {
    "reasoning": 2,
    "code": 2,
    "observation": 2,
    "solution": 3,
    "summary": 3,
    "file": 2,
    "status": 0,
    "complete": 0,
}


# --- Truncation -----------------------------------------------------------


_MAX_EVENT_CHARS = 3000  # per-event content size fed to the LLM
_MAX_KB_SNIPPET_CHARS = 600  # per-snippet size in the prompt
_MAX_KB_SNIPPETS = 4
_MAX_PREREQS = 4
_MAX_LOOK_FOR = 4
# Per-field caps, applied per section. The scientific "why" is allowed to run
# roughly twice as long as the technical one: it is asked for 4-8 sentences of
# mechanism and interpretation, where the technical "why" is asked for 2-4
# sentences of engineering rationale.
_MAX_WHAT_CHARS = {MODE_TECHNICAL: 500, MODE_SCIENTIFIC: 700}
_MAX_WHY_CHARS = {MODE_TECHNICAL: 1200, MODE_SCIENTIFIC: 2400}
_MAX_IMPACT_CHARS = 1200
_MAX_CITATIONS = 3
_CITATION_SNIPPET_MAX = 200


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n\n[…truncated for prompt size…]"


# --- JSON extraction -----------------------------------------------------


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_LOOSE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from an LLM response.

    Tries, in order: a fenced ```json``` block, a loose {...} match, then
    a plain json.loads. Returns None on any failure — the caller falls
    back to a soft-failure card.
    """
    if not text:
        return None
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = _JSON_LOOSE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


# --- Field-level validators ----------------------------------------------


def _coerce_str_list(value: Any, max_items: int) -> list[str]:
    """Coerce a value to a list[str] of bounded length. Tolerant of the
    LLM returning a single string instead of a list, or None."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [v for v in value if isinstance(v, str)]
    else:
        return []
    return [_truncate(s, 200).strip() for s in items if s.strip()][:max_items]


def _coerce_bloom(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        v = value.strip()
        # Case-insensitive match against allowed set.
        for allowed in _BLOOM_ALLOWED:
            if v.lower() == allowed.lower():
                return allowed
    return fallback


def _coerce_dok(value: Any, fallback: int) -> int:
    try:
        v = int(value)
        if v in _DOK_ALLOWED:
            return v
    except (TypeError, ValueError):
        pass
    return fallback


def _coerce_citations(
    value: Any, allowed_sources: set[str], max_items: int
) -> list[dict]:
    """Drop any citation whose source isn't in the retrieved KB set, and
    any citation without a real snippet. Never invent a source."""
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        src = item.get("source")
        if not isinstance(src, str) or src not in allowed_sources:
            continue
        snippet = item.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        page = item.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        out.append(
            {
                "source": src,
                "page": page,
                "snippet": _truncate(snippet, _CITATION_SNIPPET_MAX).strip(),
            }
        )
        if len(out) >= max_items:
            break
    return out


def _coerce_section(value: Any, mode: str) -> CardSection:
    """Validate one mode's slice of the LLM response into a `CardSection`.

    Tolerant on purpose: a small local model asked for two nested objects
    will sometimes return a string where an object was requested, or omit
    `impact` entirely. Anything unusable becomes an empty field rather
    than an exception, because a partly-filled card still teaches
    something and a raised exception costs the student the whole step.
    """
    if isinstance(value, str):
        # Model collapsed the whole section into one blob — keep it as "what"
        # rather than throwing the text away.
        return CardSection(what=_truncate(value, _MAX_WHAT_CHARS[mode]).strip())
    if not isinstance(value, dict):
        return CardSection()
    section = CardSection(
        what=_truncate(str(value.get("what", "") or ""), _MAX_WHAT_CHARS[mode]).strip(),
        why=_truncate(str(value.get("why", "") or ""), _MAX_WHY_CHARS[mode]).strip(),
        prerequisites=_coerce_str_list(value.get("prerequisites"), _MAX_PREREQS),
        look_for=_coerce_str_list(value.get("look_for"), _MAX_LOOK_FOR),
    )
    if mode == MODE_SCIENTIFIC:
        section.impact = _truncate(
            str(value.get("impact", "") or ""), _MAX_IMPACT_CHARS
        ).strip()
    return section


def _sections_from_response(data: dict, modes: tuple[str, ...]) -> dict[str, CardSection]:
    """Pull one `CardSection` per enabled mode out of a parsed response.

    Handles three shapes, in order of preference:
      1. Nested, as prompted: `{"technical": {...}, "scientific": {...}}`.
      2. Prefixed flat keys: `{"technical_what": ..., "scientific_why": ...}`,
         which weaker models produce when they flatten nested schemas.
      3. Legacy flat keys: `{"what": ..., "why": ..., "prerequisites": ...}`
         — the pre-two-mode card shape. Assigned to the first enabled mode.
         This is what keeps old cached responses and the smoke test's stub
         LLM working.
    """
    out: dict[str, CardSection] = {}
    for mode in modes:
        if isinstance(data.get(mode), dict | str):
            out[mode] = _coerce_section(data[mode], mode)
            continue
        prefixed = {
            k[len(mode) + 1 :]: v
            for k, v in data.items()
            if isinstance(k, str) and k.startswith(f"{mode}_")
        }
        if prefixed:
            out[mode] = _coerce_section(prefixed, mode)
        else:
            out[mode] = CardSection()

    if all(sec.is_empty() for sec in out.values()) and modes:
        legacy = {k: data.get(k) for k in ("what", "why", "prerequisites", "look_for", "impact")}
        if any(v for v in legacy.values()):
            out[modes[0]] = _coerce_section(legacy, modes[0])
    return out


@dataclass
class RoadmapCard:
    """A one-time, run-level preview of the agent's overall plan, shown
    before the first per-step InstructionCard. Distinct from InstructionCard:
    this previews *all* steps at a glance; InstructionCard explains one."""

    overview: str = ""
    steps: list[dict] = field(default_factory=list)  # [{"title": str, "why": str}]
    _generation_failed: bool = field(default=False, repr=False)


def _build_soft_failure_roadmap() -> RoadmapCard:
    return RoadmapCard(_generation_failed=True)


_ROADMAP_SYSTEM_PROMPT = """You are an expert tutor previewing a multi-step biomedical research task for a student, before a step-by-step walkthrough begins. The agent has just proposed its plan; your job is to turn that plan into a short roadmap the student reads once, up front, so they know where the walkthrough is headed.

You will be given:
- TASK: the student's original research task
- PLAN_TEXT: the agent's own first message, which usually includes a numbered plan (may be informal or embedded in other reasoning text)

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):
{
  "overview": "<2-3 sentences, plain language: the overall strategy for solving this task>",
  "steps": [{"title": "<short step name, a few words>", "why": "<1-2 sentences: why this step is needed>"}]
}

RULES:
1. Base "steps" on PLAN_TEXT's own numbered list. Don't invent steps it doesn't imply. If PLAN_TEXT has no clear numbered list, infer a reasonable 3-7 step breakdown from TASK and PLAN_TEXT together.
2. Each step's "why" is 1-2 sentences ONLY — this is a preview, not a full explanation. Each step gets its own detailed teaching card later in the walkthrough.
3. "overview" orients the student to the strategy, not a restatement of the task.
4. Output ONLY the JSON. No markdown code fences, no commentary, no apology.
"""


def generate_roadmap(llm, task: str, plan_text: str) -> RoadmapCard:
    """Generate a one-time roadmap card for the start of a tutor run.

    Args:
        llm: A LangChain chat model, same calling convention as
            `InstructionGenerator`.
        task: The student's original research prompt.
        plan_text: The agent's first reasoning event's content (usually
            contains its numbered plan/checklist).
    """
    if llm is None:
        return _build_soft_failure_roadmap()

    from langchain_core.messages import HumanMessage, SystemMessage

    user_prompt = (
        f"TASK: {task or '(no task provided)'}\n\n"
        "PLAN_TEXT:\n---\n"
        f"{_truncate(plan_text, _MAX_EVENT_CHARS) or '(empty)'}\n---"
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_ROADMAP_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        text = getattr(response, "content", str(response)) or ""
    except Exception as e:
        print(f"tutor: roadmap LLM call failed: {e!r}")
        return _build_soft_failure_roadmap()

    data = _extract_json(text)
    if data is None:
        return _build_soft_failure_roadmap()

    overview = _truncate(str(data.get("overview", "") or ""), 600).strip()
    steps: list[dict] = []
    raw_steps = data.get("steps")
    if isinstance(raw_steps, list):
        for item in raw_steps[:10]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            why = item.get("why")
            steps.append(
                {
                    "title": _truncate(title.strip(), 160),
                    "why": _truncate(why.strip(), 300) if isinstance(why, str) else "",
                }
            )
    return RoadmapCard(overview=overview, steps=steps)


def _build_soft_failure_card(event, modes: Any = ALL_MODES) -> InstructionCard:
    """Return a placeholder card when the LLM call fails. The renderer
    sees `_generation_failed=True` and shows a small note. The user can
    still click Continue and move on.

    The event title is put in the FIRST enabled mode's `what` so the
    fallback line renders under a heading the student actually has
    switched on.
    """
    modes = normalize_modes(modes)
    title = getattr(event, "title", None) or event.type
    card = InstructionCard(
        bloom_target=_BLOOM_DEFAULTS.get(event.type, ""),
        dok_target=_DOK_DEFAULTS.get(event.type, 0),
        modes=modes,
        _generation_failed=True,
    )
    if modes:
        card.section(modes[0]).what = f"({event.type}) {title}".strip()
    return card


# --- Prompt --------------------------------------------------------------


# Shared preamble — what the model is looking at, regardless of mode.
_PROMPT_PREAMBLE = """You are an expert tutor for a biomedical research agent. The student just watched the agent do ONE step in a multi-step research task. Produce a teaching card the student can read while the agent continues.

You will be given:
- EVENT_TYPE: one of reasoning, code, observation, solution, summary, file
- EVENT_CONTENT: the event's text (may be truncated for prompt size)
- KB_SNIPPETS: up to 4 relevant passages from a knowledge base the student has uploaded (may be empty)
- TASK: the student's original research task, for context
"""

# --- Per-mode prompt fragments -------------------------------------------
#
# Each mode contributes (a) its slice of the required JSON and (b) its own
# rules. They are deliberately written to repel each other: the technical
# block forbids biology, the scientific block forbids implementation
# detail. Without that explicit separation a single LLM call happily
# writes the same paragraph twice with different adjectives — the exact
# failure the two-mode split exists to prevent.

_MODE_SCHEMA = {
    MODE_TECHNICAL: """  "technical": {
    "what": "<1-2 sentences: in technical terms, the operation this step performs \u2014 name the actual method, tool, library, API, or algorithm being run>",
    "why": "<2-4 sentences: why THIS technique/tool/parameterization was used \u2014 what it buys you over the alternative, and what would break, be unreliable, or be impossible without this step. Technical rationale only.>",
    "prerequisites": ["<something that must ALREADY be true before this step can run: an input file or artifact from an earlier step, an installed tool/package/credential, a required data format, shape, or identifier convention>", "..."],
    "look_for": ["<a concrete technical success signal in the output, e.g. 'a non-empty DataFrame with columns hit_id, e_value, identity' or 'exit code 0 and a written .pdb file'>", "..."]
  }""",
    MODE_SCIENTIFIC: """  "scientific": {
    "what": "<2-3 sentences: WHAT is being done scientifically \u2014 the biological/chemical entities involved, the property being measured or predicted, and the scientific question this step interrogates>",
    "why": "<4-8 sentences: WHY this matters scientifically. Explain the underlying principle or mechanism, what claim the result licenses, why that claim is worth making, what a scientist would conclude from a strong vs. a weak result, and any caveat or assumption the interpretation rests on. This is the longest field on the card \u2014 teach the science, do not summarize the code.>",
    "impact": "<2-4 sentences: HOW this step's result changes the material being assembled to answer THE SPECIFIC QUERY in TASK \u2014 what it contributes to the eventual answer, which later reasoning depends on it, and what would be missing or unsupported in the final answer without it. Refer to the actual task, not to research in general.>",
    "prerequisites": ["<a scientific concept, mechanism, or piece of domain background the student must understand BEFORE this step makes sense>", "..."],
    "look_for": ["<a scientifically meaningful signal to look for in the result AND what it would mean, e.g. 'conserved residues clustering in the receptor-binding loop \u2014 evidence of functional constraint'>", "..."]
  }""",
}

_MODE_RULES = {
    MODE_TECHNICAL: """TECHNICAL SECTION RULES:
T1. TECHNICAL CONTENT ONLY. Methods, algorithms, tools, libraries, APIs, parameters, data structures, file formats, I/O, error handling, performance. Do NOT explain biology, disease relevance, or why the science matters \u2014 that belongs to a different section the student may be reading alongside this one, and duplicating it here makes both worse.
T2. BE DESCRIPTIVE ABOUT THE TECHNOLOGY BEING CARRIED OUT. Name what is actually used in EVENT_CONTENT \u2014 the specific function, package, database, algorithm, or file format \u2014 and the parameters or options that materially change the result. A student should finish "what" knowing which technology ran, and finish "why" knowing why that technology.
T3. "prerequisites" ARE STRICTLY PRE-REQUISITES: things that must already exist or already be true for this step to run at all (upstream outputs, installed dependencies, required identifiers or formats, prior configuration). They are NOT takeaways, NOT next steps, and NOT concepts the student merely finds interesting.
T4. Both halves are required: "what" is WHAT is technically being done, "why" is WHY it is being done that way.""",
    MODE_SCIENTIFIC: """SCIENTIFIC SECTION RULES:
S1. SCIENCE ONLY. Do NOT describe implementation \u2014 no function names, no library or parameter talk, no file formats. Where a technical detail is unavoidable, state it as the scientific operation it performs ("aligning the sequences", not "calling MUSCLE with default gap penalties").
S2. EMPHASIZE WHAT AND WHY. "what" names the science being done; "why" is the heart of the card \u2014 the mechanism or principle at work and why the result is scientifically meaningful. Give "why" real length; a two-line "why" here is a failure.
S3. "impact" IS REQUIRED AND MUST BE SPECIFIC TO TASK. Tie this step to the actual query the student asked: what it contributes to the answer being built, and what would be missing from that answer without it. A statement that would fit any project equally well is wrong here.
S4. "prerequisites" are the scientific concepts needed to follow this step, and "look_for" are the scientifically interpretable signals in the result \u2014 not technical success checks like "the file exists".""",
}

_BOTH_MODES_RULE = """SEPARATION RULE (both sections requested):
The two sections are read side by side under different headings. They must not overlap. If a sentence would be equally at home in either section, it belongs in neither \u2014 sharpen it until it is clearly one or the other. The technical section explains the machinery; the scientific section explains the meaning.
"""

_SHARED_RULES = """SHARED RULES:
1. CITATIONS: only cite KB_SNIPPETS you actually saw. If none are relevant to THIS step, return "citations": []. Never invent a source or page.
2. BLOOM/DOK CALIBRATION \u2014 be honest, do not inflate:
   - reasoning \u2192 usually Understand or Evaluate
   - code \u2192 usually Apply (running a procedure) or Analyze (interpreting)
   - observation \u2192 usually Analyze
   - solution / summary \u2192 usually Create or Evaluate
   Most steps are Apply/Analyze. Reserve Create for genuine synthesis steps.
3. Every "what" must stand alone \u2014 a student who skipped the prior step should still understand it.
4. "prerequisites" and "look_for" are 2-4 short bullets each, phrased so the student can check them off.
5. Output ONLY the JSON. No markdown code fences, no commentary, no apology.

Default Bloom/DOK by event_type (override only if the content clearly supports it):
  reasoning   \u2192 Understand, 2
  code        \u2192 Apply, 2
  observation \u2192 Analyze, 2
  solution    \u2192 Create, 3
  summary     \u2192 Evaluate, 3
  file        \u2192 Apply, 2
"""


def _build_system_prompt(modes: Any = ALL_MODES) -> str:
    """Assemble the per-step system prompt for the enabled modes.

    Only the enabled modes contribute schema keys and rules, so switching
    a mode off shrinks the prompt instead of paying for text the student
    will never see. Returns "" when no mode is enabled \u2014 the caller skips
    the LLM call entirely in that case.
    """
    modes = normalize_modes(modes)
    if not modes:
        return ""

    schema_parts = [_MODE_SCHEMA[m] for m in modes]
    schema_parts.append(
        '  "citations": [{"source": "<EXACT source string from KB_SNIPPETS>", '
        '"page": <int or null>, "snippet": "<\u2264200 char quote>"}]'
    )
    schema_parts.append(
        '  "bloom_target": "<one of: Remember | Understand | Apply | Analyze | Evaluate | Create>"'
    )
    schema_parts.append('  "dok_target": <1 | 2 | 3 | 4>')

    parts = [
        _PROMPT_PREAMBLE,
        "Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):\n{\n"
        + ",\n".join(schema_parts)
        + "\n}",
    ]
    parts.extend(_MODE_RULES[m] for m in modes)
    if len(modes) > 1:
        parts.append(_BOTH_MODES_RULE)
    parts.append(_SHARED_RULES)
    return "\n\n".join(part.strip() for part in parts) + "\n"


# Both-modes prompt, materialized at import so `docs/generate_card_prompt_doc.py`
# (and anything else that wants to read "the prompt") still can.
_SYSTEM_PROMPT = _build_system_prompt(ALL_MODES)


def _build_user_prompt(
    event,
    event_content_truncated: str,
    kb_snippets: list,
    task: str,
    modes: tuple[str, ...] = ALL_MODES,
) -> str:
    modes = normalize_modes(modes)
    parts = [
        f"TASK: {task or '(no task provided)'}",
        f"SECTIONS_REQUESTED: {', '.join(modes) if modes else '(none)'}",
        f"EVENT_TYPE: {event.type}",
        f"EVENT_TITLE: {getattr(event, 'title', '') or '(none)'}",
        "",
        "EVENT_CONTENT (may be truncated):",
        "---",
        event_content_truncated or "(empty)",
        "---",
        "",
    ]
    if kb_snippets:
        parts.append("KB_SNIPPETS (cite only these, or return citations: []):")
        for i, snip in enumerate(kb_snippets, 1):
            parts.append(
                f"[{i}] source={snip['source']!r} page={snip['page']!r}"
            )
            parts.append(f"    {snip['content']}")
    else:
        parts.append("KB_SNIPPETS: (none — the knowledge base is empty)")
    return "\n".join(parts)


# --- Cache ---------------------------------------------------------------


class _LRUCache(OrderedDict):
    """Tiny LRU cache; bounded to `_MAX_CACHE_ENTRIES`."""

    def __init__(self, maxlen: int = 64) -> None:
        super().__init__()
        self.maxlen = maxlen

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxlen:
            self.popitem(last=False)


# --- InstructionGenerator ------------------------------------------------


class InstructionGenerator:
    """LLM-driven teaching-card generator.

    Args:
        llm: A LangChain chat model. Used with the same calling convention
            as the rest of Biomentis (`.invoke([SystemMessage, HumanMessage])`).
        knowledge_base: An optional `KnowledgeBase`. When provided, the
            generator retrieves up to 4 KB snippets per event and requires
            citations to be drawn only from that set.
        max_event_chars: Per-event truncation size (default 3000).
        cache_size: Bounded LRU size (default 64).
        modes: Which instructional lenses to generate — any subset of
            `ALL_MODES` (`technical`, `scientific`). Defaults to both.
            The UI re-sets this whenever the student flips a mode toggle;
            an empty tuple means "generate nothing", and `generate()`
            then returns an empty card without calling the LLM.
    """

    def __init__(
        self,
        llm,
        knowledge_base=None,
        max_event_chars: int = _MAX_EVENT_CHARS,
        cache_size: int = 64,
        modes: Any = ALL_MODES,
    ) -> None:
        self.llm = llm
        self.kb = knowledge_base
        self.max_event_chars = max_event_chars
        self.modes: tuple[str, ...] = normalize_modes(modes)
        self._cache = _LRUCache(maxlen=cache_size)

    def set_modes(self, modes: Any) -> None:
        """Change the enabled instructional modes.

        Cards already in the cache are keyed by mode set, so switching a
        mode on mid-run regenerates affected cards rather than serving a
        card that is missing the newly-enabled section.
        """
        self.modes = normalize_modes(modes)

    # ---- public API -------------------------------------------------------

    def generate(self, event, task: str = "", modes: Any = None) -> InstructionCard:
        """Generate a teaching card for one `UIEvent`.

        Args:
            event: A `UIEvent` from `stream_agent_events`. Read fields:
                `type`, `content`, `title`, `file_path`, `file_kind`.
            task: The student's original research prompt; passed to the LLM
                for context. Optional.
            modes: Override the generator's configured modes for this one
                call. `None` (the default) uses `self.modes`.
        """
        modes = self.modes if modes is None else normalize_modes(modes)
        # No lens enabled → nothing to teach. Return an empty card without
        # spending an LLM call; the caller treats an empty card as "skip".
        if not modes:
            return InstructionCard(modes=())

        content = self._event_text(event)
        content_truncated = _truncate(content, self.max_event_chars)

        # Cache key — same content + same KB → same card.
        kb_sig = ""
        if self.kb is not None:
            try:
                kb_sig = self.kb.kb_signature() or ""
            except Exception:
                kb_sig = ""
        cache_key = (event.type, _content_hash(content_truncated), kb_sig, modes)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # KB retrieval. Empty queries or empty KB → no snippets.
        kb_snippets = []
        allowed_sources: set[str] = set()
        if self.kb is not None and kb_sig:
            try:
                query = self._build_retrieval_query(event, content_truncated)
                docs = self.kb.retrieve(query, k=_MAX_KB_SNIPPETS) or []
            except Exception as e:
                print(f"tutor: KB retrieval failed: {e!r}")
                docs = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                allowed_sources.add(src)
                kb_snippets.append(
                    {
                        "source": src,
                        "page": page,
                        "content": _truncate(d.page_content, _MAX_KB_SNIPPET_CHARS),
                    }
                )

        # LLM call.
        card = self._call_llm(
            event, content_truncated, kb_snippets, allowed_sources, task, modes
        )

        self._cache[cache_key] = card
        return card

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _event_text(event) -> str:
        """Flatten the event's pedagogical content into one string.

        `code` events carry `content` (the code). `observation` events
        carry `content` (the output). `file` events carry `file_path` —
        we include the path, not the bytes, since the LLM can't read an
        image. `solution` / `summary` events carry the final prose.
        """
        parts = []
        if getattr(event, "title", None):
            parts.append(f"# {event.title}")
        if getattr(event, "content", None):
            parts.append(event.content)
        if getattr(event, "file_path", None):
            kind = getattr(event, "file_kind", "file") or "file"
            parts.append(f"({kind} produced: {event.file_path})")
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _build_retrieval_query(event, content_truncated: str) -> str:
        """Use the event's first ~400 chars (or title if empty) as the
        retrieval query. No need for a separate query-rewriter LLM call —
        this is good enough for MMR over a small KB."""
        text = content_truncated[:400].strip()
        if not text and getattr(event, "title", None):
            return event.title
        return text

    def _call_llm(
        self,
        event,
        content_truncated: str,
        kb_snippets: list,
        allowed_sources: set[str],
        task: str,
        modes: tuple[str, ...] = ALL_MODES,
    ) -> InstructionCard:
        if self.llm is None:
            return _build_soft_failure_card(event, modes)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_build_system_prompt(modes)),
            HumanMessage(
                content=_build_user_prompt(
                    event, content_truncated, kb_snippets, task, modes
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            print(f"tutor: LLM call failed: {e!r}")
            return _build_soft_failure_card(event, modes)

        data = _extract_json(text)
        if data is None:
            return _build_soft_failure_card(event, modes)

        bloom_fallback = _BLOOM_DEFAULTS.get(event.type, "")
        dok_fallback = _DOK_DEFAULTS.get(event.type, 0)

        sections = _sections_from_response(data, modes)
        card = InstructionCard(
            citations=_coerce_citations(
                data.get("citations"), allowed_sources, _MAX_CITATIONS
            ),
            bloom_target=_coerce_bloom(data.get("bloom_target"), bloom_fallback),
            dok_target=_coerce_dok(data.get("dok_target"), dok_fallback),
            modes=modes,
        )
        for mode, section in sections.items():
            setattr(card, mode, section)

        # Every enabled section came back empty — the model answered, but
        # with nothing usable. That reads to the student exactly like a
        # failed call, so flag it as one and let the renderer say so.
        if card.is_empty():
            return _build_soft_failure_card(event, modes)
        return card


# --- helpers --------------------------------------------------------------


def _content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

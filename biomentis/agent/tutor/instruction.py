"""Per-step instruction cards.

`InstructionGenerator` turns each `UIEvent` from `stream_agent_events` into
a teaching card the student can read alongside the agent's work. The
card has six fields: what / why / prerequisites / look_for / citations /
bloom_target / dok_target.

Pipeline:
  1. Truncate the event's content to a bounded size (the LLM prompt has
     a context budget, and a 50k-char traceback is not useful to teach).
  2. Build a KB retrieval query from the event content (a short snippet
     is enough; we don't need a separate query-rewriter LLM call).
  3. Retrieve up to 4 KB snippets. If the KB is empty, skip retrieval
     and the card will be generated with no citations.
  4. Compose a strict-JSON prompt and call the LLM.
  5. Parse the JSON, validate every field, drop invented citations,
     coerce types where safe.
  6. Cache the result by (event_type, content_hash, kb_signature) so a
     noisy re-emission of the same step doesn't re-bill the LLM.

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


# --- InstructionCard dataclass --------------------------------------------


@dataclass
class InstructionCard:
    """A teaching card attached to one agent event."""

    what: str = ""
    why: str = ""
    prerequisites: list[str] = field(default_factory=list)
    look_for: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # [{source, page, snippet}]
    bloom_target: str = ""  # "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate" | "Create"
    dok_target: int = 0     # 1 | 2 | 3 | 4
    # Internal: True if the LLM call failed and we returned a soft-failure
    # card. The renderer can use this to show a "tutor unavailable" hint.
    _generation_failed: bool = field(default=False, repr=False)


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


def _build_soft_failure_card(event) -> InstructionCard:
    """Return a placeholder card when the LLM call fails. The renderer
    sees `_generation_failed=True` and shows a small note. The user can
    still click Continue and move on."""
    title = getattr(event, "title", None) or event.type
    return InstructionCard(
        what=f"({event.type}) {title}".strip(),
        why="",
        bloom_target=_BLOOM_DEFAULTS.get(event.type, ""),
        dok_target=_DOK_DEFAULTS.get(event.type, 0),
        _generation_failed=True,
    )


# --- Prompt --------------------------------------------------------------


_SYSTEM_PROMPT = """You are an expert tutor for a biomedical research agent. The student just watched the agent do ONE step in a multi-step research task. Produce a teaching card the student can read while the agent continues.

You will be given:
- EVENT_TYPE: one of reasoning, code, observation, solution, summary, file
- EVENT_CONTENT: the event's text (may be truncated for prompt size)
- KB_SNIPPETS: up to 4 relevant passages from a knowledge base the student has uploaded (may be empty)
- TASK: the student's original research task, for context

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):
{
  "what": "<1-2 sentences, plain language, what this step is doing>",
  "why": "<3-6 sentences: the rationale AND the reasoning trail behind it — why this step exists in the workflow, what alternative approaches exist and why this one was chosen, what would go wrong or be missed without it, and how it connects to the step before/after it>",
  "prerequisites": ["<short concept the student should know>", "..."],
  "look_for": ["<what success looks like in the output, e.g. 'a non-empty DataFrame with columns X, Y, Z'>", "..."],
  "citations": [{"source": "<EXACT source string from KB_SNIPPETS>", "page": <int or null>, "snippet": "<≤200 char quote>"}],
  "bloom_target": "<one of: Remember | Understand | Apply | Analyze | Evaluate | Create>",
  "dok_target": <1 | 2 | 3 | 4>
}

RULES:
1. CITATIONS: only cite KB_SNIPPETS you actually saw. If none are relevant to THIS step, return "citations": []. Never invent a source or page.
2. BLOOM/DOK CALIBRATION — be honest, do not inflate:
   - reasoning → usually Understand or Evaluate
   - code → usually Apply (running a procedure) or Analyze (interpreting)
   - observation → usually Analyze
   - solution / summary → usually Create or Evaluate
   Most steps are Apply/Analyze. Reserve Create for genuine synthesis steps.
3. "what" must stand alone — a student who skipped the prior step should still understand it.
4. "why" is the most important field, and this is a learning tool, not a changelog — treat it as a mini worked-explanation, not a one-liner. It must cover: why this step exists in the plan, what would go wrong or be incomplete without it, and (when relevant) what tradeoffs or alternative methods were available. Do not just restate what the code does.
5. "what" stays short (1-2 sentences). "why" gets real room (3-6 sentences) — use it. "prerequisites" and "look_for" are 2-4 short bullets.
6. Output ONLY the JSON. No markdown code fences, no commentary, no apology.

Default Bloom/DOK by event_type (override only if the content clearly supports it):
  reasoning   → Understand, 2
  code        → Apply, 2
  observation → Analyze, 2
  solution    → Create, 3
  summary     → Evaluate, 3
  file        → Apply, 2
"""


def _build_user_prompt(
    event, event_content_truncated: str, kb_snippets: list, task: str
) -> str:
    parts = [
        f"TASK: {task or '(no task provided)'}",
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
    """

    def __init__(self, llm, knowledge_base=None, max_event_chars: int = _MAX_EVENT_CHARS, cache_size: int = 64) -> None:
        self.llm = llm
        self.kb = knowledge_base
        self.max_event_chars = max_event_chars
        self._cache = _LRUCache(maxlen=cache_size)

    # ---- public API -------------------------------------------------------

    def generate(self, event, task: str = "") -> InstructionCard:
        """Generate a teaching card for one `UIEvent`.

        Args:
            event: A `UIEvent` from `stream_agent_events`. Read fields:
                `type`, `content`, `title`, `file_path`, `file_kind`.
            task: The student's original research prompt; passed to the LLM
                for context. Optional.
        """
        content = self._event_text(event)
        content_truncated = _truncate(content, self.max_event_chars)

        # Cache key — same content + same KB → same card.
        kb_sig = ""
        if self.kb is not None:
            try:
                kb_sig = self.kb.kb_signature() or ""
            except Exception:
                kb_sig = ""
        cache_key = (event.type, _content_hash(content_truncated), kb_sig)
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
        card = self._call_llm(event, content_truncated, kb_snippets, allowed_sources, task)

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
    ) -> InstructionCard:
        if self.llm is None:
            return _build_soft_failure_card(event)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_build_user_prompt(event, content_truncated, kb_snippets, task)
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            print(f"tutor: LLM call failed: {e!r}")
            return _build_soft_failure_card(event)

        data = _extract_json(text)
        if data is None:
            return _build_soft_failure_card(event)

        # Drop unknown keys defensively.
        allowed_keys = {
            "what",
            "why",
            "prerequisites",
            "look_for",
            "citations",
            "bloom_target",
            "dok_target",
        }
        data = {k: v for k, v in data.items() if k in allowed_keys}

        bloom_fallback = _BLOOM_DEFAULTS.get(event.type, "")
        dok_fallback = _DOK_DEFAULTS.get(event.type, 0)

        card = InstructionCard(
            what=_truncate(str(data.get("what", "") or ""), 500).strip(),
            why=_truncate(str(data.get("why", "") or ""), 1200).strip(),
            prerequisites=_coerce_str_list(
                data.get("prerequisites"), _MAX_PREREQS
            ),
            look_for=_coerce_str_list(data.get("look_for"), _MAX_LOOK_FOR),
            citations=_coerce_citations(
                data.get("citations"), allowed_sources, _MAX_CITATIONS
            ),
            bloom_target=_coerce_bloom(data.get("bloom_target"), bloom_fallback),
            dok_target=_coerce_dok(data.get("dok_target"), dok_fallback),
        )
        return card


# --- helpers --------------------------------------------------------------


def _content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

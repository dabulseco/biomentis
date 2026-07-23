"""Critic: a second LLM that reviews a finished agent session and emits a
structured `CritiqueCard`.

Why this exists
---------------
Biomentis has no self-improvement loop. The agent is a frozen external LLM
and every session starts from scratch. This module is the "Critic" half of
in-context reward shaping: at the end of each tutor-enabled session, the
Critic reviews the transcript and writes a structured critique that (a)
gets logged as a `kind: "critique"` event in the existing JSONL, and
(b) feeds a per-user memory of `next_session_priorities` that the agent
injects into its system prompt on the next session.

CritiqueCard is also designed to be the future DPO training corpus:
- `WeaknessKind` is an enum so the future export utility can group
  rejection reasons into preference-pair buckets.
- `evidence_quote` is verbatim from the transcript (what DPO needs).
- `overall_score` is a scalar reward signal already.
- `next_session_priorities` is a learned reward function in NL form.
- `rubric_version` + `critic_prompt_hash` let us re-score old sessions
  when the prompt evolves.

Phase A: dataclasses + stub. Phase B (this file): real JSON-mode LLM call.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError


# --- Enums ---------------------------------------------------------------


class WeaknessKind(str, Enum):
    """Taxonomy of agent failure modes the Critic can flag.

    Each value is a *kind of failure*, not a single instance — a single
    critique can list multiple `Weakness(kind=..., ...)` rows. The future
    DPO export utility groups `(prompt, trajectory)` pairs by these
    kinds to build preference datasets.
    """

    SKIPPED_PREREQUISITE = "SKIPPED_PREREQUISITE"
    KB_UNUSED = "KB_UNUSED"
    CLAIM_OVERREACH = "CLAIM_OVERREACH"
    TOOL_MISUSE = "TOOL_MISUSE"
    INCOHERENT_PLAN = "INCOHERENT_PLAN"
    POOR_ERROR_RECOVERY = "POOR_ERROR_RECOVERY"


# --- Critique card -------------------------------------------------------


class Weakness(BaseModel):
    """One concrete failure the Critic points to in the transcript."""

    kind: WeaknessKind
    step_id: int | None = None
    detail: str = ""
    # Verbatim snippet from the agent's response that motivated the flag.
    # Optional but strongly preferred: the future DPO export uses it as
    # the rejection rationale.
    evidence_quote: str | None = None
    # A single short bullet (≤ 140 chars) suggesting what to do instead
    # next session. Mirrors a row of `next_session_priorities`.
    suggested_priority: str | None = None


class Strength(BaseModel):
    """A step the agent did well. Same shape as Weakness minus the
    enum and the priority suggestion — strengths are diagnostic, not
    shaping."""

    step_id: int | None = None
    detail: str = ""


class CritiqueCard(BaseModel):
    """Structured output of one Critic call.

    All fields are designed to be self-contained: a `critique` event in
    the JSONL log should not need joins against other records to be
    useful. This makes the future DPO export a single pass over the log.
    """

    session_id: str
    user_id: str
    model_name: str                       # which LLM produced this critique
    agent_model_name: str                 # which LLM the agent was running on
    overall_score: int = Field(ge=1, le=10)
    weaknesses: list[Weakness] = Field(default_factory=list)
    strengths: list[Strength] = Field(default_factory=list)
    next_session_priorities: list[str] = Field(default_factory=list)
    notes: str = ""

    # DPO-readiness metadata. Versioning lets a future re-scorer
    # re-evaluate old sessions when the critique prompt evolves.
    rubric_version: str = "v1"
    critic_prompt_hash: str = ""

    def to_log_record(self) -> dict:
        """Serialise to a dict ready for `SessionLogger.log({...})`."""
        return self.model_dump()


# --- Prompt --------------------------------------------------------------


# Bump this string when the system prompt changes meaningfully. The
# `critic_prompt_hash` on every card is computed from this string, so
# old records can be filtered for re-scoring.
_RUBRIC_VERSION = "v1"

_SYSTEM_PROMPT = f"""You are an expert reviewer of biomedical research agents. A student just watched an LLM agent (the "agent") complete a multi-step research task. Your job is to write a structured critique that will (a) be shown to the student at the end of the session and (b) be persisted to a long-term memory that shapes the agent's behavior on FUTURE sessions via its system prompt.

You will be given:
- TASK: the student's original research prompt
- TRANSCRIPT_DIGEST: a one-bullet-per-step digest of the agent's actions
- STEP_CARDS: per-step teaching cards (what/why/look-for) the student saw
- KB_STATS: whether a knowledge base was uploaded and how many sources/chunks
- AGENT_MODEL: which LLM the agent was running

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):

{{
  "overall_score": <integer 1-10; 1=very poor, 10=excellent>,
  "weaknesses": [
    {{
      "kind": "<one of: SKIPPED_PREREQUISITE | KB_UNUSED | CLAIM_OVERREACH | TOOL_MISUSE | INCOHERENT_PLAN | POOR_ERROR_RECOVERY>",
      "step_id": <int or null, the step number this critique applies to>,
      "detail": "<1-2 sentences, what went wrong>",
      "evidence_quote": "<verbatim ≤ 200 char quote from the agent's response, or null if you can't find one>",
      "suggested_priority": "<one short bullet ≤ 140 chars for next session, or null>"
    }}
  ],
  "strengths": [
    {{"step_id": <int or null>, "detail": "<1-2 sentences, what went well>"}}
  ],
  "next_session_priorities": [
    "<a short, actionable bullet the agent should follow next time. Be specific.>"
  ],
  "notes": "<optional 1-2 sentence note for the student>"
}}

WEAKNESS KIND TAXONOMY (use exactly these strings):
  SKIPPED_PREREQUISITE: agent executed a step without first explaining a concept the step's teaching card required
  KB_UNUSED:           agent reached for generic LLM knowledge when the KB likely had a more specific answer
  CLAIM_OVERREACH:     agent asserted a conclusion without supporting evidence
  TOOL_MISUSE:         wrong tool, wrong arguments, or right tool in wrong order
  INCOHERENT_PLAN:     a later step contradicts an earlier decision without reconciling
  POOR_ERROR_RECOVERY: agent errored out without adapting the plan

RULES:
1. Be CONSERVATIVE. Only flag weaknesses you can point to a real `evidence_quote` for. If you can't find a quote, set `evidence_quote` to null and the weakness is still acceptable but will be weighted less by future tooling.
2. `evidence_quote` MUST be verbatim from the agent's response in TRANSCRIPT_DIGEST or STEP_CARDS. Do not paraphrase.
3. Use the kind enum EXACTLY (uppercase, underscores). Do not invent kinds.
4. `next_session_priorities` should be 0-7 short bullets, prioritized by impact. Each bullet must be SPECIFIC (e.g. "When BLAST results return, state the E-value threshold being used" — NOT "be more careful").
5. `overall_score` reflects the overall quality of the run for a student learning biomedical research. 1-3 = major problems, 4-6 = partial, 7-8 = good, 9-10 = excellent.
6. Output ONLY the JSON. No markdown code fences, no commentary, no apology.
"""


def _build_user_prompt(
    task: str,
    transcript_summary: str,
    step_cards: list[dict],
    kb_stats: dict | None,
    agent_model_name: str,
) -> str:
    parts: list[str] = []
    parts.append(f"TASK: {task or '(no task provided)'}")
    parts.append(f"AGENT_MODEL: {agent_model_name}")
    if kb_stats:
        parts.append(
            "KB_STATS: "
            f"sources={kb_stats.get('sources', 0)}, "
            f"chunks={kb_stats.get('chunks', 0)}"
        )
    else:
        parts.append("KB_STATS: (no knowledge base uploaded)")

    parts.append("")
    parts.append("TRANSCRIPT_DIGEST (one bullet per step):")
    parts.append("---")
    parts.append(transcript_summary or "(empty)")
    parts.append("---")

    if step_cards:
        parts.append("")
        parts.append("STEP_CARDS (per-step teaching cards the student saw):")
        parts.append("---")
        for card in step_cards:
            sid = card.get("step_id", "?")
            what = card.get("what", "")
            why = card.get("why", "")
            look_for = card.get("look_for") or []
            parts.append(f"[step {sid}] what: {what}")
            if why:
                parts.append(f"          why:  {why}")
            for lf in look_for[:2]:
                parts.append(f"          look_for: {lf}")
        parts.append("---")

    return "\n".join(parts)


# --- Helpers -------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction. Reuses the same pattern as
    `instruction._extract_json` (fenced block, then loose match, then
    plain json.loads). Returns None on any failure."""
    if not text:
        return None
    import json
    import re

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    loose = re.search(r"\{.*\}", text, re.DOTALL)
    if loose:
        try:
            return json.loads(loose.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def _coerce_weaknesses(value: Any) -> list[Weakness]:
    """Coerce a list of dicts into `Weakness` rows, dropping malformed ones.

    `kind` is forced through the `WeaknessKind` enum — unknown values
    drop the whole row (we'd rather have fewer weaknesses than invent
    taxonomy). `evidence_quote` and `suggested_priority` are truncated.
    """
    if not isinstance(value, list):
        return []
    out: list[Weakness] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind_str = item.get("kind")
        try:
            kind = WeaknessKind(kind_str)
        except (ValueError, TypeError):
            continue  # unknown kind → drop
        sid = item.get("step_id")
        try:
            sid = int(sid) if sid is not None else None
        except (TypeError, ValueError):
            sid = None
        eq = item.get("evidence_quote")
        if eq is not None and not isinstance(eq, str):
            eq = str(eq)
        if isinstance(eq, str) and len(eq) > 240:
            eq = eq[:220] + "…"
        sp = item.get("suggested_priority")
        if sp is not None and not isinstance(sp, str):
            sp = str(sp)
        if isinstance(sp, str) and len(sp) > 160:
            sp = sp[:140] + "…"
        out.append(
            Weakness(
                kind=kind,
                step_id=sid,
                detail=str(item.get("detail", "") or "")[:600],
                evidence_quote=eq if eq else None,
                suggested_priority=sp if sp else None,
            )
        )
    return out


def _coerce_strengths(value: Any) -> list[Strength]:
    if not isinstance(value, list):
        return []
    out: list[Strength] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sid = item.get("step_id")
        try:
            sid = int(sid) if sid is not None else None
        except (TypeError, ValueError):
            sid = None
        out.append(
            Strength(
                step_id=sid,
                detail=str(item.get("detail", "") or "")[:600],
            )
        )
    return out


def _coerce_priorities(value: Any, max_items: int = 7) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [v for v in value if isinstance(v, str)]
    else:
        return []
    out: list[str] = []
    for s in items:
        s = s.strip()
        if not s:
            continue
        if len(s) > 200:
            s = s[:180] + "…"
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _coerce_score(value: Any) -> int:
    try:
        v = int(value)
        return max(1, min(10, v))
    except (TypeError, ValueError):
        return 5  # neutral fallback


# --- Soft-failure builder ------------------------------------------------


def _soft_failure_card(
    *,
    session_id: str,
    user_id: str,
    model_name: str,
    agent_model_name: str,
    reason: str,
    rubric_version: str,
    prompt_hash: str,
) -> CritiqueCard:
    """A `CritiqueCard` returned when the LLM call failed or the JSON
    was malformed. The agent run is unaffected; the log will record
    this card so the failure mode is observable."""
    return CritiqueCard(
        session_id=session_id,
        user_id=user_id,
        model_name=model_name,
        agent_model_name=agent_model_name,
        overall_score=5,
        weaknesses=[],
        strengths=[],
        next_session_priorities=[],
        notes=f"(critic LLM unavailable: {reason})",
        rubric_version=rubric_version,
        critic_prompt_hash=prompt_hash,
    )


# --- Critic --------------------------------------------------------------


class Critic:
    """Reviews one finished agent session and returns a `CritiqueCard`.

    Phase A: returns a fixed stub. Phase B: a real JSON-mode LLM call
    that returns a validated `CritiqueCard` (or a soft-failure card
    when the LLM is unavailable or returns malformed JSON).

    The Critic's LLM should be — by design — different from (and
    usually larger than) the agent's. This is the "reward model ≠
    policy" separation in classical RLHF. The Streamlit sidebar picks
    the Critic's model independently.
    """

    def __init__(
        self,
        llm: Any | None = None,
        model_name: str = "stub",
        rubric_version: str = _RUBRIC_VERSION,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.rubric_version = rubric_version
        # Cached hash of the system prompt, so every card can carry
        # the same `critic_prompt_hash` without recomputing per call.
        self._prompt_hash: str = compute_prompt_hash(_SYSTEM_PROMPT)

    # ---- public API -------------------------------------------------------

    def critique(
        self,
        *,
        session_id: str,
        user_id: str,
        agent_model_name: str,
        transcript_summary: str,
        step_cards: list[dict] | None = None,
        kb_stats: dict | None = None,
        task: str = "",
    ) -> CritiqueCard:
        """Return a `CritiqueCard` for one finished session.

        Args:
            session_id: The session being critiqued.
            user_id: The student/user the session is for (memory key).
            agent_model_name: Display name of the LLM the agent used.
            transcript_summary: Pre-built digest of the agent's actions.
                Built by `SessionLogger.summary_for_critic()`.
            step_cards: Per-step teaching cards (dicts), one per
                instruction card the student saw. Optional.
            kb_stats: KBStats-like dict. Optional.
            task: The original student prompt (for context).
        """
        if step_cards is None:
            step_cards = []
        if self.llm is None:
            return _soft_failure_card(
                session_id=session_id,
                user_id=user_id,
                model_name=self.model_name,
                agent_model_name=agent_model_name,
                reason="no LLM configured",
                rubric_version=self.rubric_version,
                prompt_hash=self._prompt_hash,
            )

        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = _build_user_prompt(
            task=task,
            transcript_summary=transcript_summary,
            step_cards=step_cards,
            kb_stats=kb_stats,
            agent_model_name=agent_model_name,
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            return _soft_failure_card(
                session_id=session_id,
                user_id=user_id,
                model_name=self.model_name,
                agent_model_name=agent_model_name,
                reason=f"LLM invoke failed: {e!r}",
                rubric_version=self.rubric_version,
                prompt_hash=self._prompt_hash,
            )

        data = _extract_json(text)
        if data is None:
            return _soft_failure_card(
                session_id=session_id,
                user_id=user_id,
                model_name=self.model_name,
                agent_model_name=agent_model_name,
                reason="LLM returned non-JSON",
                rubric_version=self.rubric_version,
                prompt_hash=self._prompt_hash,
            )

        return self._build_card_from_parsed(
            data=data,
            session_id=session_id,
            user_id=user_id,
            agent_model_name=agent_model_name,
        )

    # ---- internals --------------------------------------------------------

    def _build_card_from_parsed(
        self,
        *,
        data: dict,
        session_id: str,
        user_id: str,
        agent_model_name: str,
    ) -> CritiqueCard:
        """Validate and coerce an LLM-parsed dict into a `CritiqueCard`.

        Tolerant: extra keys are dropped, bad values fall back to
        defaults, unknown `WeaknessKind` rows are dropped silently.
        The only hard requirement is the envelope (session_id etc.) —
        if even that fails, we return a soft-failure card so the engine
        never crashes.
        """
        try:
            return CritiqueCard(
                session_id=session_id,
                user_id=user_id,
                model_name=self.model_name,
                agent_model_name=agent_model_name,
                overall_score=_coerce_score(data.get("overall_score")),
                weaknesses=_coerce_weaknesses(data.get("weaknesses")),
                strengths=_coerce_strengths(data.get("strengths")),
                next_session_priorities=_coerce_priorities(
                    data.get("next_session_priorities")
                ),
                notes=str(data.get("notes", "") or "")[:1000],
                rubric_version=self.rubric_version,
                critic_prompt_hash=self._prompt_hash,
            )
        except ValidationError as e:
            return _soft_failure_card(
                session_id=session_id,
                user_id=user_id,
                model_name=self.model_name,
                agent_model_name=agent_model_name,
                reason=f"pydantic validation: {e!r}",
                rubric_version=self.rubric_version,
                prompt_hash=self._prompt_hash,
            )


# --- Module-level helpers ------------------------------------------------


def compute_prompt_hash(prompt: str) -> str:
    """sha1 of a critique prompt, for the `critic_prompt_hash` field.

    Exposed at module level so re-scorers can decide which old records
    need re-running when the prompt evolves.
    """
    return "sha1:" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

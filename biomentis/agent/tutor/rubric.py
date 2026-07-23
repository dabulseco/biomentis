"""Rubric + Bloom/DOK classifier for the tutor layer.

The `Rubric` class holds a list of teacher-authored objectives (or the
shipped default). `Rubric.classify_qa` is the LLM-driven classifier
that scores a student Q&A against the rubric and returns:

  - bloom_level: one of Remember | Understand | Apply | Analyze | Evaluate | Create
  - dok_level:   1 | 2 | 3 | 4
  - rubric_hit:  list of objective ids the answer satisfies
  - confidence:  0..1, the LLM's self-reported confidence

The classifier uses a strict-JSON prompt and the same JSON-extraction
heuristics as the instruction generator. On parse failure it returns a
"fallback" classification (Understand / DOK 1) with confidence 0 so the
caller can tell apart real classifications from failures.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from biomentis.agent.tutor.instruction import _extract_json


BLOOM_LEVELS = [
    "Remember",   # 1
    "Understand", # 2
    "Apply",      # 3
    "Analyze",    # 4
    "Evaluate",   # 5
    "Create",     # 6
]

DOK_LEVELS = {
    1: "Recall & Reproduction",
    2: "Skills & Concepts",
    3: "Strategic Thinking",
    4: "Extended Thinking",
}

# Coercion tables reused by the classifier.
_BLOOM_ALLOWED = set(BLOOM_LEVELS)
_DOK_ALLOWED = set(DOK_LEVELS.keys())


@dataclass
class RubricObjective:
    id: str
    description: str
    bloom_level: str = "Understand"
    dok_level: int = 2


@dataclass
class RubricClassification:
    """Result of classifying one Q&A against a rubric."""

    bloom_level: str = "Understand"  # Remember..Create
    dok_level: int = 2              # 1..4
    rubric_hit: list[str] = field(default_factory=list)  # objective ids
    confidence: float = 0.0         # 0..1
    raw: dict | None = None         # raw LLM output for debugging
    failed: bool = False            # True if the LLM call or parse failed


# --- Prompt --------------------------------------------------------------


_CLASSIFY_SYSTEM_PROMPT = """You are a careful educational assessor. You will be given:

  - QUESTION: a student's question to a tutor
  - ANSWER: the tutor's response
  - CONTEXT: the running research task and any KB snippets used to ground the answer
  - RUBRIC: a list of learning objectives, each with an id, a Bloom level, and a DOK level

Your job: classify the ANSWER's cognitive level against Bloom's taxonomy and Webb's DOK, and decide which rubric objectives (if any) the answer clearly satisfies.

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):
{
  "bloom_level": "<one of: Remember | Understand | Apply | Analyze | Evaluate | Create>",
  "dok_level": <1 | 2 | 3 | 4>,
  "rubric_hit": ["<objective_id>", "..."],
  "confidence": <float 0..1, your self-reported confidence in this classification>
}

RULES:
1. CALIBRATION — be conservative. Default to Understand/DOK 2 if unsure.
2. The ANSWER's level (not the QUESTION's level) is what you classify. A simple question can elicit a deep answer; a complex question can elicit a shallow answer.
3. RUBRIC_HIT — only include an objective id if the ANSWER clearly and explicitly covers it. If a rubric is empty or none of the objectives fit, return "rubric_hit": [].
4. CONFIDENCE — 0.0 means you have no idea; 1.0 means you're certain. Don't inflate.
5. Output ONLY the JSON. No markdown fences, no commentary, no apology.

Bloom level definitions (use these — don't invent new ones):
  Remember    — recall a fact, term, definition
  Understand  — explain an idea, summarize, paraphrase
  Apply       — use a procedure on a new input, run a known method
  Analyze     — break apart, compare, identify causes, interpret output
  Evaluate    — judge, critique, justify, prioritize
  Create      — synthesize, design, propose a new artifact

DOK level definitions:
  1 — Recall & Reproduction (verbatim recall, simple procedure)
  2 — Skills & Concepts (apply a skill, explain a concept)
  3 — Strategic Thinking (plan, reason, draw conclusions from evidence)
  4 — Extended Thinking (synthesis across domains, novel design)
"""


def _build_classify_user_prompt(
    question: str,
    answer: str,
    context: str,
    rubric_block: str,
) -> str:
    parts = [
        f"QUESTION:\n{question.strip() or '(empty)'}",
        "",
        f"ANSWER:\n{answer.strip() or '(empty)'}",
        "",
        f"CONTEXT:\n{context.strip() or '(no context)'}",
        "",
        f"RUBRIC:\n{rubric_block}",
    ]
    return "\n".join(parts)


# --- Rubric -------------------------------------------------------------


@dataclass
class Rubric:
    objectives: list[RubricObjective] = field(default_factory=list)
    source: str = "default"  # path to the YAML, or "default"

    @classmethod
    def default(cls) -> "Rubric":
        """Load the discipline-agnostic starter rubric shipped with the package."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "default_rubric.yaml")
        if not os.path.exists(path):
            return cls(objectives=[], source="default")
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str) -> "Rubric":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        objs = []
        for item in data.get("objectives", []):
            objs.append(
                RubricObjective(
                    id=str(item.get("id", "")),
                    description=str(item.get("description", "")),
                    bloom_level=str(item.get("bloom_level", "Understand")),
                    dok_level=int(item.get("dok_level", 2)),
                )
            )
        return cls(objectives=objs, source=path)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "objectives": [
                        {
                            "id": o.id,
                            "description": o.description,
                            "bloom_level": o.bloom_level,
                            "dok_level": o.dok_level,
                        }
                        for o in self.objectives
                    ]
                },
                f,
                sort_keys=False,
            )

    def as_prompt_block(self) -> str:
        """Render the rubric as text the LLM classifier can ingest."""
        if not self.objectives:
            return "(no rubric loaded — skip rubric_hit)"
        lines = []
        for o in self.objectives:
            lines.append(
                f"- {o.id} [{o.bloom_level} / DOK{o.dok_level}]: {o.description}"
            )
        return "\n".join(lines)

    # --- classification -------------------------------------------------

    def classify_qa(
        self,
        llm: Any,
        question: str,
        answer: str,
        context: str = "",
    ) -> RubricClassification:
        """Classify a Q&A against this rubric.

        Args:
            llm: A LangChain chat model.
            question: The student's question.
            answer: The tutor's answer.
            context: Optional context (running task, KB snippets, current step).

        Returns:
            A `RubricClassification` with bloom_level, dok_level, rubric_hit,
            confidence, and a `failed` flag. On any error, returns a
            fallback classification with `failed=True`.
        """
        if llm is None:
            return RubricClassification(failed=True)

        # Truncate inputs to bound the prompt.
        question_t = (question or "")[:2000]
        answer_t = (answer or "")[:4000]
        context_t = (context or "")[:3000]

        user_prompt = _build_classify_user_prompt(
            question_t, answer_t, context_t, self.as_prompt_block()
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            print(f"tutor: classify_qa LLM call failed: {e!r}")
            return RubricClassification(failed=True)

        data = _extract_json(text)
        if data is None:
            return RubricClassification(
                bloom_level="Understand",
                dok_level=1,
                confidence=0.0,
                failed=True,
            )

        return _coerce_classification(data, objective_ids=[o.id for o in self.objectives])


# --- helpers --------------------------------------------------------------


def _coerce_classification(
    data: dict, objective_ids: list[str]
) -> RubricClassification:
    """Coerce a parsed JSON dict into a `RubricClassification`.

    Drops unknown keys. Validates bloom/dok against allowed sets.
    Filters `rubric_hit` to known objective ids only.
    """
    allowed_keys = {"bloom_level", "dok_level", "rubric_hit", "confidence"}
    data = {k: v for k, v in data.items() if k in allowed_keys}

    # bloom_level
    bloom = data.get("bloom_level")
    if isinstance(bloom, str):
        for allowed in _BLOOM_ALLOWED:
            if bloom.strip().lower() == allowed.lower():
                bloom = allowed
                break
        else:
            bloom = "Understand"
    else:
        bloom = "Understand"

    # dok_level
    dok = data.get("dok_level")
    try:
        dok = int(dok)
        if dok not in _DOK_ALLOWED:
            dok = 2
    except (TypeError, ValueError):
        dok = 2

    # rubric_hit: list of objective ids
    raw_hit = data.get("rubric_hit") or []
    if not isinstance(raw_hit, list):
        raw_hit = []
    known = set(objective_ids)
    hit = [s for s in raw_hit if isinstance(s, str) and s in known]

    # confidence
    try:
        conf = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return RubricClassification(
        bloom_level=bloom,
        dok_level=dok,
        rubric_hit=hit,
        confidence=conf,
        raw=data,
        failed=False,
    )

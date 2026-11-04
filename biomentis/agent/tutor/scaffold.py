"""Question support: guiding a student to an answer in a few steps.

When a student asks the tutor something worth thinking about, the tutor
doesn't always answer at once. It writes the full answer first and holds it
back, then guides the student toward it one short step ("rung") at a time:
orient them, ask for a prediction, give a hint, work half of it. The answer
is released once the student gets there, asks for it, gives up, or the steps
run out.

This module holds the parts that need no LLM, so each rule can be tested on
its own:

  - `QuestionProfile` and `tier`: what kind of question it is, and how
    sophisticated.
  - `decide`: answer directly, or which steps to take. The policy lives here
    in code, not in a prompt, because a model told to hold an answer back
    gives in or leaks it within a turn or two.
  - `ScaffoldThread`: one guided exchange, with its release rules.
  - `leaked_point` and `fallback_rung`: catching a hint that gives the
    answer away, and what to say instead.

Spec: docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from biomentis.agent.tutor.rubric import BLOOM_LEVELS

LEVELS = ("off", "light", "standard", "challenge")
LEVEL_LABELS = {
    "off": "Answer directly",
    "light": "A nudge first",
    "standard": "Guide me",
    "challenge": "Challenge me",
}
CONTROLS = ("students", "teacher")

QUESTION_TYPES = ("logistics", "definition", "agent_action", "procedural", "conceptual", "evaluative")
MOVES = ("orient", "predict", "hint", "partial", "probe", "contrast", "find_flaw", "extend")

# No thread ever takes more steps than this, whatever the table says.
MAX_RUNGS = 3


def scaffold_level_default() -> str:
    """`BIOMENTIS_SCAFFOLD_LEVEL`, stripped and lowercased; unknown or unset is "standard"."""
    value = (os.environ.get("BIOMENTIS_SCAFFOLD_LEVEL") or "").strip().lower()
    return value if value in LEVELS else "standard"


def scaffold_control_default() -> tuple[str, bool]:
    """(control, locked) from `BIOMENTIS_SCAFFOLD_CONTROL`.

    "teacher" presets teacher control and locks it from the start, for
    classroom machines set up in advance. Anything else is students' choice,
    unlocked.
    """
    value = (os.environ.get("BIOMENTIS_SCAFFOLD_CONTROL") or "").strip().lower()
    return ("teacher", True) if value == "teacher" else ("students", False)


# --- Question profile and tier --------------------------------------------


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


@dataclass
class QuestionProfile:
    """What kind of question it is. Produced by the answer call (spec §7.1)."""

    type: str
    bloom: str = "Understand"
    dok: int = 2
    specific: bool = False
    shows_reasoning: bool = False
    wants_direct: bool = False

    @classmethod
    def parse(cls, value: Any) -> QuestionProfile | None:
        """A profile from the model's JSON, or None when it can't be trusted.

        Only the type is required: without it there is no policy to apply.
        Everything else falls back to a middling default.
        """
        if not isinstance(value, dict):
            return None
        qtype = str(value.get("type", "")).strip().lower()
        if qtype not in QUESTION_TYPES:
            return None
        bloom_in = str(value.get("bloom", "")).strip().lower()
        bloom = next((b for b in BLOOM_LEVELS if b.lower() == bloom_in), "Understand")
        try:
            dok = min(4, max(1, int(value.get("dok", 2))))
        except (TypeError, ValueError):
            dok = 2
        return cls(
            type=qtype,
            bloom=bloom,
            dok=dok,
            specific=_as_bool(value.get("specific", False)),
            shows_reasoning=_as_bool(value.get("shows_reasoning", False)),
            wants_direct=_as_bool(value.get("wants_direct", False)),
        )


_HIGHER_ORDER = {"Analyze", "Evaluate", "Create"}
_TOP_ORDER = {"Evaluate", "Create"}


def tier(p: QuestionProfile) -> str:
    """How sophisticated the question is: novice, developing, advanced or expert.

    "expert" (very sophisticated) takes two of three signals: the student
    shows their own reasoning, the question asks to evaluate or create, or it
    is DOK 3+. Only expert questions get a stretch question (spec §7.2).
    """
    signals = sum((p.shows_reasoning, p.bloom in _TOP_ORDER, p.dok >= 3))
    if signals >= 2:
        return "expert"
    if p.shows_reasoning or p.bloom in _HIGHER_ORDER or p.dok >= 3:
        return "advanced"
    if p.bloom in ("Remember", "Understand") and p.dok <= 2 and not p.specific:
        return "novice"
    return "developing"


# --- Policy ---------------------------------------------------------------


@dataclass(frozen=True)
class ScaffoldPlan:
    """Answer directly, or the steps to take before the answer."""

    direct: bool
    moves: tuple[str, ...] = ()
    stretch: bool = False
    reason: str = ""
    # The student asked for the answer outright under "challenge": they get
    # one step anyway, with "Show me the answer" made prominent.
    highlight_reveal: bool = False


# Spec §7.3. The expert tier uses the advanced column.
_TABLE: dict[tuple[str, str], tuple[str, ...]] = {
    ("light", "novice"): ("orient",),
    ("light", "developing"): ("predict",),
    ("light", "advanced"): ("probe",),
    ("standard", "novice"): ("orient",),
    ("standard", "developing"): ("predict", "hint"),
    ("standard", "advanced"): ("probe", "contrast"),
    ("challenge", "novice"): ("orient", "hint"),
    ("challenge", "developing"): ("predict", "partial"),
    ("challenge", "advanced"): ("probe", "find_flaw", "extend"),
}
_TABLE_LEVELS = {level for level, _ in _TABLE}


def _direct(reason: str) -> ScaffoldPlan:
    return ScaffoldPlan(direct=True, reason=reason)


def decide(level: str, profile: QuestionProfile | None, question_tier: str) -> ScaffoldPlan:
    """Answer directly, or which steps to take first (spec §7.3).

    Friction only where it pays: logistics and "what did the agent do" are
    always answered, definitions are answered except under "challenge", and
    so is an outright "just tell me" below "challenge".
    """
    if level not in _TABLE_LEVELS:
        return _direct("level_off")
    if profile is None:
        return _direct("profile_failed")
    if profile.type in ("logistics", "agent_action"):
        return _direct(f"type_{profile.type}")
    if profile.type == "definition" and level != "challenge":
        return _direct("type_definition")
    if profile.wants_direct and level != "challenge":
        return _direct("wants_direct")

    if profile.type == "definition":
        moves: tuple[str, ...] = ("predict",)
    else:
        column = "advanced" if question_tier == "expert" else question_tier
        moves = _TABLE[(level, column)]
        if profile.type == "procedural":
            moves = moves[: max(1, len(moves) - 1)]
    if profile.wants_direct:
        moves = moves[:1]
    return ScaffoldPlan(
        direct=False,
        moves=moves[:MAX_RUNGS],
        stretch=level == "challenge" and question_tier == "expert",
        reason="table",
        highlight_reveal=profile.wants_direct,
    )



# --- Giving up -------------------------------------------------------------

# Words that don't change what a reply means ("can you just tell me please").
_FILLER = {"just", "please", "can", "could", "you", "ok", "okay", "so", "um", "uh",
           "honestly", "really", "sorry", "well"}
_GIVE_UP_PHRASES = (
    r"(?:tell|give|show) me(?: the answer)?",
    r"idk",
    r"i (?:dont|do not) know",
    r"no (?:idea|clue)",
    r"i give up",
)
GIVE_UP_PATTERNS = (
    re.compile(r"(?:(?:" + "|".join(_GIVE_UP_PHRASES) + r")\s*)+"),
    re.compile(r"\?+"),
)


def is_give_up(text: str) -> bool:
    """True when the whole reply says "just give me the answer".

    The whole reply, not a phrase inside it: "I don't know if the threshold
    matters, but maybe…" is an attempt, and must not end the exchange.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if GIVE_UP_PATTERNS[1].fullmatch(raw):
        return True
    norm = re.sub(r"[^\w\s]", " ", raw.lower().replace("'", "").replace("’", ""))
    words = [w for w in norm.split() if w not in _FILLER]
    return bool(words) and GIVE_UP_PATTERNS[0].fullmatch(" ".join(words)) is not None


# --- One guided exchange ---------------------------------------------------

# Verdicts on a reply that show no attempt; two short ones in a row read as
# frustration (spec §7.5).
_NO_ATTEMPT = {"off_track", "gave_up", "none"}
_SHORT_REPLY_WORDS = 4


@dataclass
class ScaffoldThread:
    """One guided exchange, from question to released answer.

    Stored in Streamlit session state as `to_dict()` output, so a rerun
    mid-exchange re-renders it unchanged and the next reply continues it.

    `answer` holds the held-back answer: the released turn's UI fields plus
    `answer_body`, `follow_up`, `stretch_question` and `key_points`. Each
    rung is {move, message, reply, verdict, leak_retry}; `reply` and
    `verdict` are filled when the student answers that rung.
    """

    question: str
    surface: str
    step_id: int | None
    level: str
    control: str
    profile: dict
    tier: str
    plan: ScaffoldPlan
    answer: dict
    rungs: list[dict] = field(default_factory=list)
    hints_requested: int = 0
    started_at: float = field(default_factory=time.time)
    released: str | None = None
    # The research task, for the rung calls after a rerun.
    task: str = ""

    def next_move(self) -> str | None:
        """The plan's next unused move, or None once the steps are spent."""
        used = len(self.rungs)
        if used >= min(len(self.plan.moves), MAX_RUNGS):
            return None
        return self.plan.moves[used]

    def release_reason_for(self, verdict: str, reply: str) -> str | None:
        """Why the student's reply to the latest rung releases the answer, or None.

        Checked before the reply is recorded, in this order: a new question
        abandons the thread; a correct reply earns the answer; giving up, or
        a second short reply with no attempt in it, releases it out of
        frustration; and when no steps remain, the answer is due.
        """
        if verdict == "new_question":
            return "abandoned"
        if verdict == "correct":
            return "earned"
        if verdict == "gave_up" or is_give_up(reply):
            return "frustration"
        previous = next((r for r in reversed(self.rungs) if r.get("reply")), None)
        if (
            previous is not None
            and verdict in _NO_ATTEMPT
            and previous.get("verdict") in _NO_ATTEMPT
            and len(reply.split()) < _SHORT_REPLY_WORDS
            and len(previous["reply"].split()) < _SHORT_REPLY_WORDS
        ):
            return "frustration"
        if self.next_move() is None:
            return "exhausted"
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plan"]["moves"] = list(self.plan.moves)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ScaffoldThread:
        d = dict(d)
        plan = dict(d.pop("plan"))
        plan["moves"] = tuple(plan.get("moves", ()))
        return cls(plan=ScaffoldPlan(**plan), **d)


def compose_released(thread: ScaffoldThread) -> str:
    """The released answer's text.

    With nothing special about the release, this is exactly the text
    `TutorChat.ask` shows today: the answer, then its follow-up question.
    """
    ans = thread.answer
    body = str(ans.get("answer_body", "") or "")
    stretch = str(ans.get("stretch_question", "") or "").strip()
    follow_up = str(ans.get("follow_up", "") or "").strip()
    if body and thread.plan.stretch and stretch:
        body = f"{body}\n\n**Stretch question:** {stretch}"
    elif body and follow_up:
        body = f"{body}\n\n**Follow-up:** {follow_up}"
    if thread.released == "earned":
        body = f"✓ That's the core of it.\n\n{body}"
    return body


# --- Leak check and fallbacks ---------------------------------------------

# A key point shorter than this ("noise", "E-value") would match almost any
# hint, so it isn't checked.
_MIN_CHECK_WORDS = 3
_MIN_CHECK_CHARS = 15


def _normalize(text: str) -> str:
    """Lowercase, markdown emphasis and punctuation removed, whitespace collapsed."""
    text = re.sub(r"[*_`]", "", (text or "").lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def leaked_point(message: str, key_points: list[str]) -> str | None:
    """The first key point a hint states outright, or None.

    A plain string check: it catches the blatant leaks that make friction
    pointless and rarely fires on a near-miss phrasing. Matches whole words
    only, so "unexpected" doesn't leak "expected".
    """
    haystack = f" {_normalize(message)} "
    for point in key_points or []:
        norm = _normalize(point)
        if len(norm.split()) < _MIN_CHECK_WORDS and len(norm) < _MIN_CHECK_CHARS:
            continue
        if f" {norm} " in haystack:
            return norm
    return None


# Used when the rung call keeps leaking the answer: a fixed step for the move
# that can't give anything away.
FALLBACK_RUNGS = {
    "orient": "Look back at {where}. What does it tell you about your question?",
    "predict": "Before I explain: what do you expect the answer to be, and why?",
    "hint": "Think about which concept from {where} applies here. Which one would you reach for?",
    "partial": "Let's take it in two halves. Start with the first: what happens at {where}, "
    "and what does that set up?",
    "probe": "What are you assuming in your question? Does that assumption hold here?",
    "contrast": "Compare this case with the opposite one. What would change, and why?",
    "find_flaw": "Suppose someone gave a quick answer that ignored {where}. What would they get wrong?",
    "extend": "How would your answer change if the conditions at {where} were different?",
}


def fallback_rung(move: str, step_id: int | None) -> str:
    """The fixed step for `move`, pointing at the step the question is about."""
    where = f"step {step_id}" if step_id is not None else "the agent's last step"
    return FALLBACK_RUNGS.get(move, FALLBACK_RUNGS["orient"]).format(where=where)


# --- The rung call ----------------------------------------------------------

# Spec §7.4. Shown to the model with the move it is asked to make.
MOVE_DEFINITIONS = {
    "orient": "Point to where the answer lives (the step, value or knowledge-base source) and ask what it suggests.",
    "predict": "Ask the student to predict an outcome before you reveal anything.",
    "hint": "Name the concept that matters, without applying it.",
    "partial": "Work through the first half and ask the student to finish.",
    "probe": "Question an assumption in the student's question or reasoning.",
    "contrast": "Offer two cases and ask what differs between them.",
    "find_flaw": "Show a subtly wrong version of the reasoning and ask what is wrong with it.",
    "extend": "Ask how the answer would change under a different condition.",
}
VERDICTS = ("correct", "partial", "off_track", "gave_up", "new_question", "none")

_RUNG_PROMPT = """You are a patient tutor in a biomedical research workflow. A student asked a question. You already know the full answer (HELD_BACK_ANSWER), but instead of giving it, you are guiding the student toward it one short step at a time so they think it through themselves.

You will be given:
  - QUESTION: the student's question
  - TASK: the research task the agent is working on
  - RECENT_STEPS: what the agent has done recently (may be empty)
  - HELD_BACK_ANSWER and KEY_POINTS: the answer and the conclusions the student should reach. Never state these, or anything that gives them away.
  - MOVE: the kind of step to take now, with its definition
  - RUNG: which step this is, out of how many at most
  - THREAD: the earlier steps and the student's replies (may be empty)
  - LATEST_REPLY: the student's reply to your last step (empty on the first step)

Return a single JSON object with EXACTLY these keys (no prose, no markdown fences):
{"verdict": "<correct | partial | off_track | gave_up | new_question | none>", "message": "<your next step>"}

RULES:
1. VERDICT judges LATEST_REPLY against the KEY_POINTS: "correct" if it reaches them in substance (wording doesn't matter), "partial" if it is on the way, "off_track" if it is not, "gave_up" if the student is asking to be told or is plainly frustrated, "new_question" if it is a different question rather than an answer. Use "none" when LATEST_REPLY is empty.
2. MESSAGE makes the MOVE. Give the student exactly ONE thing to do: predict, look at a specific step or value, finish a sentence, or compare two cases.
3. If the reply is partly right, say what is right first, then build on it. Never answer with a bare "no" or "wrong".
4. Never state the HELD_BACK_ANSWER or any KEY_POINT, and don't hint so heavily that the student only has to copy it.
5. Keep it to 2-4 sentences. Be warm and direct, not effusive.
6. Output ONLY the JSON.
"""


class RungFailed(Exception):
    """The rung call failed twice; the caller releases the answer (fail open)."""


def _format_cards(cards: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cards or [], 1):
        bits = [f"{k}={c[k]}" for k in ("event_type", "what", "why", "impact") if c.get(k)]
        if bits:
            lines.append(f"step {i}: " + " | ".join(bits))
    return "\n".join(lines) or "(none)"


def _rung_user_prompt(thread: ScaffoldThread, move: str, task: str, cards: list[dict], latest_reply: str) -> str:
    ans = thread.answer
    history = []
    for r in thread.rungs:
        history.append(f"TUTOR ({r.get('move')}): {r.get('message', '')}")
        if r.get("reply"):
            history.append(f"STUDENT: {r['reply']}")
    return "\n\n".join(
        [
            f"QUESTION:\n{thread.question}",
            f"TASK:\n{task or '(none)'}",
            f"RECENT_STEPS:\n{_format_cards(cards)}",
            f"HELD_BACK_ANSWER (never state it):\n{ans.get('answer_body', '')}",
            "KEY_POINTS (never state them):\n" + ("\n".join(f"- {p}" for p in ans.get("key_points") or []) or "(none)"),
            f"MOVE: {move}\n{MOVE_DEFINITIONS.get(move, '')}",
            f"RUNG: {len(thread.rungs) + 1} of {min(len(thread.plan.moves), MAX_RUNGS)}",
            "THREAD:\n" + ("\n".join(history) or "(none)"),
            f"LATEST_REPLY:\n{latest_reply or '(empty)'}",
        ]
    )


def _parse_rung(text: str) -> tuple[str, str] | None:
    from biomentis.agent.tutor.instruction import _extract_json

    data = _extract_json(text or "")
    if not isinstance(data, dict):
        return None
    message = str(data.get("message", "") or "").strip()
    if not message:
        return None
    verdict = str(data.get("verdict", "") or "").strip().lower()
    return (verdict if verdict in VERDICTS else "none"), message


def next_rung(
    llm, thread: ScaffoldThread, move: str, *, task: str = "", cards: list[dict] | None = None,
    latest_reply: str = "",
) -> tuple[str, str, bool]:
    """The next guided step: (verdict on the latest reply, message, leak_retry).

    An invalid reply (a failed call, no JSON, no message) is retried once;
    two in a row raise `RungFailed`. A message that states a key point is
    regenerated once with what it gave away; if that leaks too, or fails,
    the move's fixed fallback text is used instead.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_RUNG_PROMPT),
        HumanMessage(content=_rung_user_prompt(thread, move, task, cards or [], latest_reply)),
    ]

    def call(msgs) -> tuple[str, tuple[str, str] | None]:
        try:
            text = getattr(llm.invoke(msgs), "content", "") or ""
        except Exception as e:
            print(f"tutor: rung call failed: {e!r}")
            return "", None
        return text, _parse_rung(text)

    text, parsed = call(messages)
    if parsed is None:
        text, parsed = call(messages)
    if parsed is None:
        raise RungFailed(f"no usable rung for move {move!r}")
    verdict, message = parsed
    if not latest_reply:
        verdict = "none"

    key_points = thread.answer.get("key_points") or []
    point = leaked_point(message, key_points)
    if point is None:
        return verdict, message, False
    retry = messages + [
        AIMessage(content=text),
        HumanMessage(
            content=f'Your last draft gave away a key point: "{point}". Rewrite the message so the '
            "student has to reach it themselves. Return the same JSON."
        ),
    ]
    _, again = call(retry)
    if again is not None and leaked_point(again[1], key_points) is None:
        return verdict, again[1], True
    return verdict, fallback_rung(move, thread.step_id), True

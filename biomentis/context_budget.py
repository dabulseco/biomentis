"""Keep a run's request inside the model's context window.

A long research task dies the same way every time: the window fills, and the
first sign of trouble is a 400 from the provider —

    The prompt is too long: 262960, model maximum context length: 262144

— which takes the whole run with it. The arithmetic behind that number is not
subtle. The system prompt is ~42k tokens before a word is exchanged (32k of it
the descriptions of all 224 tools, because the Streamlit app runs with the
retriever off), and every execute step appends an observation capped at 10,000
characters, roughly 2.5k tokens, that never comes back out. Call it 3-4k per
step: about 55-70 steps of runway, with `recursion_limit` set to 500. Nothing
was watching, so nothing spoke up.

This module does the watching. `fit_messages` trims a request to fit before it
is sent — oldest observations first, since those are the bulk and the least
useful once the agent has acted on them — and reports when the run is close
enough to the ceiling that the agent should stop working and write its answer.
A run that ends with a report is worth immeasurably more than one that ends
with a 400.

Two things it deliberately does not do:

  * It never mutates the run's own message history. Only the outgoing request
    is trimmed, so the transcript and the journal stay complete and the trim
    is recomputed from scratch on the next call.
  * It never guesses a limit it does not have. An unknown model gets no guard
    rather than a made-up ceiling that truncates a run early.

Where the limit comes from, in order:

  1. `BIOMNI_MAX_CONTEXT_TOKENS` — an explicit override always wins
  2. a limit learned from a previous context-length error, which is the
     provider's own number and therefore the most trustworthy source there is
  3. `KNOWN_CONTEXT_LIMITS`, matched on the model name

Ollama's `/api/show` is deliberately NOT consulted: for `deepseek-v4-pro:cloud`
it reports `context_length: 1048576` while the endpoint enforces 262144, so
trusting it would disable the guard exactly where it is needed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ContextFit",
    "HARD_STOP_AFTER_WRAP_UPS",
    "KNOWN_CONTEXT_LIMITS",
    "WRAP_UP_INSTRUCTION",
    "estimate_messages_tokens",
    "estimate_tokens",
    "fit_messages",
    "is_context_length_error",
    "learn_limit_from_error",
    "resolve_context_limit",
]

# Model-name substring -> context window in tokens. Substrings, because model
# names carry tags (`deepseek-v4-pro:cloud`) and deployment prefixes.
KNOWN_CONTEXT_LIMITS: dict[str, int] = {
    # Measured from the endpoint's own error, not from `/api/show`, which
    # reports the architecture maximum (1048576) rather than what is enforced.
    "deepseek-v4": 262_144,
    "claude": 200_000,
    "gpt-5": 400_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gemini-2.5": 1_048_576,
    "llama-3.3": 128_000,
    "llama3": 128_000,
    "qwen": 32_768,
}

# Room left for the model's own response. A request that exactly fills the
# window leaves nowhere to answer from.
DEFAULT_RESERVE_TOKENS = 8_192

# Token counts here are estimates from a different tokenizer than the model's,
# so the limit is discounted rather than trusted to the last token.
SAFETY_FRACTION = 0.95

# How full the trimmed request has to be before the agent is told to stop
# working and write its answer. Below this there is still real runway.
WRAP_UP_FRACTION = 0.9

# Never trim these: the system prompt, the task, and the recent exchange the
# model is actually reasoning about.
_KEEP_RECENT = 4

# How many times the agent may be told to finish and keep running code anyway
# before the run is ended for it. Two nudges is a fair chance; past that, each
# further step only makes it less likely the answer gets written at all.
HARD_STOP_AFTER_WRAP_UPS = 3

WRAP_UP_INSTRUCTION = (
    "You are nearly out of context for this task. Do NOT run any more code. "
    "Write your final answer now, in <solution></solution> tags, from what you "
    "have already established. Say plainly which parts of the original request "
    "you were unable to complete and what you would do next."
)

_OBSERVATION_RE = re.compile(r"^\s*<observation>", re.IGNORECASE)

# "model maximum context length: 262144", "maximum context length is 8192 tokens"
_LIMIT_IN_ERROR_RE = re.compile(r"max(?:imum)?\s+context\s+length\s*(?:is)?[:\s]\s*(\d{3,})", re.IGNORECASE)

_CONTEXT_ERROR_MARKERS = (
    "prompt is too long",
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "reduce the length of the messages",
)

_learned_limits: dict[str, int] = {}


# ----- token estimation ---------------------------------------------------


def _encoder():
    """A tokenizer to estimate with, or None to fall back to characters."""
    global _ENCODER
    try:
        return _ENCODER
    except NameError:
        pass
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string.

    Approximate by construction: the model's own tokenizer is usually not
    available locally, and cl100k is close enough for a budget that is already
    discounted by `SAFETY_FRACTION`.
    """
    if not text:
        return 0
    encoder = _encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return len(text) // 4


def _content_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("content") or ""
                if isinstance(part, str):
                    parts.append(part)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def estimate_messages_tokens(messages: list[Any]) -> int:
    """Estimate the token count of a message list, with per-message overhead."""
    # ~4 tokens of role/delimiter framing per message, the usual rule of thumb.
    return sum(estimate_tokens(_content_of(m)) + 4 for m in messages)


# ----- limits -------------------------------------------------------------


def model_name(llm: Any) -> str:
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _limits_path() -> Path:
    from biomentis.run_journal import default_run_dir

    return Path(default_run_dir()) / "context_limits.json"


def _load_learned() -> dict[str, int]:
    if _learned_limits:
        return _learned_limits
    try:
        with open(_limits_path(), encoding="utf-8") as handle:
            for name, limit in (json.load(handle) or {}).items():
                if isinstance(limit, int):
                    _learned_limits[name] = limit
    except Exception:
        pass
    return _learned_limits


def _save_learned() -> None:
    try:
        path = _limits_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_learned_limits, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"[context_budget] could not persist learned limit: {exc}")


def resolve_context_limit(llm: Any) -> int | None:
    """The model's usable context window, or None if it isn't known.

    None disables the guard. That is the right default: a guessed ceiling
    that is too low cuts a run short for no reason, and one that is too high
    protects nothing.
    """
    override = os.getenv("BIOMNI_MAX_CONTEXT_TOKENS")
    if override:
        try:
            return int(override)
        except ValueError:
            print(f"[context_budget] ignoring non-numeric BIOMNI_MAX_CONTEXT_TOKENS={override!r}")

    name = model_name(llm).lower()
    if not name:
        return None

    learned = _load_learned().get(name)
    if learned:
        return learned

    for fragment, limit in KNOWN_CONTEXT_LIMITS.items():
        if fragment in name:
            return limit
    return None


def is_context_length_error(exc: BaseException) -> bool:
    """Whether an exception is the provider refusing an over-long prompt.

    Worth telling apart from a transient failure: it is deterministic, so
    retrying it just sends the same doomed request again.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


def learn_limit_from_error(llm: Any, exc: BaseException) -> int | None:
    """Record the limit the provider named in its error, for next time.

    This is the most reliable source available — the server's own number, for
    the exact deployment being used. Learning it means a given model can only
    surprise us once.
    """
    match = _LIMIT_IN_ERROR_RE.search(str(exc))
    if not match:
        return None
    limit = int(match.group(1))
    name = model_name(llm).lower()
    if not name:
        return limit
    if _load_learned().get(name) != limit:
        _learned_limits[name] = limit
        _save_learned()
        print(f"[context_budget] learned context limit for {name}: {limit:,} tokens")
    return limit


# ----- fitting ------------------------------------------------------------


@dataclass
class ContextFit:
    """What `fit_messages` did, and what the caller should do about it."""

    messages: list[Any]
    limit: int
    budget: int
    tokens_before: int
    tokens_after: int
    trimmed_observations: int = 0
    dropped_messages: int = 0
    truncated_messages: int = 0
    must_wrap_up: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.trimmed_observations or self.dropped_messages or self.truncated_messages)

    def must_hard_stop_after(self, wrap_ups: int) -> bool:
        """Whether the run should be ended rather than nudged again.

        Only meaningful while `must_wrap_up` holds: an agent that has been
        asked to write its answer this many times and is still calling for
        more code is not going to stop by itself.
        """
        return self.must_wrap_up and wrap_ups >= HARD_STOP_AFTER_WRAP_UPS

    def summary(self) -> str:
        parts = [f"{self.tokens_before:,} -> {self.tokens_after:,} tokens (budget {self.budget:,})"]
        if self.trimmed_observations:
            parts.append(f"{self.trimmed_observations} observation(s) collapsed")
        if self.dropped_messages:
            parts.append(f"{self.dropped_messages} message(s) dropped")
        if self.truncated_messages:
            parts.append(f"{self.truncated_messages} message(s) truncated")
        if self.must_wrap_up:
            parts.append("asking the agent to finish")
        return "; ".join(parts)


def _placeholder(content: str) -> str:
    return (
        f"<observation>[earlier output dropped to stay within the context window — "
        f"{len(content):,} characters. Re-run the step if you still need it.]</observation>"
    )


def fit_messages(
    messages: list[Any],
    limit: int | None,
    reserve: int = DEFAULT_RESERVE_TOKENS,
) -> ContextFit:
    """Trim a request so it fits, oldest and bulkiest first.

    The order is chosen so the run degrades in the way that costs least: an
    observation the agent has already reasoned about is the first thing to go,
    the exchange it is reasoning about right now is the last.

    Never touches `messages` itself — a new list is returned, so the run's own
    history, the transcript, and the journal all stay complete.
    """
    tokens = estimate_messages_tokens(messages)
    if limit is None:
        return ContextFit(list(messages), 0, 0, tokens, tokens)

    budget = max(1, int(limit * SAFETY_FRACTION) - reserve)
    fit = ContextFit(list(messages), limit, budget, tokens, tokens)
    if tokens <= budget:
        fit.must_wrap_up = tokens >= budget * WRAP_UP_FRACTION
        if fit.must_wrap_up:
            fit.notes.append("within budget but close to it")
        return fit

    # Indices that may be modified: not the system prompt, not the first human
    # message (the task), not the last few turns.
    first_human = next(
        (i for i, m in enumerate(fit.messages) if type(m).__name__ == "HumanMessage"),
        None,
    )
    protected = {0, first_human} | set(range(max(0, len(fit.messages) - _KEEP_RECENT), len(fit.messages)))
    candidates = [i for i in range(len(fit.messages)) if i not in protected]

    # Pass 1: collapse observations, oldest first.
    for index in candidates:
        if fit.tokens_after <= budget:
            break
        content = _content_of(fit.messages[index])
        if not _OBSERVATION_RE.match(content):
            continue
        fit.messages[index] = type(fit.messages[index])(content=_placeholder(content))
        fit.trimmed_observations += 1
        fit.tokens_after = estimate_messages_tokens(fit.messages)

    # Pass 2: drop whole messages, oldest first. Only reached when collapsing
    # every observation was not enough.
    if fit.tokens_after > budget:
        slots: list[Any | None] = list(fit.messages)
        for index in candidates:
            if fit.tokens_after <= budget:
                break
            if slots[index] is None:
                continue
            slots[index] = None
            fit.dropped_messages += 1
            fit.tokens_after = estimate_messages_tokens([m for m in slots if m is not None])
        fit.messages = [m for m in slots if m is not None]

    # Pass 3: last resort. A single message can be larger than the whole
    # budget — one 400k-character observation, say — and no amount of dropping
    # other messages will help. Cut the largest by however much is still over,
    # repeatedly, so the request always ends up inside the limit.
    #
    # Cutting by a fixed fraction is not enough here: it can shrink a message
    # and still leave the request over budget, which is exactly the situation
    # this pass exists to rule out.
    attempts = 0
    while fit.tokens_after > budget and attempts < 20:
        attempts += 1
        # The system prompt is the contract the loop parses against; cutting
        # it costs more than any observation ever could.
        index = max(
            range(1, len(fit.messages)),
            key=lambda i: estimate_tokens(_content_of(fit.messages[i])),
            default=None,
        )
        if index is None:
            break
        content = _content_of(fit.messages[index])
        if estimate_tokens(content) < 200:
            break  # nothing left that is worth cutting
        over_by = fit.tokens_after - budget
        # ~4 characters per token, plus slack so one pass usually suffices.
        cut_chars = min(len(content) - 200, int(over_by * 4 * 1.2) + 200)
        if cut_chars <= 0:
            break
        keep_chars = max(200, len(content) - cut_chars)
        fit.messages[index] = type(fit.messages[index])(
            content=content[:keep_chars] + "\n\n[…truncated to fit the context window…]"
        )
        fit.truncated_messages += 1
        fit.tokens_after = estimate_messages_tokens(fit.messages)

    fit.must_wrap_up = fit.dropped_messages > 0 or fit.tokens_after >= budget * WRAP_UP_FRACTION
    if fit.changed:
        fit.notes.append(fit.summary())
    return fit

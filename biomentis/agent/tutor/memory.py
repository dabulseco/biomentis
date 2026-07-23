"""Per-user rolling memory of Critic priorities.

Stores a small JSON file at `data/tutor_memory/<user_id>.json` containing:

  {
    "user_id": "default",
    "n_sessions": 12,
    "last_updated": "2026-07-22T...",
    "priorities": [
      "When BLAST results come back, always state the E-value threshold and why.",
      "Prefer KB definitions over generic LLM knowledge for domain terms."
    ],
    "weakness_counts": {"KB_UNUSED": 14, "CLAIM_OVERREACH": 3},
    "rolling_window": 10,
    "priority_cap": 12
  }

The `priorities` list is what the agent's system prompt gets at the start
of the next session. The `weakness_counts` are a per-user diagnostic that
shows which failure modes recur — useful both for the sidebar view and
for the future DPO export (a user with lots of `KB_UNUSED` critiques is
exactly the cohort you'd want to A/B test a fine-tune on).

Phase A: only `load()`, `save()`, and a `path_for()` helper are wired.
Phase C: `update()` lands here — rolling-window cap, case-insensitive
dedup, counter accumulation.

Storage is intentionally tiny (≤ a few KB per user). The file is written
atomically (tmp + rename) so a crash mid-write can't corrupt the JSON.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from typing import Any

from biomentis.agent.tutor.critic import CritiqueCard, WeaknessKind


# Default cap on the priorities list. `update()` enforces this; the
# agent's system prompt size is bounded by this many bullets.
DEFAULT_PRIORITY_CAP = 12
# Rolling window for the n_sessions counter. The priorities list is its
# own implicit rolling window (new ones go to the front, old ones drop
# when the cap is hit). This field is a hint to UI / future analytics.
DEFAULT_ROLLING_WINDOW = 10
# Cap on the dedup key length, to bound the trim() cost on pathological
# inputs. (200 chars per priority is already huge in practice.)
_MAX_PRIORITY_KEY_LEN = 200


# --- Dedup key normalization ---------------------------------------------


def _priority_key(s: str) -> str:
    """A stable, case-insensitive, whitespace-normalized key for dedup.

    Two priorities that differ only in capitalization, leading/trailing
    whitespace, or a single period are treated as the same. This is
    deliberately tolerant because LLMs emit many paraphrases of the
    same advice.
    """
    if not s:
        return ""
    s = s.strip().lower()
    # Drop trailing punctuation that's often inconsistent.
    s = s.rstrip(".!?")
    # Collapse runs of whitespace.
    s = re.sub(r"\s+", " ", s)
    if len(s) > _MAX_PRIORITY_KEY_LEN:
        s = s[:_MAX_PRIORITY_KEY_LEN]
    return s


# --- update() -------------------------------------------------------------


def update(
    user_id: str,
    card: CritiqueCard,
    root: str = "./data/tutor_memory",
) -> dict[str, Any]:
    """Fold one `CritiqueCard` into the user's memory.

    Returns the updated memory dict (also persisted to disk). Steps:

    1. Build the candidate priority list for this card:
       - The card's `next_session_priorities` (in order).
       - For each `Weakness` with a non-empty `suggested_priority`, that
         priority too. These are catch-alls for cases where the LLM
         produced a `Weakness` but didn't promote the fix into the
         top-level priorities list.
    2. Prepend the candidates to the existing priorities, deduping
       case-insensitively. New priorities go to the front so the
       agent sees the most recent lesson first.
    3. Cap at `priority_cap` (default 12) by truncating from the tail.
    4. Bump `weakness_counts` by one for each `Weakness.kind` in the
       card. The `Strength` rows don't increment any counter.
    5. Bump `n_sessions` and update `last_updated`.
    6. Persist atomically.

    The function is idempotent on the same `card` (calling it twice
    with the same card adds each priority only once and bumps the
    counter by 2 instead of 1 per weakness, but doesn't break anything).
    For the future, we could hash the card to dedup exactly, but for
    Phase C this is sufficient.
    """
    data = load(user_id, root=root)
    cap = int(data.get("priority_cap", DEFAULT_PRIORITY_CAP))

    # --- 1. Build candidates from the card ----------------------------
    candidates: list[str] = []
    for p in card.next_session_priorities or []:
        if isinstance(p, str) and p.strip():
            candidates.append(p.strip())
    for w in card.weaknesses or []:
        sp = getattr(w, "suggested_priority", None)
        if isinstance(sp, str) and sp.strip():
            candidates.append(sp.strip())

    # --- 2. Dedup + prepend -----------------------------------------
    existing = list(data.get("priorities", []) or [])
    existing_keys = {_priority_key(p) for p in existing}
    merged: list[str] = []
    for c in candidates:
        k = _priority_key(c)
        if not k or k in existing_keys:
            continue
        merged.append(c)
        existing_keys.add(k)
    # New priorities go to the front. Cap at `priority_cap` by
    # truncating the tail — i.e. the oldest existing priorities drop
    # off when the cap is hit.
    new_priorities = (merged + existing)[: max(0, cap)]
    data["priorities"] = new_priorities

    # --- 3. Bump weakness counts ------------------------------------
    counts = dict(data.get("weakness_counts", {}) or {})
    for w in card.weaknesses or []:
        kind_value = w.kind.value if hasattr(w.kind, "value") else str(w.kind)
        counts[kind_value] = int(counts.get(kind_value, 0)) + 1
    data["weakness_counts"] = counts

    # --- 4. Bump session counter ------------------------------------
    data["n_sessions"] = int(data.get("n_sessions", 0)) + 1
    data["last_updated"] = datetime.now().isoformat(timespec="seconds")

    # --- 5. Persist -------------------------------------------------
    save(user_id, data, root=root)
    return data


# --- load / save / reset / path_for --------------------------------------


def path_for(user_id: str, root: str = "./data/tutor_memory") -> str:
    """Return the absolute path of the memory file for one user."""
    return os.path.abspath(os.path.join(root, f"{user_id}.json"))


def load(user_id: str, root: str = "./data/tutor_memory") -> dict[str, Any]:
    """Load the memory for a user. Returns an empty default if missing.

    The returned dict is a fresh copy — callers may mutate it freely
    before passing it back to `save()`.
    """
    path = path_for(user_id, root)
    if not os.path.exists(path):
        return _empty_memory(user_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted or unreadable — treat as empty rather than crashing
        # the agent run.
        return _empty_memory(user_id)
    # Defensive: if the file predates a schema field, fill in defaults.
    defaults = _empty_memory(user_id)
    for key, value in defaults.items():
        data.setdefault(key, value)
    return data


def save(user_id: str, data: dict[str, Any], root: str = "./data/tutor_memory") -> str:
    """Persist the memory dict for a user. Returns the file path written.

    Atomic write: write to a temp file in the same directory, then rename.
    This avoids the partial-write failure mode if the process is killed
    mid-flush.
    """
    os.makedirs(root, exist_ok=True)
    target = path_for(user_id, root)
    data.setdefault("user_id", user_id)
    data.setdefault("last_updated", datetime.now().isoformat(timespec="seconds"))
    fd, tmp = tempfile.mkstemp(prefix=f".{user_id}.", suffix=".json.tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, target)
    except Exception:
        # Clean up the temp file on failure so we don't leak it.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def reset(user_id: str, root: str = "./data/tutor_memory") -> None:
    """Wipe the memory for a user. The next `load()` returns empty defaults."""
    path = path_for(user_id, root)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _empty_memory(user_id: str) -> dict[str, Any]:
    """The shape of a fresh memory record. `save()` will fill in
    `last_updated` on write."""
    return {
        "user_id": user_id,
        "n_sessions": 0,
        "last_updated": None,
        "priorities": [],
        "weakness_counts": {},
        "rolling_window": DEFAULT_ROLLING_WINDOW,
        "priority_cap": DEFAULT_PRIORITY_CAP,
    }

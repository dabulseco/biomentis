"""Append-only JSONL session logger.

One file per session: `data/tutor_logs/<session_id>/<session_id>.jsonl`.
Each line is a self-contained JSON record. Five record kinds are written:

  {"kind": "step", "event_type": "code", "step_id": 7, "bloom_target": "Apply",
   "dok_target": 2, "instruction": {...}, "kb_citations": [...]}

  {"kind": "qa", "question": "...", "answer": "...", "bloom_level": "Analyze",
   "dok_level": 3, "rubric_hit": ["OBJ2"], "confidence": 0.82}

  {"kind": "kb", "action": "add", "source": "...", "chunks": 41}

  {"kind": "critique", "overall_score": 7, "weaknesses": [...],
   "strengths": [...], "next_session_priorities": [...],
   "rubric_version": "v1", "critic_prompt_hash": "sha1:..."}

  {"kind": "critique_error", "error": "..."}   # when the Critic LLM failed

The logger is safe to call from Streamlit's rerun model: appends are flushed
after every write, so a crash mid-session loses at most one record.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class SessionLogger:
    def __init__(self, session_id: str, path: str = "./data/tutor_logs") -> None:
        self.session_id = session_id
        self.dir = os.path.abspath(os.path.join(path, session_id))
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{session_id}.jsonl")

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        record.setdefault("session_id", self.session_id)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_json(self) -> str:
        """Return the entire log as a single JSON array (pretty-printed)."""
        with open(self.path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return json.dumps(records, indent=2, ensure_ascii=False)

    def export_csv(self) -> tuple[str, str]:
        """Return (steps_csv, qa_csv). One row per step / Q&A."""
        import csv
        import io

        with open(self.path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        steps_buf = io.StringIO()
        steps_writer = csv.writer(steps_buf)
        steps_writer.writerow(
            ["ts", "step_id", "event_type", "bloom_target", "dok_target", "what", "why"]
        )
        for r in records:
            if r.get("kind") != "step":
                continue
            instr = r.get("instruction") or {}
            steps_writer.writerow(
                [
                    r.get("ts", ""),
                    r.get("step_id", ""),
                    r.get("event_type", ""),
                    r.get("bloom_target", ""),
                    r.get("dok_target", ""),
                    instr.get("what", ""),
                    instr.get("why", ""),
                ]
            )

        qa_buf = io.StringIO()
        qa_writer = csv.writer(qa_buf)
        qa_writer.writerow(
            ["ts", "question", "answer", "bloom_level", "dok_level", "rubric_hit", "confidence"]
        )
        for r in records:
            if r.get("kind") != "qa":
                continue
            qa_writer.writerow(
                [
                    r.get("ts", ""),
                    r.get("question", ""),
                    r.get("answer", ""),
                    r.get("bloom_level", ""),
                    r.get("dok_level", ""),
                    ";".join(r.get("rubric_hit", []) or []),
                    r.get("confidence", ""),
                ]
            )
        return steps_buf.getvalue(), qa_buf.getvalue()

    # --- Critic digests --------------------------------------------------

    # Cap on the digest size fed to the Critic. ~8k chars ≈ 2k tokens,
    # which is the design target in the plan.
    _CRITIC_DIGEST_MAX_CHARS = 8000
    # Cap on per-step content in the digest.
    _CRITIC_PER_STEP_CHARS = 280

    def summary_for_critic(self) -> str:
        """Build a one-bullet-per-step digest of this session for the Critic.

        Walks the JSONL, picks out `step` records, and emits:

          [step 1] reasoning: <first 280 chars of content>
          [step 2] code:        <first 280 chars of content>
          ...
          [step N] solution:   <first 280 chars of content>
          [final]   <the final <solution>'s text or the last summary's text>

        Capped at `_CRITIC_DIGEST_MAX_CHARS` so the Critic's input budget
        is bounded. The `step_cards` argument to the Critic (separate)
        carries the more detailed per-step teaching cards; this digest
        is the agent's *own* actions, not the tutor's commentary on them.
        """
        try:
            records = self._read_all()
        except Exception:
            return ""

        steps: list[str] = []
        final_text: str = ""
        for r in records:
            kind = r.get("kind")
            if kind == "step":
                sid = r.get("step_id", "?")
                etype = r.get("event_type", "?")
                instr = r.get("instruction") or {}
                content = (instr.get("what") or "") + " | " + (instr.get("why") or "")
                # The "what"/"why" fields are tutor-generated; the agent's
                # own step content isn't currently in the log. We use the
                # tutor card as a stand-in for "what the agent did at
                # this step" — close enough for the Critic's purpose.
                content = content.strip(" |") or "(no instruction card)"
                content = content[: self._CRITIC_PER_STEP_CHARS]
                steps.append(f"[step {sid}] {etype}: {content}")
            elif kind == "qa":
                q = (r.get("question") or "").strip()
                a = (r.get("answer") or "").strip()
                if q or a:
                    steps.append(
                        f"[qa]  Q: {q[:160]}"
                    )
                    steps.append(
                        f"      A: {a[:240]}"
                    )
            elif kind == "critique":
                # Don't feed the previous critique into the next digest —
                # the Critic is meant to score the *agent's* actions,
                # not critique the previous critique.
                continue
            elif kind in ("critique_error", "kb"):
                # Skip; not relevant to the agent's behavior.
                continue

        body = "\n".join(steps)
        if final_text:
            body += "\n[final] " + final_text[: self._CRITIC_PER_STEP_CHARS]
        if len(body) > self._CRITIC_DIGEST_MAX_CHARS:
            body = (
                body[: self._CRITIC_DIGEST_MAX_CHARS - 80]
                + "\n[…truncated for prompt size…]"
            )
        return body

    def step_cards_for_critic(self) -> list[dict]:
        """Return the list of per-step `InstructionCard` payloads from this
        session. Each entry is `{step_id, event_type, what, why, look_for}`.

        Phase B: this is the *tutor's* teaching cards, which is the
        closest signal to "what the agent actually did" we have in the
        log. Phase C+ could record the raw agent output too; the digest
        shape would then be richer.
        """
        try:
            records = self._read_all()
        except Exception:
            return []
        out: list[dict] = []
        for r in records:
            if r.get("kind") != "step":
                continue
            instr = r.get("instruction") or {}
            out.append(
                {
                    "step_id": r.get("step_id"),
                    "event_type": r.get("event_type"),
                    "what": instr.get("what", ""),
                    "why": instr.get("why", ""),
                    "look_for": instr.get("look_for", []) or [],
                }
            )
        return out

    def _read_all(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

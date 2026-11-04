"""TutorEngine: orchestrator that ties KB, instruction gen, chat, log together.

The engine is *off* by default — the Streamlit sidebar flips it on after a
KB is uploaded. When off, `launch_streamlit_demo` is unaffected and the
agent runs exactly as it did before this layer existed.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from biomentis.agent.tutor import memory as critic_memory
from biomentis.agent.tutor import scaffold
from biomentis.agent.tutor.chat import ChatTurn, TutorChat
from biomentis.agent.tutor.critic import Critic, CritiqueCard
from biomentis.agent.tutor.instruction import ALL_MODES, InstructionGenerator, normalize_modes
from biomentis.agent.tutor.kb import KnowledgeBase
from biomentis.agent.tutor.log import SessionLogger
from biomentis.agent.tutor.rubric import Rubric


def turn_ui_dict(turn: ChatTurn) -> dict:
    """A tutor answer as the chat panel stores and renders it."""
    return {
        "role": "assistant",
        "content": turn.content,
        "citations": list(turn.citations),
        "bloom_level": turn.bloom_level,
        "dok_level": turn.dok_level,
        "rubric_hit": list(turn.rubric_hit),
        "confidence": turn.confidence,
        "failed": turn.failed,
        "also_consider": list(turn.also_consider),
        "plot_code": turn.plot_code,
        "plot_png": turn.plot_png,
        "plot_error": turn.plot_error,
        "structure_path": turn.structure_path,
        "plot_mode": turn.plot_mode,
    }


@dataclass
class ScaffoldStep:
    """Where a guided exchange stands after one action.

    `released_turn` is None while the thread is open, and the answer (as
    `turn_ui_dict`) once it is released. `reask` is the student's new
    question when they moved on mid-exchange; the caller handles it next.
    """

    thread: scaffold.ScaffoldThread
    released_turn: dict | None = None
    reask: str | None = None


class TutorEngine:
    """Per-session orchestrator for the instructional layer.

    Args:
        session_id: Unique id for this learning session.
        llm: A LangChain chat model — the same one the agent uses.
        path: Root for KB and log storage. Defaults to `./data`.
        critic_model_name: Display name of the LLM that will run the Critic.
            Phase A stores this on the engine; the actual LLM call is wired
            in Phase B. The plan calls for using a *different* (usually
            larger) model for the Critic than for the agent, mirroring the
            "reward model ≠ policy" separation in classical RLHF.
    """

    def __init__(
        self,
        session_id: str,
        llm=None,
        path: str = "./data",
        critic_model_name: str = "stub",
        critic_llm: Any | None = None,
        memory_root: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.llm = llm
        self.enabled: bool = False
        # Which instructional lenses the student has switched on. The
        # master `enabled` flag gates the step-by-step walkthrough (cards
        # + pause gates); `modes` gates WHAT each card teaches. Both on by
        # default, but either can be turned off independently — including
        # both, which leaves the walkthrough pacing and per-step Q&A in
        # place while generating no teaching text (and spending no tokens
        # on it).
        self.modes: tuple[str, ...] = ALL_MODES
        # Per-engine Critic instance. Phase A's Critic returns a fixed card;
        # Phase B's Critic uses a real LLM. By default the Critic shares the
        # agent's LLM; the Streamlit sidebar can pass a different
        # (usually larger) model via `critic_llm` for the "reward model ≠
        # policy" separation in classical RLHF.
        self.critic = Critic(llm=critic_llm if critic_llm is not None else llm,
                              model_name=critic_model_name)
        # Per-user memory is keyed by user_id, not session_id, so priorities
        # carry across sessions. The Streamlit side passes the user_id in.
        # Default mirrors the data path: <path>/tutor_memory. Tests can
        # override with `memory_root=...` to use a single temp dir.
        self.memory_root: str = (
            memory_root if memory_root is not None else os.path.join(path, "tutor_memory")
        )

        self.kb = KnowledgeBase(
            session_id,
            path=os.path.join(path, "tutor_kb"),
        )
        self.rubric: Rubric = Rubric.default()
        self.logger = SessionLogger(
            session_id,
            path=os.path.join(path, "tutor_logs"),
        )
        self.instruction_gen = InstructionGenerator(
            llm, knowledge_base=self.kb, modes=self.modes
        )
        # Chat is constructed last so it can hold references to the rubric
        # and logger (for Bloom/DOK classification and Q&A logging).
        self.chat = TutorChat(
            llm,
            knowledge_base=self.kb,
            rubric=self.rubric,
            logger=self.logger,
        )
        self.current_prompt: str = ""
        # The most recent priorities the engine has been asked to inject
        # into the agent's system prompt. Surfaced in the Streamlit sidebar
        # so the user can see what the agent "knows" about past sessions.
        self.active_priorities: list[str] = []

        # Snapshot of the most recently completed run, kept on the
        # engine so the chat panel can answer follow-up questions
        # about a task even after the live session_state buffer is
        # gone (page reload, fresh tab, etc.). The chat handler in
        # ui_tutor.py falls back to these when `_RUN_KEY` is empty.
        self.last_run_events: list = []
        self.last_run_prompt: str = ""
        self.last_run_answer: str = ""

        # Question support (scaffold.py): how much the chat guides before it
        # answers, and who decides. Teacher control locks at the first
        # question, or from the start when preset from the environment.
        self.scaffold_level: str = scaffold.scaffold_level_default()
        self.scaffold_control, self.scaffold_locked = scaffold.scaffold_control_default()
        self.scaffold_questions_asked: int = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    # --- instructional modes ---------------------------------------------

    def set_modes(self, modes) -> None:
        """Set which instructional lenses the cards teach.

        Called by the sidebar whenever a mode toggle flips. Kept separate
        from `enable()/disable()` because the two answer different
        questions: `enabled` is "walk me through this run step by step",
        `modes` is "and explain each step technically / scientifically /
        both / neither".
        """
        self.modes = normalize_modes(modes)
        self.instruction_gen.set_modes(self.modes)

    def modes_enabled(self) -> bool:
        """True when at least one instructional lens is on, i.e. cards
        will actually carry teaching content."""
        return bool(self.modes)

    # --- question support --------------------------------------------------

    def set_scaffold(self, level: str | None = None, control: str | None = None) -> bool:
        """Change the question-support level or who controls it.

        Returns False, changing nothing, once teacher control has locked.
        Unknown values are ignored; each effective change is logged.
        """
        if self.scaffold_locked:
            return False
        changed: dict[str, str] = {}
        if level in scaffold.LEVELS and level != self.scaffold_level:
            self.scaffold_level = changed["level"] = level
        if control in scaffold.CONTROLS and control != self.scaffold_control:
            self.scaffold_control = changed["control"] = control
        if changed:
            self._log({"kind": "scaffold_setting", **changed})
        self._maybe_lock_scaffold()
        return True

    def _maybe_lock_scaffold(self) -> None:
        if self.scaffold_control == "teacher" and self.scaffold_questions_asked > 0:
            self.scaffold_locked = True

    def start_question(
        self, question: str, *, surface: str, step_id: int | None = None, ask_kwargs: dict | None = None
    ) -> ScaffoldStep:
        """Answer a student's question, or open a guided exchange toward it.

        The full answer is written first in either case; when the policy
        guides, it is held back on the thread and the first step returned.
        """
        self.scaffold_questions_asked += 1
        self._maybe_lock_scaffold()
        ask_kwargs = dict(ask_kwargs or {})
        level = self.scaffold_level
        turn = self.chat.ask(question, step_id=step_id, scaffold_level=level, record=False, **ask_kwargs)

        profile = scaffold.QuestionProfile.parse(turn.question_profile) if turn.question_profile else None
        question_tier = scaffold.tier(profile) if profile is not None else ""
        plan = scaffold.decide(level, profile, question_tier)
        thread = scaffold.ScaffoldThread(
            question=question,
            surface=surface,
            step_id=step_id,
            level=level,
            control=self.scaffold_control,
            profile=turn.question_profile,
            tier=question_tier,
            plan=plan,
            answer=asdict(turn),
            task=ask_kwargs.get("task", "") or "",
        )
        if plan.direct or turn.recorded:
            return self._release(thread, "direct")
        try:
            verdict, message, leak_retry = self._next_rung(thread, plan.moves[0], "")
        except scaffold.RungFailed:
            return self._release(thread, "error")
        thread.rungs.append(_rung(plan.moves[0], message, leak_retry))
        return ScaffoldStep(thread=thread)

    def reply(self, thread, text: str) -> ScaffoldStep:
        """The student's reply to the latest step: the next step, or the answer.

        After the last step there is no move left, but the reply is still
        judged (with the last move), so a correct one is "earned".
        """
        thread = _as_thread(thread)
        move = thread.next_move()
        try:
            verdict, message, leak_retry = self._next_rung(thread, move or thread.plan.moves[-1], text)
        except scaffold.RungFailed:
            thread.rungs[-1].update(reply=text, verdict="none")
            return self._release(thread, "error")
        reason = thread.release_reason_for(verdict, text)
        if reason == "abandoned":
            return self._release(thread, reason, reask=text)
        thread.rungs[-1].update(reply=text, verdict=verdict)
        if reason:
            return self._release(thread, reason)
        thread.rungs.append(_rung(move, message, leak_retry))
        return ScaffoldStep(thread=thread)

    def hint(self, thread) -> ScaffoldStep:
        """The next step without a reply; the answer once no steps remain."""
        thread = _as_thread(thread)
        thread.hints_requested += 1
        move = thread.next_move()
        if move is None:
            return self._release(thread, "exhausted")
        try:
            _, message, leak_retry = self._next_rung(thread, move, "")
        except scaffold.RungFailed:
            return self._release(thread, "error")
        thread.rungs.append(_rung(move, message, leak_retry))
        return ScaffoldStep(thread=thread)

    def reveal(self, thread) -> ScaffoldStep:
        """"Show me the answer"."""
        return self._release(_as_thread(thread), "button")

    def _next_rung(self, thread, move: str, latest_reply: str):
        return scaffold.next_rung(
            self.chat.llm,
            thread,
            move,
            task=thread.task,
            cards=self.chat._recent_cards[-3:],
            latest_reply=latest_reply,
        )

    def _release(self, thread, reason: str, reask: str | None = None) -> ScaffoldStep:
        """Release the held-back answer: compose it, record it, log the thread.

        At level "off" the Q&A is recorded exactly as before question
        support existed: no thread in the record, no scaffold record.
        """
        thread.released = reason
        off = thread.level == "off"
        turn = ChatTurn(**thread.answer)
        if not turn.recorded:  # a soft failure was recorded by `ask` already
            turn.content = scaffold.compose_released(thread) or turn.content
            self.chat.record(thread.question, turn, thread.step_id, None if off else thread)
        if not off:
            self._log(
                {
                    "kind": "scaffold",
                    "surface": thread.surface,
                    "step_id": thread.step_id,
                    "question": thread.question,
                    "level": thread.level,
                    "control": thread.control,
                    "profile": thread.profile,
                    "tier": thread.tier,
                    "plan": list(thread.plan.moves),
                    "plan_reason": thread.plan.reason,
                    "rungs": thread.rungs,
                    "released": reason,
                    "hints_requested": thread.hints_requested,
                    "seconds_to_release": round(time.time() - thread.started_at, 1),
                }
            )
        return ScaffoldStep(thread=thread, released_turn=turn_ui_dict(turn), reask=reask)

    def _log(self, record: dict) -> None:
        try:
            self.logger.log(record)
        except Exception as e:  # logging must never break the chat
            print(f"tutor: failed to log {record.get('kind')}: {e!r}")

    def record_run(
        self,
        events: list,
        prompt: str = "",
        last_answer: str = "",
    ) -> None:
        """Snapshot a finished run onto the engine.

        Called by `tutor_wrapped_stream` when the run reaches
        `phase="done"`. The chat panel's `_handle_chat_turn` reads
        these attributes as a fallback when the live session_state
        run buffer has been cleared (e.g. by a Streamlit rerun that
        didn't take the same code path).

        The events list is stored as-is (UIEvent objects, not
        plain dicts) so the chat handler can reuse the same
        `getattr(ev, "type", ...) / getattr(ev, "content", ...)`
        accessors it uses on the live run.
        """
        self.last_run_events = list(events or [])
        self.last_run_prompt = prompt or ""
        self.last_run_answer = last_answer or ""
        # Keep `current_prompt` in sync so callers that read it
        # without consulting the run buffer still get the latest.
        if self.last_run_prompt:
            self.current_prompt = self.last_run_prompt

    def set_llm(self, llm) -> None:
        """Called when the sidebar model picker changes — re-thread the
        agent's LLM into the components that need it. The KB, logger,
        memory, and Critic do NOT auto-update: the Critic should be a
        separate (usually larger) model per the plan, and the sidebar
        wires it independently via `set_critic_llm()`."""
        self.llm = llm
        self.instruction_gen.llm = llm
        self.chat.llm = llm

    def set_critic_llm(self, llm) -> None:
        """Called when the sidebar 'Critic model' picker changes.

        Distinct from `set_llm()` so changing the agent model doesn't
        silently swap out the Critic. The Critic is meant to be a
        different (usually larger) model — the "reward model ≠ policy"
        separation in classical RLHF.
        """
        self.critic.llm = llm

    def set_rubric(self, rubric: Rubric) -> None:
        """Called when a teacher uploads a new rubric YAML."""
        self.rubric = rubric
        self.chat.set_rubric(rubric)

    def set_critic_model_name(self, name: str) -> None:
        """Called when the sidebar 'Critic model' picker changes."""
        self.critic.model_name = name

    # --- Critic / memory integration -------------------------------------

    def load_priorities(self, user_id: str) -> list[str]:
        """Read the current priority list for a user. Returns [] if none.

        Called by `streamlit_app.py` at session start, before
        `agent.configure(critic_priorities=...)`.
        """
        data = critic_memory.load(user_id, root=self.memory_root)
        priorities = data.get("priorities", []) or []
        self.active_priorities = list(priorities)
        return list(priorities)

    def on_session_end(
        self,
        user_id: str,
        agent_model_name: str,
        transcript_summary: str | None = None,
        step_cards: list[dict] | None = None,
        task: str = "",
    ) -> CritiqueCard | None:
        """Run the Critic over the just-finished session and persist the
        result.

        If `transcript_summary` is None, the engine asks the SessionLogger
        to build one (preferred — it reads the JSONL that's already on
        disk). If `step_cards` is None, same.

        Returns the `CritiqueCard` so the UI can show "lessons learned"
        inline. Returns `None` if the engine is disabled — there's
        nothing to critique without a transcript.
        """
        if not self.enabled:
            return None
        if step_cards is None:
            try:
                step_cards = self.logger.step_cards_for_critic()
            except Exception:
                step_cards = []
        if transcript_summary is None:
            try:
                transcript_summary = self.logger.summary_for_critic()
            except Exception:
                transcript_summary = ""

        kb_stats: dict[str, Any] = {}
        try:
            kb_stats = self.kb.stats().__dict__
        except Exception:
            # Stats can be expensive; never let it block session end.
            kb_stats = {}

        card = self.critic.critique(
            session_id=self.session_id,
            user_id=user_id,
            agent_model_name=agent_model_name,
            transcript_summary=transcript_summary,
            step_cards=step_cards,
            kb_stats=kb_stats,
            task=task or self.current_prompt,
        )
        # Write the critique event to the JSONL log. Failures here must
        # never propagate — the agent run is over and the log is best-effort.
        try:
            record = card.to_log_record()
            record["kind"] = "critique"
            self.logger.log(record)
        except Exception as e:  # pragma: no cover — defensive
            try:
                self.logger.log(
                    {
                        "kind": "critique_error",
                        "error": repr(e),
                        "user_id": user_id,
                    }
                )
            except Exception:
                pass

        # Phase C: fold the critique into the user's memory so the next
        # session's system prompt carries `next_session_priorities`.
        # Memory update failures must NOT block the agent run; the next
        # session will just have stale priorities.
        try:
            critic_memory.update(user_id, card, root=self.memory_root)
        except Exception as e:  # pragma: no cover — defensive
            try:
                self.logger.log(
                    {
                        "kind": "critique_error",
                        "stage": "memory_update",
                        "error": repr(e),
                        "user_id": user_id,
                    }
                )
            except Exception:
                pass

        return card


def _rung(move: str, message: str, leak_retry: bool) -> dict:
    return {"move": move, "message": message, "reply": "", "verdict": "none", "leak_retry": leak_retry}


def _as_thread(thread) -> scaffold.ScaffoldThread:
    """Threads come back from session state as plain dicts."""
    return thread if isinstance(thread, scaffold.ScaffoldThread) else scaffold.ScaffoldThread.from_dict(thread)

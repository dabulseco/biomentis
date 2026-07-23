"""TutorEngine: orchestrator that ties KB, instruction gen, chat, log together.

The engine is *off* by default — the Streamlit sidebar flips it on after a
KB is uploaded. When off, `launch_streamlit_demo` is unaffected and the
agent runs exactly as it did before this layer existed.
"""

from __future__ import annotations

import os
from typing import Any

from biomentis.agent.tutor import memory as critic_memory
from biomentis.agent.tutor.chat import TutorChat
from biomentis.agent.tutor.critic import Critic, CritiqueCard
from biomentis.agent.tutor.instruction import InstructionGenerator
from biomentis.agent.tutor.kb import KnowledgeBase
from biomentis.agent.tutor.log import SessionLogger
from biomentis.agent.tutor.rubric import Rubric


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
        self.instruction_gen = InstructionGenerator(llm, knowledge_base=self.kb)
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

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

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

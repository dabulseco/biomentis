"""Tutor chatbot.

`TutorChat.ask` is the LLM-driven chat path used by the Streamlit tutor
panel. The call sequence:

  1. Build a query from the user's question (no rewriting — the question
     is usually short and well-formed).
  2. If a KnowledgeBase is attached and non-empty, retrieve up to 4
     snippets. They are passed to the LLM as a numbered list and the
     LLM is told to cite only those sources.
  3. Compose a strict-JSON prompt that includes the running task, the
     last 3 instruction cards (so the tutor has continuity), the KB
     snippets, and the chat history.
  4. Call the LLM. Parse the JSON response (`{answer, follow_up,
     citations}`). Coerce types defensively.
  5. Drop any citation whose `source` isn't in the retrieved set
     (no hallucinations).
  6. Run `Rubric.classify_qa` to score the Q&A against Bloom/DOK
     and the teacher rubric. Falls back gracefully on LLM error.
  7. Log the Q&A to the `SessionLogger` (if one is attached) so the
     teacher has a record.
  8. Append the turn to `self.history` and return.

Failures at any step produce a soft-failure `ChatTurn` (a short apology
plus the question echoed back) so the UI doesn't crash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from biomentis.agent.tutor.instruction import _extract_json


_MAX_HISTORY_TURNS = 6          # user/assistant pairs to keep in the prompt
_MAX_KB_SNIPPETS = 4
_MAX_KB_SNIPPET_CHARS = 600
_MAX_CARDS_IN_CONTEXT = 3
_MAX_CITATION_SNIPPET = 200
_MAX_QUESTION_CHARS = 1000

# Two behavioral modes for the chat:
#   "chat" — KB + LLM hybrid. Use the KB as the primary source, but the
#            LLM is allowed to fill gaps with general knowledge. Citations
#            are required for KB-sourced content; LLM-only content has no
#            citation. This is the user-facing "Ask about this task" panel.
#   "tutor" — KB-only. The LLM must only answer from KB_SNIPPETS. This is
#            the strict-pedagogy mode for the per-step instruction
#            pipeline, where instructor material is the basis for the
#            answer. Default for backwards compatibility.
_ASK_MODE_CHAT = "chat"
_ASK_MODE_TUTOR = "tutor"
_VALID_ASK_MODES = (_ASK_MODE_CHAT, _ASK_MODE_TUTOR)


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict] = field(default_factory=list)
    bloom_level: str = ""  # filled in by classify_qa after the answer
    dok_level: int = 0
    rubric_hit: list[str] = field(default_factory=list)
    confidence: float = 0.0
    failed: bool = False  # True if the LLM call or parse failed


# --- Prompt --------------------------------------------------------------


_SYSTEM_PROMPT = """You are a patient tutor for a biomedical research agent. The student is running a multi-step task and asks you questions about it.

You will be given:
  - QUESTION: the student's current question
  - TASK: the original research task, for context
  - RECENT_STEPS: up to 3 teaching cards from the steps the agent has done so far (may be empty)
  - KB_SNIPPETS: up to 4 relevant passages from the student's uploaded knowledge base (may be empty)
  - HISTORY: the last few turns of this chat (may be empty)

Your job: answer the question in a way that helps the student learn. Be brief (2-4 short paragraphs or a short bulleted list). Always end with ONE follow-up question to check understanding.

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):
{
  "answer": "<your answer, plain text or simple markdown>",
  "follow_up": "<one follow-up question, or empty string if none>",
  "citations": [{"source": "<EXACT source string from KB_SNIPPETS>", "page": <int or null>, "snippet": "<≤200 char quote>"}]
}

RULES:
1. CITATIONS — only cite KB_SNIPPETS you actually saw. If none are relevant to THIS question, return "citations": []. Never invent a source.
2. If the KB doesn't address the question, say so plainly. Don't invent content from prior knowledge; be honest about what the KB does and doesn't say.
3. Reference RECENT_STEPS when the question is about a step the agent just did ("you just ran BLAST — the result was…").
4. HISTORY is provided for continuity. Don't repeat yourself.
5. Keep the answer short. The student is in the middle of a workflow, not reading an essay.
6. Output ONLY the JSON. No markdown fences, no commentary, no apology.
"""


# Chat mode: KB + LLM hybrid. The LLM is allowed to fill gaps with
# general knowledge, but citations are reserved for KB-sourced content.
_SYSTEM_PROMPT_CHAT = """You are a helpful assistant for a biomedical research workflow. The student is running a multi-step task and asks you questions about it. You have access to their uploaded knowledge base (KB_SNIPPETS) and to your own general knowledge.

You will be given:
  - QUESTION: the student's current question
  - TASK: the original research task, for context
  - RECENT_STEPS: up to 3 teaching cards from the steps the agent has done so far (may be empty)
  - RECENT_TRANSCRIPT: the last few events the agent produced (raw reasoning, code, observation output, summary). Present after a run finishes, so you can answer "what did the agent do?" questions. May be empty.
  - AGENT_FINAL_ANSWER: the agent's final solution to the task. Present after a run finishes, so you can answer "what was the result?" or "summarize the agent's answer." May be empty.
  - KB_SNIPPETS: up to 4 relevant passages from the student's uploaded knowledge base (may be empty)
  - HISTORY: the last few turns of this chat (may be empty)

Your job: answer the question accurately and concisely. Be brief (2-4 short paragraphs or a short bulleted list). When the KB covers the question, lead with KB content. When the KB only partially covers it (or is empty), use your general knowledge to fill gaps — but be clear about which is which. Always end with ONE follow-up question.

Return a single JSON object with EXACTLY these keys (no other keys, no prose, no markdown fences):
{
  "answer": "<your answer, plain text or simple markdown>",
  "follow_up": "<one follow-up question, or empty string if none>",
  "citations": [{"source": "<EXACT source string from KB_SNIPPETS>", "page": <int or null>, "snippet": "<≤200 char quote>"}]
}

RULES:
1. CITATIONS — cite KB_SNIPPETS for any content that came from the KB. Do not cite content that came from your general knowledge (no citation). It is fine for "citations" to be an empty array if your answer used no KB content.
2. KB FIRST — when KB_SNIPPETS address the question, prefer that content over your general knowledge. If the KB partially answers, lead with the KB portion, then add general knowledge to fill the gap, and cite only the KB portion.
3. NEVER INVENT CITATIONS — every citation's "source" must be the EXACT string from a KB_SNIPPETS entry. Never fabricate a source, page, or snippet.
4. If both the KB and your knowledge are insufficient, say so plainly.
5. Reference RECENT_STEPS when the question is about a step the agent just did ("you just ran BLAST — the result was…").
6. Reference RECENT_TRANSCRIPT and AGENT_FINAL_ANSWER when the student asks about a finished run ("what databases did you use?", "summarize the result"). These are the agent's actual output, so you can answer confidently.
7. HISTORY is provided for continuity. Don't repeat yourself.
8. Keep the answer short. The student is in the middle of a workflow, not reading an essay.
9. Output ONLY the JSON. No markdown fences, no commentary, no apology.
"""


def _build_user_prompt(
    question: str,
    task: str,
    cards: list[dict],
    transcript: str = "",
    last_answer: str = "",
    *,
    kb_snippets: list[dict],
    history: list[ChatTurn],
) -> str:
    parts = [f"QUESTION:\n{question.strip() or '(empty)'}"]

    if task:
        parts.append("")
        parts.append(f"TASK:\n{task.strip()}")

    if cards:
        parts.append("")
        parts.append("RECENT_STEPS (most recent last):")
        for i, c in enumerate(cards, 1):
            bits = []
            if c.get("event_type"):
                bits.append(f"[{c['event_type']}]")
            if c.get("what"):
                bits.append(f"what={c['what']}")
            if c.get("why"):
                bits.append(f"why={c['why']}")
            parts.append(f"{i}. " + " | ".join(bits))

    if transcript:
        parts.append("")
        parts.append("RECENT_TRANSCRIPT (last 5 events the agent produced, most recent last):")
        parts.append(transcript)

    if last_answer:
        parts.append("")
        parts.append("AGENT_FINAL_ANSWER (the agent's final solution to the task):")
        parts.append(last_answer)

    if kb_snippets:
        parts.append("")
        parts.append("KB_SNIPPETS (cite only these, or return citations: []):")
        for i, snip in enumerate(kb_snippets, 1):
            parts.append(
                f"[{i}] source={snip['source']!r} page={snip['page']!r}"
            )
            parts.append(f"    {snip['content']}")
    else:
        parts.append("")
        parts.append("KB_SNIPPETS: (none — the knowledge base is empty)")

    if history:
        parts.append("")
        parts.append("HISTORY (most recent last):")
        # Only keep the last N turns to bound the prompt.
        recent = history[-(_MAX_HISTORY_TURNS * 2):]
        for turn in recent:
            role = "STUDENT" if turn.role == "user" else "TUTOR"
            parts.append(f"{role}: {turn.content}")

    return "\n".join(parts)


# --- TutorChat -----------------------------------------------------------


class TutorChat:
    """Stateful, KB-grounded tutor chatbot.

    Args:
        llm: A LangChain chat model. Same calling convention as the rest
            of Biomentis.
        knowledge_base: An optional `KnowledgeBase`.
        rubric: An optional `Rubric` for Q&A classification. If None,
            a default Rubric is used.
        logger: An optional `SessionLogger`. When set, every Q&A is
            logged to JSONL.
    """

    def __init__(
        self,
        llm,
        knowledge_base=None,
        rubric=None,
        logger=None,
    ) -> None:
        self.llm = llm
        self.kb = knowledge_base
        from biomentis.agent.tutor.rubric import Rubric
        self.rubric = rubric if rubric is not None else Rubric.default()
        self.logger = logger
        self.history: list[ChatTurn] = []
        self._recent_cards: list[dict] = []  # populated by the wrapper

    def set_rubric(self, rubric) -> None:
        self.rubric = rubric

    def set_logger(self, logger) -> None:
        self.logger = logger

    def push_recent_card(self, card_dict: dict) -> None:
        """Called by the wrapper each time a new step card is generated.
        Keeps a short rolling list for chat continuity."""
        self._recent_cards.append(card_dict)
        if len(self._recent_cards) > _MAX_CARDS_IN_CONTEXT * 2:
            self._recent_cards = self._recent_cards[-(_MAX_CARDS_IN_CONTEXT * 2):]

    def reset(self) -> None:
        self.history = []
        self._recent_cards = []

    # --- public API -------------------------------------------------------

    def ask(
        self,
        question: str,
        context: str = "",
        task: str = "",
        transcript: str = "",
        last_answer: str = "",
        mode: str = _ASK_MODE_TUTOR,
    ) -> ChatTurn:
        """Ask the tutor a question.

        Args:
            question: The student's question.
            context: Free-form context (e.g. the current step's title or a
                hint about what's on screen).
            task: The running research task, for grounding.
            transcript: Recent agent-run events (raw reasoning, code,
                observation output, summary) so the LLM can answer
                post-run follow-up questions like "what databases did
                you use?" without re-running the task. Only consumed in
                `chat` mode.
            last_answer: The agent's final solution to the task, used
                when the student asks "summarize the agent's answer" or
                "what was the result?". Only consumed in `chat` mode.
            mode: One of "chat" (KB + LLM hybrid — LLM is allowed to fill
                gaps with general knowledge; citations are reserved for KB
                content) or "tutor" (KB-only — strict grounding; the LLM
                must only answer from KB_SNIPPETS). The "tutor" mode is
                the default for backwards compatibility.

        Returns:
            A `ChatTurn` with the answer, citations, and (if classifiable)
            Bloom/DOK/rubric_hit labels.
        """
        if mode not in _VALID_ASK_MODES:
            # Defensive: treat unknown modes as the safer KB-only path.
            mode = _ASK_MODE_TUTOR
        question = (question or "")[:_MAX_QUESTION_CHARS]
        if not question.strip():
            return ChatTurn(
                role="assistant",
                content="(empty question — type something to ask the tutor)",
                failed=True,
            )

        # 1. KB retrieval.
        kb_snippets: list[dict] = []
        allowed_sources: set[str] = set()
        if self.kb is not None:
            try:
                kb_sig = self.kb.kb_signature() or ""
            except Exception:
                kb_sig = ""
            if kb_sig:
                try:
                    docs = self.kb.retrieve(question, k=_MAX_KB_SNIPPETS) or []
                except Exception as e:
                    print(f"tutor: KB retrieval failed in chat: {e!r}")
                    docs = []
                for d in docs:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    allowed_sources.add(src)
                    content = d.page_content
                    if len(content) > _MAX_KB_SNIPPET_CHARS:
                        content = content[: _MAX_KB_SNIPPET_CHARS - 50] + "…"
                    kb_snippets.append(
                        {"source": src, "page": page, "content": content}
                    )

        # 2. Build the prompt. `transcript` and `last_answer` are only
        # consumed in chat mode (the "tutor" mode is KB-only by design —
        # we don't want the LLM to leak the agent's prior output as if
        # it were KB content). Pass empty strings in tutor mode so the
        # chat-mode system prompt is the only place those sections
        # surface.
        user_prompt = _build_user_prompt(
            question=question,
            task=task,
            cards=self._recent_cards[-_MAX_CARDS_IN_CONTEXT:],
            transcript=transcript if mode == _ASK_MODE_CHAT else "",
            last_answer=last_answer if mode == _ASK_MODE_CHAT else "",
            kb_snippets=kb_snippets,
            history=self.history,
        )

        # 3. Call the LLM.
        if self.llm is None:
            return self._soft_failure(question, "no LLM configured")

        from langchain_core.messages import HumanMessage, SystemMessage

        # Pick the system prompt by mode. The "tutor" mode is KB-only; the
        # "chat" mode is KB + LLM hybrid.
        system_prompt = (
            _SYSTEM_PROMPT_CHAT if mode == _ASK_MODE_CHAT else _SYSTEM_PROMPT
        )
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            print(f"tutor: chat LLM call failed: {e!r}")
            return self._soft_failure(question, f"LLM call failed: {e!r}")

        # 4. Parse JSON.
        data = _extract_json(text)
        if data is None:
            return self._soft_failure(question, "tutor returned an unparseable response")

        # 5. Coerce answer, follow_up, citations.
        answer = str(data.get("answer", "") or "").strip()
        follow_up = str(data.get("follow_up", "") or "").strip()
        if follow_up and answer:
            answer = f"{answer}\n\n**Follow-up:** {follow_up}"

        citations = self._coerce_citations(
            data.get("citations") or [], allowed_sources
        )

        # 6. Classify.
        cls = self.rubric.classify_qa(
            self.llm,
            question=question,
            answer=answer,
            context="\n".join(
                [
                    f"task: {task}" if task else "",
                    f"step context: {context}" if context else "",
                ]
            ),
        )

        turn = ChatTurn(
            role="assistant",
            content=answer or "(tutor returned an empty answer)",
            citations=citations,
            bloom_level=cls.bloom_level,
            dok_level=cls.dok_level,
            rubric_hit=list(cls.rubric_hit),
            confidence=cls.confidence,
            failed=cls.failed,
        )

        # 7. Log.
        if self.logger is not None:
            try:
                self.logger.log(
                    {
                        "kind": "qa",
                        "question": question,
                        "answer": answer,
                        "bloom_level": cls.bloom_level,
                        "dok_level": cls.dok_level,
                        "rubric_hit": list(cls.rubric_hit),
                        "confidence": cls.confidence,
                        "failed": cls.failed,
                        "citations": citations,
                    }
                )
            except Exception as e:
                print(f"tutor: failed to log Q&A: {e!r}")

        # 8. Append to history.
        self.history.append(ChatTurn(role="user", content=question))
        self.history.append(turn)
        return turn

    # --- internals --------------------------------------------------------

    @staticmethod
    def _coerce_citations(
        value: Any, allowed_sources: set[str]
    ) -> list[dict]:
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
                    "snippet": snippet.strip()[:_MAX_CITATION_SNIPPET],
                }
            )
        return out

    def _soft_failure(self, question: str, reason: str) -> ChatTurn:
        turn = ChatTurn(
            role="assistant",
            content=(
                f"_(tutor unavailable — {reason}). "
                f"You asked: \"{question[:200]}\" — rephrase or try again.)_"
            ),
            failed=True,
        )
        self.history.append(ChatTurn(role="user", content=question))
        self.history.append(turn)
        if self.logger is not None:
            try:
                self.logger.log(
                    {
                        "kind": "qa",
                        "question": question,
                        "answer": turn.content,
                        "bloom_level": "Understand",
                        "dok_level": 1,
                        "rubric_hit": [],
                        "confidence": 0.0,
                        "failed": True,
                        "error": reason,
                    }
                )
            except Exception as e:
                print(f"tutor: failed to log soft-failure Q&A: {e!r}")
        return turn

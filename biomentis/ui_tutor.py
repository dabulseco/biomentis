"""Tutor UI: KB uploader, rubric picker, tutor chat panel, and render hooks.

This module is the *only* part of the tutor layer that imports Streamlit.
The `TutorEngine` in `biomentis.agent.tutor.engine` is framework-agnostic, so
importing `biomentis.agent.tutor` is safe even without Streamlit.

`install_renderers()` is called once at the top of `streamlit_app.py` to
wire the rich instruction-card and pause-gate renderers into
`a1.launch_streamlit_demo` (which calls them by name).

`tutor_wrapped_stream` is the optional `stream_fn` passed to
`launch_streamlit_demo`. When the tutor is disabled it is a no-op passthrough;
when enabled it drives the agent to completion, buffers all events into a
session-state queue, and yields them in batches — one batch per Continue
click, ending with a "paused" gate and an instruction card.

Run-state model (lives in `st.session_state["biomni_tutor_run"]`):

    {
        "prompt":  str,                # the user prompt for this run
        "thread_id": int,              # the LangGraph thread id
        "events": [UIEvent, ...],      # all events from the agent
        "cursor": int,                 # next event index to yield
        "step_id": int,                # current pedagogical step id
        "phase": "buffering" | "ready" | "done",
    }

Phase transitions on each submit / continue:

    no run            → buffering: drive stream_agent_events → ready
    ready, cursor=0   → yield batch [0, first_instruction_idx] + card + pause
    ready, continue   → yield batch [cursor, next_instruction_idx] + card + pause
    ready, no more    → done: yield complete, clear state
"""

from __future__ import annotations

import os
import time
from typing import Any

# Imports are deferred to the point of use so that importing this module
# without Streamlit (e.g. for unit tests) is safe.

# Event types we attach an instruction card to. Status / complete are
# pass-through — they don't carry new pedagogical content.
_INSTRUCTION_BEARING = {"reasoning", "code", "observation", "solution", "file", "summary"}

_RUN_KEY = "biomni_tutor_run"
_CONTINUE_KEY = "biomni_tutor_continue_step_id"
_PENDING_KEY = "biomni_tutor_pending_continue"


# ----- 1. renderer installation -------------------------------------------


def install_renderers() -> None:
    """Install the rich instruction-card and pause-gate renderers into
    `biomentis.agent.a1` so `launch_streamlit_demo` can call them.

    Called once at the top of `streamlit_app.py`. No-op if Streamlit isn't
    installed, since the renderers depend on it.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    import biomentis.agent.a1 as a1_mod

    def _render_instruction_card(container: Any, card: Any) -> None:
        """Render a structured `InstructionCard` into a Streamlit container.

        Falls back to a minimal "tutor unavailable" hint if the LLM call
        failed; the dispatcher in `ui_tutor` attaches a `_generation_failed`
        flag to the card in that case.

        All expanders carry the `step_id` in their label so the same card
        re-rendered across reruns doesn't collide on Streamlit's
        auto-generated expander keys.
        """
        import streamlit as st

        # step_id should be set by the dispatcher; if not, fall back to a
        # per-render counter so we never collide on expander/button keys.
        step_id = getattr(card, "step_id", None)
        if step_id is None:
            counter = st.session_state.setdefault("biomni_tutor_card_counter", 0)
            st.session_state["biomni_tutor_card_counter"] = counter + 1
            step_id = f"unkeyed_{counter}"
        if getattr(card, "_generation_failed", False):
            with container.container():
                st.warning(
                    f"🎓 Teaching card unavailable for this step "
                    f"(LLM call failed). The agent's work above is unchanged."
                )
                if card.what:
                    st.caption(card.what)
            return

        with container.container():
            st.markdown(f"**🎓 What:** {card.what or '(no summary)'}")
            if card.why:
                st.markdown(f"**Why:** {card.why}")
            if card.prerequisites:
                with st.expander(f"Prerequisites · step {step_id}", expanded=False):
                    for p in card.prerequisites:
                        st.markdown(f"- {p}")
            if card.look_for:
                with st.expander(f"What to look for in the output · step {step_id}", expanded=False):
                    for p in card.look_for:
                        st.markdown(f"- {p}")
            if card.citations:
                with st.expander(
                    f"Sources ({len(card.citations)}) · step {step_id}", expanded=False
                ):
                    for c in card.citations:
                        src = c.get("source", "unknown")
                        page = c.get("page")
                        snippet = c.get("snippet", "")
                        page_str = f" (p. {page})" if page else ""
                        st.markdown(
                            f"- **{src}{page_str}** — _{snippet[:200]}_"
                        )
            else:
                st.caption("📚 _No KB citations for this step._")
            tags = []
            if card.bloom_target:
                tags.append(f"Bloom: **{card.bloom_target}**")
            if card.dok_target:
                tags.append(f"DOK: **{card.dok_target}**")
            if tags:
                st.caption(" · ".join(tags))

    def _render_pause_gate(container: Any, step_id: Any) -> None:
        """Render a Continue button. On click, sets the resume flag and
        triggers a rerun.

        The button key is namespaced by a per-render counter so that the
        same `paused` event (with the same `step_id`) can be re-rendered
        across reruns without a duplicate-key error. The resume flag
        itself is keyed on `step_id`, so the counter only affects the
        widget identity, not the resume logic.
        """
        import streamlit as st

        counter = st.session_state.setdefault("biomni_tutor_pause_counter", 0)
        st.session_state["biomni_tutor_pause_counter"] = counter + 1

        with container.container():
            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button(
                    "▶ Continue",
                    key=f"biomni_tutor_continue_{step_id}_{counter}",
                    use_container_width=True,
                ):
                    st.session_state[_CONTINUE_KEY] = step_id
                    st.session_state[_PENDING_KEY] = True
                    st.rerun()
            with cols[1]:
                st.caption("Read the teaching card above, then continue.")

    a1_mod._RENDER_INSTRUCTION_CARD = _render_instruction_card
    a1_mod._RENDER_PAUSE_GATE = _render_pause_gate


# ----- 2. tutor_wrapped_stream --------------------------------------------


def tutor_wrapped_stream(agent, text_input, files, history_messages, thread_id):
    """Wrap `stream_agent_events` to insert per-step instruction cards and
    pause gates.

    Behavioral model (see module docstring for state shape):

    - Tutor disabled → pass-through.
    - Tutor enabled, no buffered run → drive `stream_agent_events` to
      completion, storing all events in `st.session_state[_RUN_KEY]`,
      then yield the first batch.
    - Tutor enabled, run in progress, Continue click → yield the next batch.

    Yields the same `UIEvent` objects the inner stream produces, plus
    an `instruction` event and a `paused` event after each
    instruction-bearing event in the batch. The dispatch in
    `launch_streamlit_demo` converts these into transcript entries and
    calls the renderers we installed.
    """
    # Lazy imports — keep `biomentis.ui_tutor` importable on systems without
    # Streamlit.
    from biomentis.ui_core import stream_agent_events
    import streamlit as st

    tutor = st.session_state.get("biomni_tutor")
    if tutor is None or not tutor.is_enabled():
        # Tutor not active — behave as the unwrapped stream.
        yield from stream_agent_events(agent, text_input, files, history_messages, thread_id)
        return

    # Mark which prompt this run is for; the chat panel uses it.
    tutor.current_prompt = text_input

    run = st.session_state.get(_RUN_KEY)

    # If there's an in-flight run for a *different* prompt, cancel it —
    # the user is starting a new run.
    if run and run.get("prompt") != text_input:
        st.session_state[_RUN_KEY] = None
        run = None

    # Phase 1: drive the agent to completion if no buffered run exists.
    # We measure how long this takes (typically seconds-to-minutes) and
    # report it on the final "Done" entry. Buffered events for later
    # Continue clicks are already in memory by then, so subsequent
    # batches yield in microseconds — those don't get their own timer
    # because it would always read 0.001s and add no information.
    if run is None or run.get("phase") == "buffering":
        _t_run_start = time.monotonic()
        run = _start_new_run(agent, text_input, files, history_messages, thread_id, tutor)
        st.session_state[_RUN_KEY] = run
        st.session_state.biomni_tutor_run_started_at = _t_run_start

    # Phase 2: yield a batch of events up to the next instruction-bearing event.
    yield from _yield_next_batch(run, tutor)

    # Phase 3: if we've exhausted all events, yield a `complete`.
    # NOTE: we deliberately do NOT clear `st.session_state[_RUN_KEY]` here.
    # The chat panel needs the buffered events to answer follow-up
    # questions about the just-completed task (e.g. "summarize the
    # databases used"). The "different prompt" check at the top of
    # `tutor_wrapped_stream` (line 206) still clears the buffer when
    # the user starts a new run with a different prompt.
    if run["phase"] == "done":
        from biomentis.ui_core import UIEvent, format_duration
        _t_started = st.session_state.get("biomni_tutor_run_started_at")
        _duration = time.monotonic() - _t_started if _t_started is not None else 0.0
        yield UIEvent(
            type="complete",
            content=(
                "Tutor run complete — every step has been walked through.  ·  "
                f"total time: {format_duration(_duration)}"
            ),
            channel="inner",
            title="✅ Done",
        )


# ----- 3. run management --------------------------------------------------


def _start_new_run(agent, text_input, files, history_messages, thread_id, tutor) -> dict:
    """Drive `stream_agent_events` to completion, collecting all events.

    Returns a run-state dict in phase "ready". A spinner keeps the user
    informed while the agent works (this can take seconds to minutes
    for complex tasks).
    """
    from biomentis.ui_core import stream_agent_events
    import streamlit as st

    run: dict = {
        "prompt": text_input,
        "thread_id": thread_id,
        "events": [],
        "cursor": 0,
        "step_id": 0,
        "phase": "buffering",
    }
    st.session_state[_RUN_KEY] = run

    # We don't drive the agent from inside a `st.spinner` block here —
    # the spinner would block reruns from re-entering the wrapper, which
    # we don't want. Instead, we just collect events; the script's own
    # spinner (if any) can wrap the submit handler.
    #
    # Important: we iterate the inner generator *fully* before returning
    # so that on the next script execution, the events list is complete
    # and the wrapper can yield them in batches.
    for event in stream_agent_events(agent, text_input, files, history_messages, thread_id):
        run["events"].append(event)
        # Log KB-relevant events so the analytics have them even before
        # we attach a card. The card is generated when the user advances
        # to the event in the wrapper.
        if event.type in _INSTRUCTION_BEARING:
            try:
                _log_event_seen(run, event, tutor)
            except Exception as e:
                print(f"tutor: log_event_seen failed: {e!r}")

    run["phase"] = "ready"
    return run


def _yield_next_batch(run: dict, tutor):
    """Yield events from `run["cursor"]` up to and including the next
    instruction-bearing event. Attach a card + pause gate to that event.
    Advance the cursor. If no more instruction-bearing events, yield the
    rest and mark the run done.
    """
    from biomentis.ui_core import UIEvent

    events = run["events"]
    cursor = run["cursor"]
    if cursor >= len(events):
        run["phase"] = "done"
        return

    # Find the next instruction-bearing event at or after `cursor`.
    end = len(events)
    for i in range(cursor, len(events)):
        if events[i].type in _INSTRUCTION_BEARING:
            end = i + 1
            break
    else:
        # No more instruction-bearing events; yield the rest and finish.
        end = len(events)

    # Yield every event in [cursor, end).
    for i in range(cursor, end):
        event = events[i]
        yield event

        # If this was the instruction-bearing event of the batch, also
        # yield the card + pause gate.
        if event.type in _INSTRUCTION_BEARING:
            run["step_id"] += 1
            card = _generate_or_get_card(event, run, tutor)
            yield _make_instruction_event(event, card, run["step_id"])
            yield _make_paused_event(run["step_id"])
            run["cursor"] = i + 1
            return

    # If we walked past the end without finding an instruction-bearing event,
    # we're done. Snapshot the run onto the engine so the chat panel can
    # still answer follow-up questions about this task even if the
    # session_state buffer is later cleared (e.g. on app reload).
    run["cursor"] = end
    if end >= len(events):
        run["phase"] = "done"
        try:
            _record_run_snapshot(run, tutor)
        except Exception as e:
            print(f"tutor: record_run_snapshot failed: {e!r}")


def _record_run_snapshot(run: dict, tutor) -> None:
    """Copy a finished run's events + answer onto the engine.

    The chat panel reads from `st.session_state[_RUN_KEY]` first; if
    that's gone (page reload, fresh tab, etc.) it falls back to the
    engine's `last_run_*` snapshot. This is the defense-in-depth
    half of the post-run context fix.
    """
    events = list(run.get("events") or [])

    # Pick the agent's final answer: most recent 'solution' first,
    # then most recent 'summary'.
    last_answer = ""
    for ev in reversed(events):
        if ev.type == "solution" and (getattr(ev, "content", "") or "").strip():
            last_answer = ev.content.strip()
            break
    if not last_answer:
        for ev in reversed(events):
            if ev.type == "summary" and (getattr(ev, "content", "") or "").strip():
                last_answer = ev.content.strip()
                break
    if last_answer and len(last_answer) > 4000:
        last_answer = last_answer[:4000] + "…"

    tutor.record_run(
        events=events,
        prompt=run.get("prompt", ""),
        last_answer=last_answer,
    )


# ----- 4. card generation + logging ---------------------------------------


_CARD_CACHE: dict[tuple, Any] = {}


def _generate_or_get_card(event, run, tutor) -> Any:
    """Generate (or fetch from cache) an InstructionCard for an event.

    Caching is keyed on (event_type, content_hash, kb_signature) so that
    re-emitting a near-identical event in the same run doesn't re-bill
    the LLM. The cache is shared across the whole run.
    """
    from biomentis.agent.tutor.instruction import InstructionCard

    content = _event_text_for_hash(event)
    kb_sig = ""
    if tutor.kb is not None:
        try:
            kb_sig = tutor.kb.kb_signature() or ""
        except Exception:
            kb_sig = ""
    key = (event.type, _hash(content), kb_sig)
    if key in _CARD_CACHE:
        return _CARD_CACHE[key]
    card = tutor.instruction_gen.generate(event, task=run.get("prompt", ""))
    _CARD_CACHE[key] = card

    # Log the full step with the card so analytics can see it.
    try:
        _log_step_with_card(run, event, card, tutor)
    except Exception as e:
        print(f"tutor: log_step_with_card failed: {e!r}")

    # Push a compact summary of the card to the chat so the tutor
    # has continuity. The full InstructionCard is too verbose for the
    # chat prompt; the chat only needs what/why + the labels.
    try:
        tutor.chat.push_recent_card(
            {
                "event_type": event.type,
                "title": getattr(event, "title", None),
                "what": card.what,
                "why": card.why,
                "bloom_target": card.bloom_target,
                "dok_target": card.dok_target,
            }
        )
    except Exception as e:
        print(f"tutor: push_recent_card failed: {e!r}")
    return card


def _event_text_for_hash(event) -> str:
    parts = []
    if getattr(event, "title", None):
        parts.append(event.title)
    if getattr(event, "content", None):
        parts.append(event.content)
    if getattr(event, "file_path", None):
        parts.append(f"file:{event.file_path}")
    return "\n".join(parts)


def _log_event_seen(run: dict, event, tutor) -> None:
    """Light log entry written as soon as we see an event. The full card
    is added later via `_log_step_with_card` when it's generated."""
    tutor.logger.log(
        {
            "kind": "event_seen",
            "event_type": event.type,
            "step_id": _step_id_for(event, run),
            "title": getattr(event, "title", None),
        }
    )


def _log_step_with_card(run: dict, event, card, tutor) -> None:
    """Full step log: event type, card contents, Bloom/DOK target, KB citations."""
    tutor.logger.log(
        {
            "kind": "step",
            "event_type": event.type,
            "step_id": run["step_id"],
            "title": getattr(event, "title", None),
            "bloom_target": card.bloom_target,
            "dok_target": card.dok_target,
            "instruction": {
                "what": card.what,
                "why": card.why,
                "prerequisites": list(card.prerequisites),
                "look_for": list(card.look_for),
            },
            "kb_citations": list(card.citations),
            "generation_failed": getattr(card, "_generation_failed", False),
        }
    )


def _step_id_for(event, run) -> int:
    """Best-effort step id for an event we've just seen but haven't
    yielded yet. Returns the next step id without mutating `run`."""
    return run["step_id"] + 1


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


# ----- 5. event builders --------------------------------------------------


def _make_instruction_event(prior_event: Any, card: Any, step_id: int = 0) -> Any:
    from biomentis.ui_core import UIEvent

    bits = []
    if card.what:
        bits.append(f"**What:** {card.what}")
    if card.why:
        bits.append(f"**Why:** {card.why}")
    if card.prerequisites:
        bits.append("**Prereqs:** " + "; ".join(card.prerequisites))
    if card.look_for:
        bits.append("**Look for:** " + "; ".join(card.look_for))
    if card.bloom_target:
        bits.append(f"**Bloom:** {card.bloom_target}")
    if card.dok_target:
        bits.append(f"**DOK:** {card.dok_target}")
    content = "\n\n".join(bits) if bits else "(tutor card empty)"
    ev = UIEvent(
        type="instruction",
        content=content,
        channel="inner",
        title="🎓 Teaching note",
    )
    ev.card = card
    ev.step_id = step_id
    # Also store step_id on the card so the renderer can include it in
    # expander labels and keep the same card re-renderable across reruns.
    try:
        card.step_id = step_id
    except Exception:
        pass
    return ev


def _make_paused_event(step_id: int) -> Any:
    from biomentis.ui_core import UIEvent

    ev = UIEvent(
        type="paused",
        content="Click Continue to advance.",
        channel="inner",
        title="⏸ Paused",
    )
    ev.step_id = step_id
    return ev


# ----- 6. Streamlit sub-panels --------------------------------------------


def _build_model_choices() -> list[tuple[str, str]]:
    """Return the same [(source, model), ...] list the main app uses.

    Phase D: the Critic model picker shows the same provider list as the
    agent's model picker (Ollama first, then cloud providers whose API
    keys are set). The Critic's LLM is independent of the agent's —
    changing one does not affect the other (the "reward model ≠ policy"
    separation).
    """
    from biomentis.ui_core import list_available_providers

    providers = list_available_providers()
    choices: list[tuple[str, str]] = []
    if "Ollama" in providers:
        for m in providers["Ollama"]:
            choices.append(("Ollama", m))
    for source, models in providers.items():
        if source == "Ollama":
            continue
        for m in models:
            choices.append((source, m))
    return choices


def render_tutor_sidebar(tutor) -> None:
    """Render the tutor's KB, rubric, self-improvement, and log panels.

    Args:
        tutor: A `TutorEngine` instance held in `st.session_state`.

    The sidebar is organized by feature, top-to-bottom:

      1. Header + Enable toggle
      2. Knowledge base    (default expanded)
      3. Rubric            (default collapsed — most users use the default)
      4. Self-improvement  (default collapsed — advanced)
      5. Session log       (default collapsed — used at session end)

    Each section is a collapsible expander. The four section renderers
    are independently testable / replaceable.
    """
    import streamlit as st

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎓 Tutor (optional)")

        # Enable / disable the tutor. The toggle gates step-cards + pauses
        # only; the KB / Rubric / Self-improvement / Log controls stay
        # available even when disabled so the user can keep using the
        # tutor chat with a KB-grounded context.
        tutor.enabled = st.toggle(
            "Enable instructional mode",
            value=tutor.enabled,
            help=(
                "When on, every agent step yields a teaching card and "
                "the agent pauses for you to read it. When off, the agent "
                "runs as before and this section is KB upload + chat only."
            ),
            key="biomni_tutor_enabled",
        )

        with st.expander("Knowledge base", expanded=True):
            _render_kb_panel(tutor)

        with st.expander("Rubric", expanded=False):
            _render_rubric_panel(tutor)

        with st.expander("Self-improvement", expanded=False):
            _render_self_improvement_panel(tutor)

        with st.expander("Session log", expanded=False):
            _render_log_export(tutor)


# ----- 6a. Knowledge-base panel ------------------------------------------


def _render_kb_panel(tutor) -> None:
    """KB uploader + URL ingest + Clear button + stats."""
    import streamlit as st

    st.caption(
        "Upload course materials. The tutor chat and instruction cards "
        "use this KB to ground their answers."
    )
    uploaded = st.file_uploader(
        "Upload course materials",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        key="biomni_tutor_kb_uploader",
        help="PDF, PowerPoint, Word, or plain text. Each file is chunked and embedded locally.",
    )
    url_text = st.text_area(
        "Or paste URLs (one per line)",
        value="",
        key="biomni_tutor_kb_urls",
        height=70,
        label_visibility="visible",
    )
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        add_clicked = st.button(
            "📥 Add to KB",
            use_container_width=True,
            key="biomni_tutor_kb_add",
        )
    with col_clear:
        clear_clicked = st.button(
            "🧹 Clear KB",
            use_container_width=True,
            key="biomni_tutor_kb_clear",
        )

    if add_clicked:
        _kb_add(tutor, uploaded, url_text)
    if clear_clicked:
        tutor.kb.clear()
        st.toast("Knowledge base cleared.", icon="🧹")
        st.rerun()

    # KB stats
    try:
        stats = tutor.kb.stats()
    except Exception as e:
        stats = None
        st.caption(f"_(KB stats unavailable: {e!r})_")
    if stats is not None:
        if stats.sources == 0:
            st.caption("_No documents indexed yet._")
        else:
            st.caption(
                f"📚 {stats.sources} source(s) · {stats.chunks} chunk(s)"
                + (f" · last updated {stats.last_updated}" if stats.last_updated else "")
            )
            with st.expander("Indexed sources", expanded=False):
                for s in stats.source_names:
                    st.markdown(f"- {s}")


# ----- 6b. Rubric panel ---------------------------------------------------


def _render_rubric_panel(tutor) -> None:
    """Teacher-rubric YAML uploader + reset + objectives view."""
    import streamlit as st

    st.caption(
        "Teacher-input rubric (Bloom + DOK per objective). The default "
        "rubric is discipline-agnostic; upload a YAML to override."
    )
    rubric_file = st.file_uploader(
        "Rubric YAML",
        type=["yaml", "yml"],
        key="biomni_tutor_rubric",
        accept_multiple_files=False,
    )
    col_reset, _ = st.columns([1, 1])
    with col_reset:
        reset_clicked = st.button(
            "↺ Reset to default",
            use_container_width=True,
            key="biomni_tutor_rubric_reset",
        )

    if rubric_file is not None:
        _rubric_load_from_upload(tutor, rubric_file)
    if reset_clicked:
        tutor.rubric = tutor.rubric.default()
        try:
            tutor.set_rubric(tutor.rubric)
        except Exception:
            pass
        st.toast("Rubric reset to default.", icon="↺")
        st.rerun()

    if tutor.rubric.objectives:
        st.caption(
            f"Loaded {len(tutor.rubric.objectives)} objective(s) from {tutor.rubric.source}"
        )
        with st.expander("Objectives", expanded=False):
            for o in tutor.rubric.objectives:
                st.markdown(
                    f"- **{o.id}** [{o.bloom_level} / DOK{o.dok_level}]: {o.description}"
                )


# ----- 6c. Self-improvement panel (Critic + memory + last critique) ------


def _render_self_improvement_panel(tutor) -> None:
    """Critic model picker, per-user memory viewer, and last critique.

    Grouped under one header because they're all aspects of the same
    workflow: a second LLM reviews finished sessions, writes priorities
    into per-user memory, and the next session's system prompt picks
    them up. The "Last critique" expander is the most-recent output of
    this workflow and lives next to the controls that produced it.
    """
    import streamlit as st

    # Lazy imports so this module stays importable without Streamlit.
    from biomentis.agent.tutor import critic_memory
    from biomentis.config import default_config
    from biomentis.llm import get_llm

    if not tutor.enabled:
        st.caption("Enable the tutor above to activate self-improvement.")

    # 1. User ID (top of section — it's the memory key for everything else)
    st.caption(
        "A second LLM reviews each finished session and writes priorities "
        "the agent sees on the next session. Off by default (stub Critic)."
    )
    mem_user_id = st.text_input(
        "User ID",
        value=st.session_state.get("biomni_tutor_user_id", "default"),
        key="biomni_tutor_user_id_input",
        help="Same user_id across sessions = priorities carry over.",
    )
    if mem_user_id != st.session_state.get("biomni_tutor_user_id"):
        st.session_state.biomni_tutor_user_id = mem_user_id

    # 2. Critic model picker
    critic_choices = _build_model_choices()
    critic_labels = [f"{src}: {mdl}" for src, mdl in critic_choices]
    critic_labels_with_stub = ["(stub — no review)"] + critic_labels

    def _critic_label_for(name: str) -> str:
        if not name or name == "stub":
            return "(stub — no review)"
        for lbl in critic_labels:
            if lbl.endswith(f": {name}") or lbl.endswith(name):
                return lbl
        return name

    critic_current = tutor.critic.model_name if tutor.critic else "stub"
    critic_idx = (
        critic_labels_with_stub.index(_critic_label_for(critic_current))
        if _critic_label_for(critic_current) in critic_labels_with_stub
        else 0
    )
    selected_critic_label = st.selectbox(
        "Critic model",
        options=critic_labels_with_stub,
        index=critic_idx,
        key="biomni_critic_model",
        help=(
            "Stub disables critique entirely. Pick a model to enable "
            "the Critic; the next session-end will be reviewed."
        ),
    )
    # If the choice changed, re-thread the Critic.
    if selected_critic_label == "(stub — no review)":
        new_name = "stub"
    else:
        new_name = selected_critic_label.split(": ", 1)[-1]
    if new_name != critic_current:
        tutor.set_critic_model_name(new_name)
        if new_name == "stub":
            tutor.set_critic_llm(None)
            st.toast("Critic disabled (stub).", icon="🛑")
        else:
            src, mdl = selected_critic_label.split(": ", 1)
            try:
                critic_llm = get_llm(mdl, source=src, config=default_config)
                tutor.set_critic_llm(critic_llm)
                st.toast(f"Critic set to {selected_critic_label}.", icon="🤖")
            except Exception as e:
                st.error(f"Failed to load Critic LLM: {e}")

    # 3. Memory viewer
    try:
        mem = critic_memory.load(mem_user_id, root=tutor.memory_root)
    except Exception as e:
        st.error(f"Memory load failed: {e!r}")
        mem = None
    if mem is not None:
        n = mem.get("n_sessions", 0)
        priorities = mem.get("priorities", []) or []
        counts = mem.get("weakness_counts", {}) or {}
        if n == 0:
            st.caption(
                "_No sessions critiqued yet — pick a Critic model and run a session._"
            )
        else:
            st.caption(
                f"📓 {n} session(s) critiqued · "
                f"{len(priorities)} priority(ies) · "
                f"{len(counts)} weakness kind(s) seen"
            )
        with st.expander(
            "Active priorities (go into next session's system prompt)",
            expanded=False,
        ):
            if priorities:
                for p in priorities:
                    st.markdown(f"- {p}")
            else:
                st.caption("_(empty)_")
        with st.expander("Weakness counts", expanded=False):
            if counts:
                # Sort by count desc, then by name for stable order
                for kind, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    bar = "▮" * min(int(c), 24)
                    st.markdown(f"- **{kind}** × {c}  `{bar}`")
            else:
                st.caption("_(empty)_")
        col_apply, col_reset = st.columns([1, 1])
        with col_apply:
            apply_clicked = st.button(
                "↻ Reload into engine",
                use_container_width=True,
                key="biomni_tutor_memory_reload",
                help=(
                    "Re-read the memory file into the live TutorEngine. "
                    "The agent's system prompt won't update until the next session."
                ),
            )
        with col_reset:
            reset_mem_clicked = st.button(
                "🗑 Reset memory",
                use_container_width=True,
                key="biomni_tutor_memory_reset",
                help="Wipe this user's memory file. Cannot be undone.",
            )
        if apply_clicked:
            priorities_loaded = tutor.load_priorities(mem_user_id)
            st.toast(
                f"Loaded {len(priorities_loaded)} priority(ies) for '{mem_user_id}'.",
                icon="↻",
            )
        if reset_mem_clicked:
            try:
                critic_memory.reset(mem_user_id, root=tutor.memory_root)
            except Exception as e:
                st.error(f"Reset failed: {e!r}")
            else:
                st.toast(f"Memory for '{mem_user_id}' wiped.", icon="🗑")
                st.rerun()

    # 4. Last critique card (the Critic's most recent output)
    last_card = st.session_state.get("biomni_tutor_last_card")
    if last_card is not None:
        with st.expander("Last critique (this session)", expanded=False):
            st.markdown(
                f"**Score:** {last_card.overall_score}/10 · "
                f"**Model:** `{last_card.model_name}`"
            )
            if last_card.weaknesses:
                st.markdown("**Weaknesses:**")
                for w in last_card.weaknesses:
                    sid = f"step {w.step_id}" if w.step_id is not None else "?"
                    eq = w.evidence_quote or ""
                    st.markdown(
                        f"- `{w.kind.value}` ({sid}): {w.detail}"
                        + (f"\n  > _{eq}_" if eq else "")
                    )
            if last_card.strengths:
                st.markdown("**Strengths:**")
                for s in last_card.strengths:
                    sid = f"step {s.step_id}" if s.step_id is not None else "?"
                    st.markdown(f"- ({sid}): {s.detail}")
            if last_card.next_session_priorities:
                st.markdown("**Next-session priorities (now in memory):**")
                for p in last_card.next_session_priorities:
                    st.markdown(f"- {p}")
            if last_card.notes:
                st.caption(last_card.notes)


# ----- 6d. Session log export --------------------------------------------


def _render_log_export(tutor) -> None:
    """Three download buttons for the session log: JSON, steps CSV, Q&A CSV.

    Each button reads the JSONL via `SessionLogger.export_json` /
    `export_csv` and exposes the result as a `st.download_button`. Empty
    logs are reported with a small note instead of a download.
    """
    import streamlit as st
    from datetime import datetime

    log_path = tutor.logger.path
    if not os.path.exists(log_path):
        st.caption("_No log file yet — run the agent to populate it._")
        return
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
    except Exception:
        line_count = 0
    if line_count == 0:
        st.caption("_Log is empty — no events recorded yet._")
        return

    st.caption(f"📝 {line_count} record(s) in {os.path.basename(log_path)}")

    try:
        json_payload = tutor.logger.export_json()
    except Exception as e:
        st.error(f"Log export failed: {e!r}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{tutor.session_id}_{ts}"

    col_json, col_steps, col_qa = st.columns(3)
    with col_json:
        st.download_button(
            "📥 JSON",
            data=json_payload,
            file_name=f"{base}_tutor_log.json",
            mime="application/json",
            use_container_width=True,
            key="biomni_tutor_export_json",
        )
    try:
        steps_csv, qa_csv = tutor.logger.export_csv()
    except Exception as e:
        st.error(f"CSV export failed: {e!r}")
        return
    with col_steps:
        st.download_button(
            "📥 Steps CSV",
            data=steps_csv,
            file_name=f"{base}_steps.csv",
            mime="text/csv",
            use_container_width=True,
            key="biomni_tutor_export_steps",
        )
    with col_qa:
        st.download_button(
            "📥 Q&A CSV",
            data=qa_csv,
            file_name=f"{base}_qa.csv",
            mime="text/csv",
            use_container_width=True,
            key="biomni_tutor_export_qa",
        )


def _kb_add(tutor, uploaded_files, url_text: str) -> None:
    """Shared logic for the sidebar Add button: save uploads, ingest, toast."""
    import streamlit as st
    import sys

    print(
        f"[tutor KB] _kb_add invoked: uploaded={len(uploaded_files) if uploaded_files else 0} file(s), url_text_len={len(url_text or '')}",
        file=sys.stderr,
    )

    saved_paths: list[str] = []
    if uploaded_files:
        save_dir = os.path.join(tutor.kb.path, "raw")
        os.makedirs(save_dir, exist_ok=True)
        for up in uploaded_files:
            dest = os.path.join(save_dir, up.name)
            with open(dest, "wb") as f:
                f.write(up.getbuffer())
            saved_paths.append(dest)
            print(f"[tutor KB] saved {dest} ({os.path.getsize(dest)} bytes)", file=sys.stderr)

    urls: list[str] = []
    if url_text and url_text.strip():
        for line in url_text.splitlines():
            line = line.strip()
            if line and line.startswith(("http://", "https://")):
                urls.append(line)

    if not saved_paths and not urls:
        # Loud UI message + a log line so this is never silent.
        print("[tutor KB] no files or URLs selected", file=sys.stderr)
        st.warning("Nothing to add — pick a file or paste a URL.")
        return

    n_chunks = 0
    if saved_paths:
        try:
            n_chunks += tutor.kb.add_files(saved_paths)
            print(f"[tutor KB] add_files returned {n_chunks} chunks", file=sys.stderr)
        except Exception as e:
            print(f"[tutor KB] add_files raised: {e!r}", file=sys.stderr)
            st.error(f"KB file ingest failed: {e!r}")
    if urls:
        try:
            n_chunks += tutor.kb.add_urls(urls)
            print(f"[tutor KB] add_urls returned {n_chunks} total chunks", file=sys.stderr)
        except Exception as e:
            print(f"[tutor KB] add_urls raised: {e!r}", file=sys.stderr)
            st.error(f"KB URL ingest failed: {e!r}")

    # Always render an in-page success message (toasts disappear after 4s
    # and are easy to miss). The toast is a bonus.
    if n_chunks > 0:
        st.success(f"Indexed {n_chunks} chunk(s) from {len(saved_paths)} file(s) and {len(urls)} URL(s).")
    else:
        st.warning(
            "0 chunks indexed. The file may be empty, image-based (scanned), or password-protected. "
            "See the Streamlit log (`tail -f /tmp/streamlit.log`) for details."
        )
    st.toast(f"Indexed {n_chunks} chunk(s).", icon="📚")
    # NOTE: do NOT call st.rerun() here — the toast + success messages
    # render on the current run. The next user interaction (file change,
    # button click) will naturally re-render. st.rerun() can interrupt
    # toast display and cause silent no-ops in some Streamlit versions.


def _rubric_load_from_upload(tutor, uploaded) -> None:
    """Parse a teacher-uploaded YAML rubric and replace the engine's rubric."""
    import streamlit as st

    try:
        data = uploaded.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Couldn't read rubric: {e!r}")
        return
    try:
        import yaml
        from biomentis.agent.tutor.rubric import Rubric, RubricObjective

        parsed = yaml.safe_load(data) or {}
        objs = []
        for item in parsed.get("objectives", []):
            objs.append(
                RubricObjective(
                    id=str(item.get("id", "")),
                    description=str(item.get("description", "")),
                    bloom_level=str(item.get("bloom_level", "Understand")),
                    dok_level=int(item.get("dok_level", 2)),
                )
            )
        tutor.rubric = Rubric(objectives=objs, source=uploaded.name)
        # Re-thread the rubric into the chat so subsequent Q&As are
        # classified against it.
        try:
            tutor.set_rubric(tutor.rubric)
        except Exception:
            pass
        st.toast(f"Loaded {len(objs)} objective(s) from {uploaded.name}.", icon="📋")
    except Exception as e:
        st.error(f"Rubric parse failed: {e!r}")


# ----- 7. Tutor chat panel ------------------------------------------------


def render_tutor_chat_panel(tutor) -> None:
    """Render the persistent "Ask about this task" chat panel.

    This panel is decoupled from the "Enable instructional mode" toggle.
    It always answers questions about the running task. The answering
    strategy is KB + LLM hybrid: when the student has uploaded course
    materials to the sidebar's Knowledge base, those materials ground
    the answer (and citations are shown for the KB portion). When the
    KB is empty, the panel falls back to the selected LLM's general
    knowledge with no citations. The Tutor (Instructional Mode) toggle
    controls per-step instruction cards and pauses, NOT this chat.

    The panel is also what the tutor chat history is rendered into; the
    per-Q&A Bloom/DOK/rubric badges come straight from `ChatTurn`.
    """
    import streamlit as st

    st.session_state.setdefault("biomni_tutor_chat_history", [])

    # Resolve the KB state for the status line. We read the engine's KB
    # stats once per render — cheap (the engine caches the index).
    kb_status = ""
    try:
        stats = tutor.kb.stats()
        if stats is not None and getattr(stats, "sources", 0) > 0:
            kb_status = (
                f"📚 KB: {stats.sources} source(s) · {stats.chunks} chunk(s)"
            )
        else:
            kb_status = "📚 KB: empty — answers use the selected LLM only"
    except Exception:
        kb_status = "📚 KB: (unavailable)"

    tutor_status = (
        "🎓 Tutor mode: on (instructional cards active)"
        if tutor.enabled
        else "🎓 Tutor mode: off (chat still works)"
    )

    with st.container():
        st.markdown("### 💬 Ask about this task")
        st.caption(
            "Ask a question about the running task. Upload a KB in the "
            "sidebar to ground answers in your course materials — "
            "otherwise the selected LLM will answer on its own."
        )
        # Status line — one row, low contrast. Tells the user exactly
        # what's powering the chat at the moment.
        st.caption(f"{kb_status}  ·  {tutor_status}")

        for turn in st.session_state.biomni_tutor_chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("citations"):
                    with st.expander(
                        f"Sources ({len(turn['citations'])})", expanded=False
                    ):
                        for c in turn["citations"]:
                            src = c.get("source", "?")
                            page = c.get("page")
                            st.markdown(
                                f"- **{src}**"
                                + (f" (p. {page})" if page else "")
                            )
                else:
                    # LLM-only answer — make it explicit that there are
                    # no KB citations, so the user can tell the difference
                    # between "the LLM said X" and "the KB said X."
                    if turn["role"] == "assistant" and not turn.get("failed"):
                        st.caption(
                            "_No KB sources cited — this answer used the "
                            "selected LLM's general knowledge._"
                        )
                # Bloom / DOK / rubric_hit badges (only for assistant turns
                # that have non-empty labels, i.e. the classifier ran).
                badges = []
                if turn.get("bloom_level"):
                    badges.append(f"Bloom: **{turn['bloom_level']}**")
                if turn.get("dok_level"):
                    badges.append(f"DOK: **{turn['dok_level']}**")
                if turn.get("rubric_hit"):
                    badges.append(
                        "Rubric: " + ", ".join(f"**{r}**" for r in turn["rubric_hit"])
                    )
                if turn.get("confidence") is not None and turn.get("confidence") > 0:
                    badges.append(f"conf: {turn['confidence']:.0%}")
                if badges:
                    st.caption(" · ".join(badges))
                if turn.get("failed"):
                    st.caption("⚠️ tutor was unable to fully answer this question")

        # Multi-line prompt input. `st.chat_input` is hard-coded to a single
        # line by Streamlit, so we replace it with a text area + Ask button
        # so users can paste long questions without truncation. Submit is
        # via the Ask button or Ctrl+Enter / Cmd-Enter.
        st.session_state.setdefault("biomni_tutor_chat_draft", "")
        with st.form(key="biomni_tutor_chat_form", clear_on_submit=True):
            chat_text = st.text_area(
                "Your question",
                value=st.session_state.biomni_tutor_chat_draft,
                key="biomni_tutor_chat_textarea",
                height=110,
                label_visibility="collapsed",
                placeholder=(
                    "Ask a question about the running task. Submit with the "
                    "Ask button or Ctrl+Enter / Cmd-Enter."
                ),
            )
            ask_clicked = st.form_submit_button("Ask", use_container_width=False)
        user_q = chat_text.strip() if ask_clicked else ""
        st.session_state.biomni_tutor_chat_draft = "" if ask_clicked else chat_text
        if user_q:
            _handle_chat_turn(tutor, user_q)


def _handle_chat_turn(tutor, question: str) -> None:
    """Phase-4 chat: KB retrieval + LLM call + Bloom/DOK/rubric classification."""
    import streamlit as st

    # 1. Append the user turn to the visible history immediately.
    st.session_state.biomni_tutor_chat_history.append(
        {"role": "user", "content": question}
    )

    # 2. Build context for the LLM.
    #
    # The tutor's chat panel must be able to answer follow-up questions
    # about the just-completed task (e.g. "summarize the databases
    # used"). For that, it needs:
    #   - the original task prompt  (tutor.current_prompt)
    #   - the actual transcript     (the events' .content, not just titles)
    #   - the agent's final answer  (the last 'solution' or 'summary')
    #
    # The previous implementation only forwarded a type/title hint, so
    # the LLM had no real context to work with.
    step_context = ""
    transcript = ""
    last_answer = ""
    run = st.session_state.get("biomni_tutor_run")

    # Fallback to the engine's last-run snapshot if session_state
    # somehow lost the live run (e.g. page reload mid-session).
    if not (run and run.get("events")):
        engine_events = list(getattr(tutor, "last_run_events", []) or [])
        if engine_events:
            run = {
                "events": engine_events,
                "prompt": tutor.last_run_prompt or tutor.current_prompt,
                "cursor": len(engine_events),
            }

    if run and run.get("events"):
        # Single-event type/title hint — kept for the rubric classifier
        # (it still consumes a "step context" string).
        cursor = run.get("cursor", 0) or len(run["events"])
        for ev in reversed(run["events"][:cursor]):
            if ev.type in {"reasoning", "code", "observation", "solution", "summary", "file"}:
                step_context = (
                    f"step_id={run.get('step_id', '?')}, "
                    f"event_type={ev.type}, "
                    f"title={getattr(ev, 'title', '') or '(none)'}"
                )
                break

        # Real transcript: the last 5 instruction-bearing events'
        # actual content. This is what the LLM needs to summarize the
        # task or name the databases that were used.
        recent_events = [
            ev
            for ev in run["events"]
            if ev.type in {"reasoning", "code", "observation", "solution", "summary", "file"}
        ][-5:]
        transcript_parts: list[str] = []
        for ev in recent_events:
            title = getattr(ev, "title", "") or ev.type
            content = (getattr(ev, "content", "") or "").strip()
            if len(content) > 1200:
                content = content[:1200] + "…"
            if content:
                transcript_parts.append(f"[{ev.type}] {title}\n{content}")
        transcript = "\n\n".join(transcript_parts)

        # Last 'solution' (preferred) or 'summary' is the agent's
        # final answer. Pass it explicitly so the LLM can quote it.
        for ev in reversed(run["events"]):
            if ev.type == "solution" and (getattr(ev, "content", "") or "").strip():
                last_answer = ev.content.strip()
                break
        if not last_answer:
            for ev in reversed(run["events"]):
                if ev.type == "summary" and (getattr(ev, "content", "") or "").strip():
                    last_answer = ev.content.strip()
                    break
        if last_answer and len(last_answer) > 2000:
            last_answer = last_answer[:2000] + "…"

    _t_chat_start = time.monotonic()
    try:
        turn = tutor.chat.ask(
            question=question,
            context=step_context,
            task=tutor.current_prompt or "",
            transcript=transcript,
            last_answer=last_answer,
            mode="chat",  # KB + LLM hybrid; chat panel is decoupled from tutor
        )
    except Exception as e:
        # Defensive: TutorChat.ask is supposed to handle its own errors,
        # but if something leaks out, don't crash the UI.
        st.session_state.biomni_tutor_chat_history.append(
            {
                "role": "assistant",
                "content": f"_(tutor crashed: {e!r})_",
                "failed": True,
            }
        )
        st.rerun()
        return

    # Surface the wall-clock cost of this chat turn so users can tell
    # whether a slow response is the LLM or downstream retrieval. We
    # append to the stored answer so it renders inline; the chat
    # history renderer doesn't need to know about timing.
    from biomentis.ui_core import format_duration
    _chat_duration = time.monotonic() - _t_chat_start
    _answer_with_duration = f"{turn.content}  ·  (took {format_duration(_chat_duration)})"

    # 3. Append the assistant turn to the visible history.
    st.session_state.biomni_tutor_chat_history.append(
        {
            "role": "assistant",
            "content": _answer_with_duration,
            "citations": list(turn.citations),
            "bloom_level": turn.bloom_level,
            "dok_level": turn.dok_level,
            "rubric_hit": list(turn.rubric_hit),
            "confidence": turn.confidence,
            "failed": turn.failed,
        }
    )
    st.rerun()


# ----- 8. submit-handler integration --------------------------------------


def consume_pending_continue() -> bool:
    """If the user clicked Continue since the last script execution, this
    returns True and clears the flag. The submit handler in
    `launch_streamlit_demo` should call this before deciding whether to
    run the agent.

    Returns:
        True if a continue was pending (caller should drive the wrapper
        in continue-mode with the same prompt). False otherwise.

    NOTE: This hook is wired in Phase 3c; for now the wrapper itself
    checks the session state directly. We expose the helper for the
    `streamlit_app.py` and a1.py integration in 3c.
    """
    import streamlit as st

    if st.session_state.pop(_PENDING_KEY, False):
        return True
    return False


def get_current_run_prompt() -> str | None:
    """If a tutor run is in progress, return its prompt (so the submit
    handler can drive the wrapper with the same prompt instead of a new
    one). Returns None if no run is in progress."""
    import streamlit as st

    run = st.session_state.get(_RUN_KEY)
    if not run:
        return None
    if run.get("phase") == "done":
        return None
    return run.get("prompt")

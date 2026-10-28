"""Streamlit launcher for Biomentis.

The user picks the model from a sidebar dropdown populated from
`ollama list` (with any cloud providers whose API keys are set in the
environment as secondary options). The first Ollama model in the list
becomes the default — same ordering `ollama list` itself uses.

The agent is cached in `st.session_state` rather than `@st.cache_resource`
so the model switch takes effect immediately, without re-running the script.

The optional tutor layer (`biomentis.ui_tutor`) is wired in below. When
the user keeps the tutor disabled, behavior is identical to a tutor-less
run — no instruction cards, no pauses, no logging.
"""

import os

import streamlit as st

from biomentis.agent import A1
from biomentis.agent.tutor import TutorEngine
from biomentis.config import default_config
from biomentis.ui_core import list_available_providers
from biomentis.ui_repl import render_repl_panel
from biomentis.ui_settings import render_anthropic_key_field, render_ncbi_email_field, render_ollama_key_field
from biomentis.ui_tutor import (
    install_renderers,
    render_tutor_chat_panel,
    render_tutor_sidebar,
    tutor_wrapped_stream,
)

# Install the rich instruction-card and pause-gate renderers into
# `biomentis.agent.a1` once at import time. No-op if Streamlit isn't installed.
install_renderers()

# --- Page setup (must be the first Streamlit call) ---
st.set_page_config(page_title="Biomentis AI Agent", layout="wide")


# --- Sidebar: model picker ----------------------------------------------
def _build_model_choices() -> list[tuple[str, str]]:
    """Return a [(source, model), ...] list.

    Ollama models come first (in the order `ollama list` returns them).
    Cloud providers whose API keys are set follow.
    """
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


def _default_selection(choices: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Pick the initial dropdown selection.

    A model named in BIOMNI_LLM (with BIOMNI_SOURCE, if set) wins when it is
    available; otherwise the first Ollama model, otherwise the first choice.
    """
    if not choices:
        return None
    configured = os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL")
    # The environment, not default_config: that was built when biomentis.config
    # was imported, before the agent module loaded .env, so it had already
    # resolved to a local Ollama model and its source said "Ollama".
    source = os.getenv("BIOMNI_SOURCE")
    if configured:
        for c in choices:
            if c[1] == configured and (not source or c[0] == source):
                return c
    return choices[0]


with st.sidebar:
    st.header("Model")

    choices = _build_model_choices()
    if not choices:
        st.error(
            "No models available. Start the Ollama daemon (`ollama serve`) "
            "and pull at least one model (`ollama pull qwen2.5:14b`), "
            "or set a cloud-provider API key in your .env."
        )
        st.stop()

    default_choice = _default_selection(choices)
    default_index = choices.index(default_choice) if default_choice in choices else 0

    # Build a list of human-readable labels for the dropdown, but keep the
    # (source, model) tuple as the underlying value via index-based mapping.
    labels = [f"{src}: {mdl}" for src, mdl in choices]
    selected_idx = st.selectbox(
        "Model",
        options=range(len(choices)),
        index=default_index,
        format_func=lambda i: labels[i],
        key="biomni_model_idx",
        help="Set the default with BIOMNI_LLM in .env; otherwise the first Ollama model. "
        "Pull more with `ollama pull <name>`.",
    )
    selected_source, selected_model = choices[selected_idx]

    st.caption(f"Active: **{selected_model}** via **{selected_source}**")

    # How hard the agent works before an answer is accepted. Read below when
    # the agent is configured, so a change applies from the next run.
    st.toggle(
        "Require evidence before answering",
        value=True,
        key="biomentis_evidence_gate",
        help="Sends the first answer back (at most twice) until the run has consulted a "
        "literature source, computed its numbers in code and made a figure.",
    )
    # Temperatures, for reproducibility. 0 is the most repeatable (the model
    # always takes its most likely next token); higher values vary more from
    # run to run. Read below when the agent is configured.
    _temperature_steps = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    st.select_slider(
        "Agent temperature",
        options=_temperature_steps,
        value=default_config.temperature if default_config.temperature in _temperature_steps else 0.2,
        key="biomentis_temperature",
        help="Code, tool calls and answers. 0 gives the most repeatable runs. Default from "
        "BIOMNI_TEMPERATURE in .env (0.2 if unset). Claude models ignore it.",
    )
    st.select_slider(
        "Self-critique temperature",
        options=_temperature_steps,
        value=default_config.creative_temperature
        if default_config.creative_temperature in _temperature_steps
        else 0.7,
        key="biomentis_creative_temperature",
        help="The agent's own review of its answer. Higher finds more varied gaps; lower is more "
        "repeatable. Default from BIOMNI_CREATIVE_TEMPERATURE (0.7 if unset).",
    )
    st.segmented_control(
        "Self-critique rounds",
        options=[0, 1, 2, 3],
        default=2,
        key="biomentis_critique_rounds",
        help="Reviews of the answer by the agent's model before it is final. Each round can "
        "send the agent back to recompute numbers or find sources. 0 turns review off.",
    )

    # NCBI contact email. Rendered here (early, before the agent is built)
    # so `Bio.Entrez.email` is in place before any Entrez-touching code runs
    # — including the agent's own generated snippets, which `run_python_repl`
    # execs in this process. Session-scoped; `.env` is what makes it stick.
    st.divider()
    st.header("NCBI")
    render_ncbi_email_field()

    # Anthropic key. Also early, so `advanced_web_search_claude` and any Claude
    # chat model see the resolved key before the agent is built. The field
    # shows the key the tools will actually use, which is the whole point: a
    # stale shell export shadowing `.env` used to be invisible from the UI.
    st.divider()
    st.header("Anthropic")
    render_anthropic_key_field()

    # Ollama API key: lists Ollama's hosted models (DeepSeek, ...) under
    # "OllamaCloud" in the picker above, reached without a local signin.
    st.divider()
    st.header("Ollama Cloud")
    render_ollama_key_field()


# --- Agent (rebuilt when the model changes) -----------------------------
def _get_or_build_agent(source: str, model: str) -> A1:
    cache_key = f"{source}::{model}"
    if st.session_state.get("biomni_agent_key") != cache_key:
        # First build, or model changed — (re)instantiate.
        agent = A1(
            path="./data",
            use_tool_retriever=False,
            expected_data_lake_files=[],
        )
        # Override the LLM with the user-selected one. set_model (not a bare
        # `agent.llm = ...`) keeps the agent's temperature and stop sequences.
        try:
            agent.set_model(model, source=source)
        except Exception as e:
            st.error(f"Failed to load model `{source}: {model}`: {e}")
            st.stop()
        st.session_state.biomni_agent = agent
        st.session_state.biomni_agent_key = cache_key

    agent = st.session_state.biomni_agent

    # TutorEngine is also session-scoped and re-threaded when the LLM
    # changes (the engine's instruction generator and chat bot use it).
    # The session_id is unique per Streamlit session so KBs and logs
    # don't bleed across browser tabs.
    import uuid

    if "biomni_tutor" not in st.session_state:
        st.session_state.biomni_tutor = TutorEngine(
            session_id=st.session_state.setdefault(
                "biomni_session_id", str(uuid.uuid4())[:8]
            ),
            llm=agent.llm,
            path=agent.path,
        )
    tutor: TutorEngine = st.session_state.biomni_tutor
    if tutor.llm is not agent.llm:
        tutor.set_llm(agent.llm)
    return agent


agent = _get_or_build_agent(selected_source, selected_model)
tutor: TutorEngine = st.session_state.biomni_tutor

# --- Critic wiring (Phase D) ---------------------------------------------
# 1. Resolve the per-user id. Default to "default" so a single-user
#    installation just works; the sidebar lets the user override this.
user_id = st.session_state.setdefault("biomni_tutor_user_id", "default")
# 2. Load any priorities the user has accumulated across prior sessions
#    and stash them on the engine. `load_priorities` is a pure read; it
#    won't fail on a missing file.
tutor.load_priorities(user_id)
# 3. Re-thread them into the agent's system prompt. `agent.configure()`
#    rebuilds the system prompt; if there are priorities we want them
#    appended on the *next* invocation, which is what `configure()` does.
# self_critic=True adds "think hard about what's missing" passes after the
# agent proposes a solution (see A1.execute_self_critic), as many as the
# sidebar's "Self-critique rounds"; the critic also checks that each number
# was computed in code. evidence_gate sends an answer back until the run has
# a source, a calculation and a figure (see biomentis/run_quality.py).
#
# Skipped while a run is in flight. `configure()` rebuilds the system prompt
# and recompiles the LangGraph app, and since runs moved onto a background
# thread (see biomentis/run_worker.py) a rerun can now land in the middle of
# one — which would mutate the agent out from under the worker driving it.
# Between turns there is nothing to disturb, which is when it matters anyway.
if not st.session_state.get("biomni_run_active"):
    # Temperatures from the sidebar; the tutor follows the rebuilt client.
    if agent.set_temperatures(
        st.session_state.get("biomentis_temperature", default_config.temperature),
        st.session_state.get("biomentis_creative_temperature", default_config.creative_temperature),
    ):
        tutor.set_llm(agent.llm)
    # The independent reviewer is the Critic model picked under Tutor →
    # Self-improvement; with the Critic off ("stub") there is no review.
    _critic = tutor.critic
    _reviewing = _critic is not None and _critic.model_name != "stub" and _critic.llm is not None
    agent.review_llm = _critic.llm if _reviewing else None
    agent.review_model_name = _critic.model_name if _reviewing else ""
    # A deselected segmented control reads None; treat it as the default.
    _critique_rounds = st.session_state.get("biomentis_critique_rounds")
    _critique_rounds = 2 if _critique_rounds is None else _critique_rounds
    agent.configure(
        self_critic=_critique_rounds > 0,
        test_time_scale_round=_critique_rounds,
        critic_priorities=list(tutor.active_priorities),
        evidence_gate=st.session_state.get("biomentis_evidence_gate", True),
    )

# Sidebar order: Model picker (above) → REPL (existing) → Tutor (new).
# The REPL panel renders its own divider; we add ours after.
render_repl_panel(agent)
render_tutor_sidebar(tutor)

# Header logo — top of the main display, left-justified. Rendered
# before the chat panel so it sits at the very top of the page (the
# chat panel is the next thing Streamlit would draw). Resolved
# relative to this file so it works regardless of CWD. Width bumped
# another 50% (300 → 450 → 675) at the user's request; aspect ratio
# preserved since the source PNG is 2720×880.
_logo_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "figs",
    "biomentis_logo_concept.png",
)
# 50% larger than the first bump (300 → 450 → 675), at the user's
# request. Aspect ratio preserved since the source is 2720×880.
if os.path.exists(_logo_path):
    st.image(_logo_path, width=675)

# Tutor chat lives in a third column on wide screens. On narrow screens
# the chat will stack below the two main panels, which is acceptable.
render_tutor_chat_panel(tutor)

# Pass `tutor_wrapped_stream` as the optional `stream_fn` kwarg. When the
# tutor is disabled, the wrapper is a transparent passthrough; when
# enabled, it injects per-step instruction cards and a pause gate.
agent.launch_streamlit_demo(stream_fn=tutor_wrapped_stream)

# --- Critic end-of-session hook (Phase D) --------------------------------
# After the agent run returns, run the Critic over the just-finished
# session and fold the result into the user's memory. This is the
# "in-context reward shaping" write side: the next session picks up
# these priorities via `tutor.load_priorities()` above.
# We guard on `tutor.enabled` AND on the Critic actually having a
# non-stub model — a stub Critic returns a soft-failure card with no
# priorities, which is wasted work.
try:
    if tutor.enabled and getattr(tutor.critic, "model_name", "stub") != "stub":
        agent_disp = f"{selected_source}: {selected_model}"
        card = tutor.on_session_end(
            user_id=user_id,
            agent_model_name=agent_disp,
            task=tutor.current_prompt or "",
        )
        if card is not None:
            # Surface the result in the sidebar for the next rerun.
            st.session_state.biomni_tutor_last_card = card
except Exception as e:
    # Never let the Critic break the Streamlit rerun loop.
    st.warning(f"Critic session-end failed: {e!r}")

# --- Embedded help chatbot ----------------------------------------------
# Third-party RAG chatbot (hosted at chatbot-embedding-ifi.onrender.com).
# The embed script appends a fixed-position launcher to document.body of
# whatever page it runs in. We use `st.html` (no iframe wrapper) so the
# launcher is fixed to the actual Streamlit viewport, not a sandboxed
# iframe. A <style> tag positions the launcher in the bottom-right.
import streamlit as _st

_CHATBOT_SRC = "https://chatbot-embedding-ifi.onrender.com/chatbot-embed.js"
_CHATBOT_ID = "9bb3fa1e-fe66-43c7-945b-5cab90083e3f"

with _st.sidebar:
    help_open = _st.toggle("❓ Help chatbot", value=False, key="biomni_help_open")

if help_open:
    _st.html(
        f"""<div id="biomni-help-bot-mount">
  <style>
    /* Push the bot's launcher into the bottom-right of the viewport. */
    #biomni-help-bot-mount ~ * [class*="launcher"],
    #biomni-help-bot-mount ~ * [class*="Launcher"],
    #biomni-help-bot-mount ~ * [id*="launcher"],
    #biomni-help-bot-mount ~ * button[aria-label*="chat" i],
    #biomni-help-bot-mount ~ * button[aria-label*="bot" i] {{
      bottom: 24px !important;
      right: 24px !important;
    }}
  </style>
  <script
    src="{_CHATBOT_SRC}"
    data-chatbot-id="{_CHATBOT_ID}"
    async
    defer>
  </script>
</div>""",
        unsafe_allow_javascript=True,
    )





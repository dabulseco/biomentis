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
from biomentis.llm import get_llm
from biomentis.ui_core import list_available_providers
from biomentis.ui_repl import render_repl_panel
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
st.set_page_config(page_title="Biomentis A1 Agent", layout="wide")


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
    """Pick the initial dropdown selection. Prefers the first Ollama model;
    falls back to whatever default_config resolved to, or the first choice."""
    if not choices:
        return None
    if choices and choices[0][0] == "Ollama":
        return choices[0]
    if default_config.llm and default_config.source:
        for c in choices:
            if c[1] == default_config.llm and c[0] == default_config.source:
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
        help="First Ollama model is selected by default. Pull more with `ollama pull <name>`.",
    )
    selected_source, selected_model = choices[selected_idx]

    st.caption(f"Active: **{selected_model}** via **{selected_source}**")


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
        # Override the LLM with the user-selected one.
        try:
            agent.llm = get_llm(model, source=source, config=default_config)
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
agent.configure(critic_priorities=list(tutor.active_priorities))

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





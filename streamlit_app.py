"""Streamlit launcher for Biomni.

The user picks the model from a sidebar dropdown populated from
`ollama list` (with any cloud providers whose API keys are set in the
environment as secondary options). The first Ollama model in the list
becomes the default — same ordering `ollama list` itself uses.

The agent is cached in `st.session_state` rather than `@st.cache_resource`
so the model switch takes effect immediately, without re-running the script.
"""

import streamlit as st

from biomni.agent import A1
from biomni.config import default_config
from biomni.llm import get_llm
from biomni.ui_core import list_available_providers
from biomni.ui_repl import render_repl_panel

# --- Page setup (must be the first Streamlit call) ---
st.set_page_config(page_title="Biomni A1 Agent", layout="wide")


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
    return st.session_state.biomni_agent


agent = _get_or_build_agent(selected_source, selected_model)

# Sidebar REPL — Python interpreter that shares the agent's namespace.
# Renders above the agent's own Streamlit demo so it's always visible.
render_repl_panel(agent)

agent.launch_streamlit_demo()

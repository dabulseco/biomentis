"""Streamlit REPL panel for Biomni.

Renders a sidebar panel that lets the user execute Python in the same
`_persistent_namespace` the agent uses, plus a one-click "Run last agent code"
button that pulls the most recent code block out of the agent's transcript
and pipes it through the same interpreter.

The panel keeps:
  - The text input (editable so you can tweak before running)
  - The most recent N (input, output) pairs as a scrollable history
  - An "auto-run" toggle that runs the agent's last code as soon as the
    agent finishes a turn
  - A "clear namespace" button that wipes `_persistent_namespace`

Copy affordance:
  - `st.code(...)` already shows a built-in copy-to-clipboard button.
  - The Run row also has an explicit "Copy code" button so it's visible
    without scrolling.
"""

from __future__ import annotations

import streamlit as st

from biomni.tool.support_tools import run_python_repl

_HISTORY_LIMIT = 20


def _ensure_repl_state() -> None:
    """Initialize REPL-related session-state keys once per session."""
    st.session_state.setdefault("biomni_repl_input", "")
    st.session_state.setdefault("biomni_repl_history", [])  # list of {"code": str, "output": str}
    st.session_state.setdefault("biomni_repl_auto_run", False)
    st.session_state.setdefault("biomni_repl_last_seen_code_id", None)


def _get_last_agent_code() -> tuple[str, str | None] | None:
    """Return (code, language) for the most recent code block the agent emitted, or None.

    Looks at the transcript the agent populates in `st.session_state.biomni_transcript`
    and finds the last entry with kind == "code". Returns the entry's id (for
    auto-run dedup) alongside the code so the caller can tell whether a new
    agent turn has finished since the last auto-run.
    """
    transcript = st.session_state.get("biomni_transcript") or []
    for entry in reversed(transcript):
        if entry.get("kind") == "code":
            return entry.get("content", ""), entry.get("language") or "python", id(entry)
    return None


def _trim_history() -> None:
    history = st.session_state.get("biomni_repl_history") or []
    if len(history) > _HISTORY_LIMIT:
        st.session_state.biomni_repl_history = history[-HISTORY_LIMIT:]


def _execute_code(code: str) -> str:
    """Run `code` through the agent's REPL and return captured output."""
    try:
        return run_python_repl(code)
    except Exception as e:  # belt-and-suspenders; run_python_repl already catches
        return f"Error: {e!r}"


def render_repl_panel(agent) -> None:
    """Render the REPL panel in the Streamlit sidebar.

    Args:
        agent: The A1 instance (unused for now, but kept for future hooks
               like "show agent's data lake path" or "load the file the
               agent just referenced").
    """
    _ensure_repl_state()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🐍 Python REPL")
        st.caption(
            "Runs the same Python the agent uses. Variables you define here "
            "are visible to the agent on its next turn, and vice versa."
        )

        # --- Auto-run toggle ----------------------------------------------
        st.session_state.biomni_repl_auto_run = st.toggle(
            "Auto-run when agent finishes",
            value=st.session_state.biomni_repl_auto_run,
            help="When on, the most recent agent code block runs in this REPL as soon as the turn completes.",
        )

        # --- Last-agent-code shortcut -------------------------------------
        last = _get_last_agent_code()
        if last is not None:
            last_code, last_lang, last_id = last
            if st.button(
                f"▶ Run last agent code ({len(last_code.splitlines())} lines)",
                use_container_width=True,
                key="biomni_repl_run_last",
            ):
                st.session_state.biomni_repl_input = last_code
                output = _execute_code(last_code)
                st.session_state.biomni_repl_history.append(
                    {"code": last_code, "output": output}
                )
                _trim_history()
                st.session_state.biomni_repl_last_seen_code_id = last_id
                st.rerun()

        # --- Editable code input -----------------------------------------
        st.session_state.biomni_repl_input = st.text_area(
            "Code",
            value=st.session_state.biomni_repl_input,
            height=180,
            key="biomni_repl_input_widget",
            label_visibility="collapsed",
            placeholder="# Type or paste Python here. Use the button above to load the agent's last code.",
        )

        # --- Action row ----------------------------------------------------
        col_run, col_copy, col_clear = st.columns([1, 1, 1])
        with col_run:
            run_clicked = st.button("▶ Run", use_container_width=True, key="biomni_repl_run")
        with col_copy:
            if st.session_state.biomni_repl_input:
                # st.code is what gives the built-in copy-to-clipboard button;
                # we re-render the current input as code so the user always
                # has a one-click copy target right next to the Run button.
                with st.popover("📋 Copy"):
                    st.code(st.session_state.biomni_repl_input, language="python")
        with col_clear:
            if st.button("🧹 Clear", use_container_width=True, key="biomni_repl_clear"):
                # Wipe the persistent namespace by re-importing the module so
                # the dict gets re-initialized.
                import importlib

                from biomni.tool import support_tools

                importlib.reload(support_tools)
                st.session_state.biomni_repl_history.append(
                    {"code": "# (namespace cleared)", "output": ""}
                )
                _trim_history()
                st.toast("REPL namespace cleared.", icon="🧹")
                st.rerun()

        if run_clicked and st.session_state.biomni_repl_input.strip():
            output = _execute_code(st.session_state.biomni_repl_input)
            st.session_state.biomni_repl_history.append(
                {"code": st.session_state.biomni_repl_input, "output": output}
            )
            _trim_history()
            st.rerun()

        # --- Output + history ---------------------------------------------
        history = st.session_state.get("biomni_repl_history") or []
        if history:
            st.markdown("**Output & history**")
            # Show most recent first
            for i, entry in enumerate(reversed(history)):
                with st.expander(
                    f"#{len(history) - i}  ·  {entry['code'].splitlines()[0][:60] if entry['code'].strip() else '(no code)'}",
                    expanded=(i == 0),
                ):
                    if entry["code"].strip():
                        st.code(entry["code"], language="python")
                    st.markdown("**Output:**")
                    st.code(entry["output"] or "(no output)", language="text")

    # --- Auto-run hook (runs even when not rendering the panel,
    # so toggling the switch takes effect on the next agent turn) ----------
    if st.session_state.biomni_repl_auto_run and last is not None:
        last_code, last_lang, last_id = last
        if st.session_state.biomni_repl_last_seen_code_id != last_id:
            output = _execute_code(last_code)
            st.session_state.biomni_repl_history.append(
                {"code": last_code, "output": output}
            )
            _trim_history()
            st.session_state.biomni_repl_last_seen_code_id = last_id
            st.toast("Auto-ran agent's last code in REPL.", icon="▶")

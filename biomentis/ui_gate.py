"""Streamlit pause panel for the completion gate (see `biomentis.gate.control`).

From round 4, or when the context window is nearly full, a run blocks in
`RoundControl.wait()` until the user picks Continue or Stop. This panel is how
they pick. It renders from `control.pending` alone, so it survives every
rerun for as long as the run waits, and it renders nothing otherwise.

`key_prefix` keeps the widget keys apart when several paused runs are shown
on one page (one panel per run in a batch).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

ROUND_REASON = "Round {n} needs your decision"
CONTEXT_REASON = "The context window is nearly full"


def pause_reason(pending: dict) -> str:
    if pending.get("reason") == "context":
        return CONTEXT_REASON
    return ROUND_REASON.format(n=pending.get("round"))


def render_round_pause(control: Any, key_prefix: str = "") -> None:
    """Show the pending decision of `control`, if any, with Continue and Stop."""
    pending = control.pending if control is not None else None
    if not pending:
        return
    findings = pending.get("findings") or []
    with st.container(border=True):
        st.warning(pause_reason(pending), icon=":material/pause_circle:")
        if pending.get("reason") == "context":
            st.caption(
                "Continue asks the agent to wrap up with what it has; Stop ends the run now."
            )
        else:
            st.caption(
                "The automatic revision rounds are used up and these findings still block the "
                "answer. Tick any you accept as is, then Continue for another round or Stop."
            )
        for finding in findings:
            with st.container(border=True):
                st.markdown(f"**{finding.check}** · {finding.issue}")
                if finding.quote:
                    st.markdown(f"> {finding.quote}")
                if finding.fix:
                    st.caption(f"Fix: {finding.fix}")
                st.checkbox("Accept as is", key=f"{key_prefix}accept_{finding.key}")
        # Callbacks, not `if st.button(...)`: the decision is made before the
        # rerun the click causes, so that rerun already draws the page without
        # the panel and goes back to following the run.
        with st.container(horizontal=True):
            st.button(
                "Continue",
                key=f"{key_prefix}round_continue",
                type="primary",
                icon=":material/play_arrow:",
                on_click=_continue,
                args=(control, [f.key for f in findings], key_prefix),
            )
            st.button("Stop", key=f"{key_prefix}round_stop", icon=":material/stop:", on_click=control.stop)


def _continue(control: Any, keys: list[str], key_prefix: str) -> None:
    accepted = {k for k in keys if st.session_state.get(f"{key_prefix}accept_{k}")}
    control.decide("continue", accepted)

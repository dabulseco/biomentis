"""Sidebar panel for recovering work from a previous run.

The background worker in `biomentis.run_worker` keeps a run alive through a
stray click, but it cannot survive the process itself dying — a crash, a
`streamlit run` restart, a closed laptop. `biomentis.run_journal` writes every
event to disk as it is produced precisely for that case; this module is the
way back in.

Three things can be recovered from a journal, in increasing order of how much
they matter:

  * the transcript, dropped back into the session so the run reads as if it
    had never been interrupted
  * the answer, which is part of that transcript
  * every code block the run generated, as one runnable file — the artifact
    the work was really for

Restoring is deliberately additive-free: it *replaces* the visible transcript
rather than merging, and it does not resume the agent. A restored run is a
record, not a live run; picking the task back up means sending the prompt
again, now with the previous attempt's code in hand.
"""

from __future__ import annotations

from typing import Any

from biomentis.run_journal import code_script, entries_for_run, list_runs, load_run

__all__ = ["render_run_recovery"]

_RESTORED_KEY = "biomni_restored_run_id"


def _describe(summary: dict[str, Any]) -> str:
    """One scannable line: when it ran, how it ended, what it produced."""
    started = (summary.get("started") or "").replace("T", " ")[:16]
    status = {
        "complete": "✅",
        "cancelled": "⏹",
        "error": "⚠️",
        "interrupted": "⚡",
    }.get(summary.get("status", ""), "•")
    bits = [f"{summary.get('events', 0)} events"]
    if summary.get("code_blocks"):
        bits.append(f"{summary['code_blocks']} code")
    if summary.get("has_answer"):
        bits.append("answer")
    return f"{status} {started} · {' · '.join(bits)}"


def render_run_recovery(container: Any, *, run_dir: str | None = None, limit: int = 10) -> None:
    """Render the recovery panel into `container` (usually `st.sidebar`).

    Safe to call on every rerun and safe to call when no runs exist — it
    renders a short explanation instead of an empty list.
    """
    import streamlit as st

    runs = list_runs(run_dir, limit=limit)

    with container.expander("🗂 Recover a previous run", expanded=False):
        if not runs:
            st.caption(
                "Runs are journaled to disk as they happen. Once you have run "
                "a task, it will be listed here — transcript, answer, and "
                "generated code — even if the app was closed mid-run."
            )
            return

        st.caption(
            "Every run is written to disk step by step. Restore one to put its "
            "transcript back on screen, or download the code it generated."
        )

        # Restoring mid-run would replace the transcript the live run is
        # actively appending to.
        run_active = bool(st.session_state.get("biomni_run_active"))
        if run_active:
            st.info("A run is in progress — restoring is disabled until it finishes.")

        for summary in runs:
            run_id = summary["run_id"]
            prompt = (summary.get("prompt") or "").strip() or "(no prompt recorded)"
            st.markdown(f"**{_describe(summary)}**")
            st.caption(prompt[:160] + ("…" if len(prompt) > 160 else ""))

            cols = st.columns(2)
            if cols[0].button(
                "↩ Restore",
                key=f"biomni_restore_{run_id}",
                disabled=run_active,
                help="Replace the on-screen transcript with this run's. Does not restart the agent.",
                use_container_width=True,
            ):
                run = load_run(summary["path"])
                st.session_state.biomni_transcript = entries_for_run(run)
                st.session_state[_RESTORED_KEY] = run_id
                # A restored run is a record, not a live one: make sure the
                # run banner and the background-run cursor agree with that.
                st.session_state.biomni_run_active = False
                st.session_state.biomni_run_started_at = None
                st.session_state.biomni_run_cursor = 0
                st.rerun()

            # Loading the file to build the download payload is cheap next to
            # a run, and it is the only way to hand Streamlit real bytes.
            cols[1].download_button(
                "⬇ Code",
                data=code_script(load_run(summary["path"])),
                file_name=f"biomentis_{run_id}.py",
                mime="text/x-python",
                key=f"biomni_code_{run_id}",
                disabled=not summary.get("code_blocks"),
                help="Every code block this run generated, as one file.",
                use_container_width=True,
            )
            st.divider()

        if st.session_state.get(_RESTORED_KEY):
            st.caption(f"Showing restored run `{st.session_state[_RESTORED_KEY]}`.")

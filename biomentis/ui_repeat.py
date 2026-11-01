"""Sidebar repeat-run tool: run one prompt K times and compare the runs.

An R&D tool over `biomentis.eval.repeat_runs.RepeatBatch`. With
``BIOMNI_REPEAT_RUNS=false`` (``default_config.repeat_runs``, read at render
time) nothing is rendered: no toggle, no expander.

* At most one active batch per process (the state it protects is per
  process), held in a module-level registry keyed by the session id (like
  `run_worker._RUNS`), so it survives reruns and long pauses with the browser
  open. It need not survive an app restart. Finished entries are dropped
  when a new batch starts.
* The button starts a launcher thread: it builds the first agent, resolves
  the rubric once (the uploaded one, else drafted with the agent's
  ``gate_llm or llm``) and starts the batch. The script never blocks on it.
* A batch and an interactive run never overlap, in any session: the button
  is disabled (and the start re-checked) while any interactive run is in
  flight, and `active_batch_session` lets the main page disable Send in
  every session while a batch is active.
* While a batch is active its progress area is a fragment that reruns every
  two seconds; with nothing running there is no auto-refresh.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from biomentis.config import default_config
from biomentis.gate.rubric import Rubric, format_rubric_markdown, prompt_sha, resolve_rubric
from biomentis.ui_gate import render_round_pause

TOGGLE_LABEL = "Show repeat-run tool"
EXPANDER_LABEL = "R&D: repeat this prompt"
BUTTON_LABEL = "Run this prompt K times"
REFRESH_SECONDS = 2


class RepeatEntry:
    """A session's batch: preparing (rubric, first agent), then running, then done."""

    def __init__(self, prompt: str, k: int, batch: Any = None):
        self.prompt = prompt
        self.k = k
        self.batch = batch
        self.rubric: Rubric | None = getattr(batch, "rubric", None)
        self.error = ""  # why the batch could not start
        self.preparing = batch is None

    def active(self) -> bool:
        return self.preparing or (self.batch is not None and not self.batch.done())


# session id -> RepeatEntry. Module-level, so a rerun cannot collect it.
_BATCHES: dict[str, RepeatEntry] = {}
_LOCK = threading.Lock()


def get_entry(session: str) -> RepeatEntry | None:
    with _LOCK:
        return _BATCHES.get(session)


def set_entry(session: str, entry: RepeatEntry) -> None:
    with _LOCK:
        _BATCHES[session] = entry


def clear_entry(session: str) -> None:
    with _LOCK:
        _BATCHES.pop(session, None)


def batch_active(session: str) -> bool:
    """True while this session's batch is preparing or has unfinished runs."""
    entry = get_entry(session)
    return entry is not None and entry.active()


def active_batch_session() -> str | None:
    """The session whose batch is active, in any session of this process, or None.

    The state a batch protects (the REPL namespace, working folder, captured
    plots) is per process, so at most one batch runs per process, and no
    interactive run in any session overlaps it.
    """
    with _LOCK:
        entries = list(_BATCHES.items())
    return next((key for key, entry in entries if entry.active()), None)


def _session_key() -> str:
    # The same key `A1.launch_streamlit_demo` uses for its background run.
    if "biomni_session_key" not in st.session_state:
        st.session_state.biomni_session_key = uuid.uuid4().hex
    return st.session_state.biomni_session_key


def blocked_reason(session: str, own_run_flag: bool = False) -> str:
    """Why a batch cannot start now ("" when it can): a batch or an interactive run anywhere."""
    from biomentis.run_worker import live_run_sessions

    busy = active_batch_session()
    if busy == session:
        return "A batch is running; wait for it to finish."
    if busy is not None:
        return "A repeat-run batch is running in another session; wait for it to finish."
    live = live_run_sessions()
    if own_run_flag or session in live:
        return "An interactive run is in progress; start a batch when it has finished."
    if live:
        return "An interactive run is in progress in another session; start a batch when it has finished."
    return ""


# ------------------------------------------------------------- starting


def _launch(entry: RepeatEntry, upload: str | None, make_agent: Callable[[], Any], settings_label: str,
            out_dir: Path) -> None:
    from biomentis.eval.repeat_runs import RepeatBatch

    try:
        first = make_agent()
        rubric = resolve_rubric(entry.prompt, upload, getattr(first, "gate_llm", None) or first.llm)
        entry.rubric = rubric
        agents = iter([first])

        def factory():
            return next(agents, None) or make_agent()

        settings = f"k={entry.k}, {settings_label}, rubric {rubric.source}"
        try:
            from biomentis.ui_core import run_settings

            described = run_settings(first)
        except Exception:
            described = ""
        if described:
            settings += f"; {described}"
        batch = RepeatBatch(entry.prompt, entry.k, rubric, factory, out_dir, settings=settings)
        batch.start()
        entry.batch = batch
    except Exception as exc:
        entry.error = f"{type(exc).__name__}: {exc}"
    finally:
        entry.preparing = False


def start_batch(session: str, prompt: str, k: int, upload: str | None, make_agent: Callable[[], Any],
                settings_label: str, out_root: Path) -> RepeatEntry | None:
    """Register a new batch for `session` and prepare and start it on a launcher thread.

    Returns None, starting nothing, while a batch is active in any session.
    Finished entries (of every session) are dropped when a new batch starts.
    """
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = Path(out_root) / f"repeat-{stamp}-{prompt_sha(prompt)[:8]}"
    entry = RepeatEntry(prompt, k)
    with _LOCK:  # check and register at once: two sessions cannot both start
        if any(e.active() for e in _BATCHES.values()):
            return None
        _BATCHES.clear()  # all finished
        _BATCHES[session] = entry
    threading.Thread(target=_launch, args=(entry, upload, make_agent, settings_label, out_dir),
                     name="repeat-launcher", daemon=True).start()
    return entry


def _on_start(session: str, upload: str | None, make_agent, settings_label: str, out_root: Path) -> None:
    # Checked again here, not only by the disabled button: the page may be stale.
    prompt = (st.session_state.get("biomentis_repeat_prompt") or "").strip()
    if not prompt or blocked_reason(session, bool(st.session_state.get("biomni_run_active"))):
        return
    k = int(st.session_state.get("biomentis_repeat_k") or 5)
    start_batch(session, prompt, k, upload, make_agent, settings_label, out_root)


# ------------------------------------------------------------- progress


def progress_rows(batch: Any) -> list[dict]:
    """One row per run: run #, status, rounds (rounds used, so far for a paused run)."""
    outcomes = {o.index: o for o in batch.outcomes()}
    paused = {i: c.pending for i, c in batch.paused()}
    states: dict[int, str] = {}
    for line in batch.waiting():  # "Run 3: continued, waiting its turn"
        head, _, state = line.partition(": ")
        try:
            states[int(head.split()[-1])] = state
        except ValueError:
            continue
    rows = []
    for i in range(1, batch.k + 1):
        if i in outcomes:
            status, rounds = outcomes[i].final_status, outcomes[i].rounds_used
        elif paused.get(i):
            round_number = paused[i].get("round")
            status = f"paused at round {round_number}"
            rounds = round_number - 1 if isinstance(round_number, int) else None
        else:
            status, rounds = states.get(i, "not started"), None
        rows.append({"Run #": i, "Status": status, "Rounds": rounds})
    return rows


def _render_progress(session: str, live: bool) -> None:
    entry = get_entry(session)
    if entry is None:
        return
    if live and not entry.active():
        st.rerun()  # the batch just ended: redraw the whole page (button, Send, no more refresh)
    if entry.error:
        st.error(f"The batch could not start: {entry.error}")
    if entry.preparing:
        st.info("Preparing the rubric and the first run…", icon=":material/hourglass_top:")
    if entry.rubric is not None:
        st.caption(f"Rubric ({entry.rubric.source}), the same for every run")
        with st.container(height=220, border=True):
            st.markdown(format_rubric_markdown(entry.rubric))
    batch = entry.batch
    if batch is None:
        return
    st.dataframe(progress_rows(batch), hide_index=True)
    for index, control in batch.paused():
        st.caption(f"Run {index}")
        render_round_pause(control, key_prefix=f"repeat{index}_")
    if batch.done():
        report = getattr(batch, "report", None)
        if report is not None and Path(report).exists():
            st.download_button(
                "Download report",
                data=Path(report).read_bytes(),
                file_name=Path(report).name,
                mime="text/markdown",
                icon=":material/download:",
                key="biomentis_repeat_report",
            )
            st.caption(f"Report: `{report}`")
        if getattr(batch, "report_error", ""):
            st.error(f"The report could not be written: {batch.report_error}")


# --------------------------------------------------------------- render


def render_repeat_runs(make_agent: Callable[[], Any], rubric_upload: str | None, settings_label: str,
                       out_root: Path = Path("runs")) -> None:
    """The sidebar toggle and the "R&D: repeat this prompt" expander.

    `make_agent` builds a fresh agent configured like the interactive one; it
    is called on the batch's threads, so it must not read `st.session_state`.
    """
    if not default_config.repeat_runs:
        return
    st.toggle(
        TOGGLE_LABEL,
        value=True,
        key="biomentis_show_repeat_runs",
        help="An R&D tool: run one prompt several times with one rubric and compare the runs. "
        "Remove it with BIOMNI_REPEAT_RUNS=false.",
    )
    if not st.session_state.get("biomentis_show_repeat_runs", True):
        return
    session = _session_key()
    with st.expander(EXPANDER_LABEL, icon=":material/repeat:"):
        prompt = st.text_area("Prompt to repeat", key="biomentis_repeat_prompt")
        st.number_input("K", min_value=1, max_value=10, value=5, step=1, key="biomentis_repeat_k",
                        help="How many runs of the prompt. They run one at a time.")
        entry = get_entry(session)
        active = entry is not None and entry.active()
        reason = blocked_reason(session, bool(st.session_state.get("biomni_run_active")))
        if not reason and not (prompt or "").strip():
            reason = "Type the prompt to repeat first."
        st.button(
            BUTTON_LABEL,
            key="biomentis_repeat_start",
            icon=":material/play_arrow:",
            disabled=bool(reason),
            on_click=_on_start,
            args=(session, rubric_upload, make_agent, settings_label, Path(out_root).resolve()),
        )
        if reason:
            st.caption(reason)
        st.fragment(_render_progress, run_every=REFRESH_SECONDS if active else None)(session, active)

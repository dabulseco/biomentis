"""Run an agent turn on a background thread so a stray click cannot kill it.

Streamlit re-executes the whole script on every widget interaction, and it
stops the previous execution to do it: `_enqueue_forward_msg` checks for a
pending rerun on *every* `st.*` call and raises `RerunException` when it finds
one. A research run is a plain `for event in stream(...)` loop whose body
calls `st.markdown`, so the first event to arrive after a click tears the loop
down, drops the only reference to the generator, and garbage-collects twenty
minutes of LangGraph state. The click can be anywhere — the sidebar model
picker, an export button, the tutor's chat box.

The fix is to stop running the agent on the script thread. A `BackgroundRun`
owns a worker thread that pulls the event stream and appends each event to a
list; the script thread only *consumes* that list. A rerun still tears down
the consumer, and now that costs nothing: the worker keeps going, and the next
script run re-attaches to the same `BackgroundRun` and picks up at the cursor
it left off. Nothing is lost, not even the step that was in flight.

Two details make this safe rather than merely convenient:

  * The worker inherits the Streamlit script run context. The tutor's
    `_advance_run_live` reads and writes `st.session_state`, so without the
    context a tutor-gated run raises `NoSessionContext` the moment it starts.
    Contexts outlive the script run they were created for, and the session
    state they point at is the live one.
  * Every event is journaled by the worker as it is produced, before any
    consumer sees it. A run whose UI never comes back — browser closed,
    process killed — is still on disk.

Runs are held in a module-level registry keyed by Streamlit session id, so
they survive the rerun that would otherwise drop the last reference to them.

Set `BIOMENTIS_BACKGROUND_RUNS=0` to fall back to running the stream inline on
the script thread (the pre-existing behavior).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from biomentis.run_journal import RunJournal

__all__ = [
    "BackgroundRun",
    "background_runs_enabled",
    "clear_run",
    "get_run",
    "start_run",
]

# session id -> BackgroundRun. Module-level, so a rerun cannot collect it.
_RUNS: dict[str, BackgroundRun] = {}
_REGISTRY_LOCK = threading.Lock()


def background_runs_enabled() -> bool:
    """Whether runs are moved off the script thread. On unless opted out."""
    return os.getenv("BIOMENTIS_BACKGROUND_RUNS", "1").strip().lower() not in ("0", "false", "no")


class BackgroundRun:
    """One agent turn, executing on its own thread.

    The event list only ever grows, and consumers track their own cursor into
    it, so any number of script runs can attach, be torn down, and re-attach
    without losing or repeating an event.
    """

    def __init__(self, prompt: str, journal: RunJournal | None = None):
        self.prompt = prompt
        self.journal = journal
        self.events: list[Any] = []
        self.error: BaseException | None = None
        self.started_at = time.monotonic()
        self.finished_at: float | None = None

        self._condition = threading.Condition()
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # ----- state ---------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def is_alive(self) -> bool:
        return not self._done.is_set()

    def elapsed(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started_at

    @property
    def run_id(self) -> str | None:
        return self.journal.run_id if self.journal is not None else None

    # ----- driving -------------------------------------------------------

    def start(self, stream_factory: Callable[[], Iterator[Any]]) -> BackgroundRun:
        """Spawn the worker. `stream_factory` is called on the worker thread.

        Building the generator on the worker rather than the caller matters
        for the tutor: `_create_run` stores the live generator in session
        state, and doing that from the thread that will drive it keeps
        creation and consumption on one thread.
        """
        thread = threading.Thread(target=self._run, args=(stream_factory,), daemon=True, name="biomentis-run")
        _attach_script_run_ctx(thread)
        self._thread = thread
        thread.start()
        return self

    def _run(self, stream_factory: Callable[[], Iterator[Any]]) -> None:
        status, detail = "complete", None
        stream: Iterator[Any] | None = None
        try:
            stream = stream_factory()
            for event in stream:
                self._append(event)
                if self._cancel.is_set():
                    status = "cancelled"
                    break
        except BaseException as exc:  # noqa: BLE001 - a worker must not die silently
            # Deliberately BaseException: Streamlit's RerunException and
            # StopException both derive from it, and one can reach here if
            # tutor code inside the stream touches a widget. Recording it is
            # far better than a thread that vanishes with no explanation.
            self.error = exc
            status, detail = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException:  # noqa: BLE001
                        pass
            self.finished_at = time.monotonic()
            if self.journal is not None:
                self.journal.finish(status, detail)
            self._done.set()
            with self._condition:
                self._condition.notify_all()

    def _append(self, event: Any) -> None:
        # Journal first: an event that reaches the disk but not the screen is
        # recoverable, an event that reaches neither is not.
        if self.journal is not None:
            self.journal.append(event)
        with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def cancel(self) -> None:
        """Ask the worker to stop after the event it is currently producing.

        There is no way to interrupt a running LLM call or a code execution
        mid-flight, so this is cooperative: the run ends at the next event
        boundary. Everything produced up to that point is kept.
        """
        self._cancel.set()
        with self._condition:
            self._condition.notify_all()

    # ----- consuming -----------------------------------------------------

    def events_from(self, index: int, poll_seconds: float = 0.25) -> Iterator[Any]:
        """Yield events from `index` onward, blocking until the run ends.

        The caller is expected to be torn down mid-iteration — that is the
        whole point — so it must persist its cursor after handling each
        event, not after the loop.
        """
        while True:
            with self._condition:
                while index >= len(self.events) and not self._done.is_set():
                    self._condition.wait(poll_seconds)
                if index >= len(self.events):
                    return  # finished, and the caller has seen everything
                event = self.events[index]
            index += 1
            yield event

    def pending_from(self, index: int) -> int:
        """How many produced-but-unconsumed events sit after `index`."""
        return max(0, len(self.events) - index)


# ----- registry -----------------------------------------------------------


def start_run(
    session_key: str,
    prompt: str,
    stream_factory: Callable[[], Iterator[Any]],
    *,
    journal: RunJournal | None = None,
) -> BackgroundRun:
    """Register and start a run for `session_key`, replacing any previous one."""
    run = BackgroundRun(prompt, journal=journal)
    with _REGISTRY_LOCK:
        previous = _RUNS.get(session_key)
        if previous is not None and previous.is_alive():
            previous.cancel()
        _RUNS[session_key] = run
    return run.start(stream_factory)


def get_run(session_key: str) -> BackgroundRun | None:
    with _REGISTRY_LOCK:
        return _RUNS.get(session_key)


def clear_run(session_key: str) -> None:
    """Forget the session's run once its events have all been consumed."""
    with _REGISTRY_LOCK:
        _RUNS.pop(session_key, None)


def _attach_script_run_ctx(thread: threading.Thread) -> None:
    """Give the worker the current Streamlit script run context, if any.

    Without it, `st.session_state` from the worker raises — which the tutor
    layer does on every step. With it, the worker reads and writes the same
    session state the script does. Absent Streamlit (tests, CLI) this is a
    no-op and the worker runs as a plain thread.
    """
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(thread, ctx)
    except Exception:
        pass

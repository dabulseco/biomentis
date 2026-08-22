"""Durable, append-only record of a single agent turn's UI events.

A run is 20 minutes of expensive work — LLM calls, executed code, the
observations that came back — and until now none of it existed anywhere but
`st.session_state.biomni_transcript`, which dies with the browser session.
`biomentis.eval.step_trace` writes diagnostics, but it truncates code at 4000
chars, observations at 2000, and never stores a solution; it answers "what
went wrong", not "what did we produce".

This module answers the second question. Every `UIEvent` is written to
`<run_dir>/<run_id>.jsonl` the instant the agent produces it, by the worker
that produced it — not by the UI consuming it — so a run is recoverable even
if nothing ever renders it. Records are one JSON object per line, flushed by
closing the handle per write, so a killed process still leaves a usable file.

Reading it back gives three things:

  * `load_run` / `list_runs` — the events of a past run, newest first
  * `entries_for_run` — those events as transcript entries, ready to drop
    straight back into `st.session_state.biomni_transcript`
  * `code_script` — every code block the run generated, concatenated into one
    runnable file, which is the artifact most worth keeping

Nothing here imports Streamlit; the journal is written from a worker thread
and read from a CLI just as happily.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RunJournal",
    "journaled",
    "code_script",
    "default_run_dir",
    "entries_for_run",
    "event_from_record",
    "list_runs",
    "load_run",
]


def default_run_dir() -> str:
    """Where run journals are written. Override with `BIOMENTIS_RUN_DIR`."""
    return os.getenv("BIOMENTIS_RUN_DIR", "runs")


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of an event field to something JSON can hold.

    Instruction/roadmap cards are dataclasses; keeping their fields means a
    restored transcript can re-render the real teaching card instead of
    falling back to plain markdown.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return {"__card__": type(value).__name__, **asdict(value)}
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


# The UIEvent fields worth persisting, plus the two the tutor layer attaches
# to instruction-bearing events after construction.
_EVENT_FIELDS = (
    "type",
    "content",
    "channel",
    "title",
    "language",
    "status",
    "duration",
    "file_path",
    "file_kind",
    "collapsible",
    "step_id",
    "run_id",
    "card",
)


def event_to_record(event: Any) -> dict[str, Any]:
    """Flatten a `UIEvent` (or anything with its attributes) to a dict."""
    return {
        field: _jsonable(getattr(event, field, None))
        for field in _EVENT_FIELDS
        if getattr(event, field, None) is not None
    }


class _RestoredEvent:
    """A `UIEvent` look-alike rebuilt from a journal record.

    Deliberately not the real `UIEvent`: that dataclass has a fixed field
    list, and a journal written by a newer version may carry fields this one
    doesn't know. Attribute access with a `None` default keeps the render
    dispatch working either way.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        self._data = dict(data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._data.get(name)

    def __repr__(self) -> str:
        return f"_RestoredEvent({self._data.get('type')!r})"


def event_from_record(record: dict[str, Any]) -> Any:
    """Rebuild an event-shaped object from a journal `event` record."""
    return _RestoredEvent(record)


class RunJournal:
    """Writes one JSONL file for one agent turn.

    Every method swallows its own IO errors: a journal that cannot be written
    must never take down the run it is recording.
    """

    def __init__(
        self,
        prompt: str,
        run_dir: str | os.PathLike[str] | None = None,
        *,
        enabled: bool = True,
        **meta: Any,
    ):
        self.enabled = enabled
        self.prompt = prompt
        self.seq = 0
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_sha(prompt or 'run')[:6]}"
        self._path: Path | None = None
        self._started = datetime.now()
        self._meta = {k: _jsonable(v) for k, v in meta.items()}

        if not enabled:
            return
        try:
            directory = Path(run_dir or default_run_dir())
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / f"{self.run_id}.jsonl"
        except Exception as exc:
            print(f"[run_journal] could not open journal: {exc}")
            self._path = None
            return

        self._write({"type": "run_start", "prompt": prompt, **self._meta})

    @property
    def path(self) -> str | None:
        return str(self._path) if self._path is not None else None

    def append(self, event: Any) -> None:
        """Record one UIEvent. Called from the worker as the event is produced."""
        self.seq += 1
        self._write({"type": "event", "seq": self.seq, "event": event_to_record(event)})

    def finish(self, status: str, detail: str | None = None) -> None:
        self._write(
            {
                "type": "run_end",
                "status": status,
                "detail": detail,
                "events": self.seq,
                "duration_s": round((datetime.now() - self._started).total_seconds(), 2),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        record.setdefault("run_id", self.run_id)
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        try:
            # Open/append/close per record: the close is the flush, which is
            # what makes a killed process still leave a readable journal.
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            print(f"[run_journal] could not write record: {exc}")


def journaled(stream: Any, journal: RunJournal | None) -> Any:
    """Wrap an event stream so every event is written as it passes through.

    The background worker journals events itself. This is for the inline path
    (`BIOMENTIS_BACKGROUND_RUNS=0`), so that opting out of the worker does not
    also opt out of durability — those are separate concerns, and the second
    one is worth having either way.
    """
    if journal is None:
        yield from stream
        return

    status, detail = "complete", None
    try:
        for event in stream:
            journal.append(event)
            yield event
    except BaseException as exc:  # noqa: BLE001 - record, then let it propagate
        status, detail = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        journal.finish(status, detail)


# ----- reading ------------------------------------------------------------


def load_run(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one journal file into `{run_id, prompt, meta, events, status, ...}`.

    Unparseable lines are skipped rather than raising — a journal truncated
    mid-write by a hard kill is still worth most of its content.
    """
    meta: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    end: dict[str, Any] = {}

    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = record.get("type")
                if kind == "run_start":
                    meta = record
                elif kind == "event":
                    events.append(record.get("event") or {})
                elif kind == "run_end":
                    end = record
    except OSError as exc:
        print(f"[run_journal] could not read {path}: {exc}")

    return {
        "path": str(path),
        "run_id": meta.get("run_id") or end.get("run_id") or Path(path).stem,
        "prompt": meta.get("prompt", ""),
        "started": meta.get("ts", ""),
        "meta": {k: v for k, v in meta.items() if k not in ("type", "prompt", "ts", "run_id")},
        "events": events,
        # A journal with no run_end was interrupted — the process died, or the
        # run is still going. Either way it is the interesting case.
        "status": end.get("status", "interrupted"),
        "detail": end.get("detail"),
        "duration_s": end.get("duration_s"),
    }


def list_runs(run_dir: str | os.PathLike[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Summarize the most recent journals, newest first."""
    directory = Path(run_dir or default_run_dir())
    if not directory.is_dir():
        return []

    paths = sorted(directory.glob("*.jsonl"), reverse=True)[:limit]
    summaries = []
    for path in paths:
        run = load_run(path)
        summaries.append(
            {
                "run_id": run["run_id"],
                "path": run["path"],
                "prompt": run["prompt"],
                "started": run["started"],
                "status": run["status"],
                "duration_s": run["duration_s"],
                "events": len(run["events"]),
                "code_blocks": sum(1 for e in run["events"] if e.get("type") == "code"),
                "has_answer": any(e.get("type") in ("solution", "summary") for e in run["events"]),
            }
        )
    return summaries


def entries_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a loaded run's events into transcript entries.

    Uses the same mapping the live UI uses, so a restored run renders
    identically to the run that produced it.
    """
    from biomentis.ui_core import transcript_entry_for_event

    entries: list[dict[str, Any]] = []
    code_entries: list[dict[str, Any]] = []
    if run.get("prompt"):
        entries.append({"panel": "main", "role": "user", "content": run["prompt"]})
    for record in run.get("events", []):
        entry = transcript_entry_for_event(event_from_record(record), code_entries)
        if entry is not None:
            entries.append(entry)
    return entries


def code_script(run: dict[str, Any]) -> str:
    """Every code block the run generated, as one annotated file.

    This is the artifact the work is really for: the generated code is the
    exercise, and it should outlive the browser tab that displayed it.
    """
    blocks = [e for e in run.get("events", []) if e.get("type") == "code"]
    header = [
        "# Code generated by Biomentis",
        f"# run: {run.get('run_id', '?')}    started: {run.get('started', '?')}",
        f"# task: {(run.get('prompt') or '').strip()[:500]}",
        f"# {len(blocks)} code block(s)",
        "",
    ]
    if not blocks:
        return "\n".join([*header, "# (this run generated no code)", ""])

    parts = ["\n".join(header)]
    for index, block in enumerate(blocks, start=1):
        language = block.get("language") or "python"
        body = block.get("content") or ""
        parts.append(f"# ===== block {index} ({language}) =====")
        if language != "python":
            # Keep it valid Python: a non-python block is preserved verbatim
            # inside a string rather than silently dropped or left to a syntax
            # error halfway down the file.
            parts.append(f'_BLOCK_{index}_{language.upper()} = r"""\n{body}\n"""')
        else:
            parts.append(body)
        parts.append("")
    return "\n".join(parts)

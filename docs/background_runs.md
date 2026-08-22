# Surviving a stray click

A run used to die if you clicked anything.

Streamlit re-executes the whole script on every widget interaction, and it
stops the previous execution to do it: `ScriptRunner._enqueue_forward_msg`
checks for a pending rerun on **every** `st.*` call and raises
`RerunException` when it finds one. A research run was a plain
`for event in stream(...)` loop whose body called `st.markdown`, so the first
event to arrive after a click tore the loop down, dropped the only reference
to the generator, and garbage-collected the LangGraph state behind it.

The click could be anywhere — the sidebar model picker, an Export button, the
tutor's chat box, a transcript tab. Twenty minutes of work, gone, and because
`RerunException` derives from `BaseException` (so user code cannot swallow it)
the `except Exception` around the loop never fired, leaving the banner
claiming "⏳ Running" forever. It read as a freeze rather than a crash.

Two things now stand between a run and that outcome.

## 1. The run is not on the script thread

`biomentis/run_worker.py`

A `BackgroundRun` owns a worker thread that pulls the event stream and appends
each event to a list. The script thread only *consumes* that list. A rerun
still tears down the consumer, and now that costs nothing: the worker keeps
going, and the next script execution re-attaches to the same `BackgroundRun`
and resumes at the cursor it left off.

```
submit ──> start_run(session_key, prompt, stream_factory)
              │
              ├─ worker thread: for event in stream(): journal it, append it
              │
              └─ script thread: for event in run.events_from(cursor): render it
                                    ^
                    stray click ────┘  torn down here, resumes here on the
                                       next script execution
```

Runs live in a module-level registry keyed by Streamlit session id, because a
run outlives the script execution that started it and cannot be held in a
local. The cursor (`st.session_state.biomni_run_cursor`) is how many of the
run's events are already in the transcript; it is what makes re-attaching
exact rather than approximate — no gaps, no repeats.

Two details make this safe rather than merely convenient:

- **The worker inherits the script run context** (`add_script_run_ctx`). The
  tutor's `_advance_run_live` reads and writes `st.session_state`, so without
  it a tutor-gated run would raise `NoSessionContext` immediately. Contexts
  outlive the script run that created them, and the session state they point
  at is the live one.
- **Every event is journaled by the worker as it is produced**, before any
  consumer sees it. A run whose UI never comes back is still on disk.

### What changed for the user

- Clicking around during a run is safe. The run banner says so.
- Because a click no longer stops a run, there is a **Stop run** button. It is
  cooperative: the run ends at the next event boundary, since there is no way
  to interrupt an LLM call or an executing block mid-flight. Everything
  produced up to that point is kept.
- Starting a second task while one is in flight is refused rather than
  silently racing the first for the agent object.
- `streamlit_app.py` skips `agent.configure()` while a run is active — it
  rebuilds the system prompt and recompiles the graph, and a rerun can now
  land in the middle of a run.

Set `BIOMENTIS_BACKGROUND_RUNS=0` to go back to running the stream inline on
the script thread.

### Both instructional modes, and neither

The worker sits at the one point every run passes through — the event loop in
`A1.launch_streamlit_demo` — so it covers research mode, a tutor-gated
walkthrough, and the tutor installed but switched off, without any of them
knowing about it. A tutor step is just a run that ends at its pause gate
instead of at `complete`; the registry is freed when the step's worker
finishes, so the next Continue click starts a fresh one.

## 2. Every step is on disk before you see it

`biomentis/run_journal.py`

One JSONL file per run, at `runs/<run_id>.jsonl` (`BIOMENTIS_RUN_DIR` to move
it). One JSON object per line, written by opening/appending/closing per record
— the close is the flush, which is what makes a killed process still leave a
readable file.

```
{"type":"run_start","run_id":"20260820-074358-e872c7","prompt":"...","mode":"research",...}
{"type":"event","seq":1,"event":{"type":"status","content":"..."}}
{"type":"event","seq":2,"event":{"type":"code","content":"import pandas...","language":"python"}}
{"type":"run_end","status":"complete","events":42,"duration_s":1284.6}
```

A journal with no `run_end` reads back as `interrupted`, which is exactly the
case worth finding after a crash.

This is deliberately *not* what `biomentis/eval/step_trace.py` records. That
writes diagnostics — it truncates code at 4000 chars, observations at 2000,
and never stores a solution, because it answers "what went wrong". The journal
answers "what did we produce", so it stores events whole.

### Recovering

The sidebar's **🗂 Recover a previous run** panel lists recent runs with when
they ran, how they ended, and what they produced:

- **↩ Restore** puts the run's transcript back on screen. It replaces the
  visible transcript and does not restart the agent — a restored run is a
  record, not a live run.
- **⬇ Code** downloads every code block the run generated as one runnable
  `.py` file, with the task and per-block headers. Non-Python blocks (R, bash)
  are preserved verbatim inside a string so the file still parses.

Reading a journal in code:

```python
from biomentis.run_journal import code_script, entries_for_run, list_runs, load_run

list_runs()                      # newest first, with status and counts
run = load_run("runs/20260820-074358-e872c7.jsonl")
entries_for_run(run)             # transcript entries, ready to render
print(code_script(run))          # every code block, as one file
```

`entries_for_run` goes through `ui_core.transcript_entry_for_event`, the same
mapping the live UI uses, so a restored run renders identically to the run
that produced it instead of drifting the first time an event type is added.

## Related fix: UI runs were never traced

`StepTrace` is on by default, but `start_run` — which is what gives the tracer
a file to write to — was only called from `A1.go` and `A1.go_stream`. The
Streamlit path goes through `ui_core.stream_agent_events`, which drives
`agent.app.stream` directly, so **no UI run had ever been traced**: the graph
dutifully recorded every step into a tracer with no path, and every
`record_execute` early-returned. `stream_agent_events` now opens and closes
the trace itself, so `python -m biomentis.eval.step_trace report traces/`
finally has app runs to report on.

## Tests

```bash
python -m biomentis.agent.tests.test_background_run
```

Covers the worker (produces with no consumer attached, exact re-attach after a
teardown, cancel, error capture), the journal (readable mid-run, interrupted
status, code recovery, restore parity with the live mapping), and the real
`launch_streamlit_demo` render loop under `AppTest` — submitting a prompt,
re-attaching to an in-flight run at a cursor, and a tutor step ending at its
gate.

## Known limits

- A run is tied to a browser session. Reload into a *new* session and the old
  worker is orphaned: it runs to completion and journals everything, but the
  new session will not re-attach to it. Recover it from the journal.
- Cancel and teardown both land at event boundaries. A single step that takes
  ten minutes takes ten minutes to stop.
- The worker and the script thread both touch `st.session_state`. They write
  largely disjoint keys, and Streamlit's session state is not lock-protected;
  this has not bitten in practice but it is the place to look if something
  ever races.

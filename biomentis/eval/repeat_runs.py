"""Repeat runs: one prompt run K times with one frozen rubric, the runs compared.

An R&D tool for seeing how much a run's answer varies. ``BIOMNI_REPEAT_RUNS=false``
disables it (the CLI then exits with code 2).

How a batch runs:

* One rubric for all runs: every agent gets the identical ``Rubric`` object
  as ``agent.preset_rubric``, so its ``draft_rubric`` node uses it as is. The
  rubric is saved once in the batch folder.
* Every run gets a fresh agent from ``agent_factory`` and its own absolute
  output folder, ``<out_dir>/run-01/``, ``run-02/``…
* Each run executes ``agent.go(prompt)`` on its own thread, but only one run
  executes at a time: the batch owns an execution token (a lock). A run holds
  it while it works and gives it up only while paused. Each agent gets a
  ``RoundControl`` subclass whose ``wait`` releases the token before blocking
  and, on Continue, takes it back (the run is "continued, waiting its turn"
  until then). On Stop it does not take it back, so a stopped run ends at once.
  The supervisor thread takes the token before starting each run, so the next
  run starts when the previous one has finished or paused.
  The token protects the process-wide state A1's code steps use: the working
  folder (``workspace.working_in``), ``structures`` current run ID, captured
  plots and the shared REPL namespace.
* REPL namespace: a fresh run starts with an empty namespace; a run that
  pauses keeps a (shallow) copy of its namespace, restored when it resumes; the
  namespace from before the batch is restored when the batch finishes. On each
  hand-over the captured plots are cleared and the structures run ID reset.
* A paused run waits on its thread for the user's Continue or Stop for as long
  as it takes; the batch never times it out.
* Each time a run finishes and others are not, the partial report
  (``reports/<date>-repeat-<sha8>-partial.md``) is rewritten. When the last
  run finishes, the final report (``reports/<date>-repeat-<sha8>.md``) is
  written, the partial report is removed, and ``done()`` becomes true. If
  writing the final report fails, ``done()`` still becomes true and the error
  is kept in ``report_error``.

CLI::

    python -m biomentis.eval.repeat_runs --prompt-file P --k 5 [--rubric R] [--rounds 1] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from biomentis.gate.control import RoundControl
from biomentis.gate.rubric import Rubric, prompt_sha, resolve_rubric, save_rubric
from biomentis.eval.repeat_report import (
    RunOutcome,
    coverage_marks,
    report_path,
    solution_text,
    write_report,
)

__all__ = ["RepeatBatch", "RunOutcome", "write_report", "run_pause_loop", "main"]

DISABLED_MESSAGE = "Repeat runs are disabled (BIOMNI_REPEAT_RUNS=false)."


def _telemetry(agent: Any) -> str:
    """The run-quality summary of a real A1 run; "" when there is nothing to summarise."""
    state = getattr(agent, "_conversation_state", None) or {}
    messages = state.get("messages") if isinstance(state, dict) else None
    if not messages:
        return ""
    try:
        from biomentis.run_quality import format_quality_summary, run_evidence
        from biomentis.ui_core import gate_summary, run_settings

        return format_quality_summary(run_evidence(messages), run_settings(agent), gate=gate_summary(agent))
    except Exception as exc:  # telemetry must never lose a run's outcome
        return f"(telemetry unavailable: {type(exc).__name__}: {exc})"


def _repl_namespace() -> dict:
    """The REPL namespace A1's code steps run in (one dict for the process)."""
    from biomentis.tool.support_tools import _persistent_namespace

    return _persistent_namespace


def _reset_step_state() -> None:
    """Clear the per-step process state a run hands over: captured plots, structures run ID."""
    from biomentis.structures import set_current_run_id
    from biomentis.tool.support_tools import clear_captured_plots

    clear_captured_plots()
    set_current_run_id(None)


class _BatchRoundControl(RoundControl):
    """A run's pause control that gives the batch's execution token up while paused."""

    def __init__(self, batch: RepeatBatch, index: int) -> None:
        super().__init__()
        self._batch = batch
        self._index = index

    def wait(self, poll_seconds: float = 0.5) -> str:
        self._batch._park(self._index)
        choice = super().wait(poll_seconds)
        if choice == "continue":
            self._batch._resume(self._index)
        return choice  # on "stop" the token is not taken back: the run ends at once


class RepeatBatch:
    """K runs of one prompt; see the module docstring."""

    def __init__(
        self,
        prompt: str,
        k: int,
        rubric: Rubric,
        agent_factory: Callable[[], Any],
        out_dir: Path,
        settings: str = "",
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.prompt = prompt
        self.k = k
        self.rubric = rubric
        self.agent_factory = agent_factory
        self.out_dir = Path(out_dir).resolve()
        self.settings = settings
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.sha = rubric.prompt_sha or prompt_sha(prompt)
        self.report: Path | None = None  # the final report, once written
        self.report_error: str = ""  # why the final report could not be written
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._token = threading.Lock()  # the execution token: one run works at a time
        self._holder: int | None = None  # the run holding the token
        self._turn_waiting: set[int] = set()  # continued runs waiting for the token
        self._snapshots: dict[int, dict] = {}  # namespaces of parked runs
        self._original_namespace: dict | None = None
        self._agents: dict[int, Any] = {}  # unfinished runs only: dropped once the outcome is in
        self._launched: set[int] = set()  # runs whose agent was built
        self._outcomes: dict[int, RunOutcome] = {}
        self._done = threading.Event()
        self._started = False

    # ------------------------------------------------------------ public

    def start(self) -> None:
        """Start the batch and return at once; runs go one after another (see module docstring)."""
        if self._started:
            raise RuntimeError("the batch has already started")
        self._started = True
        save_rubric(self.rubric, self.out_dir)
        self._original_namespace = dict(_repl_namespace())
        threading.Thread(target=self._supervise, name="repeat-supervisor", daemon=True).start()

    def paused(self) -> list[tuple[int, RoundControl]]:
        """(run index, its RoundControl) for each run waiting for Continue or Stop."""
        with self._lock:
            running = [(i, a) for i, a in sorted(self._agents.items()) if i not in self._outcomes]
        found = []
        for i, agent in running:
            control = getattr(agent, "round_control", None)
            if control is not None and control.pending is not None:
                found.append((i, control))
        return found

    def done(self) -> bool:
        """True once all K runs have finished and the final report is written (or failed)."""
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until done() or `timeout` seconds; return done()."""
        return self._done.wait(timeout)

    def outcomes(self) -> list[RunOutcome]:
        """The finished runs, in run order."""
        with self._lock:
            return [self._outcomes[i] for i in sorted(self._outcomes)]

    def waiting(self) -> list[str]:
        """One line per unfinished run: paused, continued and waiting its turn, running, not started."""
        paused = {i: c.pending for i, c in self.paused()}
        with self._lock:
            started = set(self._launched)
            finished = set(self._outcomes)
            turn = set(self._turn_waiting)
        lines = []
        for i in range(1, self.k + 1):
            if i in finished:
                continue
            if paused.get(i):
                lines.append(f"Run {i}: paused at round {paused[i]['round']}, waiting for Continue or Stop")
            elif i in turn:
                lines.append(f"Run {i}: continued, waiting its turn")
            elif i in started:
                lines.append(f"Run {i}: running")
            else:
                lines.append(f"Run {i}: not started")
        return lines

    # ---------------------------------------------------- execution token

    def _take_token(self, index: int, namespace: dict | None) -> None:
        """Block for the token, then set the process state up for run `index`."""
        self._token.acquire()
        with self._lock:
            self._holder = index
            self._turn_waiting.discard(index)
        ns = _repl_namespace()
        ns.clear()
        ns.update(namespace or {})
        _reset_step_state()

    def _give_token(self, index: int) -> None:
        with self._lock:
            if self._holder != index:
                return
            self._holder = None
        self._token.release()

    def _park(self, index: int) -> None:
        """Run `index` is pausing: keep its namespace and give the token up."""
        with self._lock:
            holding = self._holder == index
        if holding:
            self._snapshots[index] = dict(_repl_namespace())
            self._give_token(index)

    def _resume(self, index: int) -> None:
        """Run `index` was continued: wait its turn, then restore its namespace."""
        with self._lock:
            self._turn_waiting.add(index)
        self._take_token(index, self._snapshots.pop(index, None))

    # ----------------------------------------------------------- running

    def _supervise(self) -> None:
        for index in range(1, self.k + 1):
            self._take_token(index, None)  # a fresh run starts with an empty namespace
            try:
                agent = self.agent_factory()
                agent.preset_rubric = self.rubric
                agent.output_dir = str(self.out_dir / f"run-{index:02d}")
                agent.round_control = _BatchRoundControl(self, index)  # reset_gate keeps it
                agent.pause_allowed = True  # each run has its own thread, so it can wait for an answer
            except Exception as exc:
                self._give_token(index)
                self._finish(self._failed(index, exc))
                continue
            with self._lock:
                self._agents[index] = agent
                self._launched.add(index)
            threading.Thread(target=self._run, args=(index, agent), name=f"repeat-run-{index}",
                             daemon=True).start()

    def _failed(self, index: int, exc: BaseException) -> RunOutcome:
        return RunOutcome(index=index, run_id=f"run-{index:02d}",
                          final_status=f"⚠️ Run failed ({type(exc).__name__}: {exc})",
                          rounds_used=0, gate_results=[], final_answer="", telemetry="")

    def _run(self, index: int, agent: Any) -> None:
        try:
            _, content = agent.go(self.prompt)
            results = list(getattr(agent, "run_gate_results", None) or [])
            outcome = RunOutcome(
                index=index,
                run_id=str(getattr(getattr(agent, "tracer", None), "run_id", None) or f"run-{index:02d}"),
                final_status=getattr(agent, "final_status", "") or "(no status)",
                rounds_used=results[-1].round if results else 0,
                gate_results=results,
                final_answer=solution_text(content),
                telemetry=_telemetry(agent),
                coverage=coverage_marks(self.rubric, getattr(agent, "run_reviews", None)),
                claims=list(getattr(agent, "_gate_prev_claims", None) or []),
            )
        except Exception as exc:  # one broken run must not stop the batch
            outcome = self._failed(index, exc)
            outcome.gate_results = list(getattr(agent, "run_gate_results", None) or [])
        try:
            self._finish(outcome)
        finally:
            self._give_token(index)

    def _finish(self, outcome: RunOutcome) -> None:
        with self._lock:
            self._outcomes[outcome.index] = outcome
            self._agents.pop(outcome.index, None)  # the outcome has all the report needs
            all_done = len(self._outcomes) == self.k
        with self._write_lock:
            if self._done.is_set():
                return
            outcomes = self.outcomes()
            if not all_done:
                try:
                    write_report(self.prompt, self.rubric, self.settings, outcomes, self.out_dir, partial=True,
                                 waiting=self.waiting(), date=self.date, total_runs=self.k)
                except Exception as exc:  # a partial report is a convenience; the runs go on
                    print(f"repeat runs: could not write the partial report: {type(exc).__name__}: {exc}")
                return
            try:
                self.report = write_report(self.prompt, self.rubric, self.settings, outcomes, self.out_dir,
                                           date=self.date, total_runs=self.k)
                report_path(self.out_dir, self.sha, self.date, partial=True).unlink(missing_ok=True)
            except Exception as exc:
                self.report_error = f"{type(exc).__name__}: {exc}"
            finally:
                if self._original_namespace is not None:
                    ns = _repl_namespace()
                    ns.clear()
                    ns.update(self._original_namespace)
                self._done.set()


# --------------------------------------------------------------------- CLI


def pause_line(index: int, pending: dict) -> str:
    """The question asked on stdin when a run pauses."""
    blocking = [f for f in pending.get("findings", []) if getattr(f, "severity", "") == "blocking"]
    what = "; ".join(f.issue for f in blocking) or (
        "the context window is nearly full" if pending.get("reason") == "context" else "no blocking findings"
    )
    return f"Run {index} paused at round {pending['round']}: {what}. Answer [c]ontinue / [s]top: "


def run_pause_loop(batch: RepeatBatch, ask: Callable[[str], str] = input,
                   out: Callable[[str], Any] = print, poll: float = 0.5) -> None:
    """Until the batch is done, ask on stdin about each paused run and pass the answer on.

    Unrecognised input asks again; end of input (EOF) stops the run, since
    nobody is left to answer.
    """
    while not batch.done():
        for index, control in batch.paused():
            pending = control.pending
            if pending is None:
                continue
            question = pause_line(index, pending)
            while True:
                try:
                    answer = ask(question).strip().lower()
                except EOFError:
                    out(f"No input: stopping Run {index}.")
                    control.stop()
                    break
                if answer in ("c", "continue"):
                    control.decide("continue")
                    break
                if answer in ("s", "stop"):
                    control.stop()
                    break
                out("Please answer c (continue) or s (stop).")
        batch.wait(poll)


def cli_reviewer(review_model: str, agent_model: str, build: Callable[..., Any] | None = None,
                 out: Callable[[str], Any] = print) -> tuple[Any, str]:
    """(review_llm, review_model_name) for the CLI's runs, built like the app's default reviewer.

    Says which reviewer the runs get, or that there is none (so they will be
    "Not fully checked"), and warns when it is the agent's model family.
    """
    from biomentis.agent_setup import REVIEWER_NONE_MESSAGE, SAME_FAMILY_WARNING, build_reviewer
    from biomentis.independent_review import same_model_family

    review_llm, error = (None, "") if not review_model else build_reviewer(review_model, build)
    if review_llm is None:
        if error:
            out(f"Reviewer {review_model} could not be loaded ({error}).")
        out(REVIEWER_NONE_MESSAGE)
        return None, ""
    out(f"Reviewer: {review_model}")
    if same_model_family(agent_model, review_model):
        out(SAME_FAMILY_WARNING)
    return review_llm, review_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m biomentis.eval.repeat_runs",
                                description="Run one prompt K times with one rubric and compare the runs.")
    p.add_argument("--prompt-file", required=True, type=Path, help="text file with the prompt")
    p.add_argument("--k", type=int, default=5, help="number of runs (default 5)")
    p.add_argument("--rubric", type=Path, default=None,
                   help="rubric file (any format; converted if not standard). Default: drafted by the agent")
    p.add_argument("--rounds", type=int, default=1, help="planned self-critique rounds (default 1; 0 = none)")
    p.add_argument("--out", type=Path, default=None,
                   help="batch folder (default ./runs/repeat-<date>-<sha8>/)")
    return p


def main(argv: list[str] | None = None) -> int:
    from biomentis.config import BiomentisConfig

    if not BiomentisConfig().repeat_runs:  # read now, not at import
        print(DISABLED_MESSAGE, file=sys.stderr)
        return 2
    args = _parser().parse_args(argv)
    if args.k < 1:
        print("--k must be at least 1", file=sys.stderr)
        return 2
    prompt = args.prompt_file.read_text(encoding="utf-8")
    sha = prompt_sha(prompt)
    out_dir = (args.out or Path("runs") / f"repeat-{datetime.now().strftime('%Y-%m-%d')}-{sha[:8]}").resolve()
    rounds = max(0, args.rounds)

    from biomentis.agent.a1 import A1

    cfg = BiomentisConfig()
    review_llm, review_model_name = cli_reviewer(cfg.review_model, cfg.llm)

    def make_agent():
        agent = A1()  # models and settings from default_config / env, as other entry points do
        agent.review_llm = review_llm
        agent.review_model_name = review_model_name
        agent.configure(self_critic=rounds > 0, test_time_scale_round=rounds, completion_gate=True)
        return agent  # RepeatBatch allows each run to pause

    first = make_agent()
    upload = args.rubric.read_text(encoding="utf-8") if args.rubric else None
    try:
        rubric = resolve_rubric(prompt, upload, getattr(first, "gate_llm", None) or first.llm)
    except Exception as exc:
        print(f"Could not get the rubric: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    agents = iter([first])

    def factory():
        return next(agents, None) or make_agent()

    from biomentis.ui_core import run_settings

    settings = f"k={args.k}, rounds={rounds}, completion gate on, rubric {rubric.source}"
    described = run_settings(first)
    if described:
        settings += f"; {described}"
    batch = RepeatBatch(prompt, args.k, rubric, factory, out_dir, settings=settings)
    print(f"Repeat runs: {args.k} runs of prompt {sha[:8]} into {batch.out_dir}")
    batch.start()
    run_pause_loop(batch)
    if batch.report_error:
        print(f"The report could not be written: {batch.report_error}", file=sys.stderr)
        return 1
    print(f"Report: {batch.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

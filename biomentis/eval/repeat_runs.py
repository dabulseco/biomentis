"""Repeat runs: one prompt run K times with one frozen rubric, the runs compared.

An R&D tool for seeing how much a run's answer varies. ``BIOMNI_REPEAT_RUNS=false``
disables it (the CLI then exits with code 2).

How a batch runs:

* One rubric for all runs: every agent gets the identical ``Rubric`` object
  as ``agent.preset_rubric``, so its ``draft_rubric`` node uses it as is. The
  rubric is saved once in the batch folder.
* Every run gets a fresh agent from ``agent_factory`` and its own output
  folder, ``<out_dir>/run-01/``, ``run-02/``…
* Each run executes ``agent.go(prompt)`` on its own thread. A supervisor
  thread starts Run 1, and starts the next run once every started run has
  finished or is paused (``round_control.pending`` set). A paused run waits
  on its thread for the user's Continue or Stop for as long as it takes; the
  batch never times it out.
* Each time a run finishes and others are not, the partial report
  (``reports/<date>-repeat-<sha8>-partial.md``) is rewritten. When the last
  run finishes, the final report (``reports/<date>-repeat-<sha8>.md``) is
  written, the partial report is removed, and ``done()`` becomes true.

Note: a run continued after a pause runs alongside whichever run started
meanwhile. A1's code steps change the process-wide working folder, so two
runs executing code at the same moment can see each other's folder.

CLI::

    python -m biomentis.eval.repeat_runs --prompt-file P --k 5 [--rubric R] [--rounds 1] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
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
POLL_SECONDS = 0.1


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
        self.out_dir = Path(out_dir)
        self.settings = settings
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.sha = rubric.prompt_sha or prompt_sha(prompt)
        self.report: Path | None = None  # the final report, once written
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._agents: dict[int, Any] = {}
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
        """True once all K runs have finished and the final report is written."""
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until done() or `timeout` seconds; return done()."""
        return self._done.wait(timeout)

    def outcomes(self) -> list[RunOutcome]:
        """The finished runs, in run order."""
        with self._lock:
            return [self._outcomes[i] for i in sorted(self._outcomes)]

    # ----------------------------------------------------------- running

    def _idle(self) -> bool:
        """Every started run has finished or is paused."""
        with self._lock:
            running = [a for i, a in self._agents.items() if i not in self._outcomes]
        return all(getattr(getattr(a, "round_control", None), "pending", None) is not None for a in running)

    def _supervise(self) -> None:
        for index in range(1, self.k + 1):
            try:
                agent = self.agent_factory()
                agent.preset_rubric = self.rubric
                agent.output_dir = str(self.out_dir / f"run-{index:02d}")
            except Exception as exc:
                self._finish(self._failed(index, f"run-{index:02d}", exc))
                continue
            with self._lock:
                self._agents[index] = agent
            threading.Thread(target=self._run, args=(index, agent), name=f"repeat-run-{index}",
                             daemon=True).start()
            if index < self.k:
                time.sleep(POLL_SECONDS)  # let go() reset the run's state before it is polled
                while not self._idle():
                    time.sleep(POLL_SECONDS)

    def _failed(self, index: int, run_id: str, exc: BaseException) -> RunOutcome:
        return RunOutcome(index=index, run_id=run_id, final_status=f"⚠️ Run failed ({type(exc).__name__}: {exc})",
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
            outcome = self._failed(index, f"run-{index:02d}", exc)
            outcome.gate_results = list(getattr(agent, "run_gate_results", None) or [])
        self._finish(outcome)

    def _waiting(self) -> list[str]:
        paused = {i: c.pending for i, c in self.paused()}
        with self._lock:
            started = set(self._agents)
            finished = set(self._outcomes)
        lines = []
        for i in range(1, self.k + 1):
            if i in finished:
                continue
            if i in paused and paused[i]:
                lines.append(f"Run {i}: paused at round {paused[i]['round']}, waiting for Continue or Stop")
            elif i in started:
                lines.append(f"Run {i}: running")
            else:
                lines.append(f"Run {i}: not started")
        return lines

    def _finish(self, outcome: RunOutcome) -> None:
        with self._lock:
            self._outcomes[outcome.index] = outcome
            all_done = len(self._outcomes) == self.k
        with self._write_lock:
            if self._done.is_set():
                return
            outcomes = self.outcomes()
            if all_done:
                self.report = write_report(self.prompt, self.rubric, self.settings, outcomes, self.out_dir,
                                           date=self.date, total_runs=self.k)
                report_path(self.out_dir, self.sha, self.date, partial=True).unlink(missing_ok=True)
                self._done.set()
            else:
                write_report(self.prompt, self.rubric, self.settings, outcomes, self.out_dir, partial=True,
                             waiting=self._waiting(), date=self.date, total_runs=self.k)


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
    out_dir = args.out or Path("runs") / f"repeat-{datetime.now().strftime('%Y-%m-%d')}-{sha[:8]}"
    rounds = max(0, args.rounds)

    from biomentis.agent.a1 import A1

    def make_agent():
        agent = A1()  # models and settings from default_config / env, as other entry points do
        agent.configure(self_critic=rounds > 0, test_time_scale_round=rounds, completion_gate=True)
        agent.pause_allowed = True  # each run has its own thread, so it can wait for an answer
        return agent

    first = make_agent()
    upload = args.rubric.read_text(encoding="utf-8") if args.rubric else None
    rubric = resolve_rubric(prompt, upload, first.llm)

    agents = iter([first])

    def factory():
        return next(agents, None) or make_agent()

    from biomentis.ui_core import run_settings

    settings = f"k={args.k}, rounds={rounds}, completion gate on, rubric {rubric.source}"
    described = run_settings(first)
    if described:
        settings += f"; {described}"
    batch = RepeatBatch(prompt, args.k, rubric, factory, out_dir, settings=settings)
    print(f"Repeat runs: {args.k} runs of prompt {sha[:8]} into {out_dir}")
    batch.start()
    run_pause_loop(batch)
    print(f"Report: {batch.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

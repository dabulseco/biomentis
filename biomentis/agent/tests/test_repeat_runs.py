"""Tests for the repeat-run tool: one prompt run K times, the runs compared.

Stub agents stand in for A1 (no models): each has a real `RoundControl` and
is scripted either to pass or to fail its gate rounds and pause before
round 4 through the real `request` + `wait`.

What these tests pin down:

  1.  Runs go one after another; a run that pauses is parked and the next
      one starts, so Run 3 finishes while Run 2 waits
  2.  A partial report is rewritten as runs finish and lists Runs 1 and 3
      while Run 2 is paused
  3.  Stopping Run 2 finishes the batch; the final report has it as
      "⛔ Stopped by user — did not pass" (and the partial report is removed)
  4.  The report has the settings, the prompt SHA, a rubric table (✓/✗/–),
      each run's findings and a Disagreements section with total_cost_usd
      from Runs 1 and 3 (3× apart); a value within 5% is not listed
  5.  Every run gets the identical Rubric object and its own output folder;
      the rubric is saved once in the batch folder
  6.  The CLI's pause prompt re-asks on unrecognised input and passes
      Continue or Stop to the run
  7.  With BIOMNI_REPEAT_RUNS=false the CLI exits with code 2 and the message

Run with:
    python -m biomentis.agent.tests.test_repeat_runs
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


# --- Stub agent -----------------------------------------------------------

from biomentis.gate import Finding, GateResult  # noqa: E402
from biomentis.gate.control import RoundControl  # noqa: E402
from biomentis.gate.quantities import QuantityClaim  # noqa: E402
from biomentis.gate.rounds import final_label  # noqa: E402
from biomentis.gate.rubric import Rubric, RubricItem, prompt_sha  # noqa: E402
from biomentis.independent_review import Review  # noqa: E402

PROMPT = "Design a 16S metagenomics study of three soils on a $2,000 budget."

RUBRIC = Rubric(
    items=(
        RubricItem("D1", "deliverable", "A sample plan per soil"),
        RubricItem("C1", "constraint", "Total cost within $2,000"),
        RubricItem("L1", "lesson", "Replicates are needed to compare soils"),
    ),
    source="uploaded",
    prompt_sha=prompt_sha(PROMPT),
)


class StubAgent:
    """Just enough of A1 for the batch: go(), the gate's results, a real RoundControl."""

    def __init__(self, script: dict):
        self.script = script
        self.round_control = RoundControl()
        self.preset_rubric = None
        self.output_dir = None
        self.final_status = ""
        self.run_gate_results: list[GateResult] = []
        self._gate_prev_claims = None
        self.run_reviews: list[dict] = []

    def go(self, prompt):
        self.round_control.reset()
        self.run_gate_results = []
        script = self.script
        if script["kind"] == "pass":
            self.run_gate_results = [
                GateResult(0, [Finding("figure", "warning", "Figure 2", "Figure 2 is cited but was not drawn")], "ok"),
            ]
            self.run_reviews = [{"review": Review(verdict="accept", coverage=script["coverage"])}]
            self._gate_prev_claims = script["claims"]
            self.final_status = final_label(self.run_gate_results[-1], False, 0)
            return [], f"<solution>{script['answer']}</solution>"
        # "pause": rounds 0-3 fail on a blocking finding, then the run pauses before round 4
        blocking = Finding("arithmetic", "blocking", "3 × 12 = 38", "3 × 12 is 36, not 38", key="arith-1")
        self.run_gate_results = [GateResult(r, [blocking], "off") for r in range(4)]
        self._gate_prev_claims = script["claims"]
        self.round_control.request(round_number=4, findings=[blocking], reason="round")
        choice = self.round_control.wait(poll_seconds=0.05)
        stopped = choice == "stop"
        self.final_status = final_label(self.run_gate_results[-1], stopped, 3)
        return [], "<solution>Draft with 3 × 12 = 38 libraries</solution>"


def _claims(cost: float, depth: float) -> list[QuantityClaim]:
    return [
        QuantityClaim("total_cost_usd", cost, "USD", f"Total: ${cost:,.0f}"),
        QuantityClaim("reads_per_sample", depth, "reads", f"{depth:,.0f} reads per sample"),
    ]


SCRIPTS = [
    {"kind": "pass", "answer": "Plan A", "claims": _claims(600.0, 50_000.0),
     "coverage": [{"id": "D1", "status": "present", "quote": "three soils, five cores each", "issue": ""},
                  {"id": "C1", "status": "present", "quote": "Total: $600", "issue": ""},
                  {"id": "L1", "status": "missing", "quote": "", "issue": "no replicates discussed"}]},
    {"kind": "pause", "claims": []},
    {"kind": "pass", "answer": "Plan B", "claims": _claims(1800.0, 51_000.0),
     "coverage": [{"id": "[d1]", "status": "wrong", "quote": "one core per soil", "issue": "no per-soil plan"},
                  {"id": "C1", "status": "present", "quote": "Total: $1,800", "issue": ""},
                  {"id": "L1", "status": "present", "quote": "five replicates per soil", "issue": ""}]},
]


def _factory(made: list):
    scripts = iter(SCRIPTS)

    def make():
        agent = StubAgent(next(scripts))
        made.append(agent)
        return agent

    return make


def _until(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# --- Tests ----------------------------------------------------------------


def test_batch_and_report():
    print("\n[1-5] batch: pause parks a run, partial and final reports, disagreements, one rubric")
    from biomentis.eval.repeat_runs import RepeatBatch

    out = Path(tempfile.mkdtemp(prefix="repeat-test-"))
    made: list[StubAgent] = []
    batch = RepeatBatch(PROMPT, 3, RUBRIC, _factory(made), out, settings="k=3, rounds=1, stub model")
    batch.start()

    got_three = _until(lambda: len(batch.outcomes()) == 2 and len(batch.paused()) == 1)
    _check("Run 3 finishes while Run 2 is paused", got_three,
           f"outcomes={[o.index for o in batch.outcomes()]} paused={[i for i, _ in batch.paused()]}")
    _check("the paused run is Run 2", [i for i, _ in batch.paused()] == [2], str(batch.paused()))
    _check("finished runs are 1 and 3", [o.index for o in batch.outcomes()] == [1, 3],
           str([o.index for o in batch.outcomes()]))
    _check("not done while a run is paused", not batch.done())

    reports = out / "reports"
    sha8 = prompt_sha(PROMPT)[:8]
    partials = list(reports.glob(f"*-repeat-{sha8}-partial.md"))
    _check("a partial report exists", len(partials) == 1, str(list(reports.glob("*"))))
    partial = partials[0].read_text(encoding="utf-8") if partials else ""
    _check("partial report lists Runs 1 and 3", "### Run 1" in partial and "### Run 3" in partial, partial[:400])
    _check("partial report does not list Run 2 as finished", "### Run 2" not in partial, partial[:400])
    _check("partial report names the run still waiting", "Run 2" in partial and "paused" in partial.lower(),
           partial[:400])

    _, control = batch.paused()[0]
    control.decide("stop")
    finished = batch.wait(timeout=10)
    _check("batch finishes after Stop on Run 2", finished and batch.done())
    _check("three outcomes in order", [o.index for o in batch.outcomes()] == [1, 2, 3],
           str([o.index for o in batch.outcomes()]))

    finals = [p for p in reports.glob(f"*-repeat-{sha8}.md")]
    _check("final report written", len(finals) == 1, str(list(reports.glob("*"))))
    _check("partial report removed once the final one is written",
           not list(reports.glob("*-partial.md")), str(list(reports.glob("*"))))
    report = finals[0].read_text(encoding="utf-8") if finals else ""
    run2 = batch.outcomes()[1]
    _check("Run 2 outcome is stopped by user", run2.final_status == "⛔ Stopped by user — did not pass",
           run2.final_status)
    _check("final report has Run 2 as stopped by user",
           "### Run 2" in report and "⛔ Stopped by user — did not pass" in report, report[:600])
    _check("Run 2's rounds recorded", run2.rounds_used == 3, str(run2.rounds_used))
    _check("final answer kept without the solution tags", batch.outcomes()[0].final_answer == "Plan A",
           repr(batch.outcomes()[0].final_answer))

    _check("report has the settings", "k=3, rounds=1, stub model" in report)
    _check("report has the prompt SHA", prompt_sha(PROMPT) in report)

    table = [ln for ln in report.splitlines() if ln.startswith("| ")]
    row = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[2:-1]] for ln in table}
    _check("rubric table: D1 ✓ (present + quote), – (no reviewer), ✗ (wrong; ID normalised)",
           row.get("D1") == ["✓", "–", "✗"], str(row.get("D1")))
    _check("rubric table: L1 ✗, –, ✓", row.get("L1") == ["✗", "–", "✓"], str(row.get("L1")))

    _check("report lists each run's findings",
           "3 × 12 is 36, not 38" in report and "Figure 2 is cited but was not drawn" in report)

    dis = report.split("## Disagreements", 1)[1] if "## Disagreements" in report else ""
    _check("Disagreements lists total_cost_usd", "total_cost_usd" in dis, dis[:400])
    _check("Disagreements shows Run 1's and Run 3's values", "600" in dis and "1,800" in dis, dis[:400])
    _check("a value within 5% is not a disagreement", "reads_per_sample" not in dis, dis[:400])

    _check("every run got the identical Rubric object",
           len(made) == 3 and all(a.preset_rubric is RUBRIC for a in made),
           str([id(a.preset_rubric) for a in made]))
    _check("each run has its own output folder under the batch folder",
           [Path(a.output_dir) for a in made] == [out / "run-01", out / "run-02", out / "run-03"],
           str([a.output_dir for a in made]))
    _check("rubric saved once in the batch folder", (out / f"rubric-{sha8}.md").exists(),
           str(list(out.glob("*"))))


def test_cli_pause_prompt():
    print("\n[6] CLI pause prompt: re-asks on unknown input, Continue and Stop reach the run")
    from biomentis.eval.repeat_runs import RepeatBatch, run_pause_loop

    out = Path(tempfile.mkdtemp(prefix="repeat-cli-"))
    made: list[StubAgent] = []
    batch = RepeatBatch(PROMPT, 3, RUBRIC, _factory(made), out)
    batch.start()
    answers = iter(["maybe", "s"])
    asked: list[str] = []

    def ask(text: str) -> str:
        asked.append(text)
        return next(answers)

    lines: list[str] = []
    worker = threading.Thread(target=run_pause_loop, args=(batch,),
                              kwargs={"ask": ask, "out": lines.append, "poll": 0.02}, daemon=True)
    worker.start()
    worker.join(timeout=10)
    _check("pause loop returns when the batch is done", not worker.is_alive() and batch.done())
    expected = "Run 2 paused at round 4: 3 × 12 is 36, not 38. Answer [c]ontinue / [s]top: "
    _check("prompt line names the run, round and blocking findings", asked[:1] == [expected], str(asked))
    _check("unrecognised input re-prompts", len(asked) == 2, str(asked))
    _check("Stop reached Run 2", batch.outcomes()[1].final_status == "⛔ Stopped by user — did not pass",
           batch.outcomes()[1].final_status if batch.outcomes() else "no outcomes")

    batch = RepeatBatch(PROMPT, 3, RUBRIC, _factory([]), Path(tempfile.mkdtemp(prefix="repeat-cli-")))
    batch.start()
    answers = iter(["c"])
    worker = threading.Thread(target=run_pause_loop, args=(batch,),
                              kwargs={"ask": lambda _q: next(answers), "out": lines.append, "poll": 0.02},
                              daemon=True)
    worker.start()
    worker.join(timeout=10)
    status = batch.outcomes()[1].final_status if len(batch.outcomes()) == 3 else "not finished"
    _check("Continue reached Run 2 (it went on and was not stopped)",
           batch.done() and status == "⚠️ Not fully checked (reviewer off)", status)


def test_cli_disabled():
    print("\n[7] BIOMNI_REPEAT_RUNS=false: CLI exits 2 with the message")
    from biomentis.eval.repeat_runs import main

    prompt_file = Path(tempfile.mkdtemp(prefix="repeat-off-")) / "p.txt"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    old = os.environ.get("BIOMNI_REPEAT_RUNS")
    os.environ["BIOMNI_REPEAT_RUNS"] = "false"
    err, outbuf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(outbuf):
            code = main(["--prompt-file", str(prompt_file), "--k", "2"])
    finally:
        if old is None:
            os.environ.pop("BIOMNI_REPEAT_RUNS", None)
        else:
            os.environ["BIOMNI_REPEAT_RUNS"] = old
    _check("exit code 2", code == 2, str(code))
    _check("prints the disabled message",
           "Repeat runs are disabled (BIOMNI_REPEAT_RUNS=false)." in err.getvalue() + outbuf.getvalue(),
           err.getvalue() + outbuf.getvalue())


def main() -> int:
    for test in (test_batch_and_report, test_cli_pause_prompt, test_cli_disabled):
        try:
            test()
        except Exception as exc:  # a crash is a failure, not an abort
            import traceback

            traceback.print_exc()
            _check(f"{test.__name__} ran", False, f"{type(exc).__name__}: {exc}")
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

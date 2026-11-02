# Run Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop BioMentis from calling an answer finished until mechanical, rubric and reviewer checks pass. Do this through an automatic-then-human-gated round policy, and add an R&D tool that runs one prompt K times and compares the results.

**Architecture:**
- **Gate package.** A new `biomentis/gate/` package holds:
  - pure, testable checkers (arithmetic, figures, claims, quantities)
  - the rubric model
  - the round policy
  - a thread-safe pause control
- **Graph changes.** In `A1.configure` (`biomentis/agent/a1.py`), the `self_critic` → `independent_review` tail of the LangGraph graph is replaced by `gate → critique | await_decision | end`. A new `draft_rubric` node runs first.
- **UI.** `ui_core` turns gate results into events. `streamlit_app.py` renders the pause panel and the status labels.
- **Repeat-run tool.** `biomentis/eval/repeat_runs.py` drives K agents with the same rubric and writes a comparison report.

**Tech Stack:** Python 3.11 (conda env `biomni`), LangGraph, langchain-core messages, Streamlit (`AppTest` for UI tests), hand-rolled `_check()` test scripts (no pytest).

**Spec:** `docs/superpowers/specs/2026-10-31-run-robustness-design.md`

## Global Constraints

**Environment and tests**
- Python: `/opt/anaconda3/envs/biomni/bin/python`. Package: `biomentis`. Never import `biomni.*`.
- Tests are plain scripts in `biomentis/agent/tests/`, using the `_check(name, cond, detail)` harness copied from `test_run_quality.py`. Each is run as:
  `PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni /opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.<name>`
  A test script exits non-zero when any check fails.
- Add every new test command to the Tests list in `CLAUDE.md`.

**Gate behaviour**
- Everything is derived from the run's message history (`run_quality.current_run`), so live and traced runs behave the same.
- Automatic round limit: **3 rounds total**, regardless of the planned number. From round 4, the run **pauses before each round** and waits until the user picks **Continue** or **Stop**.
- The first draft's gate is **round 0**. Planned rounds always run.
- Final status labels, verbatim:
  - `✅ Passed all checks after N rounds`
  - `⛔ Stopped by user — did not pass`
  - `⚠️ Not fully checked (reviewer off)`
- Finding severities: `"blocking"` and `"warning"`. Only blocking findings fail the gate. A user's **accept as is** resolves a blocking finding for that run.
- A paused run must not be ended by any timeout. It must survive Streamlit reruns and hours of waiting with the browser open. It need **not** survive an app restart.
- The agent-drafted (or converted) rubric is frozen for the run. Revisions never modify it.

**Settings**
- New env settings, following `BiomentisConfig` conventions in `biomentis/config.py`:
  - `BIOMNI_COMPLETION_GATE` (default `true`)
  - `BIOMNI_REPEAT_RUNS` (default `true`; `false` removes the repeat-run tool from the UI and makes its CLI exit with a message)
  - `BIOMNI_REVIEW_MODEL` (default reviewer when no Critic model is chosen; empty = none)
- Tutor chat Q&A and tutor step gates are unchanged. The gate applies only to A1 research runs with the completion gate on.

## Review Focus

1. **Numbers that look like arithmetic but aren't claims.** Inputs such as `2×250 bp`, `1.6×10¹⁷ copies`, `$1,015`, `0.0375 °C/mmHg` must produce **no** arithmetic finding; only statements with `=` are claims. Test in Task 2.
2. **Rounded results.** `= 0.0663 °C` and `≈ 0.53` must pass when they are correct to the stated precision. Expressions with unsupported operators (`√`, `ln`) are skipped, not flagged. Test in Task 2.
3. **The existing Stop run button pressed during a round-4 pause.** It must end the run as `⛔ Stopped by user — did not pass` within a second, not hang the worker. Test in Task 7.
4. **Reviewer call fails twice (exception or invalid JSON) inside a gate.** The reviewer-dependent checks are reported as a warning `review failed`. The final label becomes `⚠️ Not fully checked (review failed)`. No rounds are spent trying to fix a reviewer outage. Test in Task 12.
5. **Very long answers** (the metagenomics Run 3 final answer is about 65k characters). Quantity extraction is chunked by `##`/`###` section with no silent truncation, and every chunk's claims are merged. Test in Task 5.

---

## Phase 1: Labels and mechanical checks

### Task 1: Fix the revision label

**Files:**
- Modify: `biomentis/ui_core.py:415-433` (solution title)
- Test: `biomentis/agent/tests/test_revision_labels.py`

**Interfaces:**
- Produces: `revision_title(round_number: int, source: str) -> str` in `ui_core.py`.
  - `source` ∈ `{"first", "self_critique", "checks", "reviewer"}`.
  - Returns `"✅ Answer"` for `first`, else `f"✅ Answer (round {n}: revised after {label})"`, where label is `self-critique` / `failed checks` / `independent review`.
- Produces: `revision_source(messages_before_solution: list[BaseMessage]) -> str`. It finds the nearest loop HumanMessage before the solution: `CRITIC_FEEDBACK_PREFIX` → `self_critique`; `REVIEW_FEEDBACK_PREFIX` → `reviewer`; `GATE_FEEDBACK_PREFIX` (Task 8) → `checks`.

- [ ] **Step 1: Write the failing test.** Build a message list: user prompt → solution → critic feedback → solution → reviewer feedback → solution. Assert the three emitted titles are:
  - `"✅ Answer"`
  - `"✅ Answer (round 1: revised after self-critique)"`
  - `"✅ Answer (round 2: revised after independent review)"`

  The metagenomics Run 3 bug was that the third title said "self-critique #2".
- [ ] **Step 2: Run it and confirm it fails.**
  `$P -m biomentis.agent.tests.test_revision_labels` → `✗` on the third title.
- [ ] **Step 3: Implement** `revision_title` and `revision_source`, and use them in the solution branch of `stream_agent_events`. Import `GATE_FEEDBACK_PREFIX` lazily; until Task 8 it is defined in `run_quality.py` as `"The completion checks found problems"`.
- [ ] **Step 4: Run the test**, then `test_solution_streaming` and `test_run_quality`. All pass.
- [ ] **Step 5: Commit.**
  `git add biomentis/ui_core.py biomentis/run_quality.py biomentis/agent/tests/test_revision_labels.py && git commit -m "fix: label each revision by what triggered it"`

### Task 2: Gate types and the arithmetic checker

**Files:**
- Create: `biomentis/gate/__init__.py`, `biomentis/gate/arithmetic.py`
- Create: `biomentis/agent/tests/fixtures/robustness/` (text snippets copied from the run exports; see the step below)
- Test: `biomentis/agent/tests/test_gate_arithmetic.py`

**Interfaces:**
- Produces, in `biomentis/gate/__init__.py`:
  ```python
  @dataclass
  class Finding:
      check: str          # "arithmetic" | "quantity" | "figure" | "claim" | "evidence" | "rubric" | "review" | "regression"
      severity: str       # "blocking" | "warning"
      quote: str          # exact text from the answer
      issue: str
      fix: str = ""
      key: str = ""       # stable id for "accept as is", e.g. sha1(check+quote)[:10]
  @dataclass
  class GateResult:
      round: int
      findings: list[Finding]
      reviewer_state: str  # "ok" | "off" | "failed"
      @property
      def blocking(self) -> list[Finding]: ...   # excludes keys in accepted
      accepted: set[str] = field(default_factory=set)
      @property
      def passed(self) -> bool: ...              # no blocking findings
  ```
- Produces: `check_arithmetic(answer: str) -> list[Finding]` in `gate/arithmetic.py`.
  - A claim is a line segment `expr = number`, where `expr` uses only numbers, `+ - × x * / ( )` and `·`.
  - Comma thousands separators are allowed; units and `$` are stripped.
  - Tolerance: half a unit in the last stated decimal place of the result, or 0.5% relative, whichever is larger.
  - A mismatch gives a blocking `Finding(check="arithmetic")` whose `fix` holds the computed value.

- [ ] **Step 1: Create fixtures.** Save short `.txt` snippets in `biomentis/agent/tests/fixtures/robustness/`. Copy verbatim from the exports listed in the spec header:
  - `meta_run3_library_sum.txt`: the "24+3+4+2+2+2+4 = 39 ✓   48+5+4+2+2+2+4 = 65 ✓" line
  - `bp_run1_correct_sums.txt`: Run 1's `150.00 + 0.0375(760-749.82)` style lines and "3 + 4 + 5 + 6 + 1 + 5 + 4 + 2 + 3 = 33" from metagenomics Run 1
- [ ] **Step 2: Write the failing test.** Assert:
  - `check_arithmetic` on `meta_run3_library_sum.txt` returns 2 blocking findings, with fixes containing `41` and `67`.
  - The "= 33" sum returns none.
  - `"2×250 bp"`, `"1.6×10¹⁷ copies"`, `"$1,015/field sample"`, `"k = 0.0375 °C mmHg⁻¹"` return none (Review Focus 1).
  - `"0.92/(0.512 × 0.9993) = 1.80"` returns none, and `"= 1.70"` for the same expression returns one.
  - `"√(0.0593² + 0.0296²) = 0.0663"` returns none (Review Focus 2).
- [ ] **Step 3: Run it and confirm it fails** (`ModuleNotFoundError: biomentis.gate`).
- [ ] **Step 4: Implement** the dataclasses and `check_arithmetic`. Evaluate with a tiny recursive-descent parser over the allowed tokens, never `eval`.
- [ ] **Step 5: Run the test.** All pass.
- [ ] **Step 6: Commit** (`feat: completion gate types and arithmetic check`).

### Task 3: Figure integrity

**Files:**
- Create: `biomentis/gate/figures.py`
- Modify: `biomentis/run_quality.py` (`RunEvidence`, `run_evidence`)
- Test: `biomentis/agent/tests/test_gate_figures.py`

**Interfaces:**
- Consumes: `Finding` (Task 2); `run_quality.unmatched_figure_refs`.
- Produces, in `run_quality.py`:
  - `RunEvidence.figure_records: list[FigureRecord]`, where `FigureRecord(step: int, title_line: str, code: str)`. `step` is the message index of the observation; `code` is the step's code.
  - `RunEvidence.last_solution_index: int`.
- Produces, in `gate/figures.py`: `check_figures(answer: str, ev: RunEvidence) -> list[Finding]`.
  - **Cited but not drawn:** blocking. Uses `unmatched_figure_refs`, *without* the once-per-run limit.
  - **Stale:** blocking. A figure drawn *before* the message index of the previous `<solution>` (the earlier candidate) whose title line contains a number with ≥3 significant figures or a `$`/`%` amount that does not appear in `answer`.
  - **Garbled title:** warning. The figure's code passes a title string containing an unescaped `$` followed by a digit to `title`/`set_title`/`suptitle`.

- [ ] **Step 1: Write the failing test.** Use synthetic message histories:
  - (a) Answer cites `Fig 14, Fig 15` and only Figs 1–13 were drawn → 2 blocking findings mentioning 14 and 15.
  - (b) Fig 12 drawn before the first solution with title `"Fig 12a — First-year outlay $158.00; reagents are only $77.00"`; final answer says `$414` and never `$158` → 1 blocking "stale" finding.
  - (c) The same figure redrawn after the revision with `$414` → no finding.
  - (d) Code `ax.set_title(f"outlay ${total:.2f}; reagents ${r:.2f}")` → 1 warning.
  - (e) A figure with only small numbers (`"Fig 3 — 3 traces"`) drawn before the revision → no stale finding.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement** the record capture in `run_evidence` (keep `figure_titles` for existing callers) and `check_figures`.
- [ ] **Step 4: Run** this test plus `test_run_quality` and `test_step_plots`. All pass.
- [ ] **Step 5: Commit** (`feat: figure integrity checks (missing, stale, garbled title)`).

### Task 4: Claim checks — unsourced "verified", unlabelled simulated data, evidence

**Files:**
- Create: `biomentis/gate/claims.py`
- Test: `biomentis/agent/tests/test_gate_claims.py`

**Interfaces:**
- Consumes: `Finding`; `run_quality.run_evidence`, `missing_evidence`, `LITERATURE_TOOLS`.
- Produces:
  - `check_verified_claims(answer: str, run_text: str) -> list[Finding]`. A sentence containing `verified|confirmed` *and* a version-like token (`\b\d+\.\d+(\.\d+)?\b`) is blocking unless the sentence contains a URL, or the version string appears in an observation in `run_text`. Other "verified" sentences give no finding.
  - `check_simulated_labels(answer: str, run_code: list[str]) -> list[Finding]`. Applies only if any code step uses `np.random`, `random.`, `default_rng` or `rng.`. Then each answer heading or table caption matching `(?i)sample data|example data|sample .*table` must have one of `simulated|illustrative|synthetic|hypothetical|invented|not real` within the heading line or the next 300 characters. Otherwise it is blocking.
  - `check_evidence(ev: RunEvidence) -> list[Finding]`. Wraps `missing_evidence(ev)`: `literature`, `calculation`, `figure` → blocking; unplotted tables → warning.

- [ ] **Step 1: Write the failing test.**
  - `"FAPROTAX 1.2.12 (current release, Louca lab download page — verified)."` with no URL and no `1.2.12` in the observations → 1 blocking finding.
  - The same sentence with `1.2.12` present in an observation → none.
  - `"24+3+... = 41 ✓ verified by addition"` → none (no version token).
  - Boiling-point Run 2's `"Sample data table (class pool; barometer 748 mmHg; correction applied to all)"` with code using `np.random` → 1 blocking finding.
  - Run 1's `"Sample data table — simulated for illustration"` → none.
  - A run with no literature tool → a blocking `evidence` finding.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement** the three functions.
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: gate checks for unsourced verification, unlabelled simulated data, evidence`).

### Task 5: Quantity extraction, consistency, derivations and regressions

**Files:**
- Create: `biomentis/gate/quantities.py`
- Test: `biomentis/agent/tests/test_gate_quantities.py`

**Interfaces:**
- Consumes: `Finding`.
- Produces:
  ```python
  @dataclass
  class QuantityClaim:
      name: str            # normalised by the extractor, e.g. "libraries_total_n3", "nacl_i_app_1m"
      value: float
      unit: str
      quote: str
      condition: str = ""  # e.g. "1.00 mol/kg", "n=3 design"
      derivation: str = "" # optional expression over other claim names, e.g. "groups_per_solute * trials_per_group"
  EXTRACT_SYSTEM_PROMPT: str
  def extract_quantities(answer: str, llm, max_chunk_chars: int = 12000) -> list[QuantityClaim]
  def check_consistency(claims: list[QuantityClaim]) -> list[Finding]
  def check_derivations(claims: list[QuantityClaim]) -> list[Finding]
  def check_regressions(prev: list[QuantityClaim], new: list[QuantityClaim], recomputed_values: set[float]) -> list[Finding]
  ```
- **Extraction:** chunk by `##`/`###` headings into pieces of at most `max_chunk_chars`. Call `llm` once per chunk with `EXTRACT_SYSTEM_PROMPT`, asking for JSON `{"quantities":[...]}`. Merge, de-duplicating identical `(name, value, quote)`. An unparsable chunk reply is retried once, then skipped with a printed note; there is no silent cap.
- **Consistency:**
  - Same `name` *and* same `condition`, with values differing beyond 0.5% → blocking.
  - Same `name` with a different or empty `condition` → warning. This covers NaCl i, which legitimately varies with concentration.
- **Derivations:** when every name in a claim's `derivation` exists, evaluate it with the Task 2 parser (variables substituted). A mismatch beyond 0.5% → blocking.
- **Regressions:** for each name in both lists whose value changed beyond 0.5%, if the new value is not in `recomputed_values` (numbers printed by code steps since the previous candidate) → blocking `regression` finding, quoting both values.
- **Code-output cross-check:** also produces `check_against_outputs(claims: list[QuantityClaim], output_numbers: set[float]) -> list[Finding]`. A claim with a `derivation`, or whose name ends in `_total`/`_n`/`_cost`, that matches no number printed by any code step (within 0.5%) → warning `"not found in any code output"`. This is the spec's "numbers that disagree with the run's code output"; the reviewer judges the rest. Test: `libraries_total=39` with output numbers `{41.0}` → 1 warning.

- [ ] **Step 1: Write the failing test.** Use a fake LLM (`SimpleNamespace(invoke=lambda msgs: AIMessage(content=json))`) that returns canned claims.
  - Three claims `nacl_i_app` 1.80 / 1.84 / 1.86, all with condition `"1.00 mol/kg"` → blocking.
  - `nacl_i_app` 1.78 at `"0.50 mol/kg"` vs 1.87 at `"1.00 mol/kg"` → warning only.
  - `values_per_solute=8` with `derivation="groups_per_solute * trials_per_group"`, where `groups_per_solute=1` and `trials_per_group=2` → blocking, fix `2`.
  - Regression: prev `glucose_class_g=40`, new `glucose_class_g=159`, `recomputed_values` lacks 159 → blocking. With 159 present → none.
  - Chunking: a 30,000-character answer with 5 sections → the fake LLM is called ≥3 times, and claims from the first and last sections are both present (Review Focus 5).
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.** `EXTRACT_SYSTEM_PROMPT` must ask for:
  - every numeric quantity that a design, budget, count or result depends on;
  - a stable snake_case `name`;
  - `condition` whenever the value depends on a condition;
  - `derivation` whenever the answer shows how the value follows from others.
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: quantity extraction with consistency, derivation and regression checks`).

## Phase 2: Round policy and pause

### Task 6: Round policy (pure logic)

**Files:**
- Create: `biomentis/gate/rounds.py`
- Test: `biomentis/agent/tests/test_gate_rounds.py`

**Interfaces:**
- Produces:
  ```python
  AUTO_ROUND_LIMIT = 3
  def next_action(round_done: int, planned: int, result: GateResult) -> str
      # returns "critique" | "await_decision" | "end"
  def final_label(result: GateResult, stopped: bool, rounds: int) -> str
  ```
- **`next_action` rules:**
  - If `round_done < planned` → `"critique"`.
  - Else if `result.passed` → `"end"`.
  - Else if `round_done < AUTO_ROUND_LIMIT` → `"critique"`.
  - Else → `"await_decision"`.
  - **Reviewer state:** when `reviewer_state in ("off", "failed")`, only blocking findings from checks other than `rubric`/`review` count for the "else" branches.
- **`final_label`:**
  - `stopped` → `"⛔ Stopped by user — did not pass"`.
  - `reviewer_state == "off"` → `"⚠️ Not fully checked (reviewer off)"`.
  - `"failed"` → `"⚠️ Not fully checked (review failed)"`.
  - Passed → `f"✅ Passed all checks after {rounds} rounds"` (use `1 round` when rounds == 1).

- [ ] **Step 1: Write the failing test.** A table of cases:
  - planned=1, round 0 fails → critique
  - planned=1, round 1 fails → critique
  - round 2 fails → critique
  - round 3 fails → await_decision
  - round 4 fails → await_decision
  - planned=2, round 1 passes → critique (planned rounds always run)
  - planned=2, round 2 passes → end
  - planned=0, round 0 passes → end
  - reviewer off with only `rubric` blocking at round 1, planned=1 → end
  - plus every `final_label` branch, verbatim
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: round policy for the completion gate`).

### Task 7: Pause control

**Files:**
- Create: `biomentis/gate/control.py`
- Modify: `biomentis/run_worker.py` (`BackgroundRun.cancel` also stops a waiting control)
- Test: `biomentis/agent/tests/test_round_control.py`

**Interfaces:**
- Produces:
  ```python
  class RoundControl:
      def request(self, round_number: int, findings: list[Finding], reason: str) -> None   # reason: "round" | "context"
      def wait(self, poll_seconds: float = 0.5) -> str   # blocks; returns "continue" | "stop"; no timeout
      def decide(self, choice: str, accepted_keys: set[str] = frozenset()) -> None
      def stop(self) -> None                             # == decide("stop")
      @property
      def pending(self) -> dict | None                   # {"round", "findings", "reason"} while waiting
      accepted: set[str]                                 # accumulated "accept as is" keys
  ```
  It uses a `threading.Condition`. `wait` loops on `cond.wait(poll_seconds)` until a decision is set.
- `BackgroundRun` gains `round_control: RoundControl | None`, and `cancel()` calls `round_control.stop()` when one is set.

- [ ] **Step 1: Write the failing test.**
  - A worker thread calls `wait()`; the main thread sleeps 1.5 s (a stand-in for a long wait), then calls `decide("continue", {"abc"})`. The worker returns `"continue"` and `control.accepted == {"abc"}`.
  - A second wait, then `BackgroundRun.cancel()` on a run holding the control → returns `"stop"` within 1 s (Review Focus 3).
  - `pending` is non-None only while waiting.
  - Assert there is no `timeout` parameter on `wait`, and that `config.timeout_seconds` is never read by the module (grep the source text).
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** this test plus `test_background_run`. All pass.
- [ ] **Step 5: Commit** (`feat: pause control for round decisions`).

### Task 8: Wire the gate into the agent graph

**Files:**
- Modify: `biomentis/agent/a1.py`:
  - `configure` (~l.1764)
  - `generate` evidence block (~l.2020–2040)
  - `execute_self_critic` (~l.2320)
  - `independent_review` (~l.2360)
  - graph construction (~l.2432–2481)
- Modify: `biomentis/config.py` (new settings); `biomentis/run_quality.py` (`GATE_FEEDBACK_PREFIX`, summary lines)
- Create: `biomentis/gate/runner.py`
- Test: `biomentis/agent/tests/test_gate_graph.py`

**Interfaces:**
- Consumes: Tasks 2–7.
- Produces, in `gate/runner.py`:
  ```python
  def run_gate(agent, messages: list[BaseMessage], round_number: int, rubric=None) -> GateResult
  def gate_feedback(result: GateResult) -> str     # starts with GATE_FEEDBACK_PREFIX; lists blocking findings then warnings
  ```
  `run_gate` calls:
  - `check_arithmetic`
  - `check_figures`
  - `check_verified_claims`, `check_simulated_labels`, `check_evidence`
  - `extract_quantities` (with `agent.llm`) + `check_consistency` + `check_derivations`
  - `check_regressions` against the previous candidate's claims, cached on `agent._gate_prev_claims`
  - the reviewer (Task 12), when `agent.review_llm` is set

  It applies `RoundControl.accepted`.
- **`configure(...)` signature** gains `completion_gate: bool | None = None` (default `default_config.completion_gate`) and stores `self.round_control = RoundControl()`. `evidence_gate` stays accepted but is ignored when `completion_gate` is on.
- **New graph:**
  - `START → draft_rubric → generate`
  - `generate` (on solution) `→ gate`
  - `gate → critique | await_decision | END` (via `next_action`)
  - `critique → generate`
  - `await_decision → critique | finalize`
  - `finalize → END`

  With `completion_gate=False`, the old graph is built unchanged.
- **Node behaviour:**
  - `gate` appends `agent.run_gate_results: list[GateResult]`.
  - `await_decision` calls `round_control.request(...)` then `wait()`.
  - `finalize` sets `agent.final_status = final_label(...)`.
- **Context limit:** in `generate`, when `_fit_request` reports wrap-up and the gate is on, route to `await_decision` with `reason="context"` instead of silently wrapping up. On `continue`, append `WRAP_UP_INSTRUCTION` and proceed as today.
- **Settings:** `BiomentisConfig.completion_gate` (env `BIOMNI_COMPLETION_GATE`), `repeat_runs` (env `BIOMNI_REPEAT_RUNS`), `review_model` (env `BIOMNI_REVIEW_MODEL`).
- **Tracing:** after each gate, call the step tracer (`biomentis/eval/step_trace.py`, the same recorder `generate`/`execute` use) with a `gate` record: `{"round", "passed", "reviewer_state", "blocking": [check names], "warnings": n}`. `python -m biomentis.eval.step_trace report traces/` then counts gate failures by check. Test (d) in Step 1: the trace JSONL of run (a) holds two `gate` records, the first with `passed: false`.

- [ ] **Step 1: Write the failing test.** Drive the compiled graph with a scripted fake LLM (pattern: `test_independent_review.py`), with `review_llm=None` and `planned=1`:
  - **(a)** The first answer contains `"24+3 = 28"`. The gate fails, a critique message is appended, and the next answer has `"24+3 = 27"` → passes after round 1. `final_status == "⚠️ Not fully checked (reviewer off)"`.
  - **(b)** The answer keeps the error. After round 3 the graph blocks in `await_decision`; `agent.round_control.pending["round"] == 4`. A test thread calls `decide("stop")` → the run ends with `final_status == "⛔ Stopped by user — did not pass"`.
  - **(c)** With `completion_gate=False`, the old behaviour holds (`test_independent_review` still passes).
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement** `runner.py`, the settings and the graph changes. Keep `MAX_EVIDENCE_ASKS` only on the `completion_gate=False` path.
- [ ] **Step 4: Run** this test plus `test_run_quality`, `test_independent_review`, `test_context_budget`, `test_runaway_generation` and `test_run_completion`. All pass.
- [ ] **Step 5: Commit** (`feat: completion gate and round policy in the agent graph`).

### Task 9: Pause panel, status labels and sidebar

**Files:**
- Modify: `biomentis/ui_core.py` (new event kinds `round_pause`, `gate_result`, `final_status`)
- Modify: `streamlit_app.py` (render them; sidebar toggle replacing `biomentis_evidence_gate`; pass `completion_gate` to `configure`)
- Test: `biomentis/agent/tests/test_round_pause_ui.py` (AppTest harness, pattern: `test_tab_focus.py`)

**Interfaces:**
- Consumes: `agent.round_control`, `agent.run_gate_results`, `agent.final_status`.
- **`ui_core` events:**
  - After each gate, `UIEvent(type="gate_result", channel="inner", title=f"🧪 Checks after round {n}", content=<markdown list of findings>)`.
  - On a pause, `UIEvent(type="round_pause", channel="main", ...)`.
  - At the end, `UIEvent(type="final_status", channel="main", content=final_status)`. It is rendered **above** the final answer, and the export's main HTML begins with it.
- **Pause panel** (main panel, rendered while `round_control.pending`):
  - reason text: `"Round {n} needs your decision"`, or `"The context window is nearly full"`
  - the blocking findings, each with a checkbox `"Accept as is"` (key `accept_<finding.key>`)
  - buttons `Continue` and `Stop`, calling `round_control.decide("continue", accepted)` / `round_control.stop()`
- **Sidebar:** a toggle `"Completion checks"` (key `biomentis_completion_gate`, default from config). Its help text is the spec's one-line description.
- **Run-quality summary** adds:
  - `**Status:** <final_status>`
  - `**Rounds:** used X of planned Y`
  - `**Accepted as is:** n findings`
  - `**Rubric:** uploaded | agent-drafted` (Task 11 fills this)

- [ ] **Step 1: Write the failing test.** An AppTest harness with a stub agent whose `round_control.pending` is set:
  - The panel shows "Round 4 needs your decision", 2 findings and both buttons.
  - Ticking one "Accept as is" and clicking Continue calls `decide("continue", {key})`.
  - A rerun with no click keeps the panel (survives reruns).
  - A finished stub with `final_status="⛔ Stopped by user — did not pass"` renders that text before the answer and in the export HTML.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** this test plus `test_tab_focus`, `test_solution_streaming` and `test_step_plots`. All pass.
- [ ] **Step 5: Manual check:** `streamlit run streamlit_app.py`; set rounds=1; prompt: `What is 24+3+4+2+2+2+4? State it as a sum, deliberately write = 39 in your answer.` Leave the pause open 10+ minutes, then Continue. The run proceeds.
- [ ] **Step 6: Commit** (`feat: round pause panel, gate results and final status in the UI`).

## Phase 3: Rubric and reviewer

### Task 10: Rubric model and file format

**Files:**
- Create: `biomentis/gate/rubric.py`
- Test: `biomentis/agent/tests/test_rubric.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class RubricItem: id: str; kind: str; text: str      # kind: "deliverable" | "constraint" | "lesson"
  @dataclass(frozen=True)
  class Rubric:
      items: tuple[RubricItem, ...]
      source: str                                    # "uploaded" | "converted" | "agent-drafted"
      prompt_sha: str
  def parse_rubric_markdown(text: str) -> Rubric | None   # None if not in the standard format
  def format_rubric_markdown(r: Rubric) -> str
  def save_rubric(r: Rubric, folder: Path) -> Path        # writes rubric-<prompt_sha[:8]>.md
  ```
- **Format** (the designer's pick):
  - three `##` sections: `Deliverables`, `Constraints`, `Core lessons`
  - each item a line `- [D3] text` / `- [C2] text` / `- [L1] text`
  - IDs stable and unique
- `format_rubric_markdown(parse_rubric_markdown(x)) == x` for canonical input.

- [ ] **Step 1: Write the failing test.**
  - Round-trip a 3-section rubric.
  - Prose input returns `None`.
  - Duplicate IDs raise `ValueError`.
  - The dataclasses are frozen (assigning to `.text` raises).
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: rubric model and markdown format`).

### Task 11: Draft or convert the rubric at run start, and the upload control

**Files:**
- Modify: `biomentis/gate/rubric.py` (LLM functions)
- Modify: `biomentis/agent/a1.py` (`draft_rubric` node; `A1.set_rubric_upload(text: str | None)`)
- Modify: `streamlit_app.py` (rubric upload)
- Test: `biomentis/agent/tests/test_rubric_draft.py`

**Interfaces:**
- Produces:
  ```python
  DRAFT_SYSTEM_PROMPT: str
  CONVERT_SYSTEM_PROMPT: str
  def draft_rubric(prompt: str, llm) -> Rubric                  # source="agent-drafted"
  def convert_rubric(upload: str, prompt: str, llm) -> Rubric   # source="converted"
  def resolve_rubric(prompt: str, upload: str | None, llm) -> Rubric
      # parse_rubric_markdown(upload) -> source="uploaded"; else convert; else draft
  ```
- **The `draft_rubric` node:**
  - calls `resolve_rubric` once per run;
  - stores `agent.run_rubric`;
  - saves it to the session output folder (`biomentis/workspace.py` output-folder helper);
  - appends **no** HumanMessage the model could edit; the rubric reaches the model only through the critique and gate feedback text (Task 13);
  - emits a `UIEvent(type="rubric", channel="main", title="📋 Rubric for this run (frozen)")`.
- **`DRAFT_SYSTEM_PROMPT`** asks for:
  - deliverables: every item the prompt explicitly asks for;
  - constraints: every stated limit;
  - core lessons: 3–6 principles that any correct answer must convey whatever its specific approach.
- **Upload:** `st.file_uploader("Rubric (optional)", type=["md","txt","yaml","yml"], key="biomentis_rubric_upload")` in the sidebar near "Self-critique rounds".

- [ ] **Step 1: Write the failing test** (fake LLM):
  - With the boiling-point prompt text (copy it into a fixture) and no upload, `resolve_rubric` returns `source == "agent-drafted"` and calls the LLM with `DRAFT_SYSTEM_PROMPT`.
  - A canonical upload → `"uploaded"` with no LLM call.
  - A prose upload → `"converted"`.
  - Graph test: `agent.run_rubric` is identical (`is`) before and after a critique round.
  - The saved file exists and re-parses equal.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** this test plus `test_gate_graph` and `test_output_folder`. All pass.
- [ ] **Step 5: Commit** (`feat: frozen per-run rubric (uploaded, converted or agent-drafted)`).

### Task 12: Reviewer: rubric coverage, default reviewer, failure handling, same-family warning

**Files:**
- Modify: `biomentis/independent_review.py`
- Modify: `biomentis/gate/runner.py` (call the reviewer)
- Modify: `streamlit_app.py` (default reviewer from `BIOMNI_REVIEW_MODEL`; sidebar warning)
- Test: extend `biomentis/agent/tests/test_independent_review.py`

**Interfaces:**
- Produces:
  - `REVIEW_SYSTEM_PROMPT` gains a `COVERAGE` step and JSON key `"coverage": [{"id","status":"present"|"missing"|"wrong","quote","issue"}]`.
  - `Review.coverage: list[dict]`.
  - `build_review_prompt(inputs, assumption_checks, rubric: Rubric | None = None)` appends a `RUBRIC:` section.
  - `coverage_findings(review: Review, rubric: Rubric) -> list[Finding]`:
    - `missing`/`wrong` → blocking `rubric` finding;
    - `present` with an empty `quote` → blocking (spec: "present" without a quote counts as missing);
    - a rubric ID absent from `coverage` → blocking.
  - `review_findings(review: Review) -> list[Finding]`: `needs_rework` → one blocking `review` finding per problem; `accept_with_corrections` problems → warnings.
  - `same_model_family(a: str, b: str) -> bool`: compares the vendor token (the first segment of the name split on `[-:/ ]`, lowercased).
    - `("deepseek-v4.1-flash:1017-cloud", "deepseek-v4.1-flash:cloud")` → True
    - `("deepseek-v4.1-flash:cloud", "gemma4:31b-cloud")` → False
  - `review_inputs(messages, focus_numbers: set[str] = frozenset())`: steps whose output contains any focus number are kept in full; the rest are clipped as now.
- **In `run_gate`:**
  - no `review_llm` → `reviewer_state="off"`;
  - a reviewer reply unusable after the existing retry, or an exception → `reviewer_state="failed"` plus a warning finding `"review failed: <error>"` (Review Focus 4);
  - otherwise `"ok"`, adding coverage and review findings.

  The old one-correction-round logic in `independent_review` is used only when `completion_gate=False`.

- [ ] **Step 1: Write the failing tests.**
  - Coverage JSON with `L2 missing` → 1 blocking finding naming `L2`.
  - Rubric ID `D5` absent from coverage → blocking.
  - `present` with empty quote → blocking.
  - `same_model_family` cases as above.
  - The reviewer raises an exception twice → `GateResult.reviewer_state == "failed"`, and `final_label(...)` → `"⚠️ Not fully checked (review failed)"`.
  - `focus_numbers={"0.0593"}` keeps the full output of the step that printed it.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** In `streamlit_app.py`:
  - when no Critic model is chosen and `default_config.review_model` is set, build that model with `get_llm` and use it as `agent.review_llm`;
  - show `st.sidebar.warning("Reviewer and agent are the same model family — they may share blind spots.")` when `same_model_family` is true.
- [ ] **Step 4: Run** `test_independent_review`, `test_gate_graph` and `test_gate_rounds`. All pass.
- [ ] **Step 5: Commit** (`feat: reviewer checks rubric coverage; default reviewer; failure and same-family handling`).

## Phase 4: Self-critique

### Task 13: Critique that preserves what passed

**Files:**
- Modify: `biomentis/agent/a1.py` (`execute_self_critic` → the critique node of Task 8)
- Test: `biomentis/agent/tests/test_gate_critique.py`

**Interfaces:**
- Consumes: `GateResult`, `Rubric`, `gate_feedback`.
- Produces: `CRITIQUE_PROMPT_TEMPLATE` in `biomentis/gate/runner.py`, with placeholders `{task}`, `{rubric}`, `{findings}`, `{passed_quantities}`. It replaces the "Critic harshly… think hard what are missing" prompt.
- **Instruction order:**
  1. Fix every blocking finding.
  2. Check each rubric item.
  3. Then look for missing work.

  The prompt must include this sentence verbatim: `"Do not change any value listed under PASSED QUANTITIES unless a code step in this round shows it is wrong; if you change one, recompute it in <execute></execute> first."`
- `{passed_quantities}`: the previous candidate's `QuantityClaim`s that produced no finding, as `name = value unit`.
- The critique still runs on `self.creative_llm` (temperature unchanged, user-set).

- [ ] **Step 1: Write the failing test.**
  - The critique prompt sent to a capturing fake LLM contains the verbatim sentence above, the rubric item IDs, the text of a blocking finding, and `glucose_class_g = 40 g`.
  - It does not contain `"Critic harshly"`.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** this test plus `test_run_quality` (the "critic checks numbers" assertions may need updating to the new wording; keep their intent) and `test_gate_graph`. All pass.
- [ ] **Step 5: Commit** (`feat: critique works from findings and rubric and keeps verified values`).

## Phase 5: Repeat-run tool

### Task 14: Repeat-run core, CLI and report

**Files:**
- Create: `biomentis/eval/repeat_runs.py`, `biomentis/eval/repeat_report.py`
- Test: `biomentis/agent/tests/test_repeat_runs.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class RunOutcome:
      index: int; run_id: str; final_status: str; rounds_used: int
      gate_results: list[GateResult]; final_answer: str; telemetry: str
  class RepeatBatch:
      def __init__(self, prompt: str, k: int, rubric: Rubric, agent_factory: Callable[[], A1], out_dir: Path): ...
      def start(self) -> None          # runs sequentially; a run that pauses is parked (its thread waits) and the next run starts
      def paused(self) -> list[tuple[int, RoundControl]]
      def done(self) -> bool
      def outcomes(self) -> list[RunOutcome]
  def write_report(prompt: str, rubric: Rubric, settings: str, outcomes: list[RunOutcome], out_dir: Path) -> Path
  ```
- **Batch setup:**
  - The batch drafts or resolves **one** rubric before starting, and passes it to every run via `agent.run_rubric` (the `draft_rubric` node skips when it is preset).
  - Every run gets a fresh `A1` from `agent_factory`, configured identically.
- **Report** (`reports/<date>-repeat-<prompt_sha[:8]>.md`):
  - settings and prompt SHA
  - a rubric table: rows = item IDs, columns = runs, cells ✓/✗/–
  - per-run status, rounds and telemetry
  - all findings per run
  - a **Disagreements** section: quantity names present in ≥2 runs whose values differ by >5%, listing each run's value
  - a partial report (`…-partial.md`) is rewritten each time a run finishes, while others are paused
- **CLI:**
  `python -m biomentis.eval.repeat_runs --prompt-file P --k 5 [--rubric R] [--rounds 1]`
  - When a run pauses, it prints `Run i paused at round n: <blocking findings>. Answer [c]ontinue / [s]top:` and waits on stdin. This is how an unattended batch waits for the user.
  - Exits with `"Repeat runs are disabled (BIOMNI_REPEAT_RUNS=false)."` when the setting is off.

- [ ] **Step 1: Write the failing test.** Use an `agent_factory` returning stub agents whose fake LLMs are scripted:
  - Run 1 passes, Run 2 pauses at round 4, Run 3 passes.
  - Assert that while Run 2 is paused, Run 3 still completes, and a partial report exists listing Runs 1 and 3.
  - After `decide("stop")` on Run 2, `done()` is true, and the final report has Run 2 as `⛔ Stopped by user — did not pass`.
  - The Disagreements section lists `total_cost_usd` with values from runs 1 and 3 that differ by 3×.
  - All runs received the identical `Rubric` object.
  - With `BIOMNI_REPEAT_RUNS=false`, the CLI main returns exit code 2 with the message.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: repeat-run batch, CLI and comparison report`).

### Task 15: Repeat-run UI and removal switch

**Files:**
- Modify: `streamlit_app.py` (sidebar expander "R&D: repeat this prompt")
- Test: `biomentis/agent/tests/test_repeat_runs_ui.py` (AppTest)

**Interfaces:**
- Consumes: `RepeatBatch`, `write_report`, `RoundControl`, the Task 9 pause panel (reused per run).
- **UI:**
  - a sidebar toggle `"Show repeat-run tool"` (key `biomentis_show_repeat_runs`, default True)
  - when shown: number input K (1–10, default 5), button `"Run this prompt K times"`
  - a progress table (run #, status, rounds)
  - one pause panel per paused run
  - a link to the report when done
- When `default_config.repeat_runs` is False, neither the toggle nor the expander is rendered.

- [ ] **Step 1: Write the failing test.**
  - With `BIOMNI_REPEAT_RUNS=false`, the AppTest finds no element labelled "Run this prompt K times" and no "Show repeat-run tool" toggle.
  - With it true and the toggle off, the button is hidden; toggle on, it is shown.
  - With a stub batch holding one paused run, a pause panel keyed by run index is shown, and Continue calls that run's control.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the test.** All pass.
- [ ] **Step 5: Commit** (`feat: repeat-run tool in the sidebar, removable via BIOMNI_REPEAT_RUNS`).

### Task 16: Regression suite, docs and acceptance

**Files:**
- Create: `biomentis/agent/tests/test_robustness_regressions.py`, `docs/run_robustness.md`
- Modify: `CLAUDE.md` (test list); `.env.example` if present (new settings)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the regression test.** One `_check` per spec §8 fixture, running the real checker functions on the fixture snippets (fake LLM only for quantity extraction):
  - false sum "= 39"
  - NaCl 1.80 / 1.84 / 1.86
  - "n = 8 per solute"
  - Figs 14–16 missing
  - stale $158 figure
  - "FAPROTAX 1.2.12 verified" without a source
  - budget regression ×4
  - unlabelled sample data
  - the label fix (reuse the Task 1 helper)

  Add any snippets not yet in `fixtures/robustness/`, copied verbatim from the exports.
- [ ] **Step 2: Run it.** It passes, because the earlier tasks are in place.
  `$P -m biomentis.agent.tests.test_robustness_regressions` → all `✓`.
- [ ] **Step 3: Write `docs/run_robustness.md`,** in the voice of `docs/background_runs.md`: what the gate checks, the round policy and pause, the rubric, the labels, the repeat-run tool, and the three env settings. Add all new test commands to `CLAUDE.md`.
- [ ] **Step 4: Run every test in `CLAUDE.md`'s list.** All pass.
- [ ] **Step 5: Acceptance run (manual, spec §8.2):**
  `python -m biomentis.eval.repeat_runs --prompt-file <boiling-point prompt> --k 5 --rounds 1`, with a reviewer of a different family set via `BIOMNI_REVIEW_MODEL`.
  - **Expected:** a report in `reports/` in which every run is `✅ Passed…` or visibly paused/stopped, and no passed run contains a blocking finding when re-checked with `test_robustness_regressions`' checkers on its final answer.
  - Repeat for the metagenomics prompt.
  - Record both report paths in `docs/run_robustness.md`.
- [ ] **Step 6: Commit** (`test: regression suite from the run comparisons; docs for run robustness`).

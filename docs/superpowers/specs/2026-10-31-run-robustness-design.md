# Run robustness: completion gate, rubric, round policy, repeat-run tool

**Date:** 2026-10-31
**Status:** draft for approval
**Evidence base:** hand-written comparisons of repeated runs of the same prompt:
- Metagenomics 16S workflow, 3 runs:
  - `~/Documents/IFI/Grants/aa-Active/4-ITYC_GenLAB/BioMentis/5-Metagenomics/BioMentis_Run_Comparison.md`
- Boiling-point elevation lab, 2 runs:
  - `…/6-BoilingPoint/BioMentis_BoilingPoint_Run_Comparison.md`
- CHO workflow, 2 runs:
  - `reports/2026-10-28-cho-workflow-run-comparison.md`

---

## 1. Intent

BioMentis will eventually be used in classrooms by students and teachers. The owner is not yet convinced it performs consistently enough for that. Right now he is the only user, testing prompts for robustness.

**"Consistent" here does not mean identical output.** Five runs of a lab-design prompt may legitimately produce five different protocols: different beaker, different solute set, different timing. That variety is welcome. What must hold in every run:

1. **No calculation errors.** Masses, molalities, sums, counts, costs and timings are right, and the text, tables and figures agree with each other.
2. **Feasible within the stated constraints.** Standard equipment and modest reagents, or whatever the prompt specifies.
3. **The same core lessons are covered,** whatever the specific approach.

The harness must check these invariants on every run. It must not call an answer finished until they pass, unless a human explicitly stops it.

## 2. What the evaluations showed (and where in the code)

| Failure seen | Run(s) | Cause in the harness |
|---|---|---|
| Final answer cites figures that were never drawn (Figs 14–16) | Boiling-point Run 1 | `run_quality.missing_evidence` / `MAX_EVIDENCE_ASKS = 2` in `a1.generate`: the evidence and figure checks are spent before the last revision, so the final answer goes out unchecked. `figure_ref_asks` is asked once per run. |
| Figure shows numbers from before the revision ($158 cost vs text $414; pilot $25,330 vs $24,370) | Boiling-point Run 1, Metagenomics Run 3 | Nothing detects stale figures. |
| Self-critique turned a correct budget into one 4× too high; revision introduced 6 new bugs | Boiling-point Run 2, Metagenomics Run 2 | `execute_self_critic` prompt ("Critic harshly… think hard what are missing") at `creative_temperature` 0.7; nothing re-checks numbers that were already right. |
| Printed arithmetic false ("24+3+4+2+2+2+4 = 39"); same quantity with 3 values (NaCl i = 1.80 / 1.84 / 1.86); "n = 8 per solute" when the design gives 2 | Metagenomics Run 3, Boiling-point Run 2 | No mechanical arithmetic or consistency check. The reviewer sees ~800 chars per step in long runs (`_EVIDENCE_TOTAL_CHARS` clipping in `independent_review.review_inputs`). |
| Runs disagree on "verified" facts (QIIME 2 Docker vs conda; FAPROTAX 1.2.6 vs 1.2.12) | Metagenomics Runs 2 vs 3 | "Verified" needs no attached source. |
| Reviewer absent | Boiling-point Runs 1–2 | Reviewer runs only when a Critic model is chosen (`streamlit_app.py` ~l.249, `agent.review_llm`). |
| Reviewer ≈ agent model (`deepseek-v4.1-flash:cloud` reviewing `…:1017-cloud`) | All metagenomics runs | No warning about same-family review, though `independent_review.py`'s own docstring notes shared blind spots. |
| "Answer (revised after self-critique #3)" when only 2 critique rounds ran | Metagenomics Run 3 | `ui_core.py` ~l.430 labels every new `<solution>` as a self-critique revision, including the reviewer's correction round. |
| Ran out of context; wrote no files despite "executable workflow" | Metagenomics Runs 2, 3 | `context_budget` wrap-up ends the run quietly; no deliverable contract. |
| Sample data not labelled as invented | Boiling-point Run 2, Metagenomics Runs 1–2 | No provenance tag on generated data. |
| No way to repeat a prompt and compare | — | `eval/` has a QA benchmark loader and a step tracer only. |

## 3. Goals

- **G1. Completion gate.** Every candidate final answer is checked. An answer is *finished* only when all gate checks pass.
- **G2. Rubric.** Each run is judged against a rubric of deliverables, constraints and core lessons. The rubric is either uploaded or, failing that, drafted by the agent, then frozen for the run and saved for reuse.
- **G3. Round policy with a human stop.** The run keeps revising until the gate passes. Rounds 1–3 are automatic; from round 4 the run pauses for the user.
- **G4. Safer self-critique.** A revision must not silently break what was already correct.
- **G5. Reviewer by default,** with better evidence, and a warning when it is the same model family as the agent.
- **G6. Repeat-run tool** for R&D: run a prompt K times and get a comparison report. It can be switched off in the UI and removed via `.env`.
- **G7. Honest labels:** round numbering, pass/stop status, and simulated-data tags.

## 4. Non-goals

- Making runs deterministic or identical.
- Changing the agent's model, prompts for domain content, or tool set beyond what the gate needs.
- A student/teacher-facing UI redesign. That comes later and may hide some of this.
- Grading answers for a course (the rubric checks robustness, not student work).

## 5. Anti-goals (would count as failure even if it "works")

- A gate that passes because the same model grades its own work leniently. Arithmetic and figure checks must be mechanical, and coverage judgments must quote the answer.
- Unbounded loops or cost without a human in the loop beyond round 3.
- Hiding failures: a stopped or failing run must never look like a checked answer.
- Checks so rigid they force every run onto the same protocol. The rubric checks *what must be true*, not *how* it is achieved.
- Moving goalposts: a revision cannot rewrite the rubric it is judged against.

## 6. Design

### 6.1 Rubric (G2)

**Contents.** Three lists, each item short and checkable:
- **Deliverables:** what the prompt asks for. For the boiling-point prompt: learning objectives, design, protocol, supplies, equipment, timings, data handling, pitfalls, safety, assessment, per phase. For metagenomics: sections 0–8 plus labels, cost and time per step.
- **Constraints:** standard teaching-lab equipment, flag fume-hood steps, modest cost, controls never skipped, etc.
- **Core lessons / principles:** for example "the effect depends on particle count, not identity"; "Phase 1's measured precision decides what Phase 2 can resolve"; "barometric correction matters for absolute Tb and cancels in ΔTb".

**Source.**
- **Uploaded:** the user attaches a rubric file with the prompt (sidebar or chat attachment). It is used as is.
- **Agent-drafted:** with no upload, the agent's *first* action is to draft the rubric from the prompt. This is one LLM call before any research. The draft is shown at the top of the run and saved to the session output folder as an editable file the user can upload next time.

**Frozen.** Once the run starts, the rubric cannot change. Self-critique and revisions see it, but cannot edit it.

**Format** (builder's call, with these requirements): human-readable and hand-editable (Markdown or YAML), one item per line, and stable item IDs so reports can compare runs item by item.

**Uploads in another format** (a prose document, a teacher's checklist) are converted by the model into the standard list, with IDs assigned *(designer's call)*. The converted rubric is shown to the user and frozen before the run starts. The original upload is kept alongside it.

### 6.2 Completion gate (G1)

Runs on **every** candidate final answer: first draft, every revision, and every correction. There is no per-run cap like `MAX_EVIDENCE_ASKS`. Checks:

1. **Arithmetic and consistency audit (mechanical).**
   - Every arithmetic statement printed in the answer is recomputed in code: explicit equations, "a + b + … = c" sums, table totals, products of per-unit rate × count, percentages.
   - The same named quantity stated with different values in different places is flagged (e.g. library count 39 vs 41; NaCl i 1.80 / 1.84 / 1.86).
   - Numbers that disagree with the run's code output are flagged.
   - **Extraction can be LLM-assisted, but verification is code.** Extraction means turning prose and tables into a list of {quantity, value, unit, location, stated derivation}.
   - Counts derived from the design (e.g. "n per solute = groups doing it × trials") are checked when the answer states the design inputs.
2. **Figure integrity.** Every figure the answer cites exists. Every figure drawn before the latest revision whose title or labels carry numbers absent from (or contradicting) the final answer is flagged *stale*. A figure title containing math-mode-garbled text (an unescaped `$`) is flagged.
3. **Rubric coverage.** For each rubric item, a judgment of present / missing / wrong, with a quote from the answer as evidence. "Present" without a quote counts as missing. The reviewer model does this, not the agent model.
4. **Independent review verdict** (existing `independent_review`) is not `needs_rework`.
5. **"Verified" claims carry a source.** Any claim the answer marks verified/confirmed must point to a tool output or URL from this run, or the check fails.
6. **Simulated data labelled.** Sample tables generated in code from random or invented inputs must be labelled as simulated in the answer. Detection can use the code that produced them (e.g. `np.random`, hand-entered "sample" values).

7. **Evidence checks** (folded in from today's `evidence_gate`, without the `MAX_EVIDENCE_ASKS` cap): at least one source consulted, calculations done in code, figures drawn for computed tables where they help.

**Gate result:** a structured list of findings, each with a location quote and a suggested fix. It is shown under the answer and fed to the next round.

**Two severities** *(designer's call)*:
- **Blocking:**
  - an arithmetic statement that recomputes differently
  - a number contradicting the code output
  - a cited figure missing or stale
  - a rubric item missing or wrong
  - a "verified" claim with no source
  - reviewer `needs_rework`
  - a regression (§6.4)
- **Warning:** findings that may be legitimate, e.g. one name with different values that could differ for a stated reason (NaCl i at 0.5 vs 1.0 mol/kg), or values written as approximate (≈, ~).

Only blocking findings stop an answer from passing. Warnings are listed under the answer. At a round-4+ pause the user can mark any blocking finding **"accept as is"**, which counts it as resolved for that run and records it in the run-quality summary.

**Scope** *(designer's call)*: the gate applies to full agent task runs (`A1` research runs), behind a setting that defaults to on. Tutor chat Q&A and the tutor's own step gates are unchanged.

### 6.3 Round policy (G3)

- The sidebar's **Self-critique rounds** (0–3) becomes the *planned* number of critique rounds.
- **The planned rounds always run,** even if an earlier answer already passed. That is the owner's stated model: extra rounds happen only on failure.
- **The first draft's gate is round 0.**
- **One round** = feedback (self-critique, plus gate findings, plus reviewer problems) → agent revision → gate.
- **The final answer must pass the gate after the last round.**
- If the gate fails after the planned rounds, the run **continues automatically up to round 3 in total**, regardless of the setting.
- **From round 4 on, the run pauses before each round.**
  - It shows the outstanding findings, a per-finding "accept as is" option, and two buttons: **Continue** and **Stop**.
  - It waits until the user answers.
- **Persistence (owner's decision):** a paused run does **not** need to survive an app restart. It **must** survive a long wait, hours, with the browser open:
  - page reruns,
  - the browser tab idling,
  - and no run-level timeout (e.g. `BIOMNI_TIMEOUT_SECONDS`) or LLM client idle timeout ending it.

  The run can then continue normally on **Continue**: no lost state, no expired model session.

  The waiting worker thread (or the tutor's end-at-a-gate / resume-on-Continue pattern, if that fits the graph better) is the builder's choice, within `docs/background_runs.md`.
- **Stop** delivers the latest answer labelled **"Stopped by user — did not pass"**, with the failures listed at the top of the answer, in the export and in the run-quality summary.
- **A passing answer** is labelled **"Passed all checks after N rounds"**.
- **Context limit.** If `context_budget` says the window is nearly full, the run must not quietly ship. It pauses with the same Continue/Stop choice (Continue = proceed with trimmed context) regardless of round number.

### 6.4 Self-critique changes (G4)

- The critique prompt receives the gate failures and the frozen rubric, and is told to **keep what passed**: anything the gate verified may change only if a code step in the same round shows it was wrong.
- **Regression check:** the gate compares the audited numbers of the previous candidate with the new one. A value that passed before and changed without a recomputation step in this round is flagged as a regression failure.
- The harsh, open-ended wording ("Critic harshly… think hard what are missing") is replaced by: check against the rubric, then the gate failures, then the missing items.
- The temperature stays user-set. The repeat-run tool is how the owner will test whether a lower critique temperature helps.

### 6.5 Reviewer (G5)

- **On by default.** If no Critic model is chosen, use a configured default reviewer.
- **The user may switch it off** (owner's decision). With it off:
  - the gate still runs every check that doesn't need the reviewer (1, 2, 5, 6, 7 and the regression check);
  - the planned rounds run;
  - **but no extra rounds are added for the checks that need the reviewer** (rubric coverage, verdict).
  - The answer is labelled **"Not fully checked (reviewer off)"**, never "Passed".
  - If a mechanical check is still failing, the normal round policy applies to it.
- **Same-family warning.** If the reviewer and agent models share a family (e.g. both `deepseek-v4.1-flash*`), show a warning in the run-quality summary and the sidebar.
- **Better evidence.** Instead of clipping every step evenly, give the reviewer:
  - the arithmetic-audit results,
  - the code steps whose outputs the final answer's numbers come from (full),
  - and the rest clipped as now.
- **Rubric coverage.** The reviewer also performs check 3 of the gate.
- The existing "one correction round, then re-review" logic is replaced by the round policy in §6.3.

### 6.6 Labels and telemetry (G7)

- **Fix the label** in `ui_core.py`: number answers by round, and name the source of the revision (self-critique, gate failures, or reviewer).
- **Run-quality summary adds:**
  - rubric source (uploaded / agent-drafted)
  - gate result per round
  - final status (passed / stopped by user)
  - rounds used vs planned
  - reviewer model, with the same-family warning
- `step_trace` records gate results per round so `step_trace report` can aggregate them.

### 6.7 Repeat-run tool (G6)

- **Invocation.** From the UI ("Run this prompt K times"), and as a script for unattended batches. Inputs: prompt, K (default 5), rubric (uploaded or one agent-drafted rubric shared by all K runs), and the current settings.
- **Behaviour.**
  - All K runs use the same frozen rubric and settings.
  - Each run applies the full gate and round policy.
  - **A run that reaches round 4 pauses individually and waits for the user** (owner's decision), even in an unattended batch. The other runs carry on.
  - The batch report is written once every run has passed or been stopped.
  - The waiting must not break the run's ability to continue (same requirement as §6.3).
  - A partial report of the finished runs may be shown while others wait.
  - Parallel vs sequential execution is the builder's call, respecting Ollama Cloud rate limits.
- **Report.** Saved to `reports/` as Markdown and/or HTML, modelled on the hand-written comparisons:
  - settings and prompt hash (confirming the runs are identical)
  - per-run gate results and rubric coverage (item by item)
  - pass / stop status and rounds used
  - telemetry (steps, sources, figures, errors)
  - **disagreements:** key quantities or decisions that differ across runs (e.g. read length, total cost, tool versions), marked as variation or contradiction
  - variance summary
- **Switching off.**
  - A UI toggle hides the option.
  - An env variable (name is the builder's call, following existing `BIOMNI_*` conventions) removes it entirely, for classroom builds.

## 7. Constraints

- **Environment:**
  - Python env is the `biomni` conda env (`/opt/anaconda3/envs/biomni/bin/python`).
  - Package is `biomentis`.
  - Tests are hand-rolled `_check()` scripts run with `PYTHONPATH` set (see `CLAUDE.md`), not pytest.
- **Runs:** agent turns run on a background thread. Any pause must work with that and with the run journal. Streamlit UI tests use `streamlit.testing.v1.AppTest`.
- **Message history:** everything is derived from the run's message history (as `run_quality` does now), so it works for live and traced runs.
- **Cost:** a gate pass adds LLM calls (extraction, coverage judgment). Keep them bounded per round; the expensive part should be the agent's own work, not the gate.

## 8. Success criteria

1. **Regression fixtures from real transcripts** (exports in the folders above) are caught by the gate:
   - "24+3+4+2+2+2+4 = 39"
   - NaCl i 1.80 / 1.84 / 1.86
   - "n = 8 per solute" with one solute per group × 2 trials
   - Figs 14–16 cited but not drawn
   - the stale $158 cost figure
   - "FAPROTAX 1.2.12 verified" with no source in an earlier draft
   - the budget regression (correct class quantities replaced by ×4)
   - sample data unlabelled
   - the "self-critique #3" label
2. **Repeat runs:** with the repeat-run tool, 5 runs each of the boiling-point and metagenomics prompts give:
   - no delivered "passed" answer containing an arithmetic inconsistency, a missing/stale figure, or a missing rubric item;
   - every failing run either passes by round 3, or pauses and is visibly labelled if stopped.
3. **No silent end:** turning the reviewer off, or hitting the context limit, never produces an unlabelled "finished" answer. A reviewer-off run is labelled "Not fully checked (reviewer off)".
4. **Long pause:** a run paused at round 4 for several hours, with the browser open, continues correctly on Continue.

## 9. Owner's decisions (as stated)

- Consistency of accuracy, constraints and core lessons matters more than identical output. Variation in approach is fine.
- The reviewer can be on. It was off in the boiling-point runs only to see its effect.
- Not finished without passing. Rounds continue automatically up to 3. From round 4, **pause and wait** for the user to choose Continue or Stop.
- Core lessons come from an uploaded rubric. If none is uploaded, the agent drafts them.
- The repeat-run tool is wanted for R&D, removable later via UI or `.env`.
- The reviewer may be switched off. Such runs are labelled "Not fully checked (reviewer off)".
- In unattended repeat runs, a run at round 4 waits for the user, as long as waiting doesn't harm its ability to continue.
- A paused run need not survive an app restart, but must survive a long pause with the browser open.

## 10. Calls made by the designer (check these)

- The automatic limit is 3 rounds total because the sidebar maxes out at 3, even when fewer are planned.
- The agent-drafted rubric is frozen for the run and saved for reuse, so repeat runs share one rubric.
- Self-critique rounds = 0 still runs the gate. If it fails, rounds follow §6.3 (so "0" means "no planned critique", not "no checking").
- Hitting the context limit pauses for the user regardless of round number.
- Labels:
  - "Passed all checks after N rounds"
  - "Stopped by user — did not pass"
  - "Not fully checked (reviewer off)"
- Planned rounds always run. The first draft's gate is round 0.
- Two severities (blocking / warning), plus "accept as is" for a finding at a round-4+ pause.
- The gate covers full agent task runs, behind a setting that defaults to on. Tutor Q&A is unchanged.
- Today's evidence checks are folded into the gate, uncapped.
- Non-standard rubric uploads are converted by the model, shown, and frozen.

## 11. Left to the builder

- Rubric file format (within §6.1's requirements) and where the upload control lives.
- How claims are extracted for the arithmetic audit, as long as verification is code.
- Stale-figure detection method details.
- Parallel vs sequential repeat runs; report format (Markdown, HTML, or both).
- Env variable names; default reviewer model choice.
- Phasing: suggested order is
  1. label fix + gate checks 1–2 (mechanical, cheap, catch the most)
  2. round policy + pause
  3. rubric + coverage
  4. self-critique and reviewer changes
  5. repeat-run tool

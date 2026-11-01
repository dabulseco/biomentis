# Answers that have been checked

A run used to end when the model said it was done.

Three real runs of the same two prompts (a boiling-point lab, a metagenomics
plan) came back with a printed sum that was wrong (`24+3+4+2+2+2+4 = 39`, it
is 41), the same quantity stated three ways (NaCl's van 't Hoff factor as 1.80,
1.84 and 1.86), "n = 8 per solute" where the design gives two, figures cited
but never drawn, a cost figure still showing $158 beside text that said $414,
and a "verified" version number with no source. One run's self-critique took a
correct budget and made it four times too high. Nothing noticed, because nothing
looked.

The completion gate looks. It runs on the final answer of every research run,
sends what it finds back to the agent, and only calls the answer finished when
the checks pass. If they do not pass, the label on the answer says so.

The regression suite replays those failures from text copied out of the real
exports: `biomentis/agent/tests/test_robustness_regressions.py`, with the
snippets in `biomentis/agent/tests/fixtures/robustness/`.

## 1. What the gate checks

`biomentis/gate/`

The gate reads the run's message history (`run_quality.current_run`), so a live
run and a traced one behave the same. Every check returns findings. A finding is
**blocking** or a **warning**; only blocking findings fail the gate.

| Check | Looks for | Blocks? |
|---|---|---|
| `arithmetic` | A statement with `=` whose left side is a plain expression and whose result is wrong to the stated precision (`24+3+4+2+2+2+4 = 39`). `2×250 bp`, `$1,015` and `0.0375 °C/mmHg` are not claims; `√` and `ln` are skipped, not flagged | blocking |
| `figure` | A figure number the answer cites that the run never drew (Figs 14-16) | blocking |
| `figure` | A figure drawn before the last revision whose title shows a number the final answer no longer contains ($158 vs $414) | blocking |
| `figure` | An unescaped `$` before a number in a plot title (matplotlib turns it into math) | warning |
| `claim` | "verified" or "confirmed" next to a version number, with no URL in the sentence and no such version in any code output | blocking |
| `claim` | A "sample data" table, from a run that generated random numbers, with nothing in or after the caption saying it is simulated | blocking |
| `evidence` | No literature lookup, no calculation or no figure in the whole run | blocking |
| `evidence` | A computed table that has no figure | warning |
| `quantity` | One quantity, one condition, two values (NaCl i 1.80 / 1.84 / 1.86) | blocking |
| `quantity` | One quantity with different values under different conditions | warning |
| `quantity` | A value that does not follow from the derivation the answer gives ("8 per solute" = 1 group × 2 trials = 2) | blocking |
| `quantity` | A derived total, count or cost that no code step printed | warning |
| `regression` | A quantity that changed since the previous draft when no code step since then printed the new value (the 4x budget) | blocking |
| `rubric` | A rubric item the reviewer did not mark present with a quote | blocking |
| `review` | A problem the reviewer found; blocking when its verdict is "needs rework", a warning otherwise | either |

Quantity extraction is the one step that needs a model: the agent's gate model
reads the answer, one `##`/`###` section at a time (no silent truncation, the
claims are merged), and returns each quantity with the exact text it came from.
The consistency, derivation, regression and output checks are then plain code.

Every finding quotes the text it is about. The same finding carries a stable key,
which is what **Accept as is** (below) refers to.

## 2. Rounds

`biomentis/gate/rounds.py`

The first draft is checked as **round 0**. After that:

- **Planned rounds always run.** The sidebar's "Self-critique rounds" is a floor,
  not a ceiling; a clean answer still gets its planned review.
- **Up to 3 rounds total run automatically**, whatever the planned number. While
  blocking findings remain, the findings go back to the agent, which recomputes
  and answers again.
- **From round 4, the run pauses before each round.** The pause panel lists the
  blocking findings, each with its quote and a suggested fix, and three ways out:
  - **Accept as is** (a checkbox per finding): the finding stops blocking for the
    rest of the run. Use it for a finding that is wrong or that you do not mind.
  - **Continue**: one more round.
  - **Stop**: end the run now.
- The existing **Stop run** button also works during a pause and ends the run as
  stopped within a second.

A paused run is not ended by any timeout. It waits on its thread for as long as
you take, and survives Streamlit reruns, with the browser open. It does not
survive restarting the app.

### Context limit

When the context window is nearly full the run pauses the same way, with
"The context window is nearly full". **Continue** asks the agent to wrap up with
what it has; **Stop** ends the run.

### Inline mode

`BIOMENTIS_BACKGROUND_RUNS=0` (see `background_runs.md`) runs the stream on the
script thread, so nobody could press Continue while it is blocked. A pause
**stops the run instead**, and the sidebar says so. The label is still
`⛔ Stopped by user — did not pass`; the run is never labelled passed without
passing.

### Plain CLI `go()`

Outside Streamlit nothing renders a panel. A round-4 pause waits for
`agent.round_control.decide(...)`, called from another thread:

```python
agent.round_control.decide("continue", {finding.key})   # accept those findings, go on
agent.round_control.decide("stop")
```

`agent.round_control.pending` holds the round, the findings and the reason. To
have a script stop at round 4 rather than wait, set `agent.pause_allowed = False`.

## 3. The rubric

`biomentis/gate/rubric.py`

A rubric says what the answer must **deliver**, **respect** and **teach**, as
items the reviewer can tick off. It is fixed for the run: revisions never change
it, so a run cannot pass by grading itself more kindly.

- **Upload** (sidebar, "Rubric (optional)"): `.md`, `.txt`, `.yaml` or `.yml`. A
  file already in the standard format is used as is. Any other text is converted
  by the agent's model into the standard format.
- **None uploaded:** the agent drafts one from the prompt before it starts.
- **Canonical markdown:**

  ```markdown
  # Rubric

  ## Deliverables
  - [D1] A day-by-day schedule that fits a 50-minute period

  ## Constraints
  - [C1] No fume hood

  ## Core lessons
  - [L1] Boiling is vapour pressure equal to external pressure
  ```

  IDs are `D`, `C` or `L` plus a number, and must not repeat.
- **Saved:** next to the run's other output, as `rubric-<first 8 of the prompt's SHA>.md`.
- **Used only with the gate on.** Coverage is checked by the reviewer; with no
  reviewer, rubric findings are not reported and the label says so.

## 4. The reviewer

`biomentis/independent_review.py`

A second model reads the answer and the evidence behind it. It is the only check
that can notice a missing rubric item or a wrong mechanism.

- The default is `BIOMNI_REVIEW_MODEL` when no Critic model is picked in the
  sidebar. Empty means no reviewer.
- If the reviewer is the same vendor family as the agent (`deepseek-…` reviewing
  `deepseek-…`), the sidebar warns: they may share blind spots. Pick a different
  family.
- If the reviewer's call raises, or its reply is still not usable after one
  retry, the reviewer-dependent checks are reported as a warning `review failed`
  and the run finishes as `⚠️ Not fully checked (review failed)`. No rounds are
  spent trying to fix a reviewer outage.
- With no reviewer, the rubric and review checks are skipped and the label is
  `⚠️ Not fully checked (reviewer off)`.

The revision titles also tell the truth about what sent an answer back:
`✅ Answer (round 2: revised after independent review)`, not "self-critique #3"
when the reviewer asked for it (`ui_core.revision_title`).

## 5. Status labels

Shown above the answer, in the export and in the run summary:

| Label | When |
|---|---|
| `✅ Passed all checks after N rounds` | No blocking finding is left (accepted ones do not count) |
| `⛔ Stopped by user — did not pass` | Stop was pressed, or a pause could not be offered |
| `⚠️ Not fully checked (reviewer off)` | No reviewer was set |
| `⚠️ Not fully checked (review failed)` | The reviewer failed twice in the last gate |

## 6. Repeat runs

`biomentis/eval/repeat_runs.py`, `biomentis/eval/repeat_report.py`, `biomentis/ui_repeat.py`

For seeing how much an answer varies: one prompt, K runs, one rubric, side by
side. A research tool, not part of an ordinary run.

**In the UI:** the sidebar's "Show repeat-run tool" toggle opens "R&D: repeat
this prompt". Enter K and press "Run this prompt K times". It uses the sidebar's
current model, reviewer and settings. The button is disabled while an
interactive run is going, and Send is disabled while a batch is.

**From the command line:**

```bash
python -m biomentis.eval.repeat_runs --prompt-file prompt.txt --k 5 --rounds 1
#   [--rubric rubric.md]   any format; drafted by the agent if omitted
#   [--out DIR]            default ./runs/repeat-<date>-<sha8>/
```

How it runs:

- **One rubric for every run,** resolved once and saved in the batch folder.
- **One run executes at a time.** Code steps share the working folder, the plot
  capture and the REPL, so runs take turns. A run that pauses at round 4 hands
  its turn on, so the next run starts while it waits for you; when you press
  Continue it queues for its turn again. Stop ends it at once.
- **Each run has its own output folder** (`run-01/`, `run-02/`, ...) and its own
  **REPL namespace**: a fresh run starts empty, a paused run gets its variables
  back when it resumes, and the namespace from before the batch is restored at
  the end.
- Each paused run gets its own pause panel, in the UI and at the CLI prompt.
- `BIOMNI_REPEAT_RUNS=false` removes the tool from the UI and makes the CLI exit
  with a message.

**The report** is written to `reports/<date>-repeat-<prompt sha8>.md`, rewritten
as `...-partial.md` after each run finishes while others are still going:

- the settings and the prompt's SHA;
- a **rubric table**, one row per item and one column per run: a tick when the
  run's last review marked the item present with a quote, a cross when it was
  missing, wrong or unquoted, a dash when the run had no working reviewer;
- each run's status label, rounds, telemetry and every finding;
- **Disagreements**: final-answer quantities that at least two runs report and
  that differ by more than 5%, with each run's value. These are the numbers to
  verify first.

## 7. Settings

In `.env` (see `.env.example`), read by `biomentis/config.py`:

| Setting | Default | Does |
|---|---|---|
| `BIOMNI_COMPLETION_GATE` | `true` | The gate on or off. Off, a run keeps the older evidence nudge. The sidebar's "Completion checks" toggle starts from it |
| `BIOMNI_REPEAT_RUNS` | `true` | `false` removes the repeat-run tool from the UI and makes its CLI exit with a message |
| `BIOMNI_REVIEW_MODEL` | empty | The reviewer when no Critic model is picked. Empty is no reviewer. Choose another family from the agent's |

Tutor chat answers and tutor step gates are unchanged; the gate applies to
research runs only.

## Tests

```bash
python -m biomentis.agent.tests.test_robustness_regressions
```

Replays the failures above from text copied out of the real exports, through the
real checkers. Only quantity extraction uses a fake model, one that returns
claims quoting the same snippets. The other gate tests are in `CLAUDE.md`
(`test_gate_*`, `test_rubric*`, `test_round_*`, `test_repeat_runs*`,
`test_revision_labels`, `test_independent_review`).

## Acceptance runs

Not part of the test suite: they need live models and hours. Set
`BIOMNI_REVIEW_MODEL` to a model from a different family than the agent, then:

```bash
python -m biomentis.eval.repeat_runs --prompt-file biomentis/agent/tests/fixtures/robustness/bp_prompt.txt --k 5 --rounds 1
python -m biomentis.eval.repeat_runs --prompt-file <metagenomics prompt file> --k 5 --rounds 1
```

Expected, for each: a report in `reports/` in which every run is
`✅ Passed all checks after N rounds` or is visibly paused or stopped, and no
passed run contains a blocking finding when its final answer is re-checked with
the checkers from `test_robustness_regressions`.

- Boiling-point report: not yet run
- Metagenomics report: not yet run

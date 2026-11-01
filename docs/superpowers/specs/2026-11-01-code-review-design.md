# On-demand Code Review — design

Date: 2026-11-01
Status: implemented (live click-through in the real app left to the owner)

## Why

The agent writes code to do its calculations, and nothing checks that the code
is *correct*. The existing checks ask other questions:

- the evidence gate asks whether a calculation exists;
- the self-critic (same model as the author) asks whether each number was
  computed in code;
- the independent reviewer judges the *final answer* against each step's code
  and output, but sees only the first 1,500 characters of each step's code.

A boiling-point-elevation run (2026-11-01, `deepseek-v4.1-flash`) produced a
script that ran cleanly and printed solute masses 18x too small
(`moles_water = V_water/1000/18.015`, labelled "= 0.1000 kg", printing
0.0056 kg), an uncertainty budget that counted a cancelling calibration offset
twice, and class reagent totals inconsistent with its own plan. Nothing flagged
it.

Students also copy agent code and run it on their own machines. Wrong code then
leaves the app with the app's authority behind it.

## What we are building

1. **A "Code Review" button wherever code is shown**, for every user (student,
   instructor, anyone). On click, a dedicated code-review model reviews that
   code and returns a structured report plus **a corrected, runnable script —
   always**.
2. **The automatic end-of-run reviewer sees more of each script.**
3. **A "Code review model" setting**, separate from the agent and Critic
   models, with a default chosen by a model comparison (see "Model choice").
4. **The self-critic is told to check the code, not just the numbers** — a
   free backstop on every run.

## Deployment this version targets

Each student runs the app on their own computer: one user per Streamlit
process (owner, 2026-11-01). A shared multi-student server is expected in 2–3
months; what it will need is listed under "Later versions" and is out of scope
here.

## Goals

- A student who intends to run a piece of code outside the app can ask for a
  review first, and leave with a script that runs on its own and gives the
  right answer.
- Reviews cost nothing unless someone clicks.
- A review never disturbs a run: it does not advance a paused tutor, is not
  lost if the student clicks something else, and does not block the page.
- The instructor can see which code was reviewed and what was found.

## Non-goals (this version)

- Per-student or per-class API keys, quotas, or click limits. All reviews run
  on the app owner's Ollama cloud account (`OLLAMA_API_KEY`).
- An offline/local review path as a supported default (a local model may be
  selected in the setting, but nothing is tuned for it).
- Automatically applying the corrected script, or re-running the agent's step
  with it. The corrected script is for the student to copy.
- Executing code as part of the review. The reviewer judges code and any
  output it already produced; it does not run anything.
- Automatic review of every step.

## Anti-goals (would be a failure even if it works)

- A review that "fixes" correct code to match a wrong reading of the task, or
  invents problems to have something to say.
- A corrected script that is not self-contained (depends on agent state, agent
  tools, or files the student does not have) without saying so.
- A click that kills, pauses, or advances an agent run.
- Any change to what the agent itself does when nobody clicks (apart from parts
  2 and 4, which are deliberate).

## Where the button appears

On every block of code a user can see and copy:

| Where | Rendered today by | Notes |
|---|---|---|
| Agent code steps (Python, R, Bash) | `render_entry`, `biomentis/agent/a1.py:4199` (`st.code`) | Main case. Present live and on history replay. |
| Code blocks inside the final answer | `st.markdown` of the solution, `a1.py:4268` | Fenced blocks are extracted and each gets a button below the answer. |
| Tutor chat plot code | `_render_visuals`, `biomentis/ui_tutor.py:1818` | Both the chat panel and per-step Q&A. |
| The scratch REPL panel | `biomentis/ui_repl.py` | Student-written code. *Builder's call (Claude's): included, since it is the same component.* |

## What the reviewer receives

- The code block, line-numbered, whole (no character cap up to a generous
  safety limit; see "Limits").
- The block's output, if it has run (the observation that follows it).
- The task (run prompt) and the step's purpose (the agent's reasoning text
  immediately before the code, if any).
- **Earlier code from the same run.** Agent Python steps share one persistent
  namespace (`biomentis/utils.py:1500`), so a step routinely uses names defined
  steps earlier. The reviewer gets the run's earlier code blocks (as
  `run_journal.code_script` assembles them) so it can tell which names come
  from where, and so the corrected script can define them.
- Language (python / r / bash).

For tutor plot code and REPL code, the context is just the code, its output if
any, and the task.

## What the review returns

A JSON object, parsed defensively like the tutor's other JSON replies:

```json
{
  "verdict": "ok" | "run_with_fixes" | "do_not_trust",
  "issues": [{"line": 12, "kind": "bug|mismatch|units|science|portability",
              "severity": "high|medium|low", "quote": "...",
              "problem": "...", "fix": "..."}],
  "corrected_code": "<complete script, runnable on its own>",
  "summary": "<2-3 sentences for a student>"
}
```

Review checklist (the prompt, in this order): bugs; mismatches between
comments/labels/printed output and what the code computes; units and constants
(recompute one key value by hand); science and reference values; portability
(agent-only imports such as `biomentis.tool`, files only in the run folder,
names from earlier steps, missing packages).

**The corrected script is always returned**, including when the verdict is
"ok" (then it is the original made self-contained). It must:

- run on its own with standard scientific Python (or R/Bash for those blocks);
- define anything it previously took from earlier steps;
- replace agent-only helpers with standard libraries, or keep them with a
  clear comment that they need the app;
- carry a short header comment: generated by Code Review, which model, date,
  and that it should be checked before use.

- be compatible with Python 3.10+ (no 3.12-only syntax such as reusing the
  same quote character inside an f-string).

**The corrected script is syntax-checked before it is shown.** For Python,
`ast.parse`; for Bash, `bash -n`; R is not checked. If the reply has no usable
`corrected_code`, is unreadable, or the corrected script fails the syntax
check, the review is retried once, telling the model what was wrong (the same
retry pattern as `independent_review`). After that the report is shown with
"no corrected script could be produced" (or, for a script that still fails the
check, the script with a warning naming the syntax error). *(Added after the
model comparison: both MiniMax models returned corrected scripts that did not
run — a 3.12-only f-string and an indentation error.)*

## How it looks

Under each code block, a small **Code Review** button. After a click:

- the button becomes "Reviewing… (model name)" with a spinner;
- when done, a collapsible "Code review — <verdict>" panel appears under the
  code: the summary, the issues as a list (severity, line, quote, problem,
  fix), and the corrected script in `st.code` (which has Streamlit's copy
  button) plus a "Download .py" button;
- a review that failed shows the reason (no key, quota exhausted, model
  unreachable, unreadable reply) and a "Try again" button.

A second click on the same code shows the stored review instantly.

## How it runs

- **Off the script thread.** The review call can take a minute. A Streamlit
  script run is interrupted by any other click, which would drop an in-flight
  call — the same failure that moved agent runs onto a background thread
  (`docs/background_runs.md`). Reviews run on a small worker pool; the UI shows
  progress and picks up the result on a later rerun (polling with
  `st.fragment(run_every=...)` while any review is pending).
- **Keyed by content and context.** A review is identified by a hash of
  (code, output, earlier-run code, task, model). The context is in the key
  because the corrected script defines names from earlier steps: two runs can
  produce an identical block (`print(df.head())`) with different histories.
  *(Claude's call.)* Results are cached in a process-wide store, so reruns,
  replays and second clicks reuse them; a different model gets its own review.
- **Reloaded after a restart.** On startup the store is filled from the
  `*.reviews.jsonl` sidecars, so history replay still shows stored reviews.
  *(Claude's call.)*
- **Never touches the agent.** No agent state is read or written except the
  transcript/journal data needed for context. The review call uses its own LLM
  client.

## Tying a code block to its run

Transcript code entries carry only content, language and title today. When the
UI builds a code entry (`transcript_entry_for_event`, the consumer side), it
also records the `run_id` and the block's index among that run's code blocks.
The journal and the producer thread are untouched, so the run-loop invariants
hold. The index locates the block's earlier code (`run_journal.code_script`),
its output (the next observation) and the reasoning before it. Entries replayed
from older runs without these fields fall back to "code and output only"
context. *(Claude's call.)*

## Where reviews are stored

- **Session:** the process-wide cache above, keyed by hash; the transcript
  entry is not modified.
- **On disk:** `runs/<run_id>.reviews.jsonl`, one line per finished review
  (hash, block locator, model, verdict, issues, corrected code, timestamp).
  This is a separate file because `runs/<run_id>.jsonl` must be written only
  by the run's producer thread (CLAUDE.md, run-loop invariants). Reviews for
  code outside a run (REPL, tutor chat with no run) go to the session's
  output folder.
- **Export:** each reviewed code entry carries its review in the exported
  transcript (a review block under the code, like the Q&A extras), and the
  corrected scripts are written into the export folder as
  `code_review_<n>.py`.

## Model and account

- A new setting **"Code review model"** in the main sidebar (not inside the
  tutor's settings, since every user gets the button). It lists **Ollama models
  only** — cloud and installed local ones — and has no "off" entry. Its
  default is fixed by the model comparison below. *(Claude's call: the owner
  said all reviews run through the owner's Ollama cloud account.)*
- All reviews use the app owner's Ollama cloud key. With no key, or with the
  quota exhausted, the button shows that error; nothing else is affected.
- Thinking is on for cloud models that support it (existing
  `_ollama_reasoning` behaviour).

## Model choice

Candidates (owner's choice): `gpt-oss:120b-cloud`, `minimax-m3:cloud`,
`minimax-m3.1-flash:cloud`, and local `gpt-oss:20b`. Each reviews three
scripts with known bugs — the real boiling-point script and two seeded ones
(qPCR standard curve: Avogadro off 10x, 20 µL vs "2 µL", ln for log10;
Arrhenius half-life: kJ not converted, °C not K, minutes labelled hours).
Scored on bugs found, false alarms, whether the corrected script runs and
prints the right numbers, and time. The winner is the default.

The comparison is run separately from the build, before it; the build only
sets the winning model as the default.

### Model comparison results (2026-11-01)

One review per model per script, temperature 0.2, the Appendix A prompt.
Corrected scripts were executed and their numbers checked.

| | gpt-oss:120b-cloud | minimax-m3.1-flash:cloud | minimax-m3:cloud | gpt-oss:20b (local) |
|---|---|---|---|---|
| qPCR: 3 planted bugs found | 3/3 | 3/3 | 3/3 | 3/3 |
| Arrhenius: 3 planted bugs found | 3/3 | 3/3 | 3/3 | 3/3 |
| Boiling: mass bug (18x) found | yes | yes | yes | yes |
| Boiling: uncertainty budget (accuracy bound / cancelling offset) | resolution only | yes, both | yes, both | resolution only |
| Boiling: class-quantity plan contradictions, hydrates | no | yes | yes | partly |
| Corrected scripts run and give right numbers | 3/3 | 2/3 (boiling script uses 3.12-only f-string; correct when run on 3.13) | 2/3 (boiling script has an indentation error) | 3/3 |
| Time per review | 5–14 s | 63–343 s | 76–424 s | 38–231 s |
| Output tokens per review | 1.9k–6.3k | 7.5k–37.6k | 10.4k–44.7k | 3.7k–20.5k |
| Noise | low | "do not trust" on every script; NumPy flagged as a portability issue each time | same | low |

**Default: `gpt-oss:120b-cloud`.** It caught every planted code bug, every
corrected script ran and gave the right numbers, a review takes seconds, and it
uses roughly 5–7x fewer output tokens (owner's quota) than either MiniMax
model. Its weakness is depth: it fixed the boiling-point code but not the
uncertainty-budget reasoning or the plan's internal contradictions. The
MiniMax models review the *science* far more deeply but are slow, costly in
tokens, and returned broken corrected scripts — which the syntax check and
retry now address. `minimax-m3.1-flash:cloud` stays selectable in the setting
for a deeper review. Small sample (3 scripts, 1 run each): revisit if reviews
in class miss things.

## Part 2 — automatic reviewer limits

In `biomentis/independent_review.py`:

- `_STEP_CODE_CHARS` 1,500 → 4,000.
- `_clip` keeps the head **and the tail** of long code (about 60/40), with the
  elided count in the middle, instead of cutting the tail off — final tables
  and totals are usually at the end.
- `_EVIDENCE_TOTAL_CHARS` (90,000) and the per-step scaling stay, so long runs
  still shrink each step's share.

## Part 4 — self-critic checks the code

One added paragraph in `execute_self_critic`'s prompt (`a1.py:2319`): for each
script, compare printed values with the comments and labels that describe
them, check units and conversions, and recompute one key value by hand. No
extra calls.

## Limits

- Code sent for review: whole block up to 40,000 characters; earlier-run
  context up to 30,000 (head-and-tail clipped); output up to 8,000.
- Worker pool: 2 concurrent reviews per process; further clicks queue (one
  user per process this version, so the queue is that user's own clicks).
- Review call timeout: 5 minutes.

## Implications for users

- **Students:** a button under code; 15–60 s per review; a report and a
  corrected script to copy or download. No key, no setup. The run is not
  affected.
- **Instructors:** reviews appear in the journal sidecar and the export.
- **Owner:** every click is one call on the owner's Ollama cloud quota
  (roughly 3–15k tokens in, 1–4k out). There is no per-student limit in this
  version.

## Appendix A — review prompt used in the model comparison

The starting point for the production prompt; the build extends it with the
earlier-code context, "always return corrected_code", the language rule and the
Python 3.10 rule.

```text
You are a careful scientific code reviewer. A student is about to run the SCRIPT below on their own computer, outside the research agent that wrote it. Find every error that would make its results wrong or stop it from running, then give a corrected script.

Check, in this order:
1. BUGS: wrong formulas, wrong variables, wrong function (e.g. natural log where log10 is meant), off-by-one, logic errors.
2. MISMATCHES: a comment, variable name, or printed label that disagrees with what the code actually computes (e.g. a comment says "2 uL" but the code uses 20; a value printed as "h" that is in minutes). Compare the OUTPUT, if given, against the labels.
3. UNITS AND CONSTANTS: unit conversions, physical constants, magnitudes. Recompute at least one key value by hand and say whether it is physically plausible.
4. SCIENCE: assumptions or reference values that are wrong for the field.
5. PORTABILITY: anything that will not work outside the agent (agent-only imports, files that may not exist, missing packages).

Be specific: quote the line, say what is wrong, and give the correct value. Do not report style issues. Do not invent problems.

Return ONLY a JSON object, no markdown fences:
{
  "verdict": "ok" | "run_with_fixes" | "do_not_trust",
  "issues": [{"line": <int or null>, "kind": "bug"|"mismatch"|"units"|"science"|"portability", "severity": "high"|"medium"|"low", "quote": "<code text>", "problem": "<what is wrong>", "fix": "<what it should be>"}],
  "corrected_code": "<the complete corrected script, runnable as-is>",
  "summary": "<2-3 sentences for a student>"
}
```

## Left to the builder

- Exact widget keys, layout spacing, and the fragment polling interval.
- How fenced code blocks are pulled out of the final answer.
- Whether the worker pool lives in its own module or alongside
  `run_worker.py`.
- Test structure, within the project's `_check()` harness and `AppTest`
  conventions.

## Later versions

Per-user/per-class keys, a supported offline review model, click limits, and an
optional "review every step automatically" mode.

For the shared-server deployment (2–3 months out): a pool sized for many users
with queue position shown; file locking on the sidecars; reviews outside a run
stored server-side where the instructor can reach them, not in the student's
Downloads folder.

# Scaffolded Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a student asks the tutor a question worth thinking about, guide them to the answer in a few short steps instead of answering at once. Scale the challenge with the question's sophistication. Never leave the student stuck.

**Architecture:**
- **Pure logic.** A new module `biomentis/agent/tutor/scaffold.py` holds the logic that needs no LLM:
  - question profile and tier
  - policy table
  - thread state machine and release rules
  - leak check
  - fallback rung texts

  It also holds the one new LLM call, the rung call.
- **`TutorChat.ask`** gains a `scaffold_level` argument. Above `off`, the same answer call also returns the question profile, key points and (under `challenge`) a stretch question, and logging/history are deferred until the answer is released.
- **`TutorEngine`** gains the level and control settings, plus four methods that run a thread: `start_question`, `reply`, `hint`, `reveal`.
- **UI.** `ui_tutor.py` adds the sidebar control and renders open threads on both surfaces: the page chat and the per-step box.

**Tech Stack:** Python 3.11 (conda env `biomni`), langchain-core messages, Streamlit (`AppTest` for UI tests), hand-rolled `_check()` test scripts (no pytest).

**Spec:** `docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md`

## Global Constraints

**Environment and tests**
- Python: `/opt/anaconda3/envs/biomni/bin/python`. Package: `biomentis`. Never import `biomni.*`.
- Tests are plain scripts using the `_check(name, cond, detail)` harness copied from `test_chat_visuals.py`, and exit non-zero on any failure. Run as:
  `PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni /opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.<name>`
- Stub LLMs return canned JSON in call order, like `_LLM` in `test_chat_visuals.py`. No test may reach a real model.
- Add every new test command to the Tests list in `CLAUDE.md`.

**Behaviour that must hold**
- **Level `off` is today's chat, byte for byte:**
  - the same system prompt and user prompt;
  - the same `ChatTurn.content`;
  - the same `qa` log record;
  - history appended inside `ask()`.
- **Fail open.** A malformed profile means a direct answer. A rung call that fails twice releases the answer. No code path leaves a thread open with no way forward.
- **Released answers look like today's.** Once released, a turn dict on either surface carries the same keys as today's (`content`, `citations`, `also_consider`, Bloom/DOK badges, visuals), so the existing renderers and the HTML export need no change.
- **Hard cap of 3 rungs** per thread, whatever the policy table says.
- **Session-state keys** keep the existing `biomni_tutor_` prefix used throughout `ui_tutor.py`.

**Settings**
- These follow the `BIOMENTIS_PLOT_MODE` precedent: module-level helpers in `scaffold.py`, like `plots.plot_mode_default`, not `BiomentisConfig`. This deviates from spec §9, which said `config.py`; Task 10 updates the spec.
  - `BIOMENTIS_SCAFFOLD_LEVEL` ∈ `off|light|standard|challenge`. Unset or unknown means `standard`.
  - `BIOMENTIS_SCAFFOLD_CONTROL` ∈ `students|teacher`. When set to `teacher`, the control is preset and hidden. Unset or unknown means students choose.

## Review Focus

1. **Off is unchanged.** `ask(..., scaffold_level="off")` must send exactly the messages it sends today. Task 4 compares against a capture taken *before* `chat.py` is modified.
2. **Leak-check false positives.** A key point like "noise" or "E-value" would match almost any hint. Only key points of at least 3 words, or at least 15 characters after normalization, are checked. Task 3.
3. **Short correct replies are not frustration.** "false positives" is 2 words and correct. The two-short-replies rule fires only when both replies are judged `off_track`, `gave_up` or `none`. Task 2.
4. **The teacher lock survives reruns and widget resets.** Once locked, no widget interaction or session-state edit made through the UI can unlock it for the rest of the session. Task 7.
5. **A thread survives reruns.** An open thread is plain data in session state, not a generator, so a rerun mid-thread re-renders it unchanged and the next reply continues it. Tasks 8–9.

---

## Phase 1: Pure logic (no LLM)

### Task 1: Profile, tier and policy

**Files:**
- Create: `biomentis/agent/tutor/scaffold.py`
- Test: `biomentis/agent/tests/test_scaffold.py`

**Interfaces:**
- `LEVELS = ("off", "light", "standard", "challenge")`; `LEVEL_LABELS = {"off": "Answer directly", "light": "A nudge first", "standard": "Guide me", "challenge": "Challenge me"}`
- `QUESTION_TYPES = ("logistics", "definition", "agent_action", "procedural", "conceptual", "evaluative")`
- `MOVES = ("orient", "predict", "hint", "partial", "probe", "contrast", "find_flaw", "extend")`
- `@dataclass QuestionProfile(type, bloom, dok, specific, shows_reasoning, wants_direct)`
- `QuestionProfile.parse(value) -> QuestionProfile | None`: coerces the type, clamps dok to 1–4, and validates bloom against `rubric.BLOOM_LEVELS`. Returns `None` when `value` is not a dict or the type is unknown.
- `tier(p) -> "novice" | "developing" | "advanced" | "expert"`, following spec §7.2. `expert` means at least two of: `shows_reasoning`; bloom ∈ {Evaluate, Create}; dok ≥ 3.
- `@dataclass(frozen=True) ScaffoldPlan(direct: bool, moves: tuple[str, ...], stretch: bool, reason: str)`
- `decide(level, profile | None, tier) -> ScaffoldPlan`, following spec §7.3:
  - `profile is None` → `direct`, reason `profile_failed`;
  - `off` → direct;
  - logistics or agent_action → direct;
  - definition → direct, except under `challenge`, where it gets `("predict",)`;
  - `wants_direct` under light or standard → direct;
  - otherwise the table, with procedural minus one rung (minimum 1) and a hard cap of 3;
  - `stretch` is true only for `challenge` + `expert` and a non-direct plan.
- `scaffold_level_default()` and `scaffold_control_default() -> (control, locked)`, from the env vars in Global Constraints.

- [ ] **Step 1: Write the failing tests:**
  - every cell of the spec §7.3 table;
  - each tier rule, including the boundary where `advanced` has only one of the three `expert` signals;
  - each direct rule;
  - procedural reduction (light/novice stays 1);
  - the cap of 3;
  - `stretch` only for challenge + expert;
  - `parse` of junk (`None`, a list, an unknown type, dok 9 → 4);
  - env defaults with unset, valid and junk values.
- [ ] **Step 2: Run and confirm they fail** (ImportError, caught per test by the `run_all` try/except).
- [ ] **Step 3: Implement.** Keep the table as one literal dict `{(level, tier_column): moves}` so it reads like the spec.
- [ ] **Step 4: Run and confirm all pass.**

### Task 2: Thread state and release rules

**Files:**
- Modify: `biomentis/agent/tutor/scaffold.py`
- Test: `biomentis/agent/tests/test_scaffold.py`

**Interfaces:**
- `GIVE_UP_PATTERNS`: regexes for "just tell me", "tell me the answer", "idk", "i don't know", "no idea", "i give up", and a reply that is only "?" or "??". Case-insensitive and whole-reply aware: "I don't know if X, but…" is **not** a give-up.
- `is_give_up(text) -> bool`
- `@dataclass ScaffoldThread`:
  - fields: `question`, `surface` ("chat" | "step"), `step_id`, `level`, `control`, `profile` (dict), `tier`, `plan`, `answer` (dict: the held-back turn's UI-dict fields plus `answer_body`, `follow_up`, `stretch_question`, `key_points`), `rungs: list[dict]` (move, message, reply, verdict, leak_retry), `hints_requested`, `started_at`, `released: str | None`;
  - `to_dict()` / `from_dict()`: the thread is stored in session state as a plain dict (Review Focus 5).
- `ScaffoldThread.next_move() -> str | None`: the plan's next unused move, or `None` once the cap is spent.
- `ScaffoldThread.release_reason_for(verdict, reply) -> str | None`, applying spec §7.5 in this order:
  1. `new_question` → `abandoned`;
  2. `correct` → `earned`;
  3. `gave_up` or `is_give_up(reply)` → `frustration`;
  4. this reply and the previous one both under 4 words, with verdicts in {off_track, gave_up, none} → `frustration`;
  5. no next move → `exhausted`;
  6. otherwise `None`.
- `compose_released(thread) -> str`, the released answer text:
  - `earned` → `"✓ That's the core of it.\n\n"` + body;
  - otherwise the body alone;
  - then `**Follow-up:** <follow_up>`, or `**Stretch question:** <stretch_question>` when `plan.stretch` and one exists.

  With no reason and no stretch, this must equal today's `answer + "\n\n**Follow-up:** " + follow_up` composition (Review Focus 1).

- [ ] **Step 1: Write the failing tests:**
  - each release reason, in order of precedence;
  - "false positives" judged `correct` releases as `earned`, not `frustration`;
  - two replies "um ok" and "dunno" judged off_track → `frustration`;
  - "I don't know if the threshold matters, but maybe…" is not a give-up;
  - a dict round-trip keeps every field;
  - `compose_released` for each case, including stretch.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass.**

### Task 3: Leak check and fallback texts

**Files:**
- Modify: `biomentis/agent/tutor/scaffold.py`
- Test: `biomentis/agent/tests/test_scaffold.py`

**Interfaces:**
- `_normalize(text)`: lowercase, strip markdown emphasis and punctuation, collapse whitespace.
- `leaked_point(message, key_points) -> str | None`: the first checkable key point (≥ 3 words or ≥ 15 normalized characters, per Review Focus 2) whose normalized form occurs in the normalized message.
- `FALLBACK_RUNGS: dict[str, str]`: one fixed text per move, with `{where}` filled from the thread. For example:
  - `orient`: "Look back at {where}. What does it tell you about your question?"
  - `predict`: "Before I explain: what do you expect to happen, and why?"
  - `hint`: "Think about which concept from {where} applies here. Which one would you reach for?"

  `{where}` is "step N" when `step_id` is set, else "the agent's last step".

- [ ] **Step 1: Write the failing tests:**
  - a verbatim leak is caught;
  - case, punctuation and `**bold**` variants are caught;
  - a short key point ("noise") is never checked;
  - no leak → `None`;
  - every move has a fallback, and the fallbacks format with and without `step_id`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass.**

## Phase 2: LLM calls

### Task 4: `ask()` with a scaffold level

**Files:**
- Modify: `biomentis/agent/tutor/chat.py`
- Test: `biomentis/agent/tests/test_scaffold_chat.py`

**Interfaces:**
- **New `ChatTurn` fields,** all defaulting to empty: `answer_body: str`, `follow_up: str`, `question_profile: dict`, `key_points: list[str]`, `stretch_question: str`.
- **`ask(..., scaffold_level: str = "off", record: bool = True)`:**
  - When `scaffold_level != "off"`, `_SCAFFOLD_ADDENDUM` is appended to the system prompt. It asks for three extra JSON keys:
    - `question_profile` (spec §7.1 schema);
    - `key_points` (1–3 short phrases);
    - only when the level is `challenge`, `stretch_question`: "one question that goes beyond what was asked; asked only if the question is very sophisticated, else empty string".
  - The parsed values are set on the turn. `answer_body` and `follow_up` are always set, at every level, from the existing locals.
  - When `record=False`, the `qa` log record and the history append are skipped.
- **New `TutorChat.record(question, turn, step_id=None, thread=None)`:** writes the same `qa` record `ask()` writes today, plus `"scaffold": thread.to_dict()` when a thread is given, and appends to history.
  - With a thread, the user turn's history content becomes the question plus the student's replies, so later questions see the exchange as it happened.
  - Step 7 of `ask()` is refactored to call it. **No behaviour change at `off`.**

- [ ] **Step 1: Capture the baseline *before* touching `chat.py`.** A helper in the test runs `ask()` with a recording stub on fixed inputs (KB empty, two history turns, two recent cards). Save the messages, the `ChatTurn` and the `qa` log record to `biomentis/agent/tests/fixtures/scaffold_off_baseline.json`. Commit that fixture with the test.
- [ ] **Step 2: Write the failing tests:**
  - `off` is identical to the baseline (messages, turn, log record);
  - `standard` adds the addendum, and the profile and key points are parsed onto the turn;
  - `stretch_question` is requested only at `challenge`;
  - a malformed `question_profile` leaves `{}`;
  - `record=False` writes no log and leaves the history unchanged;
  - `record()` with a thread writes `qa` with `scaffold`, and the history holds the replies.
- [ ] **Step 3: Run and confirm they fail** (all but the baseline equality).
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run `test_scaffold_chat`, `test_chat_visuals`, `test_card_text_length`, `test_tutor_card_cache` and `tutor/tests/test_smoke_e2e.py`. All must pass.**

### Task 5: The rung call

**Files:**
- Modify: `biomentis/agent/tutor/scaffold.py`
- Test: `biomentis/agent/tests/test_scaffold.py`

**Interfaces:**
- **`_RUNG_PROMPT`**, the system prompt. Its content:
  - **Role:** a tutor guiding a student to an answer it already knows.
  - **Inputs:**
    - QUESTION, TASK, RECENT_STEPS;
    - HELD_BACK_ANSWER and KEY_POINTS, marked "never state these";
    - MOVE with its one-line definition from spec §7.4;
    - RUNG "n of m";
    - THREAD (prior rungs and replies);
    - LATEST_REPLY.
  - **Rules:**
    - judge LATEST_REPLY (`none` on rung 1);
    - acknowledge what is right in a partial reply;
    - never say just "no";
    - exactly one thing for the student to do;
    - 2–4 sentences;
    - for `orient`, name the step, value or KB source where the answer lives;
    - output only JSON `{"verdict": ..., "message": ...}`.
- **`next_rung(llm, thread, move, *, task, cards, latest_reply) -> tuple[str, str, bool]`** returns (verdict, message, leak_retry):
  - one retry on an invalid reply (unparseable, or empty message);
  - on a leak, one regeneration with "your last draft gave away: <point>", then `FALLBACK_RUNGS[move]`;
  - raises `RungFailed` after two failed calls, so the caller releases the answer (fail open).
- An unknown verdict is coerced to `none`.

- [ ] **Step 1: Write the failing tests (stub LLM):**
  - a good reply → (verdict, message);
  - the prompt contains the move definition and "never state";
  - unparseable then good → succeeds on the retry;
  - two failures → `RungFailed`;
  - leak then clean → the second message, with `leak_retry=True`;
  - leak twice → the fallback text;
  - an unknown verdict → `none`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass.**

### Task 6: Engine methods that run a thread

**Files:**
- Modify: `biomentis/agent/tutor/engine.py`
- Test: `biomentis/agent/tests/test_scaffold_engine.py`

**Interfaces:**
- **`TutorEngine.__init__`** sets three attributes:
  - `self.scaffold_level = scaffold_level_default()`;
  - `self.scaffold_control, self.scaffold_locked = scaffold_control_default()`;
  - `self.scaffold_questions_asked = 0`.
- **`set_scaffold(level=None, control=None) -> bool`:**
  - when `scaffold_locked`, returns False and changes nothing;
  - logs a `scaffold_setting` record on every effective change;
  - sets `scaffold_locked = True` when control is `teacher` and `scaffold_questions_asked > 0`. `start_question` re-checks this, so choosing Teacher before the first question locks it at that question.
- **`ScaffoldStep`**, a dataclass `(thread: ScaffoldThread | None, released_turn: dict | None, reask: str | None)`:
  - `released_turn` is the UI turn dict (today's keys);
  - `reask` holds a new question to handle when a thread was abandoned.
- **`start_question(question, *, surface, step_id, ask_kwargs) -> ScaffoldStep`:**
  1. Call `chat.ask(question, scaffold_level=self.scaffold_level, record=False, **ask_kwargs)`.
  2. Run `decide`.
  3. If direct or the turn failed: `chat.record(...)`, write a `scaffold` record with `released="direct"`, and return the released turn.
  4. Otherwise build the thread, call `next_rung` for the first move, and return the thread. On `RungFailed`, release with `released="error"`.
- **`reply(thread, text) -> ScaffoldStep`:**
  1. Call `next_rung` for the next move.
  2. Check `release_reason_for(verdict, text)` *before* using the new message.
  3. On a release reason: release. For `abandoned`, return `reask=text`.
  4. Otherwise append the rung and return the thread.
- **`hint(thread)`:** counts a hint and advances one rung without a reply, or releases as `exhausted` when no moves remain.
- **`reveal(thread)`:** releases as `button`.
- **Releasing:**
  - fills `thread.released`;
  - composes the content (Task 2);
  - calls `chat.record(question, turn, step_id, thread)`;
  - logs a `scaffold` record (spec §10, with `seconds_to_release`);
  - returns the released turn dict.

  The turn dict is built by one shared function, `_turn_to_ui_dict(turn, content)`, moved here from the two dict literals in `ui_tutor.py`, so both surfaces stay identical.

- [ ] **Step 1: Write the failing tests (stub LLM and a temp `SessionLogger`):**
  - a direct question → released turn, one `qa` record, one `scaffold` record `direct`;
  - a conceptual question at standard → a thread with rung 1 and no `qa` record yet;
  - a correct reply → `earned`, with ✓ in the content;
  - `reveal` → `button`;
  - `hint` twice at standard/developing → `exhausted` on the second;
  - "just tell me" → `frustration`;
  - a `new_question` verdict → `abandoned` plus `reask`;
  - `RungFailed` on rung 1 → `error`, with the answer released;
  - challenge + expert → a stretch question in the content;
  - **lock:** Teacher chosen, first question asked, then `set_scaffold(level="off")` returns False and the level is unchanged;
  - an env preset is locked from the start;
  - a `scaffold_setting` record is written per change.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass. Re-run the Task 4 list.**

## Phase 3: UI

### Task 7: Sidebar control

**Files:**
- Modify: `biomentis/ui_tutor.py` (`render_tutor_sidebar`, new `_render_scaffold_control`)
- Test: `biomentis/agent/tests/test_scaffold_ui.py`

**Behaviour:** placed directly under the plot-mode control.
- **Before the lock:**
  - a radio "Question support set by", with options Students / Teacher (key `biomni_tutor_scaffold_control`, default from the engine);
  - a radio "Question support" over the four labels (key `biomni_tutor_scaffold_level`).

  Both call `tutor.set_scaffold`.
- **Under Teacher, before the first question:** a caption, "Locks for this session when the first question is asked."
- **Locked:** the radios are not rendered at all; render only the caption "Question support: **Guide me**, set by your teacher". Removing the widgets, rather than disabling them, is what makes Review Focus 4 hold: session-state edits to a widget key that is never rendered are ignored, because the engine is the source of truth.
- **Env preset:** the same locked caption from the first render.

- [ ] **Step 1: Write the failing AppTest tests** (with an `AppTest.from_function` harness that builds a `TutorEngine` with a stub LLM, like `test_plot_mode._harness`):
  - the defaults are Students and Guide me;
  - all four levels are selectable, including Answer directly;
  - after choosing Teacher and calling `start_question` once, the radios are gone and the caption shows;
  - setting `at.session_state["biomni_tutor_scaffold_level"] = "off"` and re-running leaves the engine's level unchanged;
  - with `BIOMENTIS_SCAFFOLD_CONTROL=teacher` and `BIOMENTIS_SCAFFOLD_LEVEL=challenge`, it is locked with "Challenge me" from the first run.

  Per `CLAUDE.md`, stage widget values and assertions in the same `at.run()`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass.**

### Task 8: Page chat panel

**Files:**
- Modify: `biomentis/ui_tutor.py` (`render_tutor_chat_panel`, `_handle_chat_turn`)
- Test: `biomentis/agent/tests/test_scaffold_ui.py`

**Behaviour:**
- **Context building.** `_handle_chat_turn` keeps its context-building code, but moves it into `_chat_ask_kwargs(tutor) -> dict`, which also adds `mode="chat"`, `plot_mode` and `search_dirs`. It then calls `tutor.start_question(..., surface="chat", ask_kwargs=...)`.
- **Open thread.** Stored at `st.session_state["biomni_tutor_scaffold_thread"]` as a dict. It renders after the history:
  - each rung as an assistant message under the caption `🧭 Guided · step {n} of up to {m}`;
  - each reply as a user message;
  - under the latest rung, two buttons: `💡 Hint` (key `biomni_tutor_scaffold_hint`) and `Show me the answer` (key `biomni_tutor_scaffold_reveal`).
- **Input while a thread is open.** The Ask form's placeholder becomes "Your thinking… (or ask something new)", and submitted text goes to `tutor.reply(thread, text)`.
- **On release:**
  - append to `biomni_tutor_chat_history` the user turn (the question) and the rungs, as display-only turns with role `assistant`/`user` and a `scaffold_rung: True` flag so the export can label them;
  - then the released turn;
  - clear the thread.

  On `reask`, handle the new text as a fresh question in the same click.
- **Timing.** The `(took …)` suffix stays on released answers. Rungs get none.

- [ ] **Step 1: Write the failing AppTest tests** (harness app rendering `render_tutor_chat_panel` with a stub-LLM engine whose replies are queued per test):
  - a definition at standard → answered directly, no thread;
  - a conceptual question → rung 1 with the caption "step 1 of up to 2" and both buttons;
  - Show me the answer → the answer appears with its citations expander, and the thread is gone;
  - a correct reply → ✓ answer;
  - a new question mid-thread → the old thread is in history as abandoned, and the new question is processed;
  - a rerun with no input → the thread is re-rendered unchanged (Review Focus 5);
  - level off → today's flow (one assistant turn, no caption).
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass, plus `test_chat_visuals` and `test_tab_focus`.**

### Task 9: Per-step box

**Files:**
- Modify: `biomentis/ui_tutor.py` (`_render_step_qa`, `_handle_step_qa`)
- Test: `biomentis/agent/tests/test_scaffold_ui.py`

**Behaviour:**
- The thread lives at `run.setdefault("step_qa_thread", {})[step_id]`. It renders inside the step's "Questions about this step" expander, below the past Q&A, with the same caption and buttons. Button keys are suffixed with `{run_id}_{step_id}`, as the existing form keys are.
- The step form's text input feeds `reply` while a thread is open.
- On release, the existing `run["step_qa"][step_id]` entry gets `question` and `answer`, in today's shape, plus `"rungs": [...]`. Under the Q/A pair, the expander lists the rungs as "🧭 Guided steps (n)", collapsed.
- `ask_kwargs` for this surface are exactly today's `_handle_step_qa` arguments.

- [ ] **Step 1: Write the failing AppTest tests:**
  - a conceptual step question → a thread inside the step's expander;
  - Show me the answer → the step's Q/A entry in today's shape, with the rungs listed;
  - two different steps keep independent threads;
  - level off → today's behaviour.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm all pass.**

### Task 10: Docs and full suite

**Files:**
- Modify: `docs/tutor.md` (new section "Question support", written for teachers: the levels, who controls them, the env presets, what gets logged)
- Modify: `docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md` §9 (settings live in `scaffold.py` helpers, not `config.py`)
- Modify: `CLAUDE.md` (add the four test commands below)
- Modify: `biomentis/ui_core.py` HTML export only if a test shows `scaffold_rung` turns rendering badly. Otherwise leave it alone.

```
$P -m biomentis.agent.tests.test_scaffold         # question support: policy, tiers, release rules, leak check, rung call
$P -m biomentis.agent.tests.test_scaffold_chat    # question support: ask() unchanged at off, profile/key points/stretch, deferred record
$P -m biomentis.agent.tests.test_scaffold_engine  # question support: threads end to end, teacher lock, logging
$P -m biomentis.agent.tests.test_scaffold_ui      # question support: sidebar control + lock, page chat and step box threads
```

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run every test command in `CLAUDE.md`.** Report any failure by name, including failures this work did not cause.
- [ ] **Step 3: Live check with the user's local Ollama model.** Ask one logistics question, one definition, one conceptual question at Guide me, and one sophisticated question with reasoning at Challenge me. Confirm:
  - the first two are answered directly;
  - the third gets 1–2 rungs;
  - the fourth gets probe → find_flaw → extend, and a stretch question after the release;
  - the session log has a `scaffold` record for each.

  Report the per-rung latencies.
- [ ] **Step 4: Run `python3 ~/streamlit-component-registry/scripts/registry.py check requirements.txt`** and report the result. No components are added, so it should be unchanged.

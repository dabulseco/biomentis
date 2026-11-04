# Scaffolded answers to student questions

**Date:** 2026-11-04
**Status:** reviewed 2026-11-04 (open questions resolved, §13)
**Scope:** the two places students ask the tutor questions: the page-level
"Ask about this task" chat (`_render_chat_panel` → `_handle_chat_turn`) and
the per-step "Ask about this step" box (`_render_step_qa` → `_handle_step_qa`),
both in `biomentis/ui_tutor.py`, both answered by `TutorChat.ask` in
`biomentis/agent/tutor/chat.py`.

---

## 1. Intent

When a student asks a question that is worth thinking about, the tutor should
not always answer it outright. It should first help the student build the
answer: orient them, ask them to predict, give a hint, work part of it. The
full answer comes once they get there or once enough friction has been spent.

The friction must be **productive**: enough that the student has to think, never
so much that they give up or resent the tool. And it must be **adaptive**: the
kind of question decides whether there is any friction at all, and a more
sophisticated question earns a harder, more interesting exchange, not just a
slower one.

## 2. Decisions already made

| Decision | Choice |
|---|---|
| Where friction applies | **Only inside the tutor chat** (both surfaces). The agent's answer, transcript and Executor tab are untouched. A student who scrolls up to find and understand the answer is doing productive work too. |
| Who sets the level | Configurable per session: **Students choose** (default) or **Teacher sets**. Teacher mode locks after the first question; no PIN. |
| Default level | **Guide me** (`standard`). |
| Students may pick "Answer directly" | Yes, and the choice is logged. |
| Definitions | Direct, except under **Challenge me**, where they get one guided step. |
| Stretch question | Only under **Challenge me**, and only for *very* sophisticated questions (the `expert` tier, §7.2). |
| Approach | Classify the question, run a hint ladder whose rules live in code, draw hints from a full answer generated first and held back, and scale the challenge with the question's sophistication. Level setting and logging from the start. |

## 3. Goals

- **G1. Classify every question** by type and sophistication before deciding how to respond.
- **G2. Answer directly** when friction would only frustrate: logistics, "what did the agent do", definitions (except under Challenge me), and anything when the level is Off.
- **G3. Hint ladder** for questions worth thinking about, with the rung, the cap and the release rules enforced in code rather than in the prompt.
- **G4. Consistent hints:** every hint is drawn from a full answer generated up front and held back, and a hint that gives away the answer is caught.
- **G5. Adaptive challenge:** a more sophisticated question gets more rungs and harder moves (probe an assumption, find the flaw, extend) instead of more of the same.
- **G6. No dead ends:** a "Show me the answer" button is always there, there is a hard cap on rungs, and frustration signals release the answer early.
- **G7. Level control:** four levels; Students choose or Teacher sets, chosen when the session starts.
- **G8. Teacher-visible record:** every thread is logged with its classification, rungs, the student's replies and why the answer was released.

## 4. Non-goals

- Hiding or delaying anything outside the chat (the agent's answer, the Executor tab, figures).
- Friction on the main prompt box that starts an agent run.
- Grading students. Verdicts on replies exist to choose the next rung, not to score anyone.
- *Phase 2, not this spec:* naming specific misconceptions in a reply, a learner model that persists across sessions, and a teacher-written policy file ("always scaffold statistics questions in this lab"). The design leaves room for each (§12).

## 5. Anti-goals (failure even if it "works")

- A student who asks "where is the export button?" gets a Socratic question.
- A student is told "try again" with nothing new to go on.
- The hint quietly states the answer, so the friction is theatre.
- The tutor withholds an answer the student has visibly earned (they said it, roughly).
- A student cannot tell how far they are from the answer or how to skip ahead.
- A strong question gets the same treatment as a vague one.

## 6. User experience

### 6.1 Levels

The student-facing name is **Question support**.

| Level | Label | Behavior |
|---|---|---|
| `off` | Answer directly | Today's behavior. No ladder. |
| `light` | A nudge first | At most one rung, then the answer. |
| `standard` | Guide me | One to two rungs, depending on the question. |
| `challenge` | Challenge me | Up to three rungs, harder moves, one rung on definitions, and a stretch question after the answer for very sophisticated questions. |

Default level: `standard`. `BIOMENTIS_SCAFFOLD_LEVEL` can change the default.

### 6.2 Who controls the level

A sidebar control in the 🎓 Tutor section, **Question support set by: Students / Teacher**, defaulting to *Students*.

- **Students:** the student sees the level picker, including Answer directly, and can change it at any time. Every change is logged, so the teacher can see who switched to "Answer directly" and when.
- **Teacher:** the picker is replaced by a read-only line ("Question support: Guide me, set by your teacher").

So that a student can't simply flip it back:
- Once *Teacher* is chosen and the first question has been asked, the control locks for the rest of the session.
- `BIOMENTIS_SCAFFOLD_CONTROL=teacher` (with `BIOMENTIS_SCAFFOLD_LEVEL`) presets and locks it from `.env`, for classroom machines the teacher sets up in advance. With this set, the control never appears.

No teacher PIN: the lock after the first question plus the `.env` preset is enough.

### 6.3 A scaffolded exchange

Here is a scaffolded exchange on the page-level chat:

```
Student: Why did the agent filter BLAST hits at E < 1e-5 instead of 0.05?

🧭 Guided · step 1 of up to 2                         [💡 Hint] [Show me the answer]
Good question to stop on. Before I explain: the search ran against ~250 million
sequences. If you used 0.05, roughly how many hits would you expect by chance
alone? Look at what an E-value counts.

Student: like 0.05 of them? so a lot of false ones

✓ That's the core of it.
<full answer, as today: citations, "Also worth knowing", figures, Bloom/DOK badges>
```

Rules for every rung:

- **Say what kind of help this is and how far the answer is.** The caption reads "step 1 of up to 2", and the button is always visible. Being open about the friction is what keeps it from feeling like a wall.
- **Give exactly one thing to do:** predict, look at a specific step or value, finish a sentence, or compare two cases.
- **Acknowledge a partial reply** ("Right about X. Now what about Y?"), and never respond with a bare "no".
- **💡 Hint** advances one rung without needing an attempt. **Show me the answer** releases it immediately.
- After a release, the answer renders exactly as today's answers do, so nothing downstream changes.

The per-step box ("Ask about this step") works the same way inside the step's expander. Its single-line input becomes the reply box while a thread is open there.

### 6.4 When the student types something unrelated mid-thread

The rung call judges each reply (§8.3). If the reply is a new question rather than an attempt (`verdict = new_question`):
- the open thread is closed and logged as `released: "abandoned"`,
- the new text is handled as a fresh question.

The student never has to press "New question".

## 7. Classification and policy

### 7.1 The question profile

The profile is produced by the same LLM call that writes the held-back answer (§8.1), so it adds no extra call:

```json
"question_profile": {
  "type": "logistics | definition | agent_action | procedural | conceptual | evaluative",
  "bloom": "Remember | Understand | Apply | Analyze | Evaluate | Create",
  "dok": 1,
  "specific": true,
  "shows_reasoning": false,
  "wants_direct": false
}
```

- `specific`: the question names a step, value, tool or result.
- `shows_reasoning`: the student offers their own hypothesis or explanation ("I think X because Y…").
- `wants_direct`: an explicit "just tell me" or an urgent, time-boxed request.

The Bloom levels reuse the existing vocabulary (`rubric.BLOOM_LEVELS`).

### 7.2 Sophistication tier

The tier is a deterministic function in code, so it can be unit-tested:

| Tier | Rule |
|---|---|
| `expert` | At least **two** of: `shows_reasoning`; bloom ∈ {Evaluate, Create}; dok ≥ 3 |
| `advanced` | Not `expert`, and `shows_reasoning`, **or** bloom ∈ {Analyze, Evaluate, Create}, **or** dok ≥ 3 |
| `novice` | bloom ∈ {Remember, Understand} **and** dok ≤ 2 **and** not `specific` |
| `developing` | everything else |

### 7.3 Policy: direct or ladder

`plan = decide(level, profile, tier)` is a pure function.

**Always direct:**
- level `off`;
- type ∈ {logistics, agent_action};
- type `definition` at any level except `challenge`;
- `wants_direct` at level `light` or `standard`.

Under `challenge`, a `definition` question gets a single rung, `predict` ("what do you think it means here, given step N?"), whatever its tier.

At level `challenge`, `wants_direct` still gets one rung, but "Show me the answer" is highlighted.

**Ladder otherwise.** Each cell gives the moves in order, and the number of moves is the rung cap:

| | novice | developing | advanced |
|---|---|---|---|
| **light** | orient | predict | probe |
| **standard** | orient | predict → hint | probe → contrast |
| **challenge** | orient → hint | predict → partial | probe → find_flaw → extend |

The `expert` tier uses the `advanced` column.

`procedural` questions use the same table with the cap reduced by one, minimum one. How-to questions benefit less from long ladders.

At level `challenge` with an `expert` question, and only then, the released answer's follow-up becomes a **stretch question** that goes beyond what was asked. At every other level and tier the follow-up stays as today.

**Hard cap:** never more than 3 rungs, whatever the table says.

### 7.4 The moves

| Move | What the tutor does |
|---|---|
| `orient` | Points to where the answer lives (the step, value or KB page) and asks what it suggests |
| `predict` | Asks the student to predict an outcome before revealing it |
| `hint` | Names the concept that matters, without applying it |
| `partial` | Works the first half and asks the student to finish |
| `probe` | Questions an assumption in the student's question or reasoning |
| `contrast` | Offers two cases and asks what differs |
| `find_flaw` | Shows a subtly wrong version and asks what's wrong |
| `extend` | Asks how the answer would change under a different condition |

When the knowledge base has passages relevant to the question, `orient` points to the source ("page 9 of your reading covers this").

### 7.5 When the answer is released

The thread releases the answer and logs the reason:

| `released` reason | Trigger |
|---|---|
| `direct` | The policy chose direct |
| `earned` | The latest reply is judged `correct` |
| `button` | The student pressed Show me the answer |
| `exhausted` | The rung cap was reached; the next reply or 💡 Hint releases it |
| `frustration` | The reply is judged `gave_up`, **or** matches the give-up patterns ("just tell me", "idk", "i don't know", "no idea", "?"), **or** two replies in a row are under 4 words with no attempt in them |
| `abandoned` | The student moved on to a new question |

An `earned` release opens with an acknowledgment ("✓ That's the core of it.") before the full answer.

## 8. Generation

### 8.1 The held-back answer

When the level is not `off`, the existing `TutorChat.ask` call is extended. The same prompt and the same JSON gain two keys:
- `question_profile` (§7.1)
- `key_points`: 1–3 short phrases, each a conclusion the student should reach.

Everything else (KB retrieval, citations, "Also worth knowing", plots, and `classify_qa`) runs exactly as today, so the held-back answer is a complete `ChatTurn`.

**Cost:**
- *Direct questions:* the same number of LLM calls as today. The profile rides along in the call that was happening anyway.
- *Scaffolded questions:* one extra call per rung.
- *Latency:* the first rung waits for the full answer and then the rung call. With a local Ollama model that is roughly 5–15 s more than today for scaffolded questions only. A later optimization (§12) could start the first rung before the answer.

### 8.2 The rung call

This is a new prompt, `_RUNG_PROMPT`, in a new module `biomentis/agent/tutor/scaffold.py`.

- **Input:**
  - the question, task and recent steps;
  - the held-back answer and its key points;
  - the move to make and the rung number with the cap;
  - the thread so far;
  - the student's latest reply.
- **Output:** strict JSON:

```json
{"verdict": "correct | partial | off_track | gave_up | new_question | none",
 "message": "<the rung text>"}
```

`verdict` judges the student's latest reply, and is `none` on the first rung. The message must contain exactly one thing for the student to do.

### 8.3 Leak check

Before a rung is shown, the code checks whether any key point appears in the message, after normalizing case, punctuation and whitespace:
- on a match, the rung is regenerated once with "your last draft gave away: <point>";
- if it still leaks, a fixed fallback for that move is used (e.g. for `orient`: "Look back at step N. What does it tell you about this?").

A cheap string check is enough here. It catches the blatant leaks that make friction pointless, and rarely fires on a near-miss phrasing.

### 8.4 Failure handling

| Failure | What happens |
|---|---|
| Profile missing or malformed | Answer directly, and log `profile_failed` |
| Rung call fails twice | Release the answer, and log `released: "error"` |

A tutor that fails must fail *open*: the student gets the answer, never a stuck thread.

## 9. Code layout

| Where | What |
|---|---|
| `biomentis/agent/tutor/scaffold.py` (new) | `QuestionProfile`, `tier()`, `decide()` → `ScaffoldPlan`, `ScaffoldThread` (state: question, plan, rung, held-back `ChatTurn`, replies, verdicts, released reason), `next_rung()` (rung call + leak check), `is_give_up()`, `_RUNG_PROMPT`, fallback texts |
| `biomentis/agent/tutor/chat.py` | `ask(..., scaffold_level="off")`: when not off, adds `question_profile` and `key_points` to the requested JSON and sets them on `ChatTurn` (new fields `question_profile: dict`, `key_points: list[str]`) |
| `biomentis/agent/tutor/engine.py` | `TutorEngine.scaffold_level`, `scaffold_control`; `start_thread(turn)` / `reply(thread, text)` / `hint(thread)` / `reveal(thread)` |
| `biomentis/ui_tutor.py` | Sidebar "Question support" control + lock; chat panel and step Q&A render open threads (caption, 💡 Hint, Show me the answer), route input to `reply` while a thread is open; released threads append the held-back turn exactly as `_handle_chat_turn` does today |
| `biomentis/config.py` | `BIOMENTIS_SCAFFOLD_LEVEL`, `BIOMENTIS_SCAFFOLD_CONTROL` |
| `docs/tutor.md` | A section on question support for teachers |

The open thread lives in `st.session_state`:
- for the page chat, at `biomni_tutor_scaffold_thread`;
- for the step box, at `run["step_qa_thread"][step_id]`.

It survives reruns the same way chat history does today.

## 10. Logging

Each thread writes one `scaffold` record when it closes, next to the existing `qa` record for the released answer:

```json
{"kind": "scaffold", "surface": "chat | step", "step_id": null,
 "question": "...", "level": "standard", "control": "students",
 "profile": {...}, "tier": "developing", "plan": ["predict", "hint"],
 "rungs": [{"move": "predict", "message": "...", "reply": "...", "verdict": "partial",
            "leak_retry": false}],
 "released": "earned", "hints_requested": 0, "seconds_to_release": 74}
```

Changes to the level or control mode are logged as `scaffold_setting` records, like today's `plot_mode` records. The session-log export already includes all of these.

## 11. Testing

Everything runs on the usual `_check()` harness with stub LLMs. Tests go in `biomentis/agent/tests/test_scaffold.py`, which is also added to the `CLAUDE.md` list.

1. **`tier()` and `decide()`:** every cell of the §7.3 table; the `expert` two-of-three rule; definitions direct except one `predict` rung under `challenge`; the stretch question only for `challenge` + `expert`; logistics and agent_action always direct; `wants_direct` per level; procedural reduction; hard cap of 3.
2. **Release rules:** each reason in §7.5, including the "two short replies" rule and the give-up patterns, with no false positives on short correct replies such as "false positives".
3. **Leak check:** a leaking rung is regenerated once and then falls back; normalization catches case and punctuation variants.
4. **Fail open:** a bad profile leads to a direct answer; a rung call failing twice leads to a release with `error`.
5. **`ask()` with `scaffold_level="off"`** sends the same prompt as today. This guards against regressions in the existing chat.
6. **UI (AppTest):**
   - the default control is Students with level `standard`;
   - Teacher mode shows the read-only line and locks after the first question;
   - the env preset hides the control;
   - a scaffolded question shows the caption and both buttons;
   - Show me the answer releases the answer;
   - a reply judged correct releases it;
   - a new question mid-thread abandons the old one;
   - the step box behaves the same as the page chat.
7. **Logging:** one `scaffold` record per thread with the fields above, and `scaffold_setting` records on change.

## 12. Later (phase 2)

- **Misconception diagnosis:** a `partial` or `off_track` verdict names the specific misconception, and the next rung targets it.
- **Learner model:** concepts a student has demonstrated get less friction later in the session, and optionally across sessions (stored like `memory.py`).
- **Teacher policy file:** a markdown file of per-assignment rules that override `decide()` for matching question types or topics.
- **Faster first rung:** start an `orient` rung from the question alone while the held-back answer is generated in the background.

## 13. Resolved questions

| # | Question | Decision |
|---|---|---|
| Q1 | Teacher lock strength | Lock after the first question plus the `.env` preset. No PIN. |
| Q2 | Default level | `standard` (Guide me). |
| Q3 | Can students pick Answer directly? | Yes, logged. |
| Q4 | Scaffold definitions under Challenge me? | Yes: one `predict` rung. |
| Q5 | Who gets the stretch question? | Only `expert`-tier questions under Challenge me. |

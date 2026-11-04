# Biomentis-Tutor User Guide

The tutor is an optional instructional layer over the Biomentis A1 agent. It
turns every step the agent takes into a teaching moment, lets students
ask follow-up questions in a chatbot that's grounded in their course
materials, and logs everything for later analysis against Bloom's
taxonomy, Webb's DOK, and a teacher-supplied rubric.

> **The tutor is opt-in.** With it disabled, Biomentis runs exactly as
> before — the agent streams events, the user submits follow-ups, and
> the rest of the UI is unaffected. Turning the tutor on adds a
> sidebar panel, a chat column, and a per-step pause for each
> teaching card.

---

## Quick start

1. Install the optional tutor dependencies (one-time):

    ```bash
    pip install pypdf python-pptx python-docx tiktoken chromadb sentence-transformers
    ```

    On the first upload, the embedding model
    (`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB) downloads to
    your local Hugging Face cache. After that, KB ingestion is fully
    offline.

2. Start Biomentis:

    ```bash
    streamlit run streamlit_app.py
    ```

3. In the **🎓 Tutor (optional)** sidebar:
    - Flip **Enable instructional mode** on.
    - Choose the **instruction modes** — 🔧 **Technical details**,
      🔬 **Scientific content**, both, or neither (see
      [Instruction modes](#instruction-modes)).
    - Upload one or more course materials (PDF / PPTX / DOCX / TXT) and
      click **📥 Add to KB**. Optionally paste URLs (one per line) and
      add those too.
    - (Optional) Upload a rubric YAML. Otherwise the discipline-agnostic
      default rubric is used.

4. Submit a task in the main prompt area. The agent will run to
   completion; events stream into the right panel. After every
   reasoning / code / observation / solution / file event, a **🎓
   Teaching note** appears, carrying one colored box per enabled
   instruction mode plus the shared **Sources** from the uploaded KB and
   the **Bloom** / **DOK** target levels.

   Click **▶ Continue** to advance to the next step.

5. Use the **💬 Ask the tutor** column on the right to ask follow-up
   questions. Every Q&A is automatically classified against your
   rubric and logged.

6. At any point, download the **Session log** (JSON or two CSVs) from
   the tutor sidebar for grading, analysis, or export to a learning
   record store.

---

## What the tutor does (and doesn't)

**Does:**

- Adds a KB ingestion pipeline (PDF, PPTX, DOCX, TXT, URLs) backed by
  a local Chroma vector index.
- Generates a teaching card for every reasoning, code, observation,
  solution, summary, and file event the agent emits, through two
  independently-switchable lenses (technical / scientific).
- Pauses the run between cards so the student can read at their own
  pace.
- Grounds both the cards and the tutor chat in the KB, with strict
  citation whitelisting (no invented sources).
- Logs every step (with its card) and every Q&A (with Bloom/DOK
  classification and rubric hits) to a per-session JSONL file.
- Exposes the log as JSON + two CSVs for download.

**Doesn't (yet):**

- Pause the *agent itself* in LangGraph — the agent runs to
  completion, and the tutor drip-feeds events to the student. The
  Continue button is a UI affordance, not a thread interrupt.
- Stream live during the agent run. The first submit may take
  seconds to minutes while the agent works; events and cards appear
  once the agent finishes.
- Track individual learners across sessions. Each Streamlit session
  has its own `biomentis_session_id`; the KB and log live under
  `data/tutor_kb/<session_id>/` and `data/tutor_logs/<session_id>/`.

---

## Instruction modes

A teaching card explains a step through two independent lenses. Each has
its own toggle in the sidebar, and all four combinations are valid — turn
on one, both, or neither.

| | 🔧 Technical details | 🔬 Scientific content |
|---|---|---|
| Color | blue | violet |
| **What** | The operation being run — the actual method, tool, library, API, or algorithm, and the parameters that matter | The science being done — the entities involved, the property measured, the question interrogated |
| **Why** | Why *this* technique or tool: what it buys over the alternative, what breaks without it | Why it matters scientifically: the mechanism or principle, what claim the result licenses, what a strong vs. weak result would mean, and the caveats |
| **How this builds the answer** | — | How this step's result changes the material being assembled to answer *your specific query*, and what the final answer would be missing without it |
| **Prerequisites** | What must already exist or be installed for the step to run — upstream outputs, dependencies, required formats | Background concepts you need in order to follow the science |
| **What to look for** | How to tell the step technically succeeded | Scientifically meaningful signals in the result, and what they mean |

The two are prompted to repel each other: the technical lens is told not to
explain biology, and the scientific lens is told not to name functions or
parameters. Read side by side they complement rather than repeat; read alone
either one stands on its own.

Only the enabled modes are requested from the LLM, so running one lens costs
roughly half of running both. With **both modes off**, the walkthrough still
paces the run step by step and the per-step **Ask about this step** box still
works — no teaching cards are generated and no tokens are spent on them.

Switching a mode on mid-run regenerates the affected cards: card caching is
keyed by the mode set, so you never get a card that is silently missing the
lens you just enabled.

---

## Question support

When a student asks the tutor chat a question worth thinking about, the
tutor can guide them toward the answer in a few short steps instead of
answering at once. This applies to both chat surfaces, **Ask about this
task** and **Ask about this step**, and nowhere else: the agent's own
answer, the transcript and the teaching cards are never held back. A
student who scrolls up to find and understand the answer there is doing
useful work too.

### Levels

| Level | What the student gets |
|---|---|
| **Answer directly** | Today's chat: the answer at once. |
| **A nudge first** | One short step (orient, predict or probe), then the answer. |
| **Guide me** (default) | Up to two steps, scaled to the question. |
| **Challenge me** | Up to three steps, harder ones for sophisticated questions, and a stretch question after the answer for the most sophisticated. |

Whatever the level, these are always answered directly:
- questions about the app or the run ("where is the export button?");
- requests for the agent to do something.

Definitions ("what is an E-value?") are also answered directly, except
under **Challenge me**, where they get one prediction step first. A student
who says they want the answer straight is answered directly below
**Challenge me**; under it they get a single step with the way out
highlighted.

The tutor reads each question's type and sophistication (Bloom level,
depth, whether it shows the student's own reasoning) from the same call
that writes the answer. A more sophisticated question gets harder steps:
probing the student's reasoning, finding a flaw, extending the idea. A
question that only recalls a fact gets an orienting step. There are never
more than three steps.

### Never stuck

Every guided step shows **💡 Hint** (the next step) and **Show me the
answer**. The answer is also released when:
- the student's reply gets it right; the answer opens with "✓ That's the
  core of it.";
- the student says they give up ("idk", "just tell me", "?");
- the student sends two very short replies in a row that miss;
- the steps run out;
- the student asks something else, which is then answered as a new
  question;
- a step can't be written (the model fails twice).

The steps never state the answer. Each one is checked against the
answer's key points and rewritten if it gives one away.

### Who sets it

The sidebar's **Question support set by** switch is **Students** by
default, which lets students pick any level, **Answer directly** included.
Every change is logged.

Set it to **Teacher** at the start of a session and choose the level. The
setting locks when the first question is asked, the controls disappear, and
the sidebar shows "Question support: **Guide me**, set by your teacher".
The lock lasts for the rest of the session.

To preset a class, set these before launching the app:

```bash
export BIOMENTIS_SCAFFOLD_LEVEL=standard   # off | light | standard | challenge
export BIOMENTIS_SCAFFOLD_CONTROL=teacher  # students | teacher
```

With `BIOMENTIS_SCAFFOLD_CONTROL=teacher` the level is locked from the
first page load. An unset or unknown level means **Guide me**; an unset or
unknown control means students choose.

### What gets logged

Each guided exchange writes a `scaffold` record when its answer is
released, next to the usual `qa` record (see [Log schema](#log-schema)).
It records:
- the steps and the student's replies;
- the tutor's verdict on each reply;
- why the answer was released;
- how many hints were asked for;
- the time from question to answer.

At **Answer directly** the log is exactly what it was before question
support existed.

---

## Knowledge base

### Supported file types

| Type | Loader                                  | Notes                                       |
|------|-----------------------------------------|---------------------------------------------|
| PDF  | `pypdf`                                 | Per-page metadata preserved                 |
| PPTX | `python-pptx`                           | Slide text concatenated                      |
| DOCX | `python-docx`                           | Paragraph text concatenated                 |
| TXT  | Plain read                              | Treated as one document                     |
| URL  | `requests` + `beautifulsoup4`           | HTML stripped, main text extracted          |

### Chunking

Documents are split into ~500-token chunks with an 80-token overlap
using `tiktoken` (falls back to whitespace tokenization if `tiktoken`
isn't installed). Each chunk carries metadata (`source`, `page` for
PDFs, `chunk_id`).

### Embeddings

Default model: **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim,
CPU-friendly, ~80 MB). The first `add_files` call downloads the model
to your local Hugging Face cache. To swap models, construct
`KnowledgeBase(..., embedding_model="your-model-id")` and pass your
own `TutorEngine` to `biomentis.ui_tutor.install_renderers()`.

### Citation rules

- The instruction generator and the tutor chat both receive the
  retrieved KB snippets inline in their prompts.
- The LLM is told to cite only those snippets, with a strict prompt:
  *"Never invent a source. If none are relevant, return
  `citations: []`."*
- On the parse side, the `InstructionCard` and `ChatTurn` validators
  drop any citation whose `source` isn't in the retrieved set. You
  can trust the displayed sources.

### Storage layout

```
data/
├── tutor_kb/
│   └── <session_id>/
│       ├── raw/                # original uploaded files
│       └── index/              # Chroma persistent index
└── tutor_logs/
    └── <session_id>/
        └── <session_id>.jsonl  # append-only log
```

To share a KB across sessions, copy the `raw/` + `index/` directories
into a new `tutor_kb/<other_session_id>/`. The KB is re-loaded from
disk on first use.

---

## Rubric

The default rubric (`biomentis/agent/tutor/default_rubric.yaml`) is
discipline-agnostic and has 5 objectives spanning Remember through
Create. It looks like this:

```yaml
objectives:
  - id: GEN1
    description: "Identify the question being asked and the relevant biological / chemical context."
    bloom_level: Understand
    dok_level: 1
  - id: GEN2
    description: "Locate the right database, tool, or method for the question."
    bloom_level: Apply
    dok_level: 2
  # ... etc.
```

To use a course-specific rubric, upload a YAML in the tutor sidebar.
The schema is:

```yaml
objectives:
  - id: <unique short id, e.g. "CHIKV-1">
    description: <one-sentence objective>
    bloom_level: <Remember | Understand | Apply | Analyze | Evaluate | Create>
    dok_level: <1 | 2 | 3 | 4>
```

The rubric is used in two places:

1. **Q&A classification** — every tutor-chat answer is scored
   against the rubric's objectives; matching ids appear in
   `rubric_hit` in the log.
2. **(Implicitly) Instruction generation** — the rubric's Bloom
   levels inform the default Bloom targets for steps, so a teacher
   can shape the cognitive profile of the cards just by writing the
   rubric.

Click **↺ Reset to default** to restore the default rubric.

---

## Log schema

Every session writes a JSONL file at
`data/tutor_logs/<session_id>/<session_id>.jsonl`. Three record kinds:

### `step` — one teaching card

```json
{
  "kind": "step",
  "event_type": "code",
  "step_id": 4,
  "title": "Run BLAST",
  "bloom_target": "Apply",
  "dok_target": 2,
  "modes": ["technical", "scientific"],
  "instruction": {
    "what": "Calls run_blast() against the local PDB sequence database.",
    "why": "BLAST is the standard heuristic aligner at this query size.",
    "prerequisites": ["A FASTA query sequence", "BLAST E-value interpretation"],
    "look_for": ["A non-empty DataFrame with columns: hit_id, e_value, identity"]
  },
  "sections": {
    "technical": {
      "what": "Calls run_blast() against the local PDB sequence database.",
      "why": "BLAST is the standard heuristic aligner at this query size.",
      "prerequisites": ["A FASTA query sequence", "The pdb BLAST database is installed"],
      "look_for": ["A non-empty DataFrame with columns: hit_id, e_value, identity"]
    },
    "scientific": {
      "what": "Searches for structural homologs of the CHIKV E2 glycoprotein.",
      "why": "Homology above the twilight zone implies a shared fold, which is what makes template-based modeling of the epitope defensible.",
      "prerequisites": ["Sequence-structure relationship", "E-value as a significance measure"],
      "look_for": ["Hits concentrated in the receptor-binding domain"],
      "impact": "Supplies the structural templates the nanobody ranking downstream depends on."
    }
  },
  "kb_citations": [{"source": "course_notes.pdf", "page": 12, "snippet": "E-values below 1e-10 indicate strong homology."}],
  "generation_failed": false,
  "ts": "2026-07-19T10:11:21",
  "session_id": "abc12345"
}
```

`modes` records which instructional lenses were enabled when the card was
generated. `sections` is the per-mode split — the technical lens covers the
technology being carried out, the scientific lens covers what the step means
and how it builds the answer. `instruction` is a flattened view of the same
content (technical first, scientific as fallback) kept so the CSV export and
the Critic digest read unchanged.

### `qa` — one tutor-chat exchange

```json
{
  "kind": "qa",
  "question": "What does BLAST do?",
  "answer": "BLAST finds homologous sequences...",
  "bloom_level": "Apply",
  "dok_level": 2,
  "rubric_hit": ["GEN2"],
  "confidence": 0.78,
  "failed": false,
  "citations": [{"source": "course_notes.txt", "page": null, "snippet": "E-values below 1e-10 indicate strong homology."}],
  "ts": "2026-07-19T10:11:21",
  "session_id": "abc12345"
}
```

A guided exchange's `qa` record also carries the whole exchange under
`"scaffold"`, written when the answer is released.

### `scaffold` — one guided exchange (question support)

```json
{
  "kind": "scaffold",
  "surface": "chat",
  "step_id": null,
  "question": "Why filter BLAST hits at E < 1e-5?",
  "level": "standard",
  "control": "students",
  "profile": {"type": "conceptual", "bloom": "Apply", "dok": 2, "specific": true,
              "shows_reasoning": false, "wants_direct": false},
  "tier": "developing",
  "plan": ["predict", "hint"],
  "plan_reason": "table",
  "rungs": [{"move": "predict", "message": "What would you expect by chance in a database this size?",
             "reply": "more random hits", "verdict": "partial", "leak_retry": false}],
  "released": "button",
  "hints_requested": 0,
  "seconds_to_release": 41.2,
  "ts": "2026-11-04T10:11:21",
  "session_id": "abc12345"
}
```

`surface` is `chat` (Ask about this task) or `step` (Ask about this step).
`released` is one of:
- `direct`: no steps;
- `earned`: a correct reply;
- `button`: Show me the answer;
- `exhausted`: the steps ran out;
- `frustration`: the student gave up, or sent two short replies that
  missed;
- `abandoned`: the student asked something else;
- `error`: a step couldn't be written.

A `scaffold_setting` record (`level` and/or `control`) is written on each
change to the sidebar setting.

### `event_seen` — pre-card log (lightweight, for stream analytics)

```json
{
  "kind": "event_seen",
  "event_type": "code",
  "step_id": 4,
  "title": "Run BLAST",
  "ts": "2026-07-19T10:11:21",
  "session_id": "abc12345"
}
```

### Exports

The sidebar's **📥 JSON** button writes the full log as a single
JSON array. **📥 Steps CSV** and **📥 Q&A CSV** write one row per
step / Q&A, with the columns shown in the schema above.

---

## Example classroom workflow

1. **Pre-class.** The instructor uploads a few course PDFs (or
   points the KB at the textbook's online chapters) and uploads a
   custom YAML rubric mapping the day's objectives to Bloom/DOK
   levels. (Each student gets their own session, but the rubric can
   be reused — see the storage layout above.)

2. **In-class.** Each student runs a small task, e.g.
   *"Find 3 candidate nanobodies against CHIKV E2."* The agent
   streams events; the student reads each card, clicks Continue,
   and asks follow-up questions in the chat column.

3. **Post-class.** The student exports their session log (JSON) and
   submits it. The instructor can:

   - Read the per-step `instruction.why` to see what the student
     was *supposed* to learn at each point.
   - Score the `qa` records against the rubric by looking at
     `bloom_level`, `dok_level`, and `rubric_hit`.
   - Aggregate across students (by `session_id`) to spot common
     misunderstandings — e.g. every student is stuck at the same
     step with `confidence < 0.4` in their Q&A classifications.

4. **Iteration.** The instructor updates the rubric for next week
   based on what came up. The default Bloom/DOK calibration in
   `InstructionGenerator` (in `biomentis/agent/tutor/instruction.py`)
   can also be tuned per event type if the cards are too shallow
   or too deep.

---

## Troubleshooting

**"Knowledge base" is empty / "📥 Add to KB" hangs.**

The embedding model probably hasn't downloaded yet. The first
ingest downloads `all-MiniLM-L6-v2` (~80 MB) — give it a minute.
If you're behind a proxy, set `HF_TOKEN` or pre-download with
`huggingface-cli download sentence-transformers/all-MiniLM-L6-v2`.

**"Teaching card unavailable for this step (LLM call failed)."**

The LLM call timed out or returned malformed JSON. The agent's
work is unaffected — just click Continue and move on. The log
record for that step has `generation_failed: true` so you can
count these in the analytics.

**Tutor chat returns short answers / no KB citations.**

Two common causes:

- The KB is empty. The chat will fall back to general knowledge
  but won't be able to cite anything. Upload some files first.
- The LLM is too small to follow the JSON prompt reliably. A
  7B model will often fail; 14B+ is recommended. If the model
  returns prose instead of JSON, the chat falls back to a
  soft-failure message.

**"tutor unavailable — no LLM configured."**

The `TutorEngine` was constructed without an LLM. This happens if
the model picker failed to load. Re-select the model in the
sidebar.

**Log file is empty / no records.**

The agent hasn't run yet. Steps are logged as the agent emits
events; Q&A is logged on every chat turn.

**How do I clear a session and start over?**

- **KB only:** click **🧹 Clear KB** in the sidebar. The log
  persists; clear the log file manually if needed.
- **Everything:** delete the session's directories:
  ```bash
  rm -rf data/tutor_kb/<session_id> data/tutor_logs/<session_id>
  ```
  and reload the Streamlit app.

---

## Disabling the tutor

To remove the tutor entirely (sidebar, chat, instruction cards,
logging), set the `biomentis_tutor_enabled` toggle off in the
sidebar. The agent reverts to its original behavior.

To remove the tutor from the entry script, edit
`streamlit_app.py` and remove these lines:

```python
from biomentis.agent.tutor import TutorEngine
from biomentis.ui_tutor import (
    install_renderers, render_tutor_chat_panel,
    render_tutor_sidebar, tutor_wrapped_stream,
)
install_renderers()
# ...
tutor: TutorEngine = st.session_state.biomentis_tutor
render_tutor_sidebar(tutor)
render_tutor_chat_panel(tutor)
agent.launch_streamlit_demo(stream_fn=tutor_wrapped_stream)
```

and pass nothing (or `None`) as `stream_fn` to
`launch_streamlit_demo`. The agent and its existing UI are
otherwise unchanged.

---

## Programmatic access

If you want to use the tutor layer from a script (not the
Streamlit UI):

```python
from biomentis.agent.tutor import (
    TutorEngine, KnowledgeBase, Rubric, SessionLogger,
)
from biomentis.llm import get_llm

llm = get_llm("qwen2.5:14b", source="Ollama")
engine = TutorEngine(session_id="lab-1", llm=llm, path="./data")

# Ingest KB
engine.kb.add_files(["course_notes.pdf"])
engine.kb.add_urls(["https://www.uniprot.org/uniprotkb/Q5WBD5/entry"])

# Ask a tutor question
turn = engine.chat.ask(
    question="What does epitope conservation mean?",
    context="step 3: we're filtering variants",
    task="Find 3 candidate nanobodies.",
)
print(turn.content, turn.citations, turn.bloom_level)

# Generate an instruction card for an event
from biomentis.ui_core import UIEvent
ev = UIEvent(type="code", content="run_blast('CHIKV E2')", title="BLAST")
card = engine.instruction_gen.generate(ev, task="Find 3 nanobodies")
print(card.technical.what, card.technical.prerequisites)
print(card.scientific.why, card.scientific.impact)

# Generate only one lens (both, either, or neither is valid)
card = engine.instruction_gen.generate(ev, task="Find 3 nanobodies", modes=["scientific"])
print(card.modes, card.technical.is_empty())

# Set the modes for every subsequent card on this engine
engine.set_modes(["technical"])          # blue cards only
engine.set_modes([])                     # no cards, no LLM calls

# Inspect the log
import json
with open(engine.logger.path) as f:
    for line in f:
        record = json.loads(line)
        if record["kind"] == "step":
            print(record["step_id"], record["bloom_target"])
```

The tutor layer is framework-agnostic — only the Streamlit
adapters in `biomentis.ui_tutor.py` import Streamlit.

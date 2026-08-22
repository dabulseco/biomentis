"""Regenerate docs/tutor_instruction_cards.md.

Pulls the tutor card prompts verbatim out of
`biomentis/agent/tutor/instruction.py` (and renders real example user
messages through `_build_user_prompt`), so the reference doc cannot drift from
the source by transcription error. Run after editing either prompt:

    python docs/generate_card_prompt_doc.py

The prose sections live in this file's `doc` template -- edit them here, not
in the generated markdown.

Source line numbers are looked up at generation time by `_line()` rather than
hard-coded, since every prompt edit shifts them.
"""
import os, re, sys, types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import biomentis.agent.tutor.instruction as ins

_SRC = {
    "instruction.py": open(
        os.path.join(_REPO, "biomentis", "agent", "tutor", "instruction.py"), encoding="utf-8"
    ).read().splitlines(),
    "ui_tutor.py": open(
        os.path.join(_REPO, "biomentis", "ui_tutor.py"), encoding="utf-8"
    ).read().splitlines(),
}


def _line(pattern: str, path: str = "instruction.py") -> str:
    """`path:N` for the first line matching `pattern`, or `path` if absent."""
    rx = re.compile(pattern)
    for i, line in enumerate(_SRC[path], 1):
        if rx.search(line):
            return f"{path}:{i}"
    return path


SYS_BOTH = ins._build_system_prompt(ins.ALL_MODES).rstrip()
SYS_TECH = ins._build_system_prompt([ins.MODE_TECHNICAL]).rstrip()
SYS_SCI = ins._build_system_prompt([ins.MODE_SCIENTIFIC]).rstrip()
ROADMAP = ins._ROADMAP_SYSTEM_PROMPT.rstrip()

# Render a real example user prompt through the real builder.
ev = types.SimpleNamespace(
    type="code",
    title="Query UniProt for TP53 isoforms",
    content="from biomentis.tool.database import query_uniprot\nrows = query_uniprot('TP53')\nprint(rows.head())",
    file_path=None,
    file_kind=None,
)
EXAMPLE_KB = ins._build_user_prompt(
    ev, ev.content, [{"source": "lecture_notes.pdf", "page": 12,
                     "content": "UniProt canonical sequences are chosen per gene; isoform suffixes (-1, -2) index alternative splice products."}],
    task="Rank nanobody candidates against CHIKV E2",
)
EXAMPLE_NOKB = ins._build_user_prompt(
    ev, ev.content, [], task="Rank nanobody candidates against CHIKV E2",
    modes=[ins.MODE_SCIENTIFIC],
)

bloom = "\n".join(
    f"| `{k}` | {ins._BLOOM_DEFAULTS[k] or '—'} | {ins._DOK_DEFAULTS.get(k, 0) or '—'} |"
    for k in ins._BLOOM_DEFAULTS
)

modes_table = "\n".join(
    f"| `{m}` | {ins.MODE_META[m]['label']} | {ins.MODE_META[m]['color']} | "
    f"{ins.MODE_META[m]['blurb']} |"
    for m in ins.ALL_MODES
)

doc = f'''# Tutor instruction cards: prompts and how to change them

Working reference for the LLM prompts behind the tutor layer's teaching cards.
Prompt text below is reproduced **verbatim** from
`biomentis/agent/tutor/instruction.py`. Edit the prompts in the source, then
regenerate this file:

```bash
python docs/generate_card_prompt_doc.py
```

The prose sections come from that script's template; the prompt blocks, the
example user messages, the limit tables, and every source line reference are
read live from the module.

- Per-step prompt builder: `_build_system_prompt` — `{_line(r"^def _build_system_prompt")}`
- Per-mode schema / rules: `_MODE_SCHEMA`, `_MODE_RULES` — `{_line(r"^_MODE_SCHEMA = ")}`, `{_line(r"^_MODE_RULES = ")}`
- Run-level roadmap prompt: `_ROADMAP_SYSTEM_PROMPT` — `{_line(r"^_ROADMAP_SYSTEM_PROMPT")}`
- User-message assembly: `_build_user_prompt` — `{_line(r"^def _build_user_prompt")}`
- Response parsing / validation: `InstructionGenerator._call_llm` — `{_line(r"def _call_llm")}`
- Per-mode section parsing: `_sections_from_response` — `{_line(r"^def _sections_from_response")}`
- Renderer: `_render_instruction_card` — `{_line(r"def _render_instruction_card", "ui_tutor.py")}`

The cards are part of the tutor layer only. They never touch the agent's own
system prompt — see `docs/step_trace.md` and `engine.py` ("the agent runs
exactly as it did before this layer existed").

---

## 1. Instruction modes

A card explains one step through two independent lenses. Both are prompted in
a single LLM call; only the enabled ones contribute schema keys and rules, so
running one lens costs roughly half of running both, and running neither skips
the call entirely.

| mode | label | color | scope |
| --- | --- | --- | --- |
{modes_table}

The mode set flows: sidebar toggles (`_render_mode_toggles`,
`{_line(r"^def _render_mode_toggles", "ui_tutor.py")}`) → `TutorEngine.set_modes`
→ `InstructionGenerator.modes` → `_build_system_prompt(modes)`. It is also part
of the card cache key, so enabling a lens mid-run regenerates rather than
serving a card missing that lens.

Each mode owns its own `what` / `why` / `prerequisites` / `look_for`; the
scientific mode additionally owns `impact`. `citations`, `bloom_target` and
`dok_target` are shared — they describe the step, not the lens.

---

## 2. Per-step instruction card

One card per agent event, generated while the agent keeps working. Event types
are `reasoning`, `code`, `observation`, `solution`, `summary`, `file`.

### System prompt — both modes (verbatim)

```text
{SYS_BOTH}
```

### System prompt — technical only (verbatim)

Note the absent `SEPARATION RULE`: with one lens there is nothing to separate
from.

```text
{SYS_TECH}
```

### System prompt — scientific only (verbatim)

```text
{SYS_SCI}
```

### User message

Built by `_build_user_prompt`. The event content comes from `_event_text`,
which flattens title + content + (for `file` events) the produced path — never
file bytes, since the model cannot read an image. `SECTIONS_REQUESTED` echoes
the enabled modes so the model cannot drift onto a section it wasn't asked for.

With a KB hit, both modes requested:

```text
{EXAMPLE_KB}
```

With an empty KB, scientific mode only:

```text
{EXAMPLE_NOKB}
```

### KB retrieval query

`_build_retrieval_query` (`{_line(r"def _build_retrieval_query")}`) uses the
first 400 characters of the truncated event content as the query, falling back
to the event title if that is empty. There is no separate query-rewriter LLM
call, and retrieval does not depend on the mode set.

---

## 3. Roadmap card (shown once, before the walkthrough)

Takes the agent's first reasoning message and previews the whole plan. It is
mode-independent: the roadmap is a plan preview, not a teaching card.

### System prompt (verbatim)

```text
{ROADMAP}
```

### User message

```text
TASK: {{task or '(no task provided)'}}

PLAN_TEXT:
---
{{plan_text, truncated to _MAX_EVENT_CHARS}}
---
```

---

## 4. Everything the code enforces regardless of the prompt

### Size limits

| Limit | Value | Where | Effect |
| --- | --- | --- | --- |
| `_MAX_EVENT_CHARS` | {ins._MAX_EVENT_CHARS} | `{_line(r"^_MAX_EVENT_CHARS")}` | event content sent to the model |
| `_MAX_KB_SNIPPETS` | {ins._MAX_KB_SNIPPETS} | `{_line(r"^_MAX_KB_SNIPPETS")}` | KB passages per card |
| `_MAX_KB_SNIPPET_CHARS` | {ins._MAX_KB_SNIPPET_CHARS} | `{_line(r"^_MAX_KB_SNIPPET_CHARS")}` | chars per KB passage |
| `_MAX_PREREQS` | {ins._MAX_PREREQS} | `{_line(r"^_MAX_PREREQS")}` | prerequisite bullets kept **per mode** |
| `_MAX_LOOK_FOR` | {ins._MAX_LOOK_FOR} | `{_line(r"^_MAX_LOOK_FOR")}` | look-for bullets kept **per mode** |
| `_MAX_CITATIONS` | {ins._MAX_CITATIONS} | `{_line(r"^_MAX_CITATIONS")}` | citations kept |
| `_CITATION_SNIPPET_MAX` | {ins._CITATION_SNIPPET_MAX} | `{_line(r"^_CITATION_SNIPPET_MAX")}` | chars per citation quote |
| `what` clamp | {ins._MAX_WHAT_CHARS[ins.MODE_TECHNICAL]} technical / {ins._MAX_WHAT_CHARS[ins.MODE_SCIENTIFIC]} scientific | `{_line(r"^_MAX_WHAT_CHARS")}` | hard truncation after generation |
| `why` clamp | {ins._MAX_WHY_CHARS[ins.MODE_TECHNICAL]} technical / {ins._MAX_WHY_CHARS[ins.MODE_SCIENTIFIC]} scientific | `{_line(r"^_MAX_WHY_CHARS")}` | hard truncation after generation |
| `impact` clamp | {ins._MAX_IMPACT_CHARS} | `{_line(r"^_MAX_IMPACT_CHARS")}` | scientific only |
| roadmap `overview` | 600 | `{_line(r"^def generate_roadmap")}` | hard truncation |
| roadmap step `title` / `why` | 160 / 300 | `{_line(r"^def generate_roadmap")}` | hard truncation, max 10 steps |

**The clamps beat the prompt.** The scientific `why` is asked for 4-8 sentences
and clamped at {ins._MAX_WHY_CHARS[ins.MODE_SCIENTIFIC]} chars; asking for more
without raising the clamp yields a truncated one.

### Bloom / DOK defaults

Used as the fallback when the model's value is missing or invalid. The table at
the bottom of `_SHARED_RULES` duplicates these dicts by hand — change one and
they drift.

| event type | Bloom default | DOK default |
| --- | --- | --- |
{bloom}

Allowed values: Bloom ∈ {{{", ".join(sorted(ins._BLOOM_ALLOWED))}}}; DOK ∈ {{{", ".join(str(d) for d in sorted(ins._DOK_ALLOWED))}}}.
Validation lives in `_coerce_bloom` / `_coerce_dok`
(`{_line(r"^def _coerce_bloom")}`, `{_line(r"^def _coerce_dok")}`).

### Citations

`_coerce_citations` (`{_line(r"^def _coerce_citations")}`) filters every citation
against `allowed_sources` — the set of KB sources actually retrieved for *this*
card. The citation rule therefore holds even when the model ignores it: an
invented source is dropped, not rendered.

### JSON parsing and failure

`_extract_json` (`{_line(r"^def _extract_json")}`) accepts a bare JSON object, or
one wrapped in a markdown JSON code fence.

`_sections_from_response` (`{_line(r"^def _sections_from_response")}`) then pulls
one section per enabled mode, accepting three shapes in order: nested objects
as prompted, `technical_what`-style prefixed flat keys, and the pre-two-mode
flat `what`/`why` shape (assigned to the first enabled mode). That last path is
what keeps older cached responses and weaker local models usable.

If parsing fails, the LLM call raises, `llm` is `None`, or every enabled
section comes back empty, the generator returns a **soft-failure card**
(`_build_soft_failure_card`, `{_line(r"^def _build_soft_failure_card")}`):
`_generation_failed=True`, the event title in the first enabled mode's `what`,
default Bloom/DOK. The renderer then shows "Teaching card unavailable for this
step (LLM call failed)" and the run continues. Card generation never blocks the
agent.

### Caching

Cache key is `(event.type, content_hash(truncated_content), kb_signature,
modes)` — `{_line(r"cache_key = ")}`, mirrored in `ui_tutor._generate_or_get_card`
(`{_line(r"^def _generate_or_get_card", "ui_tutor.py")}`). **The prompt text is
not part of the key.** While iterating on wording in a live session,
already-seen events keep returning the old card. Add a hash of
`_build_system_prompt(modes)` to the key while you are editing, or restart the
Streamlit process between trials.

---

## 5. Edit checklist

**Safe to change freely** — wording, rules, calibration guidance, tone,
audience level, field descriptions. Nothing downstream parses the prose.

**Adding or renaming a field inside a mode** requires coordinated edits:

1. the mode's block in `_MODE_SCHEMA` (and its rule in `_MODE_RULES`)
2. the `CardSection` dataclass (`{_line(r"^class CardSection")}`) and
   `CardSection.to_dict`
3. `_coerce_section` (`{_line(r"^def _coerce_section")}`) — unparsed keys are
   silently dropped
4. the renderer `_render_instruction_card`
   (`{_line(r"def _render_instruction_card", "ui_tutor.py")}`), plus the export
   paths in `ui_tutor.py`: `_log_step_with_card`
   (`{_line(r"^def _log_step_with_card", "ui_tutor.py")}`) and
   `_make_instruction_event`
   (`{_line(r"^def _make_instruction_event", "ui_tutor.py")}`)

Miss step 3 and the field vanishes with no error. Miss step 4 and it is stored
but never displayed.

**Adding a third mode** additionally means: a `MODE_*` constant, an entry in
`ALL_MODES` and `MODE_META` (label, color, icon, blurb), a `CardSection` field
on `InstructionCard`, and hint text in `_PREREQ_HINT` / `_LOOK_FOR_HINT`
(`{_line(r"^_PREREQ_HINT", "ui_tutor.py")}`). The sidebar toggles, the renderer,
the prompt builder and the log all iterate `ALL_MODES`, so they pick it up for
free.

**Render order per mode box** (what the student actually sees,
`{_line(r"for mode in modes:", "ui_tutor.py")}`): mode badge → `**What:**` →
`**Why:**` → `**How this builds the answer:**` (scientific only) →
*prerequisites* expander → *what to look for* expander. Then, shared below both
boxes: *Sources* expander (or "No KB citations for this step") → Bloom/DOK
caption. Fields are omitted when empty, so a prompt change that stops producing
`look_for` silently removes that expander.

---

## 6. Ideas for the next pass

- **Externalize the prompts.** `Rubric` already loads `default_rubric.yaml`
  with a teacher-supplied override (`Rubric.from_yaml`). The same pattern for
  these prompts would allow per-course tuning without touching source, and
  make A/B comparison practical.
- **Differentiate by event type.** One prompt currently serves all six event
  types, with the calibration table doing the differentiating. A `code` card
  and a `solution` card have genuinely different teaching jobs.
- **Raise `_MAX_EVENT_CHARS`** if cards feel shallow on long code steps — the
  {ins._MAX_EVENT_CHARS}-char ceiling is a more likely cause than the prompt
  wording.
- **Level control.** Nothing in the prompt states the student's level. An
  explicit audience line (undergraduate / graduate / postdoc) is the smallest
  edit with the largest effect on card usefulness — and it would plausibly
  differ per mode.
- **Add the prompt hash to the cache key** before doing any of this.
'''

_OUT = os.path.join(_REPO, "docs", "tutor_instruction_cards.md")
with open(_OUT, "w") as f:
    f.write(doc)
print(f"wrote {_OUT}")

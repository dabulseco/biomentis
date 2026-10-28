# Biomentis — working notes for Claude Code

## Agents

Use subagents freely — `Explore` for broad searches across the codebase, `Plan`
for design work, `general-purpose` for multi-step research. No need to ask
first.

Workflows and deep-research still need an explicit ask, since they fan out
across many agents and cost accordingly.

## Environment

The project's dependencies live in the `biomni` conda env, **not** in base
Python:

```bash
/opt/anaconda3/envs/biomni/bin/python      # has streamlit, langchain, chroma, biopython…
```

Base `python3` will fail on almost any import from this repo.

The package is `biomentis` (renamed from `biomni`). Stray `biomni.*` imports
still surface occasionally — grep for them when you hit
`ModuleNotFoundError: No module named 'biomni'`.

## Tests

Plain scripts with a hand-rolled `_check()` harness — no pytest. They need the
repo on `PYTHONPATH`:

```bash
export PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni
P=/opt/anaconda3/envs/biomni/bin/python

$P -m biomentis.agent.tests.test_tab_focus          # transcript tab focus
$P -m biomentis.agent.tests.test_solution_streaming # how the answer reaches the main panel
$P -m biomentis.agent.tests.test_run_completion     # run completion + timing
$P -m biomentis.agent.tests.test_tool_imports       # tool import repair
$P -m biomentis.agent.tests.test_background_run     # background runs + journal
$P -m biomentis.agent.tests.test_stop_sequences     # turn boundaries + stop guard
$P -m biomentis.agent.tests.test_context_budget     # context window guard
$P -m biomentis.agent.tests.test_ncbi_email        # NCBI/Entrez email + sidebar field
$P -m biomentis.agent.tests.test_anthropic_key     # Anthropic key resolution + web-search fallback
$P -m biomentis.agent.tests.test_claude_models     # current Claude ids, no temperature, text-only replies
$P -m biomentis.agent.tests.test_structure_artifacts  # structure artifact registry
$P -m biomentis.agent.tests.test_structure_info  # structure metadata (no coordinates as text) + viewer models
$P -m biomentis.agent.tests.test_runaway_generation # parse-error retry cap, Ollama turn cutoff, turns in thinking or tool-call syntax, cut-off answers continued, "not yet" answers sent back
$P -m biomentis.agent.tests.test_kb_context         # knowledge base → agent + Clear KB confirm
$P -m biomentis.agent.tests.test_tutor_card_cache   # failed cards/empty chat answers retried + loop messages not shown as steps
$P -m biomentis.agent.tests.test_kb_structures      # PDB/SDF/SMILES in the KB: metadata only + View
$P -m biomentis.agent.tests.test_ollama_thinking    # per-model Ollama `think` setting (cloud on, local off)
$P -m biomentis.agent.tests.test_ollama_cloud       # Ollama API key → hosted models (DeepSeek, ...) at ollama.com
$P -m biomentis.agent.tests.test_chat_visuals       # tutor chat plots + structures
$P -m biomentis.agent.tests.test_viewer            # 3Dmol viewer page + render_structure smoke
$P -m biomentis.agent.tests.test_plot_mode         # class plot setting + run_start record
$P -m biomentis.agent.tests.test_step_plots        # plots shown in their step, or withheld under "student"
$P -m biomentis.agent.tests.test_plot_feedback     # figure summaries in observations + repeated-step guard
$P -m biomentis.agent.tests.test_show_structure    # plot guidance, show_structure tool, uploaded structures
$P biomentis/agent/tutor/tests/test_smoke_e2e.py    # tutor end-to-end
$P biomentis/agent/tutor/tests/test_critic.py       # critic + memory
```

Streamlit UI behavior is tested headlessly with `streamlit.testing.v1.AppTest`
driving a small harness app (see `test_tab_focus.py`). Note that an externally
assigned *widget* value in `AppTest` survives only the run immediately after
the assignment — stage the state and the assertion in the same `at.run()`.

## App

```bash
streamlit run streamlit_app.py
```

Agent turns run on a background thread, not the Streamlit script thread, so a
stray click can no longer kill a run — see `docs/background_runs.md`. The
render loop attaches to a `BackgroundRun` and resumes at a cursor, and every
event is journaled to `runs/<run_id>.jsonl` as it is produced. Anything that
touches the run loop should keep both properties: **a torn-down consumer must
be able to re-attach without gaps or repeats**, and the journal must be
written by the producer, not the consumer.

`streamlit_app.py` wires the tutor layer into `A1.launch_streamlit_demo` and
runs the agent with `self_critic=True`, which means **one turn can emit several
`<solution>` blocks** — the first is sent to the `self_critic` node and routed
back to `generate`. Anything consuming solutions must handle the later,
revised ones rather than latching onto the first.

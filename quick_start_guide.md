# Biomni Quick Start Guide

> **Default = local Ollama. Cloud providers (Anthropic, OpenAI, ...) are explicit opt-in.**
> Make sure `ollama serve` is running and you've pulled at least one model (`ollama pull qwen2.5:14b`). No API key needed to get started.

A practical, conda-first walkthrough for getting Biomni running locally and using it once it's up.

> Prereqs: macOS or Linux, **~3 GB free disk** for the Python env plus **~11 GB** for the data lake on first run, and a running **Ollama daemon** (recommended) — or a cloud-provider API key if you prefer.

---

## 1. Install

All three commands run from the repo root (`cd /path/to/Biomni`):

```bash
# 1. Create a fresh conda env with Python 3.11
conda create -n biomni python=3.11 -y
conda activate biomni

# 2. Install pinned runtime dependencies
pip install -r requirements.txt

# 3. Install Biomni itself (editable so local changes take effect)
pip install -e .
```

> **Heads up about env names.** The repo used to ship a `biomni_e1` conda environment; that legacy setup is preserved under `biomni_env/` for reference but is **not** the recommended path anymore. The new env is simply called `biomni`.

### 1a. Set up Ollama (the default LLM)

If you don't already have it:

```bash
# macOS
brew install ollama
# Or grab the installer from https://ollama.com

# Start the daemon (it must be running whenever you use Biomni)
ollama serve &

# Pull at least one model. Recommended starting point:
ollama pull qwen2.5:14b
# Other good options: llama3.1:8b (lighter), qwen2.5:32b or deepseek-r1:14b (stronger)
```

That's it — no `.env` file needed. When you launch Biomni it will auto-detect this Ollama model. See §1b if you want to use a cloud provider instead.

### 1b. Use a cloud provider (Anthropic, OpenAI, etc.) — opt-in

If you specifically want to use a cloud LLM, create a `.env` at the repo root:

```bash
cp .env.example .env
```

Then **uncomment** the relevant block and set the values:

```env
# Anthropic / Claude (the original default before the local-first change)
ANTHROPIC_API_KEY=sk-ant-...
BIOMNI_LLM=claude-sonnet-4-5
BIOMNI_SOURCE=Anthropic
BIOMNI_DISABLE_LOCAL_FALLBACK=true
```

```env
# OpenAI
OPENAI_API_KEY=sk-...
BIOMNI_LLM=gpt-4o
BIOMNI_SOURCE=OpenAI
BIOMNI_DISABLE_LOCAL_FALLBACK=true
```

```env
# Azure OpenAI
OPENAI_API_KEY=...
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
BIOMNI_LLM=azure-gpt-4o
BIOMNI_SOURCE=AzureOpenAI
BIOMNI_DISABLE_LOCAL_FALLBACK=true
```

For Gemini, Groq, or Bedrock, install the matching optional dep first (`pip install -r requirements-optional.txt`) and add the corresponding key to `.env`.

> **Why `BIOMNI_DISABLE_LOCAL_FALLBACK=true`?** Without it, if you also have a stale `ollama serve` running, the local-first path in `biomni/config.py` may still try to pick an Ollama model first. Setting this flag forces Biomni to respect your explicit cloud choice.

### 1c. Optional: set an NCBI email (PubMed + Entrez tools)

NCBI politely asks for an email on every Entrez request and will rate-limit requests that don't provide one. Biomni's PubMed / gene-coding-sequence tools read it in this order: an explicit argument > `default_config.ncbi_email` > the `NCBI_EMAIL` (or `BIOMNI_NCBI_EMAIL`) env var. To set it via `.env`:

```env
NCBI_EMAIL=you@example.com
```

Either variable name works. The bare-minimum placeholder `your-email@example.com` keeps things running but you'll hit throttling faster.

### Optional extras

Most users don't need these. Install any of them on top of the base install:

```bash
pip install -r requirements-optional.txt
```

That optional file covers:

- `langchain-google-genai`, `langchain-groq`, `langchain-aws` — alternate LLM providers
- `gradio>=5.0,<6.0` — alternate UI (Streamlit is the default)
- `weasyprint` — PDF export for `agent.save_conversation_history()`

### Known package conflicts

`hyperimpute`, `langchain_aws`, `cnvkit`, and `panhumanpy` are intentionally **not** in `requirements.txt` because they break the main environment. See `docs/known_conflicts.md` for install instructions. **Bedrock users** must also uncomment the relevant code in `biomni/llm.py` after installing `langchain-aws`.

---

## 2. Configure (only if you're using a cloud provider)

**If you're using the default Ollama path, skip this section.** The `default_config` singleton auto-picks the first available Ollama model at import time.

If you're using a cloud provider, your `.env` is already configured per §1b. The most useful additional variables:

```env
# Where the data lake lives (default: ./data)
BIOMNI_DATA_PATH=/path/to/your/data

# Agent execution timeout in seconds (default: 600)
BIOMNI_TIMEOUT_SECONDS=600

# Sampling temperature for the main loop -- code generation, tool selection,
# structured output, verification. Cold on purpose (default: 0.2).
BIOMNI_TEMPERATURE=0.2

# Sampling temperature for divergent calls only -- hypothesis generation and
# the self-critic's "what are we missing?" pass (default: 0.7).
BIOMNI_CREATIVE_TEMPERATURE=0.7
```

> **Why two temperatures?** The agent writes and runs Python, and sampling noise
> in that loop produces hallucinated arguments rather than insight — so the main
> loop runs cold. Creativity is bought explicitly, on the handful of calls that
> ask for ideas, via `A1.creative_llm`. Running a small local model? Try
> `BIOMNI_TEMPERATURE=0.1` and `BIOMNI_CREATIVE_TEMPERATURE=0.5` — sub-13B models
> degrade with temperature faster than frontier models do.

> **Always run Biomni from the directory that contains your `.env`** (or export the same vars in your shell). The config priority is: direct parameter to `A1(...)` > runtime `default_config` mutation > env vars > built-in defaults.

---

## 3. Launch the app

You have three options. Pick one per session.

### 3a. Streamlit UI (default)

```bash
conda activate biomni
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

The launcher is a 10-line script — it caches an `A1` agent and calls `agent.launch_streamlit_demo()`. To customize (data path, model, etc.), edit `streamlit_app.py` or write your own launcher.

### 3b. Python / notebook (programmatic)

```bash
conda activate biomni
jupyter lab   # or: jupyter notebook
```

```python
from biomentis.agent import A1

agent = A1(path="./data", llm="claude-sonnet-4-20250514")
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion.")
```

> **First run takes ~10–30 min** because the data lake (~11 GB) is downloaded into `path`. To skip the auto-download for fast testing:
> ```python
> agent = A1(path="./data", llm="claude-sonnet-4-20250514", expected_data_lake_files=[])
> ```

### 3c. Optional: Gradio UI

If you installed the optional extras and prefer Gradio, run from a Python REPL:

```python
from biomentis.agent import A1
agent = A1(path="./data", llm="claude-sonnet-4-20250514")
agent.launch_gradio_demo()        # default at http://localhost:7860
```

Useful flags: `share=True` (public link), `server_name="127.0.0.1"` (local-only), `require_verification=True` (gate behind code `Biomni2025`).

---

## 4. Workflow inside the app

Once the UI is up, the typical session goes:

### 4.1 One-time housekeeping
1. **Open the UI** in your browser (port 8501 for Streamlit, 7860 for Gradio).
2. **Verify your API key is loaded** by running a tiny prompt like `What tools do you have access to?`. If you get an auth error, double-check `.env` and restart the UI process — env vars are read at startup, not on each call.
3. **Wait for the data lake** to finish downloading on first run. The agent will appear unresponsive while this happens.

### 4.2 Daily workflow
1. **Frame the task in natural language.** Be specific about inputs (compound SMILES, file paths, gene symbols) and desired outputs (a hypothesis, a CSV, a plot). Example prompts from the README:
   - `Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.`
   - `Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis`
   - `Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
2. **Submit and watch the trace.** Biomni is ReAct-style: you'll see a stream of *Thought → Tool call → Observation* steps. Don't interrupt unless it's clearly stuck past your `BIOMNI_TIMEOUT_SECONDS` (default 600 s).
3. **Inspect outputs.** Each tool writes results to a per-task directory under `./data` (or your `BIOMNI_DATA_PATH`). The agent will also summarize the findings in the chat panel.
4. **Iterate.** Ask follow-ups in the same thread — the agent keeps short-term context. To start fresh, change the `thread_id` in the launcher (default is `42`).
5. **Save the trace** (Gradio only) when you have a result worth keeping:
   ```python
   agent.save_conversation_history("my_analysis_results.pdf")
   ```
   Requires `weasyprint`, `markdown2pdf`, or `pandoc` (see README for install options).

### 4.3 Adding external tools via MCP
If a tool you need isn't built in:
1. Create an `mcp_config.yaml` (template in `docs/mcp_integration.md`).
2. Either pre-load it in your launcher:
   ```python
   agent = A1(path="./data", llm="claude-sonnet-4-20250514")
   agent.add_mcp(config_path="./mcp_config.yaml")
   agent.go("...")
   ```
3. Or hand-add it from the UI if the front-end you chose supports it.
4. The agent will see the new tools as `<server_name>.<tool_name>` and pick them up automatically when relevant.

### 4.4 Using a local model (Ollama)
```bash
# 1. Install Ollama (https://ollama.com) and pull a model
ollama pull llama3.1:70b

# 2. In Python
from biomentis.agent import A1
agent = A1(
    path="./data",
    llm="llama3.1:70b",
    source="Ollama",
)
agent.go("Summarize the tools available in this environment.")
```

---

## 5. Shutting down and resuming

- **Stop the UI** with `Ctrl-C` in the terminal that launched it.
- **Reactivate later** with `conda activate biomni`.
- **Data lake persists** between sessions — no re-download. To wipe it, delete the `data/` directory (or whatever you set `BIOMNI_DATA_PATH` to).
- **No background daemons.** Nothing keeps running after you close the terminal.

---

## 6. Troubleshooting checklist

| Symptom | Fix |
|---|---|
| `conda: command not found` | Install Miniconda/Anaconda, then `conda init zsh` (or `bash`) and restart the shell. |
| `ModuleNotFoundError: biomni` | You forgot to `conda activate biomni`, or didn't `pip install -e .` / `pip install biomni`. |
| PubMed / NCBI tools throttled or rejected | NCBI asks for an email on every Entrez request. Set `NCBI_EMAIL=you@example.com` in `.env` (or pass `ncbi_email=` to `BiomniConfig()`). |
| UI opens but every prompt errors with auth | `.env` has an `ANTHROPIC_API_KEY` (or other cloud key) but the corresponding `BIOMNI_LLM` / `BIOMNI_SOURCE` is missing or wrong. Either remove the cloud key (revert to Ollama default) or set the matching `BIOMNI_LLM` / `BIOMNI_SOURCE` and `BIOMNI_DISABLE_LOCAL_FALLBACK=true`. |
| Agent tries Anthropic but I want Ollama | Make sure `ANTHROPIC_API_KEY` is **not** in your shell env or `.env`. The local-first fallback in `biomni/config.py` short-circuits when any cloud key is present. |
| Agent tries Ollama but I want Anthropic | Uncomment `ANTHROPIC_API_KEY` in `.env` and add `BIOMNI_LLM=claude-sonnet-4-5` + `BIOMNI_SOURCE=Anthropic` + `BIOMNI_DISABLE_LOCAL_FALLBACK=true`. |
| `Connection refused` on `localhost:11434` | `ollama serve` isn't running, or it crashed. Restart it. |
| "No models available" / `ollama list` is empty | `ollama pull qwen2.5:14b` (or any other model). |
| First run hangs for hours | It's downloading the 11 GB data lake. Set `expected_data_lake_files=[]` to skip. |
| `Gradio` version conflict | Pin to 5.x: `pip install "gradio>=5.0,<6.0"`. |
| `cnvkit` import fails | Use the Py3.10 env: `conda create -n biomni-py310 -c conda-forge -c bioconda python=3.10 cnvkit -y && conda activate biomni-py310`, then re-run `pip install -r requirements.txt`. |
| Agent times out at 600 s | Set `BIOMNI_TIMEOUT_SECONDS=1800` in `.env`, or pass `timeout=1800` to `A1(...)`. |
| Azure model not found | Prefix the name: `llm="azure-gpt-4o"`. |
| `langchain_aws` / Bedrock missing | `pip install -r requirements-optional.txt` (installs `langchain-aws`), then uncomment the Bedrock blocks in `biomni/llm.py`. |
| Security warning | Biomni runs LLM-generated code with full system privileges. **Don't run it on a machine with sensitive data** — use a sandboxed VM or container. |

---

## 7. Where to go next

- `docs/configuration.md` — full env-var and `default_config` reference
- `docs/mcp_integration.md` — adding external MCP servers / exposing Biomni as one
- `docs/known_conflicts.md` — packages intentionally left out and how to add them
- `CONTRIBUTION.md` — adding new tools, data, software, or benchmarks
- `tutorials/biomni_101.ipynb` — the single in-repo tutorial notebook
- `status_20260718.md` — overall project status and gap analysis

Happy bio-agenting. 🧬

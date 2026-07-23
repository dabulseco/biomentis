<p align="center">
  <img src="./figs/biomentis_logo_concept.png" alt="Biomentis Logo" width="600px" />
</p>

<p align="center">
<a href="https://github.com/dabulseco/biomentis">
<img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub" />
</a>
<a href="https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1">
<img src="https://img.shields.io/badge/Read-Paper-green?style=for-the-badge" alt="Paper" />
</a>
</p>



# Biomentis: A General-Purpose Biomedical AI Agent

## Overview


Biomentis is a general-purpose biomedical AI agent designed to autonomously execute a wide range of research tasks across diverse biomedical subfields. By integrating cutting-edge large language model (LLM) reasoning with retrieval-augmented planning and code-based execution, Biomentis helps scientists dramatically enhance research productivity and generate testable hypotheses.

Biomentis is a fork of [Biomni](https://github.com/snap-stanford/biomni) (Huang et al., 2025) maintained by Dylan Bulseco. It adds an **opt-in instructional layer** (Knowledge-base-grounded tutor + step-pause + Q&A logging) intended for classroom and self-study use. The original agent, tools, and benchmarks are unchanged — research-mode users see no behavioral difference.


## Quick Start

### Installation

Biomentis installs in three commands. By default it uses a **local Ollama model** — no API key required.

```bash
# 1. Create a fresh conda env with Python 3.11
conda create -n biomentis python=3.11 -y
conda activate biomentis

# 2. Install pinned runtime dependencies
pip install -r requirements.txt

# 3. Install Biomentis itself (editable so local changes take effect)
pip install -e .
```

**Set up Ollama (the default LLM):**

```bash
# macOS
brew install ollama
ollama serve &
ollama pull qwen2.5:14b    # or llama3.1:8b, deepseek-r1:14b, etc.
```

Now launch the UI:

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

The active model is shown in the sidebar. The first time you `go()` the agent will look up your local Ollama model and use it for reasoning. If you have no Ollama daemon running, the UI will start but the first `go()` call will fail loudly — that’s the signal to start `ollama serve`.

**To use a cloud provider instead** (Anthropic, OpenAI, etc.), copy and edit `.env`:

```bash
cp .env.example .env
```

Then uncomment the matching block in `.env` and set the model. Example for Anthropic:

```env
ANTHROPIC_API_KEY=sk-ant-...
BIOMNI_LLM=claude-sonnet-4-5
BIOMNI_SOURCE=Anthropic
BIOMNI_DISABLE_LOCAL_FALLBACK=true
```

> The `BIOMNI_DISABLE_LOCAL_FALLBACK=true` line matters: without it, if you also have `ollama serve` running, Biomentis’s local-first default may still try to pick an Ollama model first. Setting this flag forces Biomentis to respect your explicit cloud choice. (The `BIOMNI_*` env-var names are kept stable for backward compatibility; they apply identically to the Biomentis fork.)

For optional extras (Gemini / Groq / Bedrock providers, the Gradio UI, PDF export), see [`requirements-optional.txt`](./requirements-optional.txt). For other LLM/UI tweaks (data path, timeouts, NCBI email for PubMed tools), see [docs/configuration.md](./docs/configuration.md).

If you prefer shell environment variables over a `.env` file, the same settings work as `export ANTHROPIC_API_KEY=...` / `export BIOMNI_LLM=...` in `~/.zshrc` or `~/.bashrc`.


#### ⚠️ Known Package Conflicts

Some Python packages are not installed by default in the Biomentis environment due to dependency conflicts. If you need these features, you must install the packages manually and may need to uncomment relevant code in the codebase. See the up-to-date list and details in [docs/known_conflicts.md](./docs/known_conflicts.md).

### Basic Usage

Once inside the environment, you can start using Biomentis. With the default Ollama setup, you can omit the `llm` arg entirely and let `default_config` pick the local model:

```python
from biomentis.agent import A1

# No llm= needed — BiomentisConfig picks the first available local Ollama model
agent = A1(path=’./data’)

# Or, explicitly request a specific model (Anthropic, OpenAI, custom):
# agent = A1(path=’./data’, llm=’claude-sonnet-4-5’, source=’Anthropic’)

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

#### Controlling Datalake Loading

By default, Biomentis automatically downloads the datalake files (~11GB) when you create an agent. You can control this behavior:

```python
# Skip automatic datalake download (faster initialization)
agent = A1(path=’./data’, expected_data_lake_files=[])
```

This is useful for:
- Faster testing and development
- Environments with limited storage or bandwidth
- Cases where you only need specific tools that don’t require datalake files
If you plan on using Azure for your model, always prefix the model name with azure- (e.g. llm=’azure-gpt-4o’).

### Streamlit Interface

Launch the default web UI (Streamlit) from the repo root:

```bash
conda activate biomentis
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

The launcher is a tiny script that caches an `A1` agent and calls `agent.launch_streamlit_demo()`. The currently-active model is shown in the sidebar — it will be your local Ollama model by default, or whatever you configured in `.env`. To customize the data path, model, or which files are auto-downloaded, edit `streamlit_app.py` or write your own launcher.

**Note:** Streamlit is included in the default `requirements.txt`. No extra install step is required.

#### Optional: Gradio Interface

If you prefer Gradio, install it from `requirements-optional.txt` first:

```bash
pip install -r requirements-optional.txt   # installs gradio and friends
```

Then from a Python REPL or notebook:

```python
from biomentis.agent import A1

agent = A1(path="./data")     # uses default Ollama model
agent.launch_gradio_demo()    # default at http://localhost:7860
```

Useful flags:
- `share=True` — create a public Gradio link
- `server_name="127.0.0.1"` — local-only
- `require_verification=True` — gate the UI behind a code (default `Biomni2025`)

### Configuration Management

Biomentis includes a centralized configuration system that provides flexible ways to manage settings. You can configure Biomentis through environment variables, runtime modifications, or direct parameters.

```python
from biomentis.config import default_config
from biomentis.agent import A1

# RECOMMENDED: Modify global defaults for consistency
default_config.llm = "gpt-4"           # overrides the local-first default
default_config.source = "OpenAI"        # tells Biomentis to use OpenAI
default_config.timeout_seconds = 1200

# All agents AND database queries use these defaults
agent = A1()  # Everything uses gpt-4, 1200s timeout
```

**Note**: Direct parameters to `A1()` only affect that agent’s reasoning, not database queries. For consistent configuration across all operations, use `default_config` or environment variables.

#### NCBI / Entrez email (PubMed tools)

NCBI asks politely for an email on every Entrez request and will rate-limit requests that don’t provide one. The bare-minimum placeholder `your-email@example.com` works but you’ll hit throttling faster. Set a real address via `.env`:

```env
NCBI_EMAIL=you@example.com
```

Or programmatically:

```python
from biomentis.config import default_config
default_config.ncbi_email = "you@example.com"
```

`query_pubmed`, `get_gene_coding_sequence`, and the rest of the literature / molecular-biology tools will pick this up automatically. The resolution order is: explicit function argument → `default_config.ncbi_email` → `NCBI_EMAIL` (or `BIOMNI_NCBI_EMAIL`) env var → placeholder.

For detailed configuration options, see the **[Configuration Guide](docs/configuration.md)**.

### PDF Generation

Generate PDF reports of execution traces:

```python
from biomentis.agent import A1

# Initialize agent (uses your default Ollama model, or pass llm= explicitly)
agent = A1(path=’./data’)

# Run your task
agent.go("Your biomedical task here")

# Save conversation history as PDF
agent.save_conversation_history("my_analysis_results.pdf")
```

**PDF Generation Dependencies:**
<details>
<summary>Click to expand</summary>
For optimal PDF generation, install one of these packages:

```bash
# Option 1: WeasyPrint (recommended for best layout control)
# Conda environment (recommended)
conda install weasyprint

# System installation
brew install weasyprint  # macOS
apt install weasyprint   # Linux

# See [WeasyPrint Installation Guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) for detailed instructions.

# Option 2: markdown2pdf (Rust-based, fast and reliable)
# macOS:
brew install theiskaa/tap/markdown2pdf

# Windows/Linux (using Cargo):
cargo install markdown2pdf

# Or download prebuilt binaries from:
# https://github.com/theiskaa/markdown2pdf/releases/latest

# Option 3: Pandoc (pip installation)
pip install pandoc
```
</details>

## MCP (Model Context Protocol) Support

Biomentis supports MCP servers for external tool integration:

```python
from biomentis.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:**
For usage and implementation details, see the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/).


## Biomni-R0

**Biomni-R0** is the upstream Biomni project’s first reasoning model for biology, built on Qwen-32B with reinforcement learning from agent interaction data. It is designed to excel at tool use, multi-step reasoning, and complex biological problem-solving through iterative self-correction. Biomentis does **not** retrain or redistribute Biomni-R0; if you’d like to use it, point your `A1()` at the upstream Hugging Face artifact.

- 🤗 Model: [biomni/Biomni-R0-32B-Preview](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- 📝 Technical Report: https://biomni.stanford.edu/blog/biomni-r0-technical-report

## Biomni-Eval1

**Biomni-Eval1** is the upstream Biomni evaluation benchmark, a comprehensive 433-instance benchmark across 10 biological reasoning tasks. Biomentis ships the loader unchanged so you can score your own runs against the original ground truth.

**Resources:**
- 🤗 Dataset: [biomni/Eval1](https://huggingface.co/datasets/biomni/Eval1)
- 💻 Quick Start:
```python
from biomentis.eval import BiomentisEval1

evaluator = BiomentisEval1()
score = evaluator.evaluate(‘gwas_causal_gene_opentargets’, 0, ‘BRCA1’)
```


## 📚 Know-How Library

Biomentis inherits Biomni’s **Know-How Library** — a curated collection of best practices, protocols, and troubleshooting guides for biomedical techniques. These documents are automatically retrieved by the A1 agent when relevant to provide domain expertise and practical knowledge.

**Features:**
- Automatic retrieval based on query relevance
- Metadata tracking (authors, affiliations, licensing, commercial use)
- Compatible with commercial mode (filters non-commercial content)

### 📝 Contributing Know-How Documents

We’re actively seeking community contributions to expand our Know-How Library! Share your expertise by contributing:

- **Lab protocols** (cell culture, flow cytometry, western blotting, etc.)
- **Analysis best practices** (NGS workflows, microscopy techniques, etc.)
- **Troubleshooting guides** (common issues and solutions)
- **Experimental design guidelines** (sample size, controls, validation)
- **Domain-specific knowledge** (drug formulation, animal models, clinical trials, etc.)

Know-how documents should be practical, succinct, and include proper attribution. Use [this know-how](know_how/single_cell_annotation.md) as an example.

**To contribute:** Create a markdown file following our template and submit a pull request.

## 🤝 Contributing to Biomentis

Biomentis is an open-source fork. We welcome:

- **🔧 New Tools**: Specialized analysis functions and algorithms
- **📊 Datasets**: Curated biomedical data and knowledge bases
- **💻 Software**: Integration of existing biomedical software packages
- **📋 Benchmarks**: Evaluation datasets and performance metrics
- **📚 Know-How**: Best practices, protocols, and domain expertise
- **📚 Misc**: Tutorials, examples, and use cases
- **🔧 Update existing tools**: many current tools are not optimized - fix and replacements are welcome!

Check out the **[Contributing Guide](CONTRIBUTION.md)** for guidance on how to contribute.

If you have a particular tool/database/software in mind that you want to add, you can also submit to the [upstream Biomni form](https://forms.gle/nu2n1unzAYodTLVj6) and the Biomni team will implement them.

## 🎓 Instructional Mode (Biomentis-only feature)

Unlike the upstream Biomni, Biomentis adds an **opt-in instructional layer** designed for classroom and self-study use:

- **Per-session Knowledge base (KB)** — upload PDF, PPTX, DOCX, TXT, or URLs. The KB grounds the tutor’s answers and instruction cards in your course materials (not the agent’s general knowledge).
- **Per-step instruction cards** — every reasoning/code/observation event is followed by a "What / Why / Prerequisites / Look for" card with KB citations. A "Continue" button pauses the run between steps.
- **Tutor chat** — a persistent right-column chat panel. In Tutor (Instructional) mode, the tutor answers only from your KB. In Chat mode, the LLM fills any gaps after the KB.
- **Self-improvement / Critic** — after each session, the agent reviews its own transcript against the rubric, surfaces weaknesses, and writes them into long-term memory so future runs avoid the same mistakes. Bloom’s Taxonomy + Webb’s DOK levels classify every Q&A and step.
- **Exportable session log** — JSON or CSV per session, suitable for learning analytics.

This layer is fully off by default; if you don’t enable it, Biomentis is behaviorally identical to upstream Biomni. See [docs/tutor.md](./docs/tutor.md) for the full guide.


## Tutorials and Examples

**[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More to come!

## 🌐 Web Interface (upstream)

The upstream Biomni project operates a hosted web UI at **[biomni.stanford.edu](https://biomni.stanford.edu)**. Biomentis is the local-first, fork-and-extend counterpart and is meant to be run on your own machine via `streamlit run streamlit_app.py`.

## Important Note
- Security warning: Currently, Biomentis executes LLM-generated code with full system privileges. If you want to use it in production, please use in isolated/sandboxed environments. The agent can access files, network, and system commands. Be careful with sensitive data or credentials.
- This release was frozen as of April 15 2025, so it differs from the current web platform.
- Biomentis itself is Apache 2.0-licensed, but certain integrated tools, databases, or software may carry more restrictive commercial licenses. Review each component carefully before any commercial use.

## Acknowledgments

Biomentis is a derivative work of [Biomni](https://github.com/snap-stanford/biomni) (Huang et al., 2025). The agent, tools, evaluation benchmark, and Know-How library are reused under the Apache 2.0 License. The instructional layer (Tutor + KB + Critic) is original to Biomentis.

If you use Biomentis in research or teaching, please also cite the upstream Biomni paper:

```
@article{huang2025biomni,
  title={Biomni: A General-Purpose Biomedical AI Agent},
  author={Huang, Kexin and Zhang, Serena and Wang, Hanchen and Qu, Yuanhao and Lu, Yingzhou and Roohani, Yusuf and Li, Ryan and Qiu, Lin and Zhang, Junze and Di, Yin and others},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: biomentis` | You forgot to `conda activate biomentis`, or didn’t `pip install -e .` / `pip install biomentis`. |
| `ModuleNotFoundError: No module named ‘googlesearch’` when the agent calls `search_google` | Install the renamed package: `pip install googlesearch-python`. The literature module loads even if it’s missing — only `search_google` itself fails. |
| UI opens but every prompt errors with auth | `.env` has a cloud key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) but the matching `BIOMNI_LLM` / `BIOMNI_SOURCE` is missing or wrong. Either remove the cloud key (revert to Ollama default) or set the matching pair plus `BIOMNI_DISABLE_LOCAL_FALLBACK=true`. |
| Agent tries Anthropic but I want Ollama | Make sure no `ANTHROPIC_API_KEY` (or other cloud key) is in your shell env or `.env`. The local-first fallback in `biomentis/config.py` short-circuits only when the user hasn’t chosen a cloud model. |
| Agent tries Ollama but I want Anthropic | Uncomment `ANTHROPIC_API_KEY` in `.env` and add `BIOMNI_LLM=claude-sonnet-4-5` + `BIOMNI_SOURCE=Anthropic` + `BIOMNI_DISABLE_LOCAL_FALLBACK=true`. |
| `Connection refused` on `localhost:11434` | `ollama serve` isn’t running, or it crashed. Restart it. |
| "No models available" / `ollama list` is empty | `ollama pull qwen2.5:14b` (or any other model). |
| PubMed / NCBI tools throttled or rejected | NCBI asks for an email on every Entrez request. Set `NCBI_EMAIL=you@example.com` in `.env` (or pass `ncbi_email=` to `BiomentisConfig()`). |
| First run hangs for hours | It’s downloading the 11 GB data lake. Set `expected_data_lake_files=[]` to skip. |

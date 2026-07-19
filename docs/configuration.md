# Biomni Configuration Guide

> **Default = local Ollama.** If you have `ollama serve` running with at least one model pulled, Biomni picks that automatically — no API key, no env var, no `.env` file needed. The settings below only matter when you want to override the default (switch to a cloud provider, tune timeouts, point at a different data path, etc.).

## Quick Start

**Recommended approach**: Use environment variables or modify `default_config` for consistent behavior across your entire application.

```python
from biomni.config import default_config
from biomni.agent import A1

# Option 1: Modify global defaults (affects everything)
default_config.llm = "qwen2.5:14b"            # or any specific Ollama model name
default_config.source = "Ollama"               # optional — auto-detected
default_config.timeout_seconds = 1200

# Option 2: Use environment variables (set in .env file)
# BIOMNI_LLM=qwen2.5:14b
# BIOMNI_TIMEOUT_SECONDS=1200

agent = A1()  # Uses your configuration
```

## Configuration Methods

### 1. Environment Variables (Recommended for Production)

Create a `.env` file in your project. **Leave the cloud-key lines commented out** unless you specifically want a cloud provider — uncommenting one of them disables the Ollama default.

```bash
# Optional cloud keys — uncomment to override the local Ollama default
# ANTHROPIC_API_KEY=your_key
# OPENAI_API_KEY=your_key
# GEMINI_API_KEY=your_key
# GROQ_API_KEY=your_key
# AWS_BEARER_TOKEN_BEDROCK=your_key
# AWS_REGION=us-east-1

# Azure OpenAI
# OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Biomni Settings
BIOMNI_LLM=qwen2.5:14b                  # default: "auto" (picks first local Ollama model)
BIOMNI_SOURCE=Ollama                    # default: auto-detected from model name
BIOMNI_DISABLE_LOCAL_FALLBACK=false     # default: false; set true to force a cloud provider
BIOMNI_PATH=/path/to/data               # default: ./data
BIOMNI_TIMEOUT_SECONDS=1200             # default: 600
```

### 2. Runtime Configuration (Recommended for Scripts)

```python
from biomni.config import default_config

# Changes apply to all agents and database queries
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200
```

### 3. Direct Parameters (Use with Caution)

```python
# ⚠️ Only affects this agent's reasoning, NOT database queries
agent = A1(llm="claude-3-5-sonnet-20241022")
```

## Common Examples

### Using Different Models

```python
# Use GPT-4 everywhere
default_config.llm = "gpt-4"
agent = A1()
```

### Cost Optimization (Different Models for Agent vs Database)

```python
# Cheaper model for database queries
default_config.llm = "claude-3-5-haiku-20241022"

# More powerful model for agent reasoning
agent = A1(llm="claude-3-5-sonnet-20241022")
```

### Custom/Local Models

```python
default_config.source = "Custom"
default_config.base_url = "http://localhost:8000/v1"
default_config.api_key = "local_key"
default_config.llm = "local-llama-70b"
```

## All Available Settings

### Environment Variables

```bash
# --- Cloud provider keys (any one of these disables the Ollama default) ---
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
AWS_BEARER_TOKEN_BEDROCK=your_key
AWS_REGION=us-east-1

# Azure OpenAI
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# --- Biomni Settings ---
BIOMNI_PATH=/path/to/data                   # default: ./data
BIOMNI_TIMEOUT_SECONDS=1200                 # default: 600
BIOMNI_LLM=model_name                       # default: "auto" (picks first local Ollama model)
BIOMNI_TEMPERATURE=0.7                      # default: 0.7
BIOMNI_USE_TOOL_RETRIEVER=true              # default: true
BIOMNI_SOURCE=Anthropic                     # default: auto-detected from model name
BIOMNI_DISABLE_LOCAL_FALLBACK=false         # default: false
BIOMNI_COMMERCIAL_MODE=false                # default: false
BIOMNI_CUSTOM_BASE_URL=http://localhost:8000/v1
BIOMNI_CUSTOM_API_KEY=custom_key
OLLAMA_HOST=http://localhost:11434          # default; only set if Ollama is elsewhere
```

### Python Configuration

```python
from biomni.config import default_config

# All available settings
default_config.path = "./data"
default_config.timeout_seconds = 600
default_config.llm = "auto"                       # "auto" → first local Ollama model
default_config.source = None                      # auto-detected from model name
default_config.temperature = 0.7
default_config.use_tool_retriever = True
default_config.commercial_mode = False
default_config.base_url = None                    # for custom OpenAI-compatible servers
default_config.api_key = None
default_config.protocols_io_access_token = None
```

## Important Notes

- **For pip-installed packages**: You can't edit the package files, but you can still use environment variables or modify `default_config` at runtime
- **Configuration consistency**: Database queries always use `default_config`, regardless of agent parameters
- **Priority order**: Direct params > Runtime config > Env vars > Defaults

## Troubleshooting

**API Key Not Found**:
- Check `.env` file exists in your working directory
- Verify with: `echo $ANTHROPIC_API_KEY`

**Configuration Not Applied**:
- Changes to `default_config` only affect agents created after the change
- Direct parameters only affect that specific agent, not database queries

**Model Not Found**:
- Check spelling of model name
- For Azure, prefix with "azure-" (e.g., "azure-gpt-4o")
- Ensure you have the right API key for that provider

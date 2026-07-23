"""
Biomentis Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
from dataclasses import dataclass


@dataclass
class BiomentisConfig:
    """Central configuration for the Biomentis agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = BiomentisConfig()

        # Override specific settings
        config = BiomentisConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"
    """

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    #
    # llm="auto" is the sentinel for "no preference — pick the best local option".
    # _resolve_local_first_default() below turns this into a real Ollama model
    # name when a local daemon is running. To force a specific cloud model
    # (Anthropic, OpenAI, ...) just set BIOMNI_LLM=... and BIOMNI_SOURCE=... in
    # your .env — the local-first path will then short-circuit.
    llm: str = "auto"
    temperature: float = 0.7

    # Tool settings
    use_tool_retriever: bool = True

    # Data licensing settings
    commercial_mode: bool = False  # If True, excludes non-commercial datasets

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = None

    # Third-party integrations
    protocols_io_access_token: str | None = None

    # NCBI / Entrez
    # NCBI requires (politely) an email address on every Entrez request and
    # will rate-limit you if you don't provide one. Set via the NCBI_EMAIL
    # env var or by passing ncbi_email= when constructing BiomentisConfig.
    ncbi_email: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist."""
        # Check for environment variable overrides (optional)
        # Support both old and new names for backwards compatibility
        if os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH"):
            self.path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH")
        if os.getenv("BIOMNI_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS"))
        if os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL"):
            self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL")
        if os.getenv("BIOMNI_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("BIOMNI_USE_TOOL_RETRIEVER").lower() == "true"
        if os.getenv("BIOMNI_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("BIOMNI_COMMERCIAL_MODE").lower() == "true"
        if os.getenv("BIOMNI_TEMPERATURE"):
            self.temperature = float(os.getenv("BIOMNI_TEMPERATURE"))
        if os.getenv("BIOMNI_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("BIOMNI_CUSTOM_BASE_URL")
        if os.getenv("BIOMNI_CUSTOM_API_KEY"):
            self.api_key = os.getenv("BIOMNI_CUSTOM_API_KEY")
        if os.getenv("BIOMNI_SOURCE"):
            self.source = os.getenv("BIOMNI_SOURCE")

        # Protocols.io access token (prefer specific env vars)
        env_token = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN")
        if env_token:
            self.protocols_io_access_token = env_token

        # NCBI / Entrez email (NCBI_EMAIL is the canonical name; BIOMNI_NCBI_EMAIL
        # is accepted as a Biomni-namespaced alternative for consistency with
        # other BIOMNI_* env vars in this file).
        if os.getenv("BIOMNI_NCBI_EMAIL"):
            self.ncbi_email = os.getenv("BIOMNI_NCBI_EMAIL")
        elif os.getenv("NCBI_EMAIL"):
            self.ncbi_email = os.getenv("NCBI_EMAIL")

        self._resolve_local_first_default()

    def _resolve_local_first_default(self):
        """Default to a locally-available Ollama model.

        Resolution order:
          1. If BIOMNI_DISABLE_LOCAL_FALLBACK=true, do nothing (user wants a hard failure
             when no Ollama is running).
          2. If llm is anything other than the "auto" sentinel, the user has already
             chosen — leave it alone.
          3. If source / base_url / api_key is explicitly set, the user has chosen — leave it.
          4. Otherwise, try the local Ollama daemon. If at least one model is installed,
             pick the first one (the same model the Streamlit dropdown will show as
             index=0). If no Ollama daemon is reachable, leave llm="auto" so the caller
             gets a clear failure rather than a silent cloud default.

        Note: this function intentionally does NOT auto-pick a cloud model when a cloud
        API key is present. To use Anthropic / OpenAI / etc., set BIOMNI_LLM and
        BIOMNI_SOURCE explicitly — or pick the model in the UI dropdown.
        """
        if os.getenv("BIOMNI_DISABLE_LOCAL_FALLBACK", "").lower() == "true":
            return
        if self.llm != "auto" or self.source is not None:
            return
        if self.base_url or self.api_key:
            return

        # No cloud auto-defaults. The UI dropdown is the source of truth; this
        # resolver only handles the no-UI / programmatic case.
        try:
            from biomentis.ollama_utils import pick_default_ollama_model

            local_model = pick_default_ollama_model()
        except Exception:
            local_model = None

        if local_model:
            self.llm = local_model
            self.source = "Ollama"

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
            "ncbi_email": self.ncbi_email,
        }


# Global default config instance (optional, for convenience)
default_config = BiomentisConfig()

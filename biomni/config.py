"""
Biomni Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
from dataclasses import dataclass


@dataclass
class BiomniConfig:
    """Central configuration for Biomni agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = BiomniConfig()

        # Override specific settings
        config = BiomniConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"
    """

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    llm: str = "claude-sonnet-4-5"
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

        self._resolve_local_first_default()

    def _resolve_local_first_default(self):
        """If nothing else is configured, default to a locally available Ollama model.

        This only fires when llm/source/base_url/api_key are all still unset and no
        cloud provider API key is present in the environment, so anyone with explicit
        config or a cloud key set sees zero behavior change. A user who happens to
        explicitly pass the same string as the class default (llm="claude-sonnet-4-5")
        is indistinguishable from "unset" and will also see this fire; that collision
        is accepted as an edge case.
        """
        if os.getenv("BIOMNI_DISABLE_LOCAL_FALLBACK", "").lower() == "true":
            return
        if self.llm != "claude-sonnet-4-5" or self.source is not None:
            return
        if self.base_url or self.api_key:
            return
        cloud_key_env_vars = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "AWS_REGION")
        if any(os.getenv(var) for var in cloud_key_env_vars):
            return

        try:
            from biomni.ollama_utils import pick_default_ollama_model

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
        }


# Global default config instance (optional, for convenience)
default_config = BiomniConfig()

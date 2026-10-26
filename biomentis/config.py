"""
Biomentis Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
import re
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

    # Two temperatures, not one.
    #
    # A1 is a code-writing / code-executing loop: nearly every token it emits is
    # Python, a tool name, or a function signature, and sampling noise there
    # doesn't buy creativity — it buys hallucinated kwargs, wrong column names,
    # and retry loops. So the *base* temperature is cold, and it is what gets
    # used for code generation, tool retrieval, structured output, and every
    # verification/critique-of-a-claim pass.
    temperature: float = 0.2

    # Divergent calls — "what hypotheses should we test?", "what are we
    # missing?" — are the only place entropy earns its keep. Nothing uses this
    # implicitly; call sites opt in via A1.creative_llm.
    #
    # Note for local models: small Ollama models degrade faster with
    # temperature than frontier models do. If you run 7-13B locally, consider
    # BIOMNI_CREATIVE_TEMPERATURE=0.5 and BIOMNI_TEMPERATURE=0.1. This is left
    # as an explicit setting rather than auto-scaled per provider, so that a
    # given .env always means the same thing regardless of which model is up.
    creative_temperature: float = 0.7

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

    # Anthropic
    # Session-scoped override for ANTHROPIC_API_KEY, set by the sidebar field.
    # This is the highest-precedence tier in `resolve_anthropic_key` — see the
    # note there for why the environment is *not* the top tier.
    anthropic_key: str | None = None

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
        if os.getenv("BIOMNI_CREATIVE_TEMPERATURE"):
            self.creative_temperature = float(os.getenv("BIOMNI_CREATIVE_TEMPERATURE"))
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
            "creative_temperature": self.creative_temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
            "ncbi_email": self.ncbi_email,
        }


# Global default config instance (optional, for convenience)
default_config = BiomentisConfig()


# A deliberately loose check: "something@something.tld" with no whitespace.
# NCBI only ever uses this as a contact string, so the goal is to catch a
# typo or an empty box, not to adjudicate RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def apply_ncbi_email(email: str | None) -> bool:
    """Apply an NCBI contact email everywhere the process reads one.

    Returns True if `email` looked like an address and was applied, False if
    it was blank or malformed — in which case nothing is changed, so a bad
    keystroke in the UI can't wipe out a working value.

    Three targets, because three different code paths look in three places:

      - `default_config.ncbi_email` — read by the literature tools via
        `biomentis.tool.literature._get_ncbi_email`
      - `os.environ["NCBI_EMAIL"]` — read directly by
        `biomentis.tool.molecular_biology.get_gene_coding_sequence`, and
        inherited by any subprocess
      - `Bio.Entrez.email` — the module-level default behind Biopython's
        "Email address is not specified" warning. This is the one that covers
        *agent-written* code: `run_python_repl` execs those snippets in this
        same process, so a snippet that imports Entrez and forgets the email
        picks this up.

    Biopython is imported lazily and its absence is tolerated, so `config` does
    not become a hard dependency on it.
    """
    candidate = (email or "").strip()
    if not _EMAIL_RE.match(candidate):
        return False

    default_config.ncbi_email = candidate
    os.environ["NCBI_EMAIL"] = candidate
    try:
        from Bio import Entrez

        Entrez.email = candidate
    except Exception:
        pass
    return True


# --- Anthropic API key ---------------------------------------------------
#
# The key reaches the process by three routes that historically disagreed:
#
#   1. `.env`, loaded by `biomentis.agent.a1` with `load_dotenv(override=False)`
#   2. an exported shell variable
#   3. (removed) a `source ~/.bash_profile` subprocess in `biomentis.llm`
#
# `override=False` means an already-exported value wins, so a stale key left in
# a shell profile silently shadowed a good key in `.env` and produced
# "Error code: 401 - invalid x-api-key" with nothing pointing at the cause.
#
# `resolve_anthropic_key` inverts that for this one variable: `.env` is the
# project-local, deliberate source, so it beats an inherited shell value. The
# global `override=False` is deliberately left alone — flipping it would
# clobber intentional shell overrides for every other variable.

# Anthropic's own keys carry this prefix. It is a *hint*, not a gate: users
# pointing ANTHROPIC_BASE_URL at a gateway legitimately hold tokens that don't.
_ANTHROPIC_KEY_PREFIX = "sk-ant-"

_QUOTE_CHARS = "\"'"

# Example values like the ones `.env.example` ships with
# ("your_anthropic_api_key_here"). Copying the example file is the usual way
# to start, and a placeholder left in place was sent as the key: a 401 on
# every call, Claude models offered in the picker, and the sidebar saying
# "Applied." A placeholder counts as not set.
_PLACEHOLDER_SECRET_RE = re.compile(r"^(?:your[_-].*|<[^>]*>|sk-ant-\.\.\.|changeme|x{3,})$", re.IGNORECASE)


def _normalize_secret(raw: str | None) -> str:
    """Strip whitespace and one layer of surrounding quotes; "" for a placeholder.

    `KEY="sk-ant-..."` in a `.env` file, or a trailing newline picked up from a
    `source`d shell profile, would otherwise go into the `x-api-key` header
    verbatim and come back as a 401 that looks exactly like a revoked key.
    """
    value = _strip_secret(raw)
    return "" if _PLACEHOLDER_SECRET_RE.match(value) else value


def _strip_secret(raw: str | None) -> str:
    value = (raw or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARS:
        value = value[1:-1].strip()
    return value


def is_placeholder_secret(raw: str | None) -> bool:
    """True if `raw` is an example value such as `your_anthropic_api_key_here`."""
    return bool(_PLACEHOLDER_SECRET_RE.match(_strip_secret(raw)))


def secret_is_set(raw: str | None) -> bool:
    """True if `raw` holds a real value: not blank, and not a placeholder."""
    return bool(_normalize_secret(raw))


def looks_like_anthropic_key(key: str | None) -> bool:
    """True if `key` has the shape of a first-party Anthropic key.

    Used only to render a hint in the UI. A False here does not mean the key
    is unusable — see `_ANTHROPIC_KEY_PREFIX`.
    """
    return _normalize_secret(key).startswith(_ANTHROPIC_KEY_PREFIX)


def resolve_anthropic_key(env=None, dotenv_path: str = ".env") -> str | None:
    """Return the Anthropic key this process should use, or None.

    Precedence, highest first:

      1. `default_config.anthropic_key` — applied this session (the sidebar)
      2. `ANTHROPIC_API_KEY` in `dotenv_path` — project-local and deliberate
      3. `ANTHROPIC_API_KEY` in the environment — possibly a stale export

    Every tier is passed through `_normalize_secret`.
    """
    env = os.environ if env is None else env

    session = _normalize_secret(default_config.anthropic_key)
    if session:
        return session

    try:
        from dotenv import dotenv_values

        from_file = _normalize_secret(dotenv_values(dotenv_path).get("ANTHROPIC_API_KEY"))
        if from_file:
            return from_file
    except Exception:
        # A missing or unreadable .env is normal; fall through to the shell.
        pass

    return _normalize_secret(env.get("ANTHROPIC_API_KEY")) or None


def apply_anthropic_key(key: str | None) -> bool:
    """Apply an Anthropic key for this session.

    Returns True if `key` was usable and applied, False if it was blank or
    contained internal whitespace — in which case nothing changes, so a stray
    keystroke in the UI can't wipe out a key that already works.

    Writes both `default_config.anthropic_key` (the top precedence tier, read
    by `resolve_anthropic_key`) and `os.environ["ANTHROPIC_API_KEY"]`, which is
    what a bare `anthropic.Anthropic()` and any subprocess will pick up.

    Note the shape check is deliberately loose: anything non-blank with no
    internal whitespace is accepted, because gateway and proxy tokens are valid
    here too. `looks_like_anthropic_key` is the cosmetic check.
    """
    candidate = _normalize_secret(key)
    if not candidate or any(c.isspace() for c in candidate):
        return False

    default_config.anthropic_key = candidate
    os.environ["ANTHROPIC_API_KEY"] = candidate
    return True


def verify_anthropic_key(key: str | None) -> tuple[bool, str]:
    """Check a key against the API without spending any tokens.

    Uses `GET /v1/models` (`client.models.list`), which authenticates the key
    but runs no inference — so this can be called on a settings change without
    a cost. Returns `(ok, message)`; `message` is short and safe to render.

    A network failure returns `(False, ...)` with a message that says so, so
    the caller can distinguish "rejected" from "couldn't check".
    """
    candidate = _normalize_secret(key)
    if not candidate:
        return False, "No key set."

    try:
        import anthropic
    except ImportError:
        return False, "The `anthropic` package isn't installed."

    try:
        anthropic.Anthropic(api_key=candidate).models.list(limit=1)
        return True, "Key accepted by the Anthropic API."
    except anthropic.AuthenticationError:
        return False, "Anthropic rejected this key (401). It may be revoked or mistyped."
    except anthropic.PermissionDeniedError:
        return False, "Key authenticated but lacks permission (403)."
    except Exception as e:  # network down, proxy misconfigured, SDK too old
        return False, f"Couldn't verify the key: {type(e).__name__}."

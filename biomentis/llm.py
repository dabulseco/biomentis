import os
from typing import TYPE_CHECKING, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from biomentis.config import BiomentisConfig

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


# Client classes that end up NOT applying stop sequences, and why. Anything
# not listed here is treated as a bug by `_check_stop_sequences` below: the
# caller asked for stop sequences and the client dropped them.
#
# This registry exists because the failure mode is invisible. `ChatOllama` has
# a field named `stop` and no `stop_sequences` alias, so passing the name every
# other branch uses is accepted with no warning, no `model_kwargs`, and the
# value simply disappears. A branch that looks identical to its working
# neighbours can be doing nothing at all, and nothing says so.
STOP_SEQUENCE_EXEMPT_CLIENTS: dict[str, str] = {
    "ChatOllama": (
        "ChatOllama takes `stop=`, not `stop_sequences=`. Left unset deliberately: "
        "the default local model is a reasoning model that can emit `</execute>` "
        "while planning, and an API-side stop would cut it off mid-thought. The "
        "generate loop truncates at the first closing tag instead — see "
        "biomentis.utils.truncate_after_first_tag."
    ),
    "_ChatOpenAIResponsesNoStop": (
        "gpt-5 models reject `stop` on the Responses API, so the subclass sets it on "
        "the client and then drops it from the payload. Same local truncation applies."
    ),
}


def stop_sequences_applied(llm: BaseChatModel) -> bool:
    """Whether a constructed client actually carries stop sequences.

    The two provider SDKs spell it differently and mirror each other:
    `ChatOpenAI` has `stop` aliased to `stop_sequences`, `ChatAnthropic` has
    `stop_sequences` aliased to `stop`. Checking both is what makes this
    provider-agnostic.
    """
    return bool(getattr(llm, "stop", None) or getattr(llm, "stop_sequences", None))


def _check_stop_sequences(llm: BaseChatModel, stop_sequences: list[str] | None) -> BaseChatModel:
    """Say something when a caller asks for stop sequences and they vanish.

    Only fires for an *undeclared* drop. A known one belongs in
    `STOP_SEQUENCE_EXEMPT_CLIENTS` with its reason, which is what keeps this
    signal worth reading; if it ever prints, an SDK renamed a field or a new
    provider branch forgot to forward them.
    """
    if not stop_sequences:
        return llm
    name = type(llm).__name__
    if name in STOP_SEQUENCE_EXEMPT_CLIENTS or stop_sequences_applied(llm):
        return llm
    print(
        f"WARNING: {name} was given stop_sequences={stop_sequences!r} but carries none. "
        "The kwarg was probably silently dropped — check the field name in that SDK. "
        "Add it to biomentis.llm.STOP_SEQUENCE_EXEMPT_CLIENTS if the omission is intended."
    )
    return llm


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["BiomentisConfig"] = None,
) -> BaseChatModel:
    """Build a chat model for the given source, and verify what it was given.

    Takes the same arguments as `_build_llm`, which does the construction and
    documents them. The only thing added here is `_check_stop_sequences`, so a
    provider that drops a kwarg on the floor says so instead of looking fine.
    """
    llm = _build_llm(
        model=model,
        temperature=temperature,
        stop_sequences=stop_sequences,
        source=source,
        base_url=base_url,
        api_key=api_key,
        config=config,
    )
    return _check_stop_sequences(llm, stop_sequences)


def _build_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["BiomentisConfig"] = None,
) -> BaseChatModel:
    """
    Build a language model instance. Call `get_llm`, not this — it adds the
    stop-sequence check.

    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Sampling temperature. Defaults to the cold
                      BiomentisConfig.temperature, which is what you want for
                      code, tool calls and structured output. For divergent
                      work, pass config.creative_temperature explicitly (see
                      A1.creative_llm).
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (BiomentisConfig): Optional configuration object. If provided, unspecified parameters will use config values
    """
    # Use config values for any unspecified parameters
    if config is not None:
        if model is None:
            model = config.llm
        if temperature is None:
            temperature = config.temperature
        if source is None:
            source = config.source
        if base_url is None:
            base_url = config.base_url
        if api_key is None:
            api_key = config.api_key or "EMPTY"

    # Use defaults if still not specified. This is only the last-resort default when
    # get_llm is called without a config; the normal A1/react path resolves the
    # effective default (including the local-first Ollama fallback) via BiomentisConfig.
    if model is None:
        # "auto" is a sentinel — BiomentisConfig resolves it to a local Ollama model
        # when one is available. The auto-detect block below will see "auto" and
        # fall through to the Ollama branch (no "/" or known-cloud prefix).
        model = "auto"
    if temperature is None:
        # Matches BiomentisConfig.temperature. Cold, because the overwhelming
        # majority of calls through this function generate code or tool calls.
        temperature = 0.2
    if api_key is None:
        api_key = "EMPTY"
    # Auto-detect source from model name if not specified.
    # Order matters: explicit cloud-prefix names route to their cloud, anything
    # else defaults to Ollama so a fresh install with no API keys still works.
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        elif model[:7] == "claude-" and os.getenv("ANTHROPIC_API_KEY"):
            # Require the key to be present — otherwise an explicit "claude-..."
            # model name without a key would produce a confusing auth error
            # when Ollama would have worked fine.
            source = "Anthropic"
        elif model[:7] == "gpt-oss":
            source = "Ollama"
        elif model[:4] == "gpt-" and os.getenv("OPENAI_API_KEY"):
            source = "OpenAI"
        elif model.startswith("azure-"):
            source = "AzureOpenAI"
        elif model[:7] == "gemini-" and os.getenv("GEMINI_API_KEY"):
            source = "Gemini"
        elif "groq" in model.lower() and os.getenv("GROQ_API_KEY"):
            source = "Groq"
        elif model.startswith(
            ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
        ):
            source = "Bedrock"
        elif base_url is not None:
            source = "Custom"
        else:
            # Local-first default: anything that doesn't explicitly name a
            # cloud provider is assumed to be a local Ollama model (matches
            # Ollama's own naming conventions: "llama3.1", "qwen2.5:14b",
            # "hf.co/user/model", etc.). Requires `ollama serve` to be running.
            source = "Ollama"

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        # Newer OpenAI models (e.g., gpt-5-*) require the Responses API and may reject
        # legacy Chat Completions parameters like `stop`. Force Responses API when
        # using gpt-5 models to avoid 400 errors such as: "Unsupported parameter: 'stop'".
        use_responses = model.startswith("gpt-5")

        if use_responses:
            # Define a minimal subclass that drops the `stop` field when using the
            # Responses API, since certain models (gpt-5-*) reject it entirely.
            class _ChatOpenAIResponsesNoStop(ChatOpenAI):
                def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
                    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                    try:
                        # If this call will use the Responses API, drop `stop` to avoid 400s.
                        if hasattr(self, "_use_responses_api") and self._use_responses_api(payload):  # type: ignore[attr-defined]
                            payload.pop("stop", None)
                            # Also drop temperature for gpt-5 models as they only support default value
                            payload.pop("temperature", None)
                    except Exception:
                        # Be conservative: if anything goes wrong, still remove `stop` and `temperature`.
                        payload.pop("stop", None)
                        payload.pop("temperature", None)
                    return payload

            return _ChatOpenAIResponsesNoStop(
                model=model,
                temperature=1,  # Set to default value for gpt-5, will be removed in payload
                stop_sequences=stop_sequences,
                use_responses_api=True,
                output_version="v0",
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        API_VERSION = "2024-12-01-preview"
        model = model.replace("azure-", "")
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # Ensure ANTHROPIC_API_KEY is loaded from bash_profile if not in environment
        if not os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import subprocess

                result = subprocess.run(
                    ["bash", "-c", "source ~/.bash_profile 2>/dev/null && echo $ANTHROPIC_API_KEY"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    os.environ["ANTHROPIC_API_KEY"] = result.stdout.strip()
                    print("✓ Loaded ANTHROPIC_API_KEY from ~/.bash_profile")
            except Exception as e:
                print(f"Note: Could not load ANTHROPIC_API_KEY from bash_profile: {e}")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
        )

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        # Falls back to OLLAMA_HOST for consistency with the `ollama` package's own
        # convention; needed to reach a remote daemon or an Ollama Cloud sign-in.
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url or os.getenv("OLLAMA_HOST"),
        )

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        return ChatBedrock(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', or 'Ollama'"
        )

"""Helpers for discovering locally-running Ollama models (including Ollama Cloud

models, which proxy through the same local daemon once signed in via
`ollama signin`).

All functions here fail soft: if the `ollama` package is missing or the
daemon isn't reachable, they return an empty/None result instead of raising,
so callers (e.g. BiomniConfig's local-first default resolution) can use them
speculatively without risking a hard failure at import/construction time.
"""

import os


def _get_client(host: str | None = None):
    import ollama

    return ollama.Client(host=host or os.getenv("OLLAMA_HOST"))


def is_ollama_available(host: str | None = None, timeout: float = 1.0) -> bool:
    """Check whether an Ollama daemon is reachable at the given (or default) host."""
    try:
        _get_client(host).list()
        return True
    except Exception:
        return False


def list_ollama_models(host: str | None = None) -> list[str]:
    """List models available on the local Ollama daemon, local and cloud-signed-in alike."""
    try:
        response = _get_client(host).list()
        return [m.model for m in response.models]
    except Exception:
        return []


def pick_default_ollama_model(host: str | None = None) -> str | None:
    """Return the first available Ollama model to use as a zero-config default, if any."""
    models = list_ollama_models(host)
    return models[0] if models else None

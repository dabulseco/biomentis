"""Helpers for discovering locally-running Ollama models (including Ollama Cloud

models, which proxy through the same local daemon once signed in via
`ollama signin`).

All functions here fail soft: if the `ollama` package is missing or the
daemon isn't reachable, they return an empty/None result instead of raising,
so callers (e.g. BiomentisConfig's local-first default resolution) can use them
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


# Hosted models listed with an Ollama API key, keyed by a fingerprint of the
# key. The sidebar asks on every rerun; ollama.com is asked at most every
# _CLOUD_LIST_TTL seconds per key. A failed listing is not cached.
_CLOUD_LIST_TTL = 600.0
_cloud_list_cache: dict[str, tuple[float, list[str]]] = {}


def list_ollama_cloud_models(api_key: str | None) -> list[str]:
    """Models on Ollama's hosted API for this key, or [] (no key, or unreachable)."""
    import hashlib
    import time

    if not api_key:
        return []
    from biomentis.config import ollama_cloud_host

    host = ollama_cloud_host()
    fingerprint = hashlib.sha256(f"{host}|{api_key}".encode()).hexdigest()[:16]
    cached = _cloud_list_cache.get(fingerprint)
    if cached and time.monotonic() - cached[0] < _CLOUD_LIST_TTL:
        return list(cached[1])
    try:
        import ollama

        client = ollama.Client(host=host, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        models = sorted(m.model for m in client.list().models)
    except Exception as e:
        print(f"Note: could not list Ollama Cloud models ({type(e).__name__})")
        return []
    _cloud_list_cache[fingerprint] = (time.monotonic(), models)
    return list(models)

"""Small sidebar settings fields for the Biomentis Streamlit app.

Currently just the NCBI contact email. It lives in its own module rather than
inline in `streamlit_app.py` so the widget, its prefill, and its validation
branch can be driven directly by `AppTest` (see
`biomentis.agent.tests.test_ncbi_email`).

The field is session-scoped: it applies for as long as the app is running and
is not written to disk. Put `NCBI_EMAIL=you@example.com` in `.env` to have it
prefilled on every start.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from biomentis.config import (
    apply_anthropic_key,
    apply_ncbi_email,
    default_config,
    is_placeholder_secret,
    looks_like_anthropic_key,
    resolve_anthropic_key,
    verify_anthropic_key,
)

NCBI_EMAIL_KEY = "biomni_ncbi_email"

_HELP = (
    "NCBI asks for a contact address on every E-utilities request and "
    "rate-limits requests that omit one. Applies to this session only — set "
    "`NCBI_EMAIL=you@example.com` in .env to make it stick across restarts."
)


def render_ncbi_email_field() -> None:
    """Render the NCBI email input and apply whatever is currently in it.

    Prefills from the already-resolved config (i.e. from `NCBI_EMAIL` in
    `.env`) the first time it renders, then re-applies on every rerun so the
    value is in place before the agent runs any Entrez-touching code.
    """
    if NCBI_EMAIL_KEY not in st.session_state:
        st.session_state[NCBI_EMAIL_KEY] = default_config.ncbi_email or ""

    value = st.text_input(
        "NCBI email",
        key=NCBI_EMAIL_KEY,
        placeholder="you@example.com",
        help=_HELP,
    )

    if not value.strip():
        st.caption("⚠️ Not set — NCBI may throttle PubMed and Entrez requests.")
        return

    if apply_ncbi_email(value):
        st.caption(f"✓ Entrez requests identify as **{value.strip()}**")
    else:
        # `apply_ncbi_email` changed nothing, so whatever was applied before
        # this keystroke is still in force. Say so rather than implying the
        # setting is now empty.
        st.caption("⚠️ That doesn't look like an email — keeping the previous value.")


# --- Anthropic API key ---------------------------------------------------

ANTHROPIC_KEY_KEY = "biomni_anthropic_key"

_ANTHROPIC_HELP = (
    "Used by `advanced_web_search_claude` for multi-round agentic web search, "
    "and by Claude chat models. Without a working key the search silently "
    "degrades to a keyless DuckDuckGo scrape. Applies to this session only — "
    "set `ANTHROPIC_API_KEY=sk-ant-...` in .env to make it stick across "
    "restarts."
)

_VERDICT_KEY = "biomni_anthropic_key_verdict"


def _verify_cached(key: str) -> tuple[bool, str]:
    """Verify `key` at most once per distinct key per session.

    `verify_anthropic_key` is a real network round-trip. Streamlit reruns the
    whole script on every interaction, so an uncached call here would hit the
    API on every keystroke anywhere in the app.
    """
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    cached = st.session_state.get(_VERDICT_KEY)
    if cached and cached[0] == fingerprint:
        return cached[1], cached[2]

    ok, message = verify_anthropic_key(key)
    st.session_state[_VERDICT_KEY] = (fingerprint, ok, message)
    return ok, message


def render_anthropic_key_field(verify: bool = True) -> None:
    """Render the Anthropic key input and apply whatever is currently in it.

    Prefills from `resolve_anthropic_key`, so the box shows the key the tools
    will actually use — which is the point: when a stale shell export was
    shadowing `.env`, nothing in the UI revealed which key was in play.

    `verify=False` skips the network probe; the tests use it so the suite
    doesn't depend on having a live key or a network.
    """
    if ANTHROPIC_KEY_KEY not in st.session_state:
        st.session_state[ANTHROPIC_KEY_KEY] = resolve_anthropic_key() or ""

    value = st.text_input(
        "Anthropic API key",
        key=ANTHROPIC_KEY_KEY,
        type="password",
        placeholder="sk-ant-...",
        help=_ANTHROPIC_HELP,
    )

    if not value.strip():
        st.caption("⚠️ Not set — web search will use the keyless DuckDuckGo fallback.")
        return

    if is_placeholder_secret(value):
        st.caption(
            "⚠️ That's the example placeholder from `.env.example`, not a key. "
            "Paste your real key here, or set `ANTHROPIC_API_KEY=sk-ant-...` in .env."
        )
        return

    if not apply_anthropic_key(value):
        # `apply_anthropic_key` changed nothing, so whatever was applied before
        # this keystroke is still in force. Say so rather than implying the
        # setting is now empty.
        st.caption("⚠️ That doesn't look like a key — keeping the previous value.")
        return

    if not looks_like_anthropic_key(value):
        # Not an error: gateway/proxy tokens are legitimate when ANTHROPIC_BASE_URL
        # points somewhere other than api.anthropic.com.
        st.caption("Applied. Note it doesn't start with `sk-ant-` — expected only if you use a proxy.")
        return

    if not verify:
        st.caption("✓ Key applied for this session.")
        return

    ok, message = _verify_cached(value.strip())
    st.caption(("✓ " if ok else "⚠️ ") + message)

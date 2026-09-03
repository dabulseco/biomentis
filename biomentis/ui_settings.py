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

import streamlit as st

from biomentis.config import apply_ncbi_email, default_config

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

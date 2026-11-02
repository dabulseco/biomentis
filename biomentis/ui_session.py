"""The browser session's key for background runs and repeat-run batches, kept in the URL.

Runs and batches are held in module-level registries keyed by this key
(`run_worker._RUNS`, `ui_repeat._BATCHES`). Session state alone is not enough
to find them again: after a disconnect longer than Streamlit's session TTL
(`server.disconnectedSessionTTL`, 120 s) a reconnecting tab gets a new,
empty session state. So the key is also written to the page URL as
``?s=<key>``, and a session that finds one there adopts it: a reload or a
reconnect in the same tab re-attaches to the same run or batch.

This is a single-user app, so a run whose tab is gone for good (closed, or
opened without the ``?s=``) can also be taken over: `orphaned` says whether
the Streamlit session that last used a key is still connected, and
`adopt` makes the current session use that key.
"""

from __future__ import annotations

import re
import threading
import uuid

import streamlit as st

QUERY_PARAM = "s"
STATE_KEY = "biomni_session_key"
TAKE_OVER_LABEL = "Take over this run"
_VALID_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# biomentis session key -> id of the Streamlit session that last used it.
_OWNERS: dict[str, str] = {}
_LOCK = threading.Lock()


def _streamlit_session_id() -> str | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return ctx.session_id if ctx is not None else None
    except Exception:
        return None


def _session_is_live(streamlit_session_id: str) -> bool:
    """Whether that Streamlit session is still connected. True when it cannot be told (tests)."""
    try:
        from streamlit.runtime import Runtime

        if not Runtime.exists():
            return True
        return Runtime.instance().is_active_session(streamlit_session_id)
    except Exception:
        return True


def session_key() -> str:
    """This tab's key: the one in the URL if there is one, else the session's, else a new one.

    Called on every script run by everything that looks a run or batch up, so
    the URL and the owner registry stay current.
    """
    try:
        url_key = st.query_params.get(QUERY_PARAM)
    except Exception:
        url_key = None
    key = st.session_state.get(STATE_KEY)
    if url_key and _VALID_KEY.match(url_key) and url_key != key:
        key = url_key  # a reload or a reconnect: re-attach to this tab's run
        st.session_state[STATE_KEY] = key
    if not key:
        key = uuid.uuid4().hex
        st.session_state[STATE_KEY] = key
    if url_key != key:
        try:
            st.query_params[QUERY_PARAM] = key
        except Exception:
            pass
    owner = _streamlit_session_id()
    if owner is not None:
        with _LOCK:
            _OWNERS[key] = owner
    return key


def orphaned(key: str) -> bool:
    """Whether no connected Streamlit session uses `key` any more (its tab is gone)."""
    with _LOCK:
        owner = _OWNERS.get(key)
    if owner is None:
        return False  # never seen in this process: nothing to tell
    if owner == _streamlit_session_id():
        return False
    return not _session_is_live(owner)


def adopt(key: str) -> None:
    """Make this session use `key` (a button callback): its run or batch is shown here from now on."""
    st.session_state[STATE_KEY] = key
    st.session_state["biomni_run_cursor"] = 0  # replay the run's events into this session's transcript
    try:
        st.query_params[QUERY_PARAM] = key
    except Exception:
        pass
    owner = _streamlit_session_id()
    if owner is not None:
        with _LOCK:
            _OWNERS[key] = owner

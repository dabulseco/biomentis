"""Tests for the NCBI/Entrez email setting and its sidebar field.

NCBI asks (politely) for an email on every E-utilities request, and Biopython
warns when one isn't set:

    UserWarning: Email address is not specified.

The warning that prompted this came out of *agent-generated* code, not out of
the literature tools: `run_python_repl` `exec`s the agent's snippets in this
very process, and a snippet that does `from Bio import Entrez` inherits
whatever module-level `Entrez.email` the process has. So the fix has to reach
three places at once — `default_config`, the environment, and the live
`Bio.Entrez` module — which is what `apply_ncbi_email` does.

These tests pin down:

  1. A valid email lands in all three places
  2. `Bio.Entrez` actually stops warning — asserted against the real warning
     site (`_construct_params`), so this is the reported symptom, not a proxy
  3. Blank / malformed input is rejected and does NOT clobber a good value
  4. Surrounding whitespace is trimmed
  5. The literature tools' resolver reflects an applied value
  6. `apply_ncbi_email` survives Biopython being absent
  7. The sidebar field prefills from the resolved config
  8. Typing a valid address in the sidebar applies it
  9. Typing garbage warns in-line and leaves the previous value alone

Tests 7-9 drive the real `render_ncbi_email_field` inside a Streamlit
`AppTest` harness. Note the AppTest quirk called out in CLAUDE.md: an
externally assigned widget value survives only the run immediately after the
assignment, so each case stages its state and its assertion in one `at.run()`.

Run with:
    python -m biomentis.agent.tests.test_ncbi_email
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

from biomentis.config import apply_ncbi_email, default_config

# --- Tiny test harness (matches test_tab_focus.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _reset() -> None:
    """Clear the three targets so each test starts from a known state."""
    default_config.ncbi_email = None
    os.environ.pop("NCBI_EMAIL", None)
    try:
        from Bio import Entrez

        Entrez.email = None
    except ImportError:
        pass


def _entrez_warns() -> bool:
    """True if Biopython would emit the 'Email address is not specified'
    warning right now. Calls the real warning site; no network involved."""
    from Bio import Entrez

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Entrez._construct_params({})
    return any("Email address is not specified" in str(w.message) for w in caught)


# --- Tests ---------------------------------------------------------------


def test_valid_email_reaches_all_three_targets() -> None:
    print("\n[1] a valid email lands in config, env, and Bio.Entrez")
    _reset()
    ok = apply_ncbi_email("scientist@example.org")
    _check("returns True", ok is True, f"got {ok!r}")
    _check(
        "default_config.ncbi_email set",
        default_config.ncbi_email == "scientist@example.org",
        f"got {default_config.ncbi_email!r}",
    )
    _check(
        "NCBI_EMAIL env var set",
        os.environ.get("NCBI_EMAIL") == "scientist@example.org",
        f"got {os.environ.get('NCBI_EMAIL')!r}",
    )
    from Bio import Entrez

    _check(
        "Bio.Entrez.email set",
        Entrez.email == "scientist@example.org",
        f"got {Entrez.email!r}",
    )


def test_entrez_warning_is_silenced() -> None:
    print("\n[2] the reported UserWarning actually goes away")
    _reset()
    _check("warns before apply", _entrez_warns() is True, "no warning with email unset")
    apply_ncbi_email("scientist@example.org")
    _check("silent after apply", _entrez_warns() is False, "still warning after apply")


def test_blank_and_malformed_are_rejected() -> None:
    print("\n[3] blank / malformed input is rejected without clobbering")
    _reset()
    apply_ncbi_email("good@example.org")

    for bad in ("", "   ", None, "not-an-email", "no@tld", "two@@at.org", "a b@c.org"):
        ok = apply_ncbi_email(bad)
        _check(f"{bad!r} returns False", ok is False, f"got {ok!r}")

    _check(
        "prior value survives",
        default_config.ncbi_email == "good@example.org",
        f"got {default_config.ncbi_email!r}",
    )
    _check(
        "prior env var survives",
        os.environ.get("NCBI_EMAIL") == "good@example.org",
        f"got {os.environ.get('NCBI_EMAIL')!r}",
    )


def test_whitespace_is_trimmed() -> None:
    print("\n[4] surrounding whitespace is trimmed")
    _reset()
    ok = apply_ncbi_email("  scientist@example.org\n")
    _check("returns True", ok is True, f"got {ok!r}")
    _check(
        "stored trimmed",
        default_config.ncbi_email == "scientist@example.org",
        f"got {default_config.ncbi_email!r}",
    )


def test_literature_resolver_sees_it() -> None:
    print("\n[5] the literature tools' resolver reflects the applied value")
    _reset()
    from biomentis.tool.literature import _get_ncbi_email

    _check(
        "placeholder when unset",
        _get_ncbi_email() == "your-email@example.com",
        f"got {_get_ncbi_email()!r}",
    )
    apply_ncbi_email("scientist@example.org")
    _check(
        "resolves to applied value",
        _get_ncbi_email() == "scientist@example.org",
        f"got {_get_ncbi_email()!r}",
    )


def test_survives_missing_biopython() -> None:
    print("\n[6] apply_ncbi_email works even if Biopython isn't importable")
    _reset()
    import builtins

    real_import = builtins.__import__

    def _no_bio(name, *args, **kwargs):
        if name == "Bio" or name.startswith("Bio."):
            raise ImportError("No module named 'Bio'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_bio
    try:
        ok = apply_ncbi_email("scientist@example.org")
    finally:
        builtins.__import__ = real_import

    _check("still returns True", ok is True, f"got {ok!r}")
    _check(
        "config still set",
        default_config.ncbi_email == "scientist@example.org",
        f"got {default_config.ncbi_email!r}",
    )


# --- Streamlit harness ---------------------------------------------------

# Drives the real `render_ncbi_email_field`, so a change to the widget key,
# the prefill, or the validation branch is caught here.
_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.config import apply_ncbi_email, default_config
from biomentis.ui_settings import NCBI_EMAIL_KEY, render_ncbi_email_field

# Stage the pre-existing config value the way streamlit_app.py would have it.
_seed = st.session_state.get("seed_config_email", None)
if _seed is not None:
    default_config.ncbi_email = _seed

with st.sidebar:
    render_ncbi_email_field()

st.write("CONFIG=" + repr(default_config.ncbi_email))
st.write("WIDGET=" + repr(st.session_state.get(NCBI_EMAIL_KEY)))
'''

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _app_test():
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_ncbi_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO))
    return AppTest.from_file(path, default_timeout=60)


def _line(at, prefix: str) -> str:
    for m in at.markdown:
        if m.value.startswith(prefix):
            return m.value
    return ""


def test_sidebar_field_prefills_from_config() -> None:
    print("\n[7] the sidebar field prefills from the resolved config")
    _reset()
    at = _app_test()
    at.session_state["seed_config_email"] = "from-dotenv@example.org"
    at.run()

    _check("no exception", not at.exception, str(at.exception))
    inputs = [w for w in at.sidebar.text_input]
    _check("field rendered", len(inputs) == 1, f"got {len(inputs)} text inputs")
    if inputs:
        _check(
            "prefilled from config",
            inputs[0].value == "from-dotenv@example.org",
            f"got {inputs[0].value!r}",
        )


def test_sidebar_field_applies_a_valid_email() -> None:
    print("\n[8] typing a valid address applies it")
    _reset()
    at = _app_test()
    at.run()
    at.sidebar.text_input[0].set_value("typed@example.org").run()

    _check("no exception", not at.exception, str(at.exception))
    _check(
        "config updated",
        _line(at, "CONFIG=") == "CONFIG='typed@example.org'",
        f"got {_line(at, 'CONFIG=')!r}",
    )
    _check(
        "env updated",
        os.environ.get("NCBI_EMAIL") == "typed@example.org",
        f"got {os.environ.get('NCBI_EMAIL')!r}",
    )


def test_sidebar_field_flags_garbage_and_keeps_prior() -> None:
    print("\n[9] garbage is flagged in-line and the prior value survives")
    _reset()
    at = _app_test()
    at.session_state["seed_config_email"] = "good@example.org"
    at.run()
    at.sidebar.text_input[0].set_value("nonsense").run()

    _check("no exception", not at.exception, str(at.exception))
    _check(
        "prior value survives",
        _line(at, "CONFIG=") == "CONFIG='good@example.org'",
        f"got {_line(at, 'CONFIG=')!r}",
    )
    captions = " ".join(c.value for c in at.sidebar.caption)
    _check(
        "in-line warning shown",
        "doesn't look like an email" in captions,
        f"captions were {captions!r}",
    )


def run_all() -> int:
    print("=" * 60)
    print("NCBI email setting tests")
    print("=" * 60)
    try:
        test_valid_email_reaches_all_three_targets()
        test_entrez_warning_is_silenced()
        test_blank_and_malformed_are_rejected()
        test_whitespace_is_trimmed()
        test_literature_resolver_sees_it()
        test_survives_missing_biopython()
        test_sidebar_field_prefills_from_config()
        test_sidebar_field_applies_a_valid_email()
        test_sidebar_field_flags_garbage_and_keeps_prior()
    finally:
        _reset()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())

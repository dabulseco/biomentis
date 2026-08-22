"""Tests for the transcript tabs' focus behavior in `launch_streamlit_demo`.

The two views ("Biomentis AI Agent" / "Biomentis Executor") used to fake a
programmatic tab switch by re-ordering their labels each rerun, because
`st.tabs` had no selection API in Streamlit 1.42. That made the tabs visibly
trade places partway through a run, and — because a stateful tab's selection
is stored as the label string — silently discarded any tab the user had
picked themselves.

These tests pin the replacement down:

  1. The labels and their order are constants (never conditional on run state)
  2. Writing `_TAB_INTENT_KEY` before `st.tabs` moves the selection
  3. Writing a widget key AFTER `st.tabs` raises — this is the constraint that
     forces the intent-key indirection, so it is asserted rather than assumed
  4. The Send button's `on_click` focuses the Executor and clears the pin
  5. A manual tab click pins the selection, and a later auto-focus request is
     ignored rather than yanking the user back
  6. A new run clears the pin, earning one fresh auto-focus
  7. Changing a tab label resets the selection — the reason the run banner
     lives outside the tabs instead of as a badge on one
  8. In research mode the tabs are NOT stateful, so clicking one sends
     nothing to the server — no rerun, and therefore no torn-down run
  9. Statefulness is gated on the tutor actually driving the run

Statefulness is the knob behind 8 and 9: tracked tabs are what make
programmatic focus possible, and also what makes a click cost a rerun. A
research run streams start to finish inside one script execution, so a rerun
part-way through would lose it — hence tracked tabs only while the tutor is
gating the run, where a rerun is the normal mechanism anyway.

Tests 2-8 run the real constants and the real callbacks from `a1.py` inside a
Streamlit `AppTest` harness that mirrors the render block.

Run with:
    python -m biomentis.agent.tests.test_tab_focus
"""

from __future__ import annotations

import os
import re
import sys
import tempfile


# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


# The app under test. Mirrors the tab block in `launch_streamlit_demo`, but
# imports the real constants and the real callbacks so a change to either is
# caught here rather than only in the browser.
_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.agent.a1 import (
    _TAB_INNER_LABEL,
    _TAB_INTENT_KEY,
    _TAB_KEY,
    _TAB_MAIN_LABEL,
    _TAB_PINNED_KEY,
    _focus_executor_for_new_run,
    _note_manual_tab_switch,
)

# --- the real render block ---
_tabs_stateful = st.session_state.get("stateful", True)

if _tabs_stateful:
    _intent = st.session_state.pop(_TAB_INTENT_KEY, None)
    if _intent in (_TAB_MAIN_LABEL, _TAB_INNER_LABEL) and not st.session_state.get(_TAB_PINNED_KEY):
        st.session_state[_TAB_KEY] = _intent
    main_col, inner_col = st.tabs(
        [_TAB_MAIN_LABEL, _TAB_INNER_LABEL], key=_TAB_KEY, on_change=_note_manual_tab_switch
    )
else:
    st.session_state.pop(_TAB_INTENT_KEY, None)
    main_col, inner_col = st.tabs([_TAB_MAIN_LABEL, _TAB_INNER_LABEL], key=_TAB_KEY)
with main_col:
    st.write("main body")
with inner_col:
    st.write("inner body")

st.write(
    "SEL=" + repr(st.session_state.get(_TAB_KEY))
    + " PINNED=" + repr(st.session_state.get(_TAB_PINNED_KEY))
)
# `.open` is populated only when the tabs are registered as a widget, so it
# doubles as a probe for whether a click would cost a rerun.
st.write("OPEN=" + repr((main_col.open, inner_col.open)))

with st.form("prompt_form", clear_on_submit=True):
    st.text_input("prompt", key="p")
    st.form_submit_button("Send", on_click=_focus_executor_for_new_run)
'''

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _app_test():
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_tabs_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO))
    return AppTest.from_file(path, default_timeout=60).run()


def _open_flags(at) -> str:
    """The `OPEN=(...)` line: `(None, None)` iff the tabs are untracked."""
    for m in at.markdown:
        if m.value.startswith("OPEN="):
            return m.value
    return "(missing)"


def _selection(at) -> str:
    """The `SEL=... PINNED=...` line the harness writes each rerun."""
    for m in at.markdown:
        if m.value.startswith("SEL="):
            return m.value
    return "(missing)"


# --- Tests ----------------------------------------------------------------


def test_labels_are_constant() -> None:
    """The source must not build the label list conditionally."""
    print("\n[1] Tab labels and order are constants")
    src_path = os.path.join(_REPO, "biomentis", "agent", "a1.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()

    # Two calls: the stateful (tutor-gated) branch and the untracked
    # research-mode branch. Both must use the same constant label list.
    _check(
        "every st.tabs() call uses the two constants, in order",
        src.count("st.tabs(") == 2
        and len(
            re.findall(r"st\.tabs\(\s*\[_TAB_MAIN_LABEL,\s*_TAB_INNER_LABEL\]", src)
        )
        == 2,
        f"{src.count('st.tabs(')} calls, "
        f"{len(re.findall(r'st.tabs..s*..._TAB_MAIN_LABEL,.s*_TAB_INNER_LABEL.', src))} matching",
    )
    # The old workaround; if any of this comes back, the tabs can move again.
    for dead in ("_tab_labels", "_first_tab", "_second_tab"):
        _check(f"old reorder symbol `{dead}` is gone", dead not in src)
    _check(
        "tabs are stateful (key + on_change), which is what makes focus real",
        'key=_TAB_KEY' in src and "on_change=_note_manual_tab_switch" in src,
    )


def test_intent_moves_focus() -> None:
    print("\n[2] An intent written before st.tabs moves the selection")
    at = _app_test()
    _check("no exception on first render", not at.exception, str(at.exception))
    _check(
        "defaults to the AI Agent tab",
        "'Biomentis AI Agent'" in _selection(at),
        _selection(at),
    )

    at.session_state["biomni_tab_intent"] = "Biomentis Executor"
    at.run()
    _check(
        "intent switches the selection to the Executor",
        "SEL='Biomentis Executor'" in _selection(at),
        _selection(at),
    )
    _check(
        "intent is consumed, not sticky",
        "biomni_tab_intent" not in at.session_state,
        "intent key survived the rerun",
    )


def test_writing_widget_key_after_instantiation_raises() -> None:
    """The constraint that forces the intent indirection in the first place."""
    print("\n[3] Writing the tabs key after st.tabs raises")
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_tabs_late_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "import streamlit as st\n"
            "st.tabs(['A', 'B'], key='k', on_change='rerun')\n"
            "try:\n"
            "    st.session_state['k'] = 'B'\n"
            "    st.write('RESULT=no-raise')\n"
            "except Exception as e:\n"
            "    st.write('RESULT=' + type(e).__name__)\n"
        )
    at = AppTest.from_file(path, default_timeout=60).run()
    got = [m.value for m in at.markdown if m.value.startswith("RESULT=")]
    _check(
        "assigning a tabs key post-instantiation raises StreamlitAPIException",
        got == ["RESULT=StreamlitAPIException"],
        f"got {got}",
    )


def test_submit_focuses_executor() -> None:
    print("\n[4] The Send button's on_click focuses the Executor")
    at = _app_test()
    at.button[0].click().run()
    _check(
        "focus moved to the Executor on submit",
        "SEL='Biomentis Executor'" in _selection(at),
        _selection(at),
    )
    _check(
        "the pin is cleared for the new run",
        "PINNED=False" in _selection(at),
        _selection(at),
    )
    _check("no exception on submit", not at.exception, str(at.exception))
    _check(
        "tutor-gated tabs DO track state (the cost that buys programmatic focus)",
        _open_flags(at) == "OPEN=(False, True)",
        _open_flags(at),
    )


def test_manual_choice_is_respected() -> None:
    print("\n[5] A manual tab choice suppresses auto-focus")
    at = _app_test()

    # Stand in for the user having clicked the Executor tab. AppTest has no
    # tab-click API, and an externally-assigned widget value only survives
    # the run immediately following the assignment (a harness limitation,
    # not app behavior) — so the pin, the selection, and the competing
    # focus request all have to be staged for the SAME run.
    at.session_state["biomni_tab_pinned"] = True
    at.session_state["biomni_active_tab"] = "Biomentis Executor"
    at.session_state["biomni_tab_intent"] = "Biomentis AI Agent"
    at.run()

    # Discriminating: were the pin ignored, the intent would have pulled the
    # selection to the AI Agent tab.
    _check(
        "an auto-focus request is ignored while the user has pinned a tab",
        "SEL='Biomentis Executor'" in _selection(at),
        _selection(at),
    )
    _check(
        "the pin survives the rerun",
        "PINNED=True" in _selection(at),
        _selection(at),
    )


def test_new_run_clears_the_pin() -> None:
    print("\n[6] A new run earns one fresh auto-focus")
    at = _app_test()
    at.session_state["biomni_tab_pinned"] = True
    at.session_state["biomni_active_tab"] = "Biomentis AI Agent"
    at.run()
    _check(
        "pinned on the AI Agent tab",
        "SEL='Biomentis AI Agent'" in _selection(at) and "PINNED=True" in _selection(at),
        _selection(at),
    )

    # Discriminating: the default selection is the AI Agent tab and the
    # submit callback asks for the Executor, so if the pin were NOT cleared
    # the request would be ignored and the selection would stay put.
    at.button[0].click().run()  # submit a new run
    _check(
        "submitting clears the pin and focuses the Executor again",
        "SEL='Biomentis Executor'" in _selection(at) and "PINNED=False" in _selection(at),
        _selection(at),
    )


def test_research_mode_tabs_are_inert() -> None:
    """Research mode must not pay a rerun (and a lost run) for a tab click."""
    print("\n[8] Research-mode tabs are untracked, so a click costs nothing")
    at = _app_test()
    at.session_state["stateful"] = False
    at.session_state["biomni_tab_intent"] = "Biomentis Executor"
    at.run()

    _check("no exception with untracked tabs", not at.exception, str(at.exception))
    # `.open` is None exactly when the tabs are not registered as a widget —
    # which is what makes a click inert (nothing is sent back to the server,
    # so no rerun, so no torn-down research run).
    _check(
        "tabs report no tracked state, so a click sends nothing to the server",
        _open_flags(at) == "OPEN=(None, None)",
        _open_flags(at),
    )
    _check(
        "a stale focus request is dropped rather than left to fire later",
        "biomni_tab_intent" not in at.session_state,
        "intent survived into research mode",
    )


def test_statefulness_condition_is_wired_to_rerun_safety() -> None:
    print("\n[9] Statefulness is gated on the tutor actually driving the run")
    src_path = os.path.join(_REPO, "biomentis", "agent", "a1.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()

    _check(
        "statefulness requires BOTH a stream_fn and an enabled tutor",
        re.search(
            r"_tabs_stateful\s*=\s*stream_fn is not None and bool\(\s*getattr\(\s*_tutor,\s*[\"']enabled[\"']",
            src,
        )
        is not None,
        "the `_tabs_stateful` condition is not the expected expression",
    )
    _check(
        "on_change is only ever passed on the stateful branch",
        src.count("on_change=_note_manual_tab_switch") == 1,
        f"found {src.count('on_change=_note_manual_tab_switch')} occurrences",
    )
    _check(
        "there is a second, untracked st.tabs call for research mode",
        src.count("st.tabs(") == 2,
        f"found {src.count('st.tabs(')}",
    )


def test_label_change_resets_selection() -> None:
    """Why the run banner is rendered outside the tabs, not as a label badge."""
    print("\n[7] Changing a tab's label resets the selection")
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_tabs_label_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "import streamlit as st\n"
            "busy = st.session_state.get('busy', False)\n"
            "labels = ['A', 'B busy' if busy else 'B']\n"
            "st.tabs(labels, key='k', on_change='rerun')\n"
            "st.write('SEL=' + repr(st.session_state.get('k')))\n"
        )
    at = AppTest.from_file(path, default_timeout=60).run()
    at.session_state["k"] = "B"
    at.run()
    before = [m.value for m in at.markdown if m.value.startswith("SEL=")]
    at.session_state["busy"] = True
    at.run()
    after = [m.value for m in at.markdown if m.value.startswith("SEL=")]
    _check(
        "a label change drops the user's selection back to the first tab",
        before == ["SEL='B'"] and after == ["SEL='A'"],
        f"before={before} after={after}",
    )


def run_all() -> int:
    print("=" * 60)
    print("Transcript tab focus tests")
    print("=" * 60)
    test_labels_are_constant()
    test_intent_moves_focus()
    test_writing_widget_key_after_instantiation_raises()
    test_submit_focuses_executor()
    test_manual_choice_is_respected()
    test_new_run_clears_the_pin()
    test_label_change_resets_selection()
    test_research_mode_tabs_are_inert()
    test_statefulness_condition_is_wired_to_rerun_safety()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())

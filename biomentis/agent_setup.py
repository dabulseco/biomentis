"""How the app and the repeat-run CLI set an agent up: sidebar settings, the default reviewer.

Kept out of `streamlit_app.py` (which runs the whole app when imported) so the
CLI can share it and tests can call it. Nothing here touches Streamlit:
`settings_from_state` takes the session state as a plain mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from biomentis.config import default_config

REVIEWER_NONE_MESSAGE = "Reviewer: none (set BIOMNI_REVIEW_MODEL) — runs will be 'Not fully checked'"
SAME_FAMILY_WARNING = "Reviewer and agent are the same model family — they may share blind spots."


def build_reviewer(name: str, build: Callable[..., Any] | None = None) -> tuple[Any, str]:
    """(review_llm, error) for the default reviewer model `name`; (None, why) if it cannot be built.

    Verification runs cold, like the Critic: the agent's temperature, not
    the creative one.
    """
    if build is None:
        from biomentis.llm import get_llm as build
    try:
        return build(name, temperature=default_config.temperature, config=default_config), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def settings_from_state(state: Mapping, review_llm: Any, review_model_name: str,
                        critic_priorities: list[str]) -> dict:
    """The sidebar's agent settings, from the session state (read on the script thread)."""
    # A deselected segmented control reads None; treat it as the default.
    rounds = state.get("biomentis_critique_rounds")
    rounds = 2 if rounds is None else rounds
    # The rubric upload, for the next run (the draft_rubric node reads it).
    rubric_file = state.get("biomentis_rubric_upload")
    return dict(
        temperature=state.get("biomentis_temperature", default_config.temperature),
        creative_temperature=state.get("biomentis_creative_temperature", default_config.creative_temperature),
        review_llm=review_llm,
        review_model_name=review_model_name,
        rounds=rounds,
        critic_priorities=list(critic_priorities),
        completion_gate=state.get("biomentis_completion_gate", default_config.completion_gate),
        # Always on, as before the completion checks: with them off, the old
        # evidence nudge sends an unsupported answer back; with them on it is
        # ignored (the gate's evidence check replaces it).
        evidence_gate=True,
        rubric_upload=rubric_file.getvalue().decode("utf-8", errors="replace") if rubric_file is not None else None,
    )


def repeat_settings(s: dict) -> dict:
    """The settings for a repeat run: the interactive ones, with the completion gate always on."""
    return {**s, "completion_gate": True}


def configure_agent(agent: Any, s: dict) -> bool:
    """Apply settings to an agent; True when its temperatures changed (a new client).

    Called for the interactive agent and for each repeat run's fresh agent
    (on the batch's thread, so it reads only `s`, never the session).
    """
    changed = agent.set_temperatures(s["temperature"], s["creative_temperature"])
    agent.review_llm = s["review_llm"]
    agent.review_model_name = s["review_model_name"]
    agent.configure(
        self_critic=s["rounds"] > 0,
        test_time_scale_round=s["rounds"],
        critic_priorities=list(s["critic_priorities"]),
        completion_gate=s["completion_gate"],
        evidence_gate=s["evidence_gate"],
    )
    agent.set_rubric_upload(s["rubric_upload"])
    return changed

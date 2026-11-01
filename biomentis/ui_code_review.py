"""The "Code Review" button and report panel shown under a piece of code.

A click queues a review (`code_review_jobs.submit_review`); while it runs, only
a small fragment reruns every 2 s, so an agent run or tutor pause on the rest of
the page is untouched. The component writes no session state of its own beyond
its widget keys.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

import streamlit as st

from biomentis.code_review import CodeReview, CodeReviewRequest, review_key
from biomentis.code_review_jobs import review_status, submit_review

VERDICT_LABELS = {
    "ok": "✅ OK to run",
    "run_with_fixes": "⚠️ Run with fixes",
    "do_not_trust": "⛔ Don't trust the output",
}
_EXTENSIONS = {"python": ".py", "bash": ".sh", "r": ".R"}
POLL_SECONDS = 2


def _file_name(req: CodeReviewRequest) -> str:
    block = req.block_index if req.block_index is not None else hashlib.sha1(req.code.encode("utf-8")).hexdigest()[:8]
    return f"code_review_{block}{_EXTENSIONS.get(req.language.lower(), '.py')}"


def _plain(text: str) -> str:
    """Escape Markdown and LaTeX characters so model text shows literally."""
    return re.sub(r"([\\`*_$~#<>\[\]])", r"\\\1", text)


def _show_report(req: CodeReviewRequest, review: CodeReview, key: str) -> None:
    label = VERDICT_LABELS.get(review.verdict, review.verdict or "Done")
    with st.expander(f"Code review — {label}", expanded=True):
        if review.summary:
            st.markdown(_plain(review.summary))
        for n, issue in enumerate(review.issues, 1):
            where = f" (line {issue.line})" if issue.line is not None else ""
            st.markdown(f"**{n}. {issue.severity.upper()} · {issue.kind}{where}**")
            if issue.quote:
                st.code(issue.quote, language=req.language)
            st.markdown(f"Problem: {_plain(issue.problem)}")
            st.markdown(f"Fix: {_plain(issue.fix)}")
        if not review.corrected_code:
            st.write("No corrected script could be produced.")
            return
        if review.syntax_error:
            st.warning(f"⚠️ This corrected script has a syntax error: {review.syntax_error}")
        st.code(review.corrected_code, language=req.language)  # the code box has a copy button
        st.download_button(
            "Download corrected script",
            review.corrected_code,
            file_name=_file_name(req),
            key=f"{key}_download",
            on_click="ignore",
        )


def render_code_review(
    req: CodeReviewRequest,
    *,
    key: str,
    output_dir: str | None = None,
    llm_factory: Callable[[str], Any] | None = None,
    model: str | None = None,
) -> None:
    """Draw the review control for `req`. `llm_factory` and `model` are for tests."""
    if llm_factory is None or model is None:
        from biomentis.ui_settings import code_review_llm, code_review_model

        llm_factory = llm_factory or code_review_llm
        model = model or code_review_model()
    rkey = review_key(req, model)

    def submit() -> None:
        submit_review(req, model, llm_factory, output_dir=output_dir)

    status, review = review_status(rkey)
    if status == "none":
        if st.button("🔎 Code Review", key=f"{key}_go"):
            submit()
            st.rerun()
    elif status == "pending":

        @st.fragment(run_every=POLL_SECONDS)
        def _pending() -> None:
            if review_status(rkey)[0] != "pending":
                st.rerun()  # full rerun: draws the finished report (or the error)
            st.status(f"Reviewing… ({model})", state="running")

        _pending()
    elif status == "done" and review is not None:
        _show_report(req, review, key)
    else:
        st.error((review.error if review else "") or "The review failed.")
        if st.button("Try again", key=f"{key}_retry"):
            submit()
            st.rerun()

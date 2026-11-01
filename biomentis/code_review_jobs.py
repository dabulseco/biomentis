"""Run code reviews off the Streamlit script thread and remember the results.

A two-worker pool runs `review_code`; a lock-guarded store, keyed by
`review_key`, holds each review's state. Finished reviews are appended to a
sidecar file (`<run_dir>/<run_id>.reviews.jsonl`) so they can be reloaded after
a restart. The run's own journal (`<run_id>.jsonl`) is never written here.
Reviews without a run go to `<output_dir>/code_reviews.jsonl`, the output
folder passed to `submit_review`; with none given they go to
`<run_dir>/no-run.reviews.jsonl`, which `load_saved_reviews` reloads.
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from threading import Lock
from typing import Any, Callable

from biomentis.code_review import CodeReview, CodeReviewRequest, review_code, review_key
from biomentis.run_journal import default_run_dir

MAX_WORKERS = 2
SIDECAR_SUFFIX = ".reviews.jsonl"
OUTPUT_SIDECAR = "code_reviews.jsonl"
NO_RUN_SIDECAR = "no-run" + SIDECAR_SUFFIX

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="code-review")
_lock = Lock()  # guards _store, _futures, _seq and sidecar appends
_store: dict[str, dict] = {}  # key -> {status, review, run_id, block_index, seq}
_futures: dict[str, Future] = {}
_seq = 0


def sidecar_path(run_id: str | None, output_dir: str | None = None) -> str:
    if run_id:
        return os.path.join(default_run_dir(), f"{run_id}{SIDECAR_SUFFIX}")
    if output_dir:
        return os.path.join(output_dir, OUTPUT_SIDECAR)
    return os.path.join(default_run_dir(), NO_RUN_SIDECAR)


def _next_seq() -> int:
    global _seq
    _seq += 1
    return _seq


def _append_sidecar(key: str, req: CodeReviewRequest, review: CodeReview, output_dir: str | None) -> None:
    row = {"key": key, "run_id": req.run_id, "block_index": req.block_index, "source": req.source, **review.to_dict()}
    path = sidecar_path(req.run_id, output_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass  # the review is still in the store; saving is best-effort


def _run(key: str, req: CodeReviewRequest, model: str, llm_factory: Callable[[str], Any], output_dir: str | None) -> None:
    try:
        review = review_code(llm_factory(model), req, model)
    except Exception as e:  # llm_factory failed (no key, ...)
        review = CodeReview(error=f"Could not start the reviewer: {e}")
    review.model = review.model or model
    review.created = review.created or date.today().isoformat()
    with _lock:
        _store[key] = {
            "status": "error" if review.error else "done",
            "review": review,
            "run_id": req.run_id,
            "block_index": req.block_index,
            "seq": _next_seq(),
        }
        if not review.error:
            _append_sidecar(key, req, review, output_dir)


def submit_review(
    req: CodeReviewRequest, model: str, llm_factory: Callable[[str], Any], *, output_dir: str | None = None
) -> str:
    """Queue a review and return its key. No-op if pending or done; an error is retried.

    `output_dir` is the session's output folder, used only for reviews without a run.
    """
    key = review_key(req, model)
    with _lock:
        entry = _store.get(key)
        if entry and entry["status"] in ("pending", "done"):
            return key
        _store[key] = {
            "status": "pending",
            "review": None,
            "run_id": req.run_id,
            "block_index": req.block_index,
            "seq": _next_seq(),
        }
        try:
            _futures[key] = _executor.submit(_run, key, req, model, llm_factory, output_dir)
        except Exception as e:  # e.g. pool shut down: do not leave the key pending
            _store[key] = {
                "status": "error",
                "review": CodeReview(error=f"Could not queue the review: {e}", model=model),
                "run_id": req.run_id,
                "block_index": req.block_index,
                "seq": _next_seq(),
            }
    return key


def review_status(key: str) -> tuple[str, CodeReview | None]:
    with _lock:
        entry = _store.get(key)
        if not entry:
            return "none", None
        return entry["status"], entry["review"]


def review_for_block(run_id: str, block_index: int) -> CodeReview | None:
    """The most recent finished review of a run's code block, from any model."""
    with _lock:
        best = None
        for e in _store.values():
            if e["status"] == "done" and e["run_id"] == run_id and e["block_index"] == block_index:
                if best is None or e["seq"] > best["seq"]:
                    best = e
        return best["review"] if best else None


def load_saved_reviews(run_dir: str | None = None) -> int:
    """Fill the store from the sidecar files; returns how many reviews were loaded."""
    paths = sorted(glob.glob(os.path.join(run_dir or default_run_dir(), "*" + SIDECAR_SUFFIX)))
    count = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or not row.get("key"):
                continue
            review = CodeReview.from_dict(row)
            if review.error:
                continue
            bi = row.get("block_index")
            with _lock:
                _store[str(row["key"])] = {
                    "status": "done",
                    "review": review,
                    "run_id": row.get("run_id"),
                    "block_index": bi if isinstance(bi, int) else None,
                    "seq": _next_seq(),
                }
            count += 1
    return count


def _wait_for(key: str, timeout: float = 10.0) -> None:
    """Block until the review finishes (for tests)."""
    with _lock:
        fut = _futures.get(key)
    if fut is not None:
        fut.result(timeout=timeout)


def _clear_for_tests() -> None:
    with _lock:
        _store.clear()
        _futures.clear()

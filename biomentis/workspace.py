"""Where a conversation's output goes: one folder, by default in Downloads.

    ~/Downloads/biomentis_<stamp>/
        files/                          what the agent's code saves (its working folder)
        biomentis_main_<stamp>.html     written by "Export run to HTML"
        biomentis_inner_<stamp>.html
        figures/

The agent's code used to run in the app's own folder, so every file it saved
with a relative path (`metadata_run.csv`, a `lokoia_16s/` project tree, figure
PNGs) landed among the app's source, mixed across runs and missing from the
export. Now each code step runs inside `files/`, and the app's own paths
(`runs/`, `traces/`, `./data`) are pinned to the folder the app started in, so
they don't follow the step there.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

FILES_DIR = "files"


def launch_dir() -> str:
    """The folder the app started in; kept on `sys` so a module reload can't move it."""
    if not hasattr(sys, "_biomentis_launch_dir"):
        sys._biomentis_launch_dir = os.getcwd()
    return sys._biomentis_launch_dir


def app_path(path: str) -> str:
    """`path` made absolute against the launch folder, not the current one."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(launch_dir(), path))


def default_downloads_dir() -> str:
    """Where output folders go by default: ~/Downloads, overridable via BIOMNI_DOWNLOADS_DIR."""
    return os.getenv("BIOMNI_DOWNLOADS_DIR") or os.path.join(os.path.expanduser("~"), "Downloads")


def new_output_dir(root: str | None = None) -> str:
    """A fresh `biomentis_<stamp>` path under `root` (not created yet); `_2`, `_3`… on a clash."""
    root = root or default_downloads_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, n = os.path.join(root, f"biomentis_{stamp}"), 2
    while os.path.exists(path):
        path, n = os.path.join(root, f"biomentis_{stamp}_{n}"), n + 1
    return path


def output_stamp(output_dir: str) -> str:
    """The stamp in `biomentis_<stamp>` (with any `_2` suffix), used in the page names."""
    name = os.path.basename(os.path.normpath(output_dir))
    return name.removeprefix("biomentis_")


def files_dir(output_dir: str) -> str:
    return os.path.join(output_dir, FILES_DIR)


def prune_empty(output_dir: str) -> None:
    """Remove `files/` and then the folder itself if nothing was saved in them."""
    for path in (files_dir(output_dir), output_dir):
        try:
            os.rmdir(path)  # only succeeds when empty
        except OSError:
            return


@contextmanager
def working_in(path: str | None) -> Iterator[None]:
    """Run the body with `path` (created if needed) as the working folder; None runs it in place.

    The working folder is process-wide, so it is restored as soon as the step
    returns. Subprocesses started from `path` still import `biomentis`: the
    launch folder is put on their PYTHONPATH.
    """
    if not path:
        yield
        return
    previous = os.getcwd()
    home = launch_dir()
    entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if home not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([home, *[e for e in entries if e]])
    try:
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
    except OSError as exc:
        # A step that can't move still runs; its files land where they used to.
        print(f"[workspace] could not use {path}: {exc}")
        yield
        return
    try:
        yield
    finally:
        os.chdir(previous)


launch_dir()  # recorded at import, before any step can move the working folder

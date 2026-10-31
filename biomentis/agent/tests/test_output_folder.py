"""The conversation's output folder: where the agent's files and the export go.

Run:
    PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni \
    /opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_output_folder

The 2026-10-31 metagenomics run saved `metadata_run.csv` and a `lokoia_16s/`
project tree into the app's own folder, where they mixed with earlier runs'
files and missed the export. Covers:
  1. App paths (runs/, traces/, the data lake) are absolute, against the launch folder
  2. `working_in` moves into a folder and always back; None, or a folder that
     cannot be made, runs the step in place
  3. A run's Python and Bash steps save into <output>/files/, and Bash can still
     import biomentis from there
  4. A figure saved by relative name is found after the step, from files/
  5. A step that saves nothing leaves no empty folder behind
  6. Export writes its pages into the same folder, named by its stamp, keeps
     files/, and a second export replaces the figures instead of adding to them
  7. New folder names: stamped, `_2` on a clash; the page stamp keeps the suffix
  8. The agent is told about the folder only when there is one
"""

from __future__ import annotations

import os
import sys
import tempfile

from langchain_core.messages import AIMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))


_TMP = tempfile.mkdtemp(prefix="output_folder_")
os.environ["BIOMENTIS_ARTIFACT_DIR"] = os.path.join(_TMP, "artifacts")


def _agent(replies: list[str], output_dir: str | None):
    from biomentis.agent.a1 import A1

    seen: list = []

    class _LLM:
        def invoke(self, messages):
            seen.append(messages)
            return AIMessage(content=replies.pop(0))

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = _LLM()
    agent.configure()
    agent.output_dir = output_dir
    return agent, seen


def test_app_paths_absolute():
    print("\n[1] App paths are absolute, against the launch folder")
    from biomentis.run_journal import default_run_dir
    from biomentis.workspace import app_path, launch_dir

    _check("launch folder is where the app started", launch_dir() == os.getcwd(), launch_dir())
    _check("runs/ is absolute", default_run_dir() == os.path.join(launch_dir(), "runs"), default_run_dir())
    _check("absolute paths pass through", app_path("/x/y") == "/x/y")
    agent, _ = _agent([], None)
    _check("the data lake path is absolute", os.path.isabs(agent.path), agent.path)
    _check("the trace folder is absolute", os.path.isabs(str(agent.tracer.trace_dir)), str(agent.tracer.trace_dir))


def test_working_in():
    print("\n[2] working_in moves into a folder and always back")
    from biomentis.workspace import launch_dir, working_in

    start = os.getcwd()
    target = os.path.join(_TMP, "wi", "files")
    with working_in(target):
        inside = os.getcwd()
    _check("runs inside the folder, creating it", os.path.realpath(inside) == os.path.realpath(target), inside)
    _check("and comes back", os.getcwd() == start)
    try:
        with working_in(target):
            raise ValueError("boom")
    except ValueError:
        pass
    _check("comes back after an error", os.getcwd() == start)
    with working_in(None):
        _check("None stays put", os.getcwd() == start)
    blocker = os.path.join(_TMP, "a_file")
    open(blocker, "w").close()
    with working_in(os.path.join(blocker, "files")):
        _check("a folder that can't be made: the step runs in place", os.getcwd() == start)
    _check("subprocesses can import biomentis",
           launch_dir() in os.environ.get("PYTHONPATH", "").split(os.pathsep), os.environ.get("PYTHONPATH", ""))


def test_steps_save_into_files():
    print("\n[3] Python and Bash steps save into <output>/files/")
    output = os.path.join(_TMP, "biomentis_20261031_070000")
    start = os.getcwd()
    agent, _ = _agent(
        [
            "<execute>import os\nos.makedirs('of_test_proj', exist_ok=True)\n"
            "open('of_test_proj/params.yaml', 'w').write('a: 1')</execute>",
            "<execute>#!BASH\necho id,zone > of_test_meta.csv\n"
            "python -c \"import biomentis; print('imports ok')\" > of_test_import.txt</execute>",
            "<solution>done</solution>",
        ],
        output,
    )
    agent.go("Write files.")
    files = os.path.join(output, "files")
    _check("a Python step's file is in files/", os.path.exists(os.path.join(files, "of_test_proj", "params.yaml")),
           str(os.listdir(output) if os.path.isdir(output) else "no folder"))
    _check("a Bash step's file is in files/", os.path.exists(os.path.join(files, "of_test_meta.csv")))
    _check("nothing landed in the app folder",
           not any(os.path.exists(os.path.join(start, n)) for n in ("of_test_proj", "of_test_meta.csv", "of_test_import.txt")))
    _check("the app folder is the working folder again", os.getcwd() == start)
    check = os.path.join(files, "of_test_import.txt")
    _check("Bash imported biomentis from there", os.path.exists(check) and "imports ok" in open(check).read())


def test_relative_figure_found():
    print("\n[4] A figure saved by relative name is found from files/")
    from types import SimpleNamespace

    from biomentis.ui_core import _resolve_file_path

    output = os.path.join(_TMP, "biomentis_20261031_080000")
    os.makedirs(os.path.join(output, "files"))
    with open(os.path.join(output, "files", "fig1_budget.png"), "wb") as f:
        f.write(b"png")
    agent = SimpleNamespace(output_dir=output, path=None)
    found = _resolve_file_path(agent, "fig1_budget.png")
    _check("resolved into files/", found == os.path.join(output, "files", "fig1_budget.png"), str(found))


def test_empty_folder_pruned():
    print("\n[5] A step that saves nothing leaves no folder")
    output = os.path.join(_TMP, "biomentis_20261031_090000")
    agent, _ = _agent(["<execute>print(1 + 1)</execute>", "<solution>2</solution>"], output)
    agent.go("Add.")
    _check("no empty output folder", not os.path.exists(output), str(os.path.exists(output)))


def test_export_into_folder():
    print("\n[6] Export writes into the conversation's folder")
    from biomentis.ui_core import ExportEntry, export_run

    output = os.path.join(_TMP, "biomentis_20261031_100000")
    os.makedirs(os.path.join(output, "files"))
    with open(os.path.join(output, "files", "table.csv"), "w") as f:
        f.write("a,b\n")
    image_a, image_b = os.path.join(_TMP, "a.png"), os.path.join(_TMP, "b.png")
    for path, data in ((image_a, b"A"), (image_b, b"B")):
        with open(path, "wb") as f:
            f.write(data)

    def entries(image):
        return [ExportEntry(role="assistant", kind="image", title=None, content="", file_path=image)]

    path = export_run(entries(image_a), [], "model", output_dir=output)
    _check("returns the folder", path == output, path)
    names = sorted(os.listdir(output))
    _check("pages named by the folder's stamp",
           {"biomentis_main_20261031_100000.html", "biomentis_inner_20261031_100000.html"} <= set(names), str(names))
    _check("files/ kept", os.path.exists(os.path.join(output, "files", "table.csv")))
    export_run(entries(image_b), [], "model", output_dir=output)
    figures = os.listdir(os.path.join(output, "figures"))
    _check("a second export replaces the figures", len(figures) == 1, str(figures))
    with open(os.path.join(output, "figures", figures[0]), "rb") as f:
        _check("with the current image", f.read() == b"B")


def test_new_folder_names():
    print("\n[7] New folder names")
    from biomentis.workspace import new_output_dir, output_stamp

    root = os.path.join(_TMP, "names")
    first = new_output_dir(root)
    _check("stamped biomentis_<YYYYMMDD_HHMMSS>", len(output_stamp(first)) == 15 and output_stamp(first)[8] == "_",
           first)
    _check("not created yet", not os.path.exists(first))
    os.makedirs(first)
    second = new_output_dir(root)
    _check("a clash gets a suffix", second != first and second.startswith(first), second)
    _check("the page stamp keeps it", output_stamp(second).endswith(second[-2:]), output_stamp(second))


def test_prompt_mentions_folder():
    print("\n[8] The agent is told about the folder only when there is one")
    with_folder, seen_with = _agent(["<solution>ok</solution>"], os.path.join(_TMP, "biomentis_x"))
    with_folder.go("Hi.")
    without, seen_without = _agent(["<solution>ok</solution>"], None)
    without.go("Hi.")
    _check("with a folder: FILES guidance", "FILES:" in str(seen_with[0][0].content))
    _check("without one: none", "FILES:" not in str(seen_without[0][0].content))


def run_all() -> int:
    for test in (
        test_app_paths_absolute,
        test_working_in,
        test_steps_save_into_files,
        test_relative_figure_found,
        test_empty_folder_pruned,
        test_export_into_folder,
        test_new_folder_names,
        test_prompt_mentions_folder,
    ):
        try:
            test()
        except Exception as exc:  # a crash is a failure, not an abort
            _check(f"{test.__name__} ran", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())

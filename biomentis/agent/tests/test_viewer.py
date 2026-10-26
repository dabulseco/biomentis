"""Tests for the 3Dmol structure viewer (biomentis/ui_visuals.py).

What these tests pin down:

  1.  The viewer page loads the pinned 3Dmol build and has a fallback block
  2.  Model data can never close the inline script (no raw </script>)
  3.  Cartoon for macromolecules, sticks for everything else
  4.  render_structure runs for every fixture inside a Streamlit AppTest

Run with:
    python -m biomentis.agent.tests.test_viewer
"""

from __future__ import annotations

import os
import tempfile
import sys

from biomentis.ui_visuals import (
    THREEDMOL_URL,
    MoleculeView,
    default_style,
    viewer_html,
)

# --- Tiny test harness ----------------------------------------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


FIX = os.path.join(os.path.dirname(__file__), "fixtures")
XRAY = os.path.join(FIX, "mini_xray.pdb")
NMR = os.path.join(FIX, "mini_nmr.pdb")
SDF = os.path.join(FIX, "two_mols.sdf")
SMI = os.path.join(FIX, "aspirin.smi")


def test_viewer_html_shape():
    print("viewer_html shape")
    with open(XRAY) as f:
        html = viewer_html([MoleculeView(model_text=f.read(), fmt="pdb", style="cartoon")])
    _check("pinned 3Dmol", THREEDMOL_URL in html and "2.4.2" in html)
    _check("has fallback", "viewer-fallback" in html and "could not load" in html)
    _check("fallback text", "The 3D viewer could not load. Use the download button below." in html)
    _check("onerror handler", "onerror" in html)
    _check("try/catch", "try" in html and "catch" in html)
    cdn_tag = html.find(f'<script src="{THREEDMOL_URL}"')
    _check("showFallback defined before the CDN tag",
           0 <= html.find("function showFallback") < cdn_tag, f"{html.find('function showFallback')} vs {cdn_tag}")
    _check("watchdog armed before the CDN tag",
           0 <= html.find("setTimeout(showFallback") < cdn_tag, "")
    render_at = html.find("viewer.render();")
    _check("watchdog cleared after render",
           render_at > cdn_tag and "clearTimeout(fallbackTimer)" in html[render_at:], "")
    _check("onerror calls showFallback", 'showFallback()"></script>' in html, "")
    # Every <script> opened is closed exactly once: nothing from the data
    # introduced an extra closing tag.
    _check("script tags balanced", html.count("</script>") == html.count("<script"),
           f"{html.count('</script>')} vs {html.count('<script')}")


def test_script_injection_is_inert():
    print("script injection")
    evil = "TITLE    </script><script>alert(1)</script>\nEND\n"
    html = viewer_html([MoleculeView(evil, "pdb", "stick")])
    _check("no raw closing tag from data", "</script><script>alert" not in html)
    plain = viewer_html([MoleculeView("TITLE x\nEND\n", "pdb", "stick")])
    _check("data adds no script tags", html.count("</script>") == plain.count("</script>"))


def test_default_style():
    print("default_style")
    _check("protein → cartoon", default_style("protein") == "cartoon")
    _check("nucleic acid → cartoon", default_style("nucleic_acid") == "cartoon")
    _check("complex → cartoon", default_style("complex") == "cartoon")
    _check("small molecule → stick", default_style("small_molecule") == "stick")
    _check("unknown → stick", default_style("unknown") == "stick")


def _harness(path):
    from biomentis.ui_visuals import render_structure

    render_structure(path, key="t")
    render_structure(path, key="t2")


def test_render_structure_smoke():
    print("render_structure smoke")
    from streamlit.testing.v1 import AppTest

    for name, path in [("xray", XRAY), ("nmr", NMR), ("sdf", SDF), ("smiles", SMI),
                       ("missing", os.path.join(FIX, "nope.pdb"))]:
        try:
            at = AppTest.from_function(_harness, args=(path,), default_timeout=60).run()
            _check(f"{name}: no exception", not at.exception,
                   str([e.value for e in at.exception]))
            if name != "missing":
                _check(f"{name}: download button", len(at.get("download_button")) == 2)
            if name == "sdf":
                _check("sdf: record picker", len(at.selectbox) == 2)
        except Exception as e:  # noqa: BLE001
            _check(f"{name}: no exception", False, repr(e))


def _image_harness(path):
    from biomentis.ui_visuals import render_image_file

    render_image_file(path)


def test_missing_plot_file():
    print("missing plot file")
    import inspect

    from streamlit.testing.v1 import AppTest

    from biomentis.agent.a1 import A1
    from biomentis.ui_visuals import MISSING_IMAGE_NOTE

    at = AppTest.from_function(_image_harness, args=("/nonexistent/plot-001.png",), default_timeout=60).run()
    _check("missing png: no exception", not at.exception, str([e.value for e in at.exception]))
    _check("missing png: caption shown", [c.value for c in at.caption] == [MISSING_IMAGE_NOTE],
           str([c.value for c in at.caption]))
    png = os.path.join(tempfile.mkdtemp(), "plot-001.png")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.savefig(png)
    plt.close(fig)
    at = AppTest.from_function(_image_harness, args=(png,), default_timeout=60).run()
    _check("present png: no caption", not at.exception and len(at.caption) == 0,
           str([c.value for c in at.caption]))
    launch = inspect.getsource(A1.launch_streamlit_demo)
    body = launch[launch.find("def render_entry("):]
    branch = body[body.find('in ("image", "plot")'):]
    branch = branch[: branch.find("elif")]
    _check("render_entry uses render_image_file", "render_image_file(" in branch and "st.image(" not in branch, branch)


if __name__ == "__main__":
    test_viewer_html_shape()
    test_script_injection_is_inert()
    test_default_style()
    test_render_structure_smoke()
    test_missing_plot_file()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    sys.exit(1 if _FAILED else 0)

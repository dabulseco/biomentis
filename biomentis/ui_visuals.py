"""Structure viewer for the Streamlit app: 3Dmol.js plus a metadata panel.

`render_structure` is the one entry point the transcript, the tutor chat and
the sidebar dialog share. Coordinates reach the page only as JSON inside the
viewer's own script; the metadata panel shows header facts, never ATOM lines.

The viewer is a built-in `st.iframe` running 3Dmol.js from a pinned CDN build.
If that script cannot load, the frame shows a message and the download button still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st

from biomentis.structure_info import (
    depiction_png,
    sdf_record_count,
    structure_metadata,
    viewer_model,
)
from biomentis.structures import format_for_path, inspect_structure

THREEDMOL_URL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"

_FALLBACK_TEXT = "The 3D viewer could not load. Use the download button below."
_MACROMOLECULES = {"protein", "nucleic_acid", "complex"}


@dataclass
class MoleculeView:
    """One structure to draw. `style` is "cartoon" or "stick"."""

    model_text: str
    fmt: str
    style: str


def default_style(molecule_class: str) -> str:
    """Cartoon for proteins, nucleic acids and complexes; sticks otherwise."""
    return "cartoon" if molecule_class in _MACROMOLECULES else "stick"


def _js_json(value) -> str:
    """JSON that is safe inside an inline <script>.

    `json.dumps` leaves `</script>` and `<!--` intact, either of which would
    end or confuse the script block, so the slash and bang are escaped.
    """
    return json.dumps(value).replace("</", "<\\/").replace("<!--", "<\\!--")


_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #viewer {{ position: relative; width: 100%; height: {height}px;
             border: 1px solid #8884; border-radius: 6px; }}
  .viewer-fallback {{ display: none; font: 14px sans-serif; padding: 12px;
                      color: #666; }}
</style></head>
<body>
<div id="viewer"></div>
<div class="viewer-fallback">{fallback}</div>
<script src="{url}" onerror="showFallback()"></script>
<script>
function showFallback() {{
  document.getElementById("viewer").style.display = "none";
  document.querySelector(".viewer-fallback").style.display = "block";
}}
(function () {{
  var views = {views};
  try {{
    var viewer = $3Dmol.createViewer(document.getElementById("viewer"),
                                     {{backgroundColor: "white"}});
    views.forEach(function (v) {{
      var model = viewer.addModel(v.model, v.fmt);
      if (v.style === "cartoon") {{
        model.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
        model.setStyle({{hetflag: true}}, {{stick: {{}}}});
      }} else {{
        model.setStyle({{}}, {{stick: {{}}}});
      }}
    }});
    viewer.zoomTo();
    viewer.render();
  }} catch (e) {{
    showFallback();
  }}
}})();
</script>
</body></html>
"""


def viewer_html(views: list[MoleculeView], height: int = 420) -> str:
    """A self-contained page that draws `views` with 3Dmol (rotate and zoom on)."""
    payload = [{"model": v.model_text, "fmt": v.fmt, "style": v.style} for v in views]
    return _PAGE.format(
        height=int(height),
        fallback=_FALLBACK_TEXT,
        url=THREEDMOL_URL,
        views=_js_json(payload),
    )


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _metadata_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| | |", "|---|---|"]
    lines += [f"| **{_cell(k)}** | {_cell(v)} |" for k, v in rows]
    return "\n".join(lines)


def render_structure(path: str, *, key: str, container=None) -> None:
    """Draw `path` (PDB, SDF or SMILES): viewer, metadata, depiction, download.

    `key` must be unique per call site on a page; every widget key derives
    from it. `container` is where to draw (default: the page).
    """
    box = container if container is not None else st
    fmt = format_for_path(path)

    record = 0
    if fmt == "sdf":
        count = sdf_record_count(path)
        if count > 1:
            record = box.selectbox(
                "Record",
                range(count),
                format_func=lambda i: f"Record {i + 1} of {count}",
                key=f"rec-{key}",
            )

    left, right = box.columns(2)

    model = viewer_model(path, record)
    with left:
        if model is None:
            st.info("This structure could not be drawn.")
        else:
            text, model_fmt = model
            try:
                style = default_style(inspect_structure(path).molecule_class)
            except Exception:  # noqa: BLE001
                style = "stick"
            # st.iframe replaces the deprecated st.components.v1.html.
            st.iframe(viewer_html([MoleculeView(text, model_fmt, style)]), height=440)

    with right:
        st.markdown(_metadata_table(structure_metadata(path, record)))
        if fmt in ("sdf", "smi"):
            png = depiction_png(path, record)
            if png:
                st.image(png)

    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    box.download_button("Download structure file", data, file_name=name, key=f"dl-{key}")

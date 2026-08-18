"""UI-agnostic core for streaming an A1 agent's turn and parsing its response.

This module extracts the streaming/tag-parsing logic that used to live inline
inside the Gradio demo (`A1.launch_gradio_demo`) so it can be shared by any UI
adapter (Gradio, Streamlit, ...). It has no dependency on any UI framework and
is safe to import without either installed.

An adapter drives a turn by calling `stream_agent_events`, which yields
`UIEvent`s describing what happened (reasoning text, code execution, tool
observations, generated files, the final solution) without knowing anything
about how those events should be rendered.
"""

import base64
import html
import mimetypes
import os
import re
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage

SUPPORTED_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")

UIEventType = Literal[
    "status",
    "reasoning",
    "solution",
    "code",
    "observation",
    "file",
    "summary",
    "complete",
    # --- Tutor-layer events (Phase 1+) ---
    # "instruction": a teaching card attached to the previous event.
    # "roadmap": a one-time, run-level preview of all the steps the agent
    #   intends to take, shown before the first per-step instruction card.
    # "paused": a "Continue" gate; the stream halts until the user clicks.
    # "qa": a tutor-chat exchange (user question + assistant answer).
    # "rubric_update": a teacher rubric was loaded or replaced.
    "instruction",
    "roadmap",
    "paused",
    "qa",
    "rubric_update",
]


@dataclass
class UIEvent:
    """A single piece of an agent turn, framework-agnostic.

    channel distinguishes the "final answer" stream (main) from the
    "step-by-step execution log" stream (inner) that Biomentis's demos show
    side by side.
    """

    type: UIEventType
    content: str = ""
    channel: Literal["main", "inner"] = "inner"
    title: str | None = None
    language: str | None = None
    status: str | None = None
    duration: float | None = None
    file_path: str | None = None
    file_kind: Literal["image", "pdf"] | None = None
    collapsible: bool = False


def _resolve_file_path(agent: Any, file_path: str) -> str | None:
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    cwd_path = os.path.join(os.getcwd(), file_path)
    if os.path.exists(cwd_path):
        return cwd_path
    agent_path = getattr(agent, "path", None)
    if agent_path:
        candidate = os.path.join(agent_path, file_path)
        if os.path.exists(candidate):
            return candidate
    return None


def stream_agent_events(
    agent: Any,
    text_input: str,
    files: list[str],
    history_messages: list[BaseMessage],
    thread_id: int,
) -> Generator[UIEvent, None, None]:
    """Run one turn of the agent and yield UIEvents as its response streams in.

    Args:
        agent: An A1 instance (needs .app, .use_tool_retriever,
            ._prepare_resources_for_retrieval, .update_system_prompt_with_selected_resources, .path)
        text_input: The user's message text for this turn
        files: Paths to any files uploaded with this turn
        history_messages: Prior conversation turns as LangChain messages (not including this turn)
        thread_id: LangGraph checkpoint thread id
    """
    full_text_input = text_input
    for file_path in files:
        full_text_input += f"\n\n User uploaded this file: {file_path}\n Please use it if needed."

    agent_messages = [*history_messages, HumanMessage(content=full_text_input)]
    inputs = {"messages": agent_messages, "next_step": None}
    config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

    # `go`/`go_stream` set this themselves before streaming; this function
    # drives `agent.app.stream` directly instead of going through either,
    # so it must set it too — `execute_self_critic` reads it back.
    agent.user_task = text_input

    t = time()
    solution_found = False

    if agent.use_tool_retriever:
        yield UIEvent(type="status", content="Retrieving relevant tools, data lake items, and libraries...")
        try:
            selected_resources_names = agent._prepare_resources_for_retrieval(text_input)
            if selected_resources_names:
                agent.update_system_prompt_with_selected_resources(selected_resources_names)
        except Exception as e:
            print(f"Warning: Tool retrieval failed: {e}")
            print("Continuing without tool retrieval...")
            yield UIEvent(type="status", content="Tool retrieval unavailable, proceeding with all tools...")

    s = None
    for s in agent.app.stream(inputs, stream_mode="values", config=config):
        t_step = time() - t
        message = s["messages"][-1]

        if message.content == full_text_input:
            t = time()
            continue

        if not isinstance(message.content, str):
            continue
        content = message.content

        tag_positions = [pos for pos in (content.find(tag) for tag in ("<execute>", "<solution>", "<observation>")) if pos != -1]
        if tag_positions:
            thinking = content[: min(tag_positions)].strip()
            if thinking:
                yield UIEvent(type="reasoning", content=thinking, title="🤔 Reasoning")

        solution_match = re.search(r"<solution>(.*?)</solution>", content, re.DOTALL)
        if solution_match and not solution_found:
            solution = solution_match.group(1).strip()
            yield UIEvent(type="solution", channel="main", content=solution, title="✅ Answer")
            solution_found = True

        execute_match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL)
        if execute_match:
            code = execute_match.group(1).strip()
            language = "python"
            if code.startswith("#!R"):
                language = "r"
                code = re.sub(r"^#!R", "", code, count=1).strip()
            elif code.startswith("#!BASH") or code.startswith("#!CLI"):
                language = "bash"
                code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()
            yield UIEvent(
                type="code",
                content=code,
                language=language,
                status="pending",
                title="🛠️ Executing code...",
            )

        observation_match = re.search(r"<observation>(.*?)</observation>", content, re.DOTALL)
        if observation_match:
            observation = observation_match.group(1).strip()
            yield UIEvent(
                type="observation",
                content=observation,
                duration=t_step,
                collapsible=True,
            )

            if any(ext in observation for ext in SUPPORTED_FILE_EXTENSIONS):
                matches = re.findall(r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))", observation)
                valid_matches = [
                    m
                    for m in matches
                    if not (m.startswith("Warning:") or m.startswith("Error:") or m.startswith("'"))
                    and not m.startswith(".")
                ]

                if valid_matches:
                    yield UIEvent(type="file", title="📁 Files")
                    for raw_path in valid_matches:
                        file_path = raw_path.strip("\"'").strip()
                        abs_path = _resolve_file_path(agent, file_path)
                        if not abs_path:
                            continue
                        if file_path.lower().endswith(".pdf"):
                            yield UIEvent(
                                type="file",
                                title="📄 PDF File",
                                content=f"Found PDF at: {abs_path}",
                                file_path=abs_path,
                                file_kind="pdf",
                            )
                        else:
                            yield UIEvent(
                                type="file",
                                title="🖼️ Image Preview",
                                file_path=abs_path,
                                file_kind="image",
                            )

        t = time()

    final_content = s["messages"][-1].content if s and s.get("messages") else ""
    if not solution_found:
        solution_match = re.search(r"<solution>(.*?)</solution>", final_content, re.DOTALL)
        if solution_match:
            yield UIEvent(
                type="summary", channel="main", content=solution_match.group(1).strip(), title="✅ Solution"
            )
        else:
            cleaned = re.sub(r"<execute>.*?</execute>", "", final_content, flags=re.DOTALL)
            cleaned = re.sub(r"<observation>.*?</observation>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()
            if cleaned:
                yield UIEvent(type="summary", channel="main", content=cleaned, title="📝 Summary")
            else:
                yield UIEvent(
                    type="summary",
                    channel="main",
                    content="Task completed. Please check the execution log for details.",
                    title="📝 Summary",
                )

    yield UIEvent(type="complete", content="👈 Returning the result to the main interface...", title="🔄 Complete")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string.

    Sub-minute values use one decimal: ``"0.7s"``, ``"12.3s"``. Anything
    60s or longer uses ``"Xm Y.Ys"`` so the long-tail stays scannable.

    Examples
    --------
    >>> format_duration(0.7)
    '0.7s'
    >>> format_duration(83.4)
    '1m 23.4s'
    """
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {secs:0.1f}s"
    return f"{secs:0.1f}s"


_CLOUD_PROVIDER_CATALOG: dict[str, tuple[list[str], list[str]]] = {
    "Anthropic": (["ANTHROPIC_API_KEY"], ["claude-sonnet-4-5", "claude-opus-4-1", "claude-3-5-haiku-20241022"]),
    "OpenAI": (["OPENAI_API_KEY"], ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]),
    "Gemini": (["GEMINI_API_KEY"], ["gemini-2.5-pro", "gemini-2.5-flash"]),
    "Groq": (["GROQ_API_KEY"], ["llama-3.3-70b-versatile"]),
    "Bedrock": (["AWS_REGION"], ["anthropic.claude-3-5-sonnet-20241022-v2:0"]),
}


def list_available_providers() -> dict[str, list[str]]:
    """List providers/models available for the UI's model selector.

    Ollama models are discovered live from the local daemon (local and
    Ollama-Cloud-signed-in models alike). Cloud providers are only listed when
    their API key env var is present, and use a short illustrative catalog of
    model ids rather than a live lookup.
    """
    from biomentis.ollama_utils import list_ollama_models

    providers: dict[str, list[str]] = {}

    ollama_models = list_ollama_models()
    if ollama_models:
        providers["Ollama"] = ollama_models

    for source, (env_vars, models) in _CLOUD_PROVIDER_CATALOG.items():
        if any(os.getenv(var) for var in env_vars):
            providers[source] = models

    return providers


# --- HTML export -------------------------------------------------------

ExportEntryKind = Literal["text", "code", "observation", "image", "pdf", "instruction", "qa"]


@dataclass
class ExportEntry:
    """A single normalized, framework-agnostic message for HTML export."""

    role: Literal["user", "assistant"]
    kind: ExportEntryKind = "text"
    title: str | None = None
    content: str = ""
    language: str | None = None
    file_path: str | None = None
    # Tutor step id (set on `instruction` and `paused` events, and now on
    # the raw event + its per-step Q&A too — see a1.py's `export_entries`).
    # Used by the Streamlit renderer to give expander labels and button
    # keys a stable per-step suffix so re-renders don't collide, and by
    # `render_transcript_html` to box a step's entries together.
    step_id: int | None = None
    # Which run this step belongs to. step_id resets to 1 on every new
    # task, so on its own it can't tell two different runs' "step 1"
    # apart — run_id can. See `_StepBoxTracker` in a1.py for the live-UI
    # analog of the grouping this drives in the HTML export.
    run_id: str | None = None
    # Tutor Q&A metadata (set on `qa` entries). When present, the HTML
    # renderer appends a "Sources" list (if citations) and a metadata
    # line (Bloom / DOK / rubric) under the body so the export carries
    # the same context the live chat panel shows.
    citations: list[dict] | None = None
    bloom_level: str | None = None
    dok_level: str | None = None
    rubric_hit: list[str] | None = None
    confidence: float | None = None
    # Gap-analysis items ({"point": str, "why": str}) the tutor flagged
    # beyond the literal question. See `also_consider` on `ChatTurn`.
    also_consider: list[dict] | None = None


def ui_event_to_export_entry(event: UIEvent) -> ExportEntry:
    """Normalize a UIEvent (assistant-side) into an ExportEntry."""
    # Tutor step id (if the wrapper tagged the event).
    step_id = getattr(event, "step_id", None)
    if event.type == "file":
        if event.file_path is None:
            return ExportEntry(role="assistant", kind="text", title=event.title, content=event.content, step_id=step_id)
        return ExportEntry(
            role="assistant",
            kind=event.file_kind or "text",
            title=event.title,
            content=event.content,
            file_path=event.file_path,
            step_id=step_id,
        )
    if event.type == "instruction":
        return ExportEntry(
            role="assistant",
            kind="instruction",
            title=event.title or "🎓 Teaching note",
            content=event.content,
            step_id=step_id,
        )
    if event.type == "qa":
        return ExportEntry(
            role="assistant",
            kind="qa",
            title=event.title or "💬 Tutor Q&A",
            content=event.content,
            step_id=step_id,
        )
    if event.type == "paused":
        # Pause gates aren't content; they show up as a Continue button in
        # the live UI. Skip them in the HTML export so the page is readable
        # offline as a transcript.
        return ExportEntry(
            role="assistant",
            kind="text",
            title="⏸ Paused",
            content="(agent paused for instructional card)",
            step_id=step_id,
        )
    if event.type == "rubric_update":
        return ExportEntry(
            role="assistant",
            kind="text",
            title=event.title or "📋 Rubric updated",
            content=event.content,
            step_id=step_id,
        )
    kind: ExportEntryKind = "code" if event.type == "code" else "observation" if event.type == "observation" else "text"
    return ExportEntry(role="assistant", kind=kind, title=event.title, content=event.content, language=event.language, step_id=step_id)


_MODEL_NAME_ATTRS = ("model", "model_name", "model_id", "azure_deployment")


def get_llm_display_name(llm: Any) -> str:
    """Best-effort human-readable "provider: model" string for a LangChain chat model instance."""
    provider = type(llm).__name__.removeprefix("Chat")
    for attr in _MODEL_NAME_ATTRS:
        value = getattr(llm, attr, None)
        if value:
            return f"{provider}: {value}"
    return provider


def default_downloads_dir() -> str:
    """The directory HTML exports are saved to by default: ~/Downloads, overridable via BIOMNI_DOWNLOADS_DIR."""
    return os.getenv("BIOMNI_DOWNLOADS_DIR") or os.path.join(os.path.expanduser("~"), "Downloads")


def _embed_image_html(file_path: str) -> str:
    try:
        mime = mimetypes.guess_type(file_path)[0] or "image/png"
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" class="img-fluid rounded" alt="{html.escape(os.path.basename(file_path))}">'
    except Exception:
        return f'<div class="text-danger">Image not found: {html.escape(file_path)}</div>'


def _render_markdown(text: str) -> str:
    """Render markdown prose (tables, bold, lists, ...) to Bootstrap-styled HTML.

    Falls back to plain escaped/pre-wrapped text if the optional `markdown`
    package isn't installed, matching this codebase's existing lazy-import
    pattern for optional deps (see biomni/utils.py's convert_markdown_to_pdf).
    """
    try:
        import markdown as markdown_lib
    except ImportError:
        return f'<div style="white-space: pre-wrap;">{html.escape(text)}</div>'

    rendered = markdown_lib.markdown(text, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])
    # markdown's `tables` extension emits bare <table> tags; make them Bootstrap 5
    # tables, wrapped for horizontal scrolling on narrow viewports.
    rendered = rendered.replace(
        "<table>", '<div class="table-responsive"><table class="table table-striped table-bordered table-sm align-middle">'
    )
    rendered = rendered.replace("</table>", "</table></div>")
    return rendered


def _qa_extras_html(entry: ExportEntry) -> str:
    """Render the qa-only tail: Sources list, gap-analysis list, and a
    metadata line.

    All three pieces are optional. A Sources block appears only when
    `entry.citations` is a non-empty list; an "Also worth knowing" block
    appears only when `entry.also_consider` is non-empty. The metadata
    line appears when at least one of `bloom_level` / `dok_level` /
    `rubric_hit` / `confidence` is populated — anything else is dropped
    to keep the
    line short.
    """
    parts: list[str] = []
    citations = entry.citations or []
    if citations:
        items = []
        for c in citations:
            src = c.get("source", "?") if isinstance(c, dict) else "?"
            page = c.get("page") if isinstance(c, dict) else None
            label = f"<strong>{html.escape(str(src))}</strong>"
            if page is not None:
                label += f" <span class='text-muted'>(p. {html.escape(str(page))})</span>"
            items.append(f"<li>{label}</li>")
        parts.append(
            f"<details class='mt-2'><summary class='text-muted small'>"
            f"Sources ({len(citations)})</summary>"
            f"<ul class='small mb-0'>{''.join(items)}</ul></details>"
        )

    also_consider = entry.also_consider or []
    if also_consider:
        items = []
        for item in also_consider:
            point = item.get("point") if isinstance(item, dict) else str(item)
            why = item.get("why") if isinstance(item, dict) else None
            label = f"<strong>{html.escape(str(point))}</strong>"
            if why:
                label += f" <span class='text-muted'>— {html.escape(str(why))}</span>"
            items.append(f"<li>{label}</li>")
        parts.append(
            f"<details class='mt-2'><summary class='text-muted small'>"
            f"Also worth knowing ({len(also_consider)})</summary>"
            f"<ul class='small mb-0'>{''.join(items)}</ul></details>"
        )

    meta_bits: list[str] = []
    if entry.bloom_level:
        meta_bits.append(f"Bloom: {html.escape(entry.bloom_level)}")
    if entry.dok_level:
        meta_bits.append(f"DOK: {html.escape(str(entry.dok_level))}")
    if entry.rubric_hit:
        joined = ", ".join(html.escape(s) for s in entry.rubric_hit)
        meta_bits.append(f"Rubric: {joined}")
    if entry.confidence is not None:
        meta_bits.append(f"Confidence: {entry.confidence:.2f}")
    if meta_bits:
        parts.append(
            f"<div class='text-muted small mt-1'>{' &middot; '.join(meta_bits)}</div>"
        )

    return "".join(parts)


def _entry_body_html(entry: ExportEntry) -> str:
    if entry.kind == "code":
        language = entry.language or ""
        return f'<pre class="bg-dark text-light p-3 rounded"><code class="language-{html.escape(language)}">{html.escape(entry.content)}</code></pre>'
    if entry.kind == "observation":
        return f'<pre class="bg-secondary-subtle p-3 rounded small text-wrap">{html.escape(entry.content)}</pre>'
    if entry.kind == "image" and entry.file_path:
        return _embed_image_html(entry.file_path)
    if entry.kind == "pdf":
        return _render_markdown(entry.content) if entry.content else ""
    if entry.kind in ("instruction", "qa"):
        # Tutor-layer exports: a left "Tutor" badge, then markdown body,
        # then (for qa) a Sources list and a metadata line if the entry
        # carries them. The chat-panel Q&A turn shape adds these so the
        # export captures the same context the live panel shows.
        badge = "Tutor" if entry.kind == "qa" else "Teaching"
        badge_class = "bg-info-subtle text-info-emphasis" if entry.kind == "qa" else "bg-warning-subtle text-warning-emphasis"
        body = _render_markdown(entry.content) if entry.content else ""
        tail = _qa_extras_html(entry) if entry.kind == "qa" else ""
        return f'<div class="d-flex align-items-start gap-2"><span class="badge {badge_class} flex-shrink-0">{badge}</span><div class="flex-grow-1">{body}{tail}</div></div>'
    if entry.content:
        return _render_markdown(entry.content)
    return ""


def _render_entry_card(entry: ExportEntry) -> str:
    """Render one entry as a single chat-bubble card."""
    is_user = entry.role == "user"
    bubble_class = "bg-primary text-white" if is_user else "bg-white"
    align_class = "ms-auto" if is_user else "me-auto"
    title_html = f'<div class="fw-semibold small mb-1">{html.escape(entry.title)}</div>' if entry.title else ""
    body = _entry_body_html(entry)
    return f"""
      <div class="d-flex mb-3 {align_class}" style="max-width: 85%;">
        <div class="card {bubble_class} shadow-sm w-100">
          <div class="card-body">
            <div class="text-uppercase text-muted small mb-1">{html.escape(entry.role)}</div>
            {title_html}
            {body}
          </div>
        </div>
      </div>"""


def render_transcript_html(entries: list[ExportEntry], panel_title: str, model_name: str) -> str:
    """Render a self-contained Bootstrap 5 / HTML5 page for a chat transcript.

    Images are embedded as base64 data URIs so the exported file is portable;
    Bootstrap's CSS is pulled from its CDN, so opening the file needs internet access.

    Entries that share a `(run_id, step_id)` — a tutor step's raw output,
    its teaching card, and its "Ask about this step" Q&A — are boxed
    together in one bordered block, mirroring the live app's per-step
    grouping (see `_StepBoxTracker` in a1.py). Entries without a step_id
    (status lines, the once-per-run roadmap, plain research-mode output)
    render as standalone cards and don't interrupt an open box — the
    roadmap in particular sits between a step's raw event and its
    instruction card in the live stream, so treating it as a group-ender
    would needlessly split that step's box in two.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not entries:
        body_html = '<p class="text-muted fst-italic">No messages yet.</p>'
    else:
        blocks: list[str] = []
        box_entries: list[ExportEntry] | None = None
        box_key: tuple | None = None

        def flush_box() -> None:
            nonlocal box_entries, box_key
            if box_entries:
                inner = "".join(_render_entry_card(e) for e in box_entries)
                blocks.append(f"""
      <div class="border rounded-3 p-3 mb-3">
        <div class="text-muted small fw-semibold mb-2">📍 Step {html.escape(str(box_key[1]))}</div>
        {inner}
      </div>""")
            box_entries = None
            box_key = None

        for entry in entries:
            if entry.step_id is not None:
                key = (entry.run_id, entry.step_id)
                if key != box_key:
                    flush_box()
                    box_entries = []
                    box_key = key
                box_entries.append(entry)
            else:
                blocks.append(_render_entry_card(entry))
        flush_box()
        body_html = "".join(blocks)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(panel_title)}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-body-tertiary">
<div class="container py-4">
  <h1 class="mb-1">{html.escape(panel_title)}</h1>
  <p class="text-muted">
    Generated {html.escape(generated_at)} &middot; LLM used for this analysis: <strong>{html.escape(model_name)}</strong>
  </p>
  <hr>
  <div class="d-flex flex-column">
    {body_html}
  </div>
</div>
</body>
</html>
"""


def save_html_export(html_doc: str, filename_prefix: str, downloads_dir: str | None = None) -> str:
    """Write an exported HTML transcript to disk, returning the absolute path written."""
    target_dir = downloads_dir or default_downloads_dir()
    os.makedirs(target_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"biomni_{filename_prefix}_{timestamp}.html"
    path = os.path.join(target_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return path

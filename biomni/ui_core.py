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

import os
import re
from collections.abc import Generator
from dataclasses import dataclass
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
]


@dataclass
class UIEvent:
    """A single piece of an agent turn, framework-agnostic.

    channel distinguishes the "final answer" stream (main) from the
    "step-by-step execution log" stream (inner) that Biomni's demos show
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
    from biomni.ollama_utils import list_ollama_models

    providers: dict[str, list[str]] = {}

    ollama_models = list_ollama_models()
    if ollama_models:
        providers["Ollama"] = ollama_models

    for source, (env_vars, models) in _CLOUD_PROVIDER_CATALOG.items():
        if any(os.getenv(var) for var in env_vars):
            providers[source] = models

    return providers

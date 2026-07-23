import glob
import inspect
import os
import re
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# --- Tutor-layer renderer hooks ------------------------------------------
# The optional `biomentis.ui_tutor` module installs a rich renderer for tutor
# instruction cards and a Continue-button gate. We define them as None at
# import time so `launch_streamlit_demo` can call them conditionally without
# a circular import. The module-level globals are set by
# `biomentis.ui_tutor.install_renderers()` when the tutor sidebar is rendered.
_RENDER_INSTRUCTION_CARD: Callable[..., None] | None = None
_RENDER_PAUSE_GATE: Callable[..., None] | None = None

from biomentis.config import default_config
from biomentis.know_how import KnowHowLoader
from biomentis.llm import SourceType, get_llm
from biomentis.model.retriever import ToolRetriever
from biomentis.tool.support_tools import run_python_repl
from biomentis.tool.tool_registry import ToolRegistry
from biomentis.ui_core import (
    ExportEntry,
    format_duration,
    get_llm_display_name,
    list_available_providers,
    render_transcript_html,
    save_html_export,
    stream_agent_events,
    ui_event_to_export_entry,
)
from biomentis.utils import (
    check_and_download_s3_files,
    clean_message_content,
    convert_markdown_to_pdf,
    create_parsing_error_html,
    find_matching_execution,
    format_execute_tags_in_content,
    format_lists_in_text,
    format_observation_as_terminal,
    function_to_api_schema,
    has_execution_results,
    inject_custom_functions_to_repl,
    parse_tool_calls_from_code,
    parse_tool_calls_with_modules,
    pretty_print,
    read_module2api,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    should_skip_message,
    textify_api_dict,
)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None


class A1:
    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: SourceType | None = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        commercial_mode: bool | None = None,
        expected_data_lake_files: list | None = None,
    ):
        """Initialize the biomni agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM
            commercial_mode: If True, excludes datasets that require commercial licenses or are non-commercial only

        """
        # Use default_config values for unspecified parameters
        if path is None:
            path = default_config.path
        if llm is None:
            llm = default_config.llm
        if source is None:
            source = default_config.source
        if use_tool_retriever is None:
            use_tool_retriever = default_config.use_tool_retriever
        if timeout_seconds is None:
            timeout_seconds = default_config.timeout_seconds
        if base_url is None:
            base_url = default_config.base_url
        if api_key is None:
            api_key = default_config.api_key if default_config.api_key else "EMPTY"
        if commercial_mode is None:
            commercial_mode = default_config.commercial_mode

        # Import appropriate env_desc based on commercial_mode
        if commercial_mode:
            from biomentis.env_desc_cm import data_lake_dict, library_content_dict

            print("🏢 Commercial mode: Using commercial-licensed datasets only")
        else:
            from biomentis.env_desc import data_lake_dict, library_content_dict

            print("🎓 Academic mode: Using all datasets (including non-commercial)")

        # Store as instance attributes for later use
        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict
        self.commercial_mode = commercial_mode

        # Display configuration in a nice, readable format
        print("\n" + "=" * 50)
        print("🔧 BIOMNI CONFIGURATION")
        print("=" * 50)

        # Get the actual LLM values that will be used by the agent
        agent_llm = llm if llm is not None else default_config.llm
        agent_source = source if source is not None else default_config.source

        # Show default config (database LLM)
        print("📋 DEFAULT CONFIG (Including Database LLM):")
        config_dict = default_config.to_dict()
        for key, value in config_dict.items():
            if value is not None:
                # Special formatting for commercial_mode
                if key == "commercial_mode":
                    mode_text = "Commercial (licensed datasets only)" if value else "Academic (all datasets)"
                    print(f"  {key.replace('_', ' ').title()}: {mode_text}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Show agent-specific LLM if different from default
        if agent_llm != default_config.llm or agent_source != default_config.source:
            print("\n🤖 AGENT LLM (Constructor Override):")
            print(f"  LLM Model: {agent_llm}")
            if agent_source is not None:
                print(f"  Source: {agent_source}")
            if base_url is not None:
                print(f"  Base URL: {base_url}")
            if api_key is not None and api_key != "EMPTY":
                print(f"  API Key: {'*' * 8 + api_key[-4:] if len(api_key) > 8 else '***'}")

        print("=" * 50 + "\n")

        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "biomni_data", "benchmark")
        data_lake_dir = os.path.join(path, "biomni_data", "data_lake")

        # Create the biomni_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        if expected_data_lake_files is None:
            expected_data_lake_files = list(self.data_lake_dict.keys())

            # Check and download missing data lake files
            print("Checking and downloading missing data lake files...")
            check_and_download_s3_files(
                s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                local_data_lake_path=data_lake_dir,
                expected_files=expected_data_lake_files,
                folder="data_lake",
            )

            # Check if benchmark directory structure is complete
            benchmark_ok = False
            if os.path.isdir(benchmark_dir):
                patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
                if os.path.isdir(patient_gene_detection_dir):
                    benchmark_ok = True

            if not benchmark_ok:
                print("Checking and downloading benchmark files...")
                check_and_download_s3_files(
                    s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                    local_data_lake_path=benchmark_dir,
                    expected_files=[],  # Empty list - will download entire folder
                    folder="benchmark",
                )
        else:
            print("Skipping datalake download (load_datalake=False)")
            print("Note: Some tools may require datalake files to function properly.")

        self.path = os.path.join(path, "biomni_data")
        module2api = read_module2api()

        self.llm = get_llm(
            llm,
            stop_sequences=["</execute>", "</solution>"],
            source=source,
            base_url=base_url,
            api_key=api_key,
            config=default_config,
        )
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Initialize know-how loader
        self.know_how_loader = KnowHowLoader()

        # Filter know-how documents based on commercial mode
        if commercial_mode:
            self._filter_know_how_for_commercial_mode()

        print(f"📚 Loaded {len(self.know_how_loader.documents)} know-how documents")

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        self.configure()

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
        try:
            # Get function information
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Generate API schema using the existing utility function
            schema = function_to_api_schema(function_code, self.llm)

            # Ensure the schema has all required fields for the tool registry
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")

            # Validate and enhance the schema

            # Set default values if missing
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                # Try to extract from parameters if available
                if "parameters" in schema and isinstance(schema["parameters"], dict):
                    required_params = []
                    params = schema["parameters"]
                    if "properties" in params:
                        for param_name in params["properties"]:
                            if param_name in params.get("required", []):
                                required_params.append(param_name)
                    schema["required_parameters"] = required_params
                else:
                    schema["required_parameters"] = []

            # Add module information to the schema
            schema["module"] = module_name

            # Add the tool to the tool registry if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")
                    # Continue with adding to module2api even if registry fails

            # Add the tool to module2api structure for system prompt generation
            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            # Check if tool already exists in module2api to avoid duplicates
            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                # Update existing tool
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                # Add new tool
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Update the tool registry's document dataframe if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    # Rebuild the document dataframe
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

            # Store the original function for potential future use
            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            # Also store in _custom_tools for highlighting
            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            # Make the function available in the global namespace for execution
            import builtins

            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(
                f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval"
            )
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback

            traceback.print_exc()
            raise

    def add_mcp(self, config_path: str | Path = "./tutorials/examples/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the biomni agent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.

        Args:
            config_path: Path to the MCP configuration YAML file containing server
                        definitions and tool specifications.

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is malformed
            RuntimeError: If MCP server initialization fails
        """
        import asyncio
        import os
        import sys
        import types
        from pathlib import Path

        import nest_asyncio
        import yaml
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        nest_asyncio.apply()

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from MCP server synchronously."""
            try:

                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()

                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools

                return asyncio.run(_discover_async())
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for MCP tool execution."""
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                    async def async_tool_call():
                        async with stdio_client(server_params) as (reader, writer):
                            async with ClientSession(reader, writer) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    try:
                        loop = asyncio.get_running_loop()
                        return loop.create_task(async_tool_call())
                    except RuntimeError:
                        return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        # Initialize registries if they don't exist
        self._custom_functions = getattr(self, "_custom_functions", {})
        self._custom_tools = getattr(self, "_custom_tools", {})

        # Load and validate configuration
        try:
            config_content = Path(config_path).read_text(encoding="utf-8")
            cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e

        mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in configuration")
            return

        # Process each MCP server configuration
        for server_name, server_meta in mcp_servers.items():
            if not server_meta.get("enabled", True):
                continue

            # Validate command configuration
            cmd_list = server_meta.get("command", [])
            if not cmd_list or not isinstance(cmd_list, list):
                print(f"Warning: Invalid command configuration for server '{server_name}'")
                continue

            cmd, *args = cmd_list

            # Process environment variables
            env_vars = server_meta.get("env", {})
            if env_vars:
                processed_env = {}
                for key, value in env_vars.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        var_name = value[2:-1]
                        processed_env[key] = os.getenv(var_name, "")
                    else:
                        processed_env[key] = value
                env_vars = processed_env

            # Create module namespace for this MCP server
            mcp_module_name = f"mcp_servers.{server_name}"
            if mcp_module_name not in sys.modules:
                sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
            server_module = sys.modules[mcp_module_name]

            tools_config = server_meta.get("tools", [])

            if not tools_config:
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                    tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
                    else:
                        print(f"Warning: No tools discovered from {server_name} MCP server")
                        continue

                except Exception as e:
                    print(f"Failed to discover tools for {server_name}: {e}")
                    continue

            # Register each tool
            for tool_meta in tools_config:
                if isinstance(tool_meta, dict) and "biomentis_name" in tool_meta:
                    # Manual tool definition
                    tool_name = tool_meta.get("biomentis_name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("parameters", {})
                    # For manual tools, check if each parameter has a "required" field
                    required_param_names = []
                    for param_name, param_spec in parameters.items():
                        if param_spec.get("required", False):
                            required_param_names.append(param_name)
                else:
                    # Auto-discovered tool
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    input_schema = tool_meta.get("inputSchema", {})
                    parameters = input_schema.get("properties", {})
                    # For auto-discovered tools, get required list from inputSchema top level
                    required_param_names = input_schema.get("required", [])

                if not tool_name:
                    print(f"Warning: Skipping tool with no name in {server_name}")
                    continue

                # Create wrapper function
                wrapper_function = make_mcp_wrapper(cmd, args, tool_name, description, env_vars)

                # Add to module namespace
                setattr(server_module, tool_name, wrapper_function)

                # Build parameter lists
                required_params, optional_params = [], []
                for param_name, param_spec in parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param_spec.get("type", "string")),
                        "description": param_spec.get("description", ""),
                        "default": param_spec.get("default", None),
                    }

                    # Check if parameter is required based on the required_param_names list
                    if param_name in required_param_names:
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)

                # Create tool schema
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                    "required_parameters": required_params,
                    "optional_parameters": optional_params,
                    "module": mcp_module_name,
                    "fn": wrapper_function,
                }

                # Register in tool registry
                self.tool_registry.register_tool(tool_schema)

                # Add to module2api mapping
                if mcp_module_name not in self.module2api:
                    self.module2api[mcp_module_name] = []
                self.module2api[mcp_module_name].append(tool_schema)

                # Add to instance registries
                self._custom_functions[tool_name] = wrapper_function
                self._custom_tools[tool_name] = {
                    "name": tool_name,
                    "description": description,
                    "module": mcp_module_name,
                }

        # Update agent configuration
        self.configure()

    def get_custom_tool(self, name):
        """Get a custom tool by name.

        Args:
            name: The name of the custom tool

        Returns:
            The custom tool function if found, None otherwise

        """
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added.

        Returns:
            A list of custom tool names

        """
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool.

        Args:
            name: The name of the custom tool to remove

        Returns:
            True if the tool was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom functions
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True

        # Remove from custom tools (for highlighting)
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True

        # Remove from global namespace
        import builtins

        if hasattr(builtins, "_biomni_custom_functions") and name in builtins._biomni_custom_functions:
            del builtins._biomni_custom_functions[name]

        # Remove from tool registry
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True
                # Rebuild the document dataframe
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

        # Remove from module2api
        if hasattr(self, "module2api"):
            for tools in self.module2api.values():
                for i, tool in enumerate(tools):
                    if tool.get("name") == name:
                        del tools[i]
                        removed = True
                        break

        if removed:
            print(f"Custom tool '{name}' has been removed")
        else:
            print(f"Custom tool '{name}' was not found")

        return removed

    def add_data(self, data):
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value
                  e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
                  or {'path/to/file.txt': 'Description of the file'}

        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")

            # Initialize custom data storage if it doesn't exist
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}

            # Add each data item
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue

                # Extract filename from path for storage
                filename = os.path.basename(file_path) if "/" in file_path else file_path

                # Store the data with both the full path and description
                self._custom_data[filename] = {
                    "path": file_path,
                    "description": description,
                }

                # Also add to the data_lake_dict for consistency
                self.data_lake_dict[filename] = description

                print(f"Added data item '{filename}': {description}")
            self.configure()
            print(f"Successfully added {len(data)} data item(s) to the data lake")
            return True

        except Exception as e:
            print(f"Error adding data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_data(self, name):
        """Get a custom data item by name.

        Args:
            name: The name of the custom data item

        Returns:
            The custom data item info if found, None otherwise

        """
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added.

        Returns:
            A list of custom data item names and descriptions

        """
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item.

        Args:
            name: The name of the custom data item to remove

        Returns:
            True if the data item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom data
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True

        # Remove from data_lake_dict
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True

        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")

        return removed

    def add_software(self, software):
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value
                     e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                     or {'my_package': 'Description of the package functionality'}

        """
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")

            # Initialize custom software storage if it doesn't exist
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}

            # Add each software item
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue

                # Store the software with description
                self._custom_software[software_name] = {
                    "name": software_name,
                    "description": description,
                }

                # Also add to the library_content_dict for consistency
                self.library_content_dict[software_name] = description

                print(f"Added software '{software_name}': {description}")

            print(f"Successfully added {len(software)} software item(s) to the library")
            self.configure()
            return True

        except Exception as e:
            print(f"Error adding software: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_software(self, name):
        """Get a custom software item by name.

        Args:
            name: The name of the custom software item

        Returns:
            The custom software item info if found, None otherwise

        """
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added.

        Returns:
            A list of custom software item names and descriptions

        """
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item.

        Args:
            name: The name of the custom software item to remove

        Returns:
            True if the software item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom software
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True

        # Remove from library_content_dict
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True

        if removed:
            print(f"Custom software item '{name}' has been removed")
        else:
            print(f"Custom software item '{name}' was not found")

        return removed

    def _filter_know_how_for_commercial_mode(self):
        """Filter out know-how documents that don't allow commercial use.

        This method removes documents from the know-how loader that have
        commercial use restrictions when the agent is in commercial mode.
        """
        docs_to_remove = []

        for doc_id, doc in self.know_how_loader.documents.items():
            metadata = doc.get("metadata", {})
            commercial_use = metadata.get("commercial_use", "")

            # Check if commercial use is NOT allowed
            if "❌" in commercial_use or "Not Allowed" in commercial_use or "Non-Commercial" in commercial_use:
                docs_to_remove.append(doc_id)

        # Remove documents that don't allow commercial use
        for doc_id in docs_to_remove:
            doc_name = self.know_how_loader.documents[doc_id]["name"]
            self.know_how_loader.remove_document(doc_id)
            print(f"  ⚠️  Excluded know-how '{doc_name}' (non-commercial license)")

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        know_how_docs=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight
            know_how_docs: List of know-how documents with best practices and protocols

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Format know-how documents - include FULL content (metadata already stripped)
        know_how_formatted = []
        if know_how_docs:
            for doc in know_how_docs:
                if isinstance(doc, dict):
                    name = doc.get("name", "Unknown")
                    content = doc.get("content", "")
                    # Include full content in system prompt (metadata already removed)
                    know_how_formatted.append(f"📚 {name}:\n{content}")

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add protocol generation instructions
        prompt_modifier += """
PROTOCOL GENERATION:
If the user requests an experimental protocol, use search_protocols(), advanced_web_search_claude(), list_local_protocols(), and read_local_protocol() to generate an accurate protocol. Include details such as reagents (with catalog numbers if available), equipment specifications, replicate requirements, error handling, and troubleshooting - but ONLY include information found in these resources. Do not make up specifications, catalog numbers, or equipment details. Prioritize accuracy over completeness.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any(
            [custom_tools_formatted, custom_data_formatted, custom_software_formatted, know_how_formatted]
        )

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if know_how_formatted:
                prompt_modifier += """
📚 KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):
{know_how_docs}

IMPORTANT: These documents are ALREADY AVAILABLE in your context. You do NOT need to
retrieve them or "review" them as a separate step. You can DIRECTLY reference and use
the information from these documents to answer questions, provide protocols, suggest
parameters, and offer troubleshooting guidance.

These documents contain expert knowledge, protocols, and troubleshooting guidance.
Reference them directly for experimental design, methodology, and problem-solving.

"""

            if custom_tools_formatted:
                prompt_modifier += """
🔧 CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
📊 CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict if they exist
        if know_how_formatted:
            format_dict["know_how_docs"] = "\n\n".join(know_how_formatted)
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt

    def configure(self, self_critic=False, test_time_scale_round=0, critic_priorities=None):
        """Configure the agent with the initial system prompt and workflow.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling
            critic_priorities: Optional list of short bullets (e.g. from the
                Biomentis-Tutor Critic's `next_session_priorities`) to append
                to the system prompt. This is the "in-context reward
                shaping" hook — no model weights change, but the agent
                enters the session aware of its own past failure modes.
                When None or empty, the system prompt is unchanged.
        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Get data lake content
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # data_lake_dict and library_content_dict are already set in __init__

        # Prepare tool descriptions
        tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in self.module2api.items()}

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Add custom data items if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_with_desc.append({"name": name, "description": info["description"]})

        # Prepare library content list including custom software
        library_content_list = list(self.library_content_dict.keys())
        if hasattr(self, "_custom_software") and self._custom_software:
            for name in self._custom_software:
                if name not in library_content_list:  # Avoid duplicates
                    library_content_list.append(name)

        # Generate the system prompt for initial configuration (is_retrieval=False)
        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        # Load ALL know-how documents into initial system prompt
        # This makes best practices always available, not just when retrieved
        know_how_docs = []
        if hasattr(self, "know_how_loader") and self.know_how_loader.documents:
            for _doc_id, doc in self.know_how_loader.documents.items():
                # Use content without metadata for efficiency
                know_how_docs.append(
                    {
                        "id": doc["id"],
                        "name": doc["name"],
                        "description": doc["description"],
                        "content": doc["content_without_metadata"],
                        "metadata": doc["metadata"],
                    }
                )
            print(f"📚 Loading {len(know_how_docs)} know-how documents into system prompt")

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=False,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
            know_how_docs=know_how_docs if know_how_docs else None,
        )

        # In-context reward shaping: append the Critic's priorities from
        # prior sessions. The agent's behavior changes per session, but
        # the model weights stay frozen. Phase A wires the hook; Phase C
        # populates `critic_priorities` from the user's memory file.
        if critic_priorities:
            self.system_prompt += (
                "\n\nLessons from prior sessions (apply these):\n"
                + "\n".join(f"- {p}" for p in critic_priorities)
            )

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            # Add OpenAI-specific formatting reminders if using OpenAI models
            system_prompt = self.system_prompt
            if hasattr(self.llm, "model_name") and (
                "gpt" in str(self.llm.model_name).lower() or "openai" in str(type(self.llm)).lower()
            ):
                system_prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm.invoke(messages)

            # Normalize Responses API content blocks (list of dicts) into a plain string
            content = response.content
            if isinstance(content, list):
                # Concatenate textual parts; ignore tool_use or other non-text blocks
                text_parts: list[str] = []
                for block in content:
                    try:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            if btype in ("text", "output_text", "redacted_text"):
                                part = block.get("text") or block.get("content") or ""
                                if isinstance(part, str):
                                    text_parts.append(part)
                    except Exception:
                        # Be conservative; skip malformed blocks
                        continue
                msg = "".join(text_parts)
            else:
                # Fallback to string conversion for legacy content
                msg = str(content)

            # Enhanced parsing for better OpenAI compatibility
            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            # More flexible pattern matching for different LLM styles
            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            # Alternative patterns for OpenAI models that might use different formatting
            if not execute_match:
                # Try to find code blocks that might be intended as execute blocks
                code_block_match = re.search(r"```(?:python|bash|r)?\s*(.*?)```", msg, re.DOTALL)
                if code_block_match and not answer_match:
                    # If we found a code block and no solution, treat it as execute
                    execute_match = code_block_match

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                print("parsing error...")

                error_count = sum(
                    1 for m in state["messages"] if isinstance(m, AIMessage) and "There are no tags" in m.content
                )

                if error_count >= 2:
                    # If we've already tried to correct the model twice, just end the conversation
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["messages"].append(
                        AIMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                        )
                    )
                else:
                    # Try to correct it
                    state["messages"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            return state

        def execute(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(r"^#!CLI", "", code, count=1).strip()
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, count=1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    # Clear any previous plots before execution
                    self._clear_execution_plots()

                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                    # Plots are now captured directly in the execution entry above

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )

                # Store the execution result with the triggering message
                if not hasattr(self, "_execution_results"):
                    self._execution_results = []

                # Get any plots that were generated during this execution
                execution_plots = []
                try:
                    from biomentis.tool.support_tools import get_captured_plots

                    current_plots = get_captured_plots()
                    execution_plots = current_plots.copy()
                except Exception as e:
                    print(f"Warning: Could not capture plots from execution: {e}")
                    execution_plots = []

                # Store the execution result with metadata
                execution_entry = {
                    "triggering_message": last_message,  # The AI message that contained <execute>
                    "images": execution_plots,  # Base64 encoded images from this execution
                    "timestamp": datetime.now().isoformat(),
                }
                self._execution_results.append(execution_entry)

                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback = self.llm.invoke(messages + [HumanMessage(content=feedback_prompt)])

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        # display(Image(self.app.get_graph().draw_mermaid_png()))

    def _prepare_resources_for_retrieval(self, prompt):
        """Prepare resources for retrieval and return selected resource names.

        Args:
            prompt: The user's query

        Returns:
            dict: Dictionary containing selected resource names for tools, data_lake, and libraries
        """
        if not self.use_tool_retriever:
            return None

        # Gather all available resources
        # 1. Tools from the registry
        all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

        # 2. Data lake items with descriptions
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Create data lake descriptions for retrieval
        data_lake_descriptions = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_descriptions.append({"name": item, "description": description})

        # Add custom data items to retrieval if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_descriptions.append({"name": name, "description": info["description"]})

        # 3. Libraries with descriptions - use library_content_dict directly
        library_descriptions = []
        for lib_name, lib_desc in self.library_content_dict.items():
            library_descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items to retrieval if they exist
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                # Check if it's not already in the library descriptions to avoid duplicates
                if not any(lib["name"] == name for lib in library_descriptions):
                    library_descriptions.append({"name": name, "description": info["description"]})

        # 4. Know-how documents
        know_how_summaries = self.know_how_loader.get_document_summaries()

        # Use retrieval to get relevant resources
        resources = {
            "tools": all_tools,
            "data_lake": data_lake_descriptions,
            "libraries": library_descriptions,
            "know_how": know_how_summaries,
        }

        # Use prompt-based retrieval with the agent's LLM
        selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
        print("\n" + "=" * 60)
        print("🔍 RESOURCE RETRIEVAL")
        print("=" * 60)
        print("Using prompt-based retrieval with the agent's LLM")

        # Extract the names from the selected resources for the system prompt
        selected_resources_names = {
            "tools": selected_resources["tools"],
            "data_lake": [],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
            "know_how": [],
        }

        # Process data lake items to extract just the names
        for item in selected_resources["data_lake"]:
            if isinstance(item, dict):
                selected_resources_names["data_lake"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                # If the item already has a description, extract just the name
                name = item.split(": ")[0]
                selected_resources_names["data_lake"].append(name)
            else:
                selected_resources_names["data_lake"].append(item)

        # Process know-how documents - get the full content for selected documents
        if "know_how" in selected_resources and selected_resources["know_how"]:
            print("\n📚 Know-How Documents Retrieved:")
            for item in selected_resources["know_how"]:
                if isinstance(item, dict):
                    doc_id = item["id"]
                    doc = self.know_how_loader.get_document_by_id(doc_id)
                    if doc:
                        print(f"  ✓ {doc['name']}")
                        # Create a copy with content_without_metadata for agent context
                        doc_for_agent = {
                            "id": doc["id"],
                            "name": doc["name"],
                            "description": doc["description"],
                            "content": doc["content_without_metadata"],  # Use stripped version for agent
                            "metadata": doc["metadata"],
                        }
                        selected_resources_names["know_how"].append(doc_for_agent)
        else:
            print("\n📚 Know-How: None retrieved for this query")

        # Print summary of what was retrieved
        print("\n" + "-" * 60)
        print("📊 RETRIEVAL SUMMARY:")
        print(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
        print(f"  📊 Data Lake: {len(selected_resources_names['data_lake'])} selected")
        print(f"  ⚙️  Libraries: {len(selected_resources_names['libraries'])} selected")
        print(f"  📚 Know-How: {len(selected_resources_names['know_how'])} selected")
        print("=" * 60 + "\n")

        return selected_resources_names

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

        return self.log, message.content

    def go_stream(self, prompt) -> Generator[dict, None, None]:
        """Execute the agent with the given prompt and return a generator that yields each step.

        This function returns a generator that yields each step of the agent's execution,
        allowing for real-time monitoring of the agent's progress.

        Args:
            prompt: The user's query

        Yields:
            dict: Each step of the agent's execution containing the current message and state
        """
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

            # Yield the current step
            yield {"output": out}

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomentis.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomentis.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        # Extract know-how documents if present
        know_how_docs = selected_resources.get("know_how", [])

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=selected_resources["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
            know_how_docs=know_how_docs if know_how_docs else None,
        )

        # Print the raw system prompt for debugging
        # print("\n" + "="*20 + " RAW SYSTEM PROMPT FROM AGENT " + "="*20)
        # print(self.system_prompt)
        # print("="*70 + "\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _parse_tool_calls_from_code(self, code: str) -> list[str]:
        """Parse code to detect imported tools by looking for import statements.

        Args:
            code: The Python code to parse

        Returns:
            List of detected tool names
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_from_code(code, module2api, custom_functions)

    def _parse_tool_calls_with_modules(self, code: str) -> list[tuple[str, str]]:
        """Parse code to detect imported tools and their modules.

        Args:
            code: The Python code to parse

        Returns:
            List of tuples (tool_name, module_name)
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_with_modules(code, module2api, custom_functions)

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment.
        This makes custom tools available during code execution.
        """
        custom_functions = getattr(self, "_custom_functions", {})
        inject_custom_functions_to_repl(custom_functions)

    def create_mcp_server(self, tool_modules=None):
        """
        Create an MCP server object that exposes internal Biomentis tools.
        This gives you control over when and how to run the server.

        Args:
            tool_modules: List of module names to expose (default: all in self.module2api)

        Returns:
            FastMCP server object that you can run manually
        """
        import importlib

        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("BiomentisTools")
        modules = tool_modules or list(self.module2api.keys())

        registered_tools = 0

        for module_name in modules:
            try:
                # Import the actual module
                module = importlib.import_module(module_name)
                # Get tools for this module
                module_tools = self.module2api.get(module_name, [])

                for tool_schema in module_tools:
                    tool_name = tool_schema.get("name")
                    if not tool_name:
                        continue

                    try:
                        # Get the actual function
                        fn = getattr(module, tool_name, None)
                        if fn is None:
                            fn = getattr(self, "_custom_functions", {}).get(tool_name)

                        if fn is None:
                            print(f"Warning: Could not find function '{tool_name}' in module '{module_name}'")
                            continue

                        # Extract parameters from your specific schema format
                        required_params = tool_schema.get("required_parameters", [])
                        optional_params = tool_schema.get("optional_parameters", [])

                        # Generate the wrapper function
                        wrapper_func = self._generate_mcp_wrapper_from_biomni_schema(
                            fn, tool_name, required_params, optional_params
                        )

                        # Register with MCP
                        mcp.tool()(wrapper_func)
                        registered_tools += 1

                    except Exception as e:
                        print(f"Warning: Failed to register tool '{tool_name}': {e}")
                        continue

            except ImportError as e:
                print(f"Warning: Could not import module '{module_name}': {e}")
                continue

        print(f"Created MCP server with {registered_tools} tools")
        return mcp

    def save_conversation_history(self, filepath: str, include_images: bool = True, save_pdf: bool = True) -> None:
        """Save the complete conversation history as PDF only.

        This function generates and saves the complete conversation history from the agent's
        log and conversation state. It creates a temporary markdown file with formatted content
        including steps, code execution, observations, and optionally images, then converts it
        to PDF format. The markdown file is automatically cleaned up after PDF conversion.

        Args:
            filepath: Path where to save the PDF file (without extension). If the path doesn't
                    end with .pdf, it will be automatically appended.
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.
            save_pdf: Whether to save as PDF. Defaults to True. If False, no file is saved.

        Note:
            The function includes a 60-second timeout for PDF generation to prevent
            hanging. A temporary markdown file is created and automatically deleted.
        """
        import os
        import tempfile

        if not save_pdf:
            print("PDF saving is disabled. No file will be saved.")
            return

        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if it's not empty
            os.makedirs(directory, exist_ok=True)

        # Create PDF file path - use the user's filename and add .pdf extension
        if filepath.endswith(".pdf"):
            pdf_path = filepath
        else:
            # Remove any existing .md extension if present, then add .pdf
            base_name = filepath
            if base_name.endswith(".md"):
                base_name = base_name[:-3]  # Remove .md extension
            pdf_path = f"{base_name}.pdf"

        # Create markdown content
        markdown_content = self._generate_markdown_content(include_images)

        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(markdown_content)
            temp_markdown_path = temp_file.name

        try:
            # Add timeout for PDF generation to prevent hanging
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("PDF generation timed out")

            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                self._convert_markdown_to_pdf(temp_markdown_path, pdf_path)
                print(f"Conversation history saved as PDF: {pdf_path}")
                print(f"Total steps recorded: {len(self.log)}")
            finally:
                signal.alarm(0)  # Cancel the alarm

        except TimeoutError:
            print("Warning: PDF generation timed out after 60 seconds")
        except Exception as e:
            print(f"Warning: Could not convert to PDF: {e}")
        finally:
            # Clean up the temporary markdown file
            try:
                os.unlink(temp_markdown_path)
            except OSError:
                pass  # File might already be deleted

    def _generate_markdown_content(self, include_images: bool = True) -> str:
        """Generate markdown content from conversation history using both log and conversation state.

        This function processes the agent's conversation history from either the conversation
        state (if available) or the internal log to create a formatted markdown document.
        It handles step numbering, message processing, and content formatting.

        Args:
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.

        Returns:
            Formatted markdown string containing the complete conversation history
            with proper step numbering and content structure.
        """

        # Initialize content and tracking variables
        content = """# Biomentis Agent Conversation History

"""
        added_plots = set()
        step_number = 0
        first_human_shown = False

        # Get data source (conversation state or log)
        messages = self._get_messages_for_processing()

        # Process all messages using unified logic
        for message_data in messages:
            content, step_number, first_human_shown = self._process_message(
                message_data, content, step_number, first_human_shown, added_plots, include_images
            )

        return content

    def _get_messages_for_processing(self):
        """Get messages from conversation state or fallback to log.

        This function determines the best source for conversation messages, prioritizing
        the conversation state if available, otherwise falling back to the internal log.
        It normalizes the messages into a unified format for processing.

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        conversation_state = getattr(self, "_conversation_state", None)

        if conversation_state and hasattr(conversation_state, "get") and "messages" in conversation_state:
            print(f"DEBUG: Using conversation state with {len(conversation_state['messages'])} messages")
            return self._normalize_conversation_state_messages(conversation_state["messages"])
        else:
            print(f"DEBUG: Using self.log with {len(self.log)} entries")
            return self._normalize_log_messages(self.log)

    def _normalize_conversation_state_messages(self, messages):
        """Convert conversation state messages to unified format.

        This function takes LangChain message objects from the conversation state and
        converts them into a standardized dictionary format that the markdown generation
        system can work with. It extracts content and determines message types.

        Args:
            messages: List of LangChain message objects (HumanMessage, AIMessage, etc.)

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for message in messages:
            if hasattr(message, "content"):
                content = str(message.content)
            else:
                content = str(message)

            # Determine message type
            if isinstance(message, HumanMessage):
                msg_type = "human"
            elif isinstance(message, AIMessage):
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": message})

        return normalized

    def _normalize_log_messages(self, log_entries):
        """Convert log entries to unified format.

        This function takes internal log entries and converts them into the same
        standardized format as conversation state messages. It parses the log format
        to determine message types and extract content.

        Args:
            log_entries: List of log entry strings from the agent's internal log

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for log_entry in log_entries:
            content = str(log_entry)

            # Determine message type from log format
            if "Human Message" in content:
                msg_type = "human"
            elif "Ai Message" in content:
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": log_entry})

        return normalized

    def _process_message(self, message_data, content, step_number, first_human_shown, added_plots, include_images):
        """Process a single message and return updated state.

        This function is the main dispatcher for processing individual messages in the
        conversation history. It determines the message type and delegates to the
        appropriate processing function.

        Args:
            message_data: Dictionary containing 'content', 'type', and 'original' keys
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, updated_first_human_shown)
        """
        clean_output = clean_message_content(message_data["content"])
        msg_type = message_data["type"]

        if msg_type == "human":
            return self._process_human_message(clean_output, content, step_number, first_human_shown)
        elif msg_type == "ai":
            return self._process_ai_message(clean_output, content, step_number, added_plots, include_images)
        else:
            return self._process_other_message(
                clean_output, content, step_number, first_human_shown, added_plots, include_images
            )

    def _process_human_message(self, clean_output, content, step_number, first_human_shown):
        """Process human messages.

        This function handles human messages in the conversation history. It identifies
        parsing error messages and displays them appropriately, or formats the first
        human prompt as a special section.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter (unchanged for human messages)
            first_human_shown: Boolean flag indicating if first human message was shown

        Returns:
            Tuple of (updated_content, step_number, updated_first_human_shown)

        Note:
            Human messages don't increment the step counter as they are not considered
            steps in the agent's process.
        """
        if "each response must include thinking process" in clean_output.lower():
            parsing_error_content = create_parsing_error_html()
            content += f"{parsing_error_content}\n\n"
        elif not first_human_shown:
            content += "#### Human Prompt\n\n"
            content += f"*{clean_output}*\n\n"
            first_human_shown = True

        return content, step_number, first_human_shown  # step_number unchanged

    def _process_ai_message(self, clean_output, content, step_number, added_plots, include_images):
        """Process AI messages.

        This function handles AI messages in the conversation history. It can process
        both regular AI responses and messages containing observation tags. It handles
        step numbering, execution results, and content formatting.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, True)

        Note:
            This function can split messages containing observation tags and process
            each part separately, with observations formatted as terminal blocks.
        """
        # Check if this message contains observation tags and process accordingly
        import re

        observation_pattern = r"<observation>(.*?)</observation>"
        observation_matches = re.findall(observation_pattern, clean_output, re.DOTALL | re.IGNORECASE)

        if observation_matches:
            # Extract content before, between, and after observation tags
            parts = re.split(observation_pattern, clean_output, flags=re.DOTALL | re.IGNORECASE)

            # Process each part
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Even indices are non-observation content
                    if part.strip():
                        # This is regular content - process it normally
                        if not should_skip_message(part):
                            if part.strip():
                                step_number += 1
                                content += f"#### Step {step_number}\n\n"

                                # Handle execution results if present
                                execution_results = getattr(self, "_execution_results", None)
                                if has_execution_results(part, execution_results):
                                    content, added_plots = self._process_execution_with_results(
                                        part, content, added_plots, include_images, execution_results
                                    )
                                else:
                                    content = self._process_regular_ai_message(part, content)
                else:  # Odd indices are observation content
                    if part.strip():
                        # This is observation content - format as terminal
                        formatted_observation = format_observation_as_terminal(f"<observation>{part}</observation>")
                        if formatted_observation is not None:
                            content += f"{formatted_observation}\n\n"

            return content, step_number, True

        # Skip empty or error messages
        if should_skip_message(clean_output):
            return content, step_number, True

        if clean_output.strip():
            step_number += 1
            content += f"#### Step {step_number}\n\n"

            # Handle execution results if present
            execution_results = getattr(self, "_execution_results", None)
            if has_execution_results(clean_output, execution_results):
                content, added_plots = self._process_execution_with_results(
                    clean_output, content, added_plots, include_images, execution_results
                )
            else:
                content = self._process_regular_ai_message(clean_output, content)

        return content, step_number, True

    def _process_other_message(
        self, clean_output, content, step_number, first_human_shown, added_plots, include_images
    ):
        """Process other message types.

        This function handles message types that are neither human nor AI messages.
        It checks for observation tags and processes them accordingly, or adds the
        content as regular text.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, step_number, first_human_shown)
        """
        # Check if this is actually an observation (has <observation> tags)
        import re

        if not re.search(r"<observation>", clean_output, re.IGNORECASE):
            content += f"{clean_output}\n\n"
        return content, step_number, first_human_shown

    def _process_execution_with_results(self, clean_output, content, added_plots, include_images, execution_results):
        """Process AI message with execution results.

        This function handles AI messages that have associated execution results.
        It finds the matching execution result and adds any captured plots or images
        to the content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output
            execution_results: List of execution result dictionaries

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        matching_execution = find_matching_execution(clean_output, execution_results)

        if matching_execution:
            content = self._format_and_add_content(clean_output, content)
            content, added_plots = self._add_execution_plots(matching_execution, content, added_plots, include_images)
        else:
            content = self._format_and_add_content(clean_output, content)

        return content, added_plots

    def _format_and_add_content(self, clean_output, content):
        """Format and add content to markdown.

        This function applies formatting to AI message content before adding it to the
        markdown. It processes lists, execute tags, and tool calls to create properly
        formatted markdown content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        # Process lists first, then execute tags
        formatted_content = format_lists_in_text(clean_output)

        # Create a wrapper function for the tool parsing
        def parse_tool_calls_wrapper(code):
            return self._parse_tool_calls_with_modules(code)

        formatted_content = format_execute_tags_in_content(formatted_content, parse_tool_calls_wrapper)
        return content + f"{formatted_content}\n\n"

    def _add_execution_plots(self, matching_execution, content, added_plots, include_images):
        """Add plots from execution results.

        This function adds captured plots and images from execution results to the
        markdown content. It prevents duplicate plots from being added multiple times.

        Args:
            matching_execution: Execution result dictionary containing image data
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        if include_images and matching_execution.get("images"):
            for plot_data in matching_execution["images"]:
                if plot_data not in added_plots:
                    content += f"![Plot]({plot_data})\n\n"
                    added_plots.add(plot_data)
        return content, added_plots

    def _process_regular_ai_message(self, clean_output, content):
        """Process regular AI message without execution results.

        This function handles AI messages that don't have associated execution results.
        It applies standard formatting and adds the content to the markdown.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        return self._format_and_add_content(clean_output, content)

    def _convert_markdown_to_pdf(self, markdown_path: str, pdf_path: str) -> None:
        """Convert markdown file to PDF using weasyprint or markdown2pdf.

        This function is a wrapper around the utility function for converting markdown
        to PDF. It provides a clean interface for the agent to convert conversation
        history to PDF format.

        Args:
            markdown_path: Path to the input markdown file
            pdf_path: Path where the output PDF file should be saved

        Note:
            This function delegates to the convert_markdown_to_pdf utility function
            which handles multiple PDF conversion libraries and fallbacks.
        """
        convert_markdown_to_pdf(markdown_path, pdf_path)

    def _clear_execution_plots(self):
        """Clear execution plots before new execution.

        This function clears any previously captured plots from the execution environment
        before starting a new execution. This prevents old plots from appearing in
        new execution results.

        Note:
            This function calls the clear_captured_plots utility function and handles
            any exceptions gracefully to prevent execution failures.
        """
        try:
            from biomentis.tool.support_tools import clear_captured_plots

            clear_captured_plots()
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")

    def _generate_mcp_wrapper_from_biomni_schema(self, original_func, func_name, required_params, optional_params):
        """Generate wrapper function based on Biomentis schema format."""
        import inspect

        # Combine all parameters
        all_params = required_params + optional_params

        if not all_params:
            # No parameters
            def wrapper() -> dict:
                try:
                    result = original_func()
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__
            return wrapper

        else:
            # Has parameters
            def wrapper(**kwargs) -> dict:
                try:
                    # Build arguments dict
                    filtered_kwargs = {}

                    # Add required parameters
                    for param_info in required_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    # Add optional parameters only if provided and not None
                    for param_info in optional_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    result = original_func(**filtered_kwargs)
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            # Set function metadata
            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__

            # Create proper signature
            new_params = []

            # Map your types to Python types
            type_map = {"str": str, "int": int, "float": float, "bool": bool, "List[str]": list[str], "dict": dict}

            # Add required parameters
            for param_info in required_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                new_params.append(inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type))

            # Add optional parameters
            for param_info in optional_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                # Make it optional
                optional_type = param_type | None

                new_params.append(
                    inspect.Parameter(
                        param_name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=optional_type
                    )
                )

            # Set the signature
            wrapper.__signature__ = inspect.Signature(new_params, return_annotation=dict)

            return wrapper

    def launch_gradio_demo(self, thread_id=42, share=False, server_name="0.0.0.0", require_verification=False):
        """Launch a full-featured Gradio UI for the A1 agent (adapted from codeact_copilot).

        Args:
            thread_id: Thread ID for the conversation
            share: Whether to create a public shareable link
            server_name: Server name/IP to bind to (default: "0.0.0.0")
            require_verification: If True, requires access code verification

        Example:
            >>> agent = A1()
            >>> agent.launch_gradio_demo()
        """
        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from None

        from time import time

        self.main_history_copy = []
        self.main_export_entries: list[ExportEntry] = []
        self.inner_export_entries: list[ExportEntry] = []

        # Available access codes (if verification is required)
        available_access_codes = ["Biomni2025"]

        # Function for verification page
        def verify_access_code(code):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="Incorrect access code. Please check your access code.", visible=True),
                )

        def _history_messages_before_this_turn():
            messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    if msg["content"] not in ["Executor is working on it 👉"]:
                        messages.append(AIMessage(content=msg["content"]))
            return messages

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []
            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])

            history_messages = _history_messages_before_this_turn()

            self.main_history_copy += [{"role": "user", "content": text_input}]
            main_history.append(ChatMessage(role="user", content=text_input if text_input else "[Uploaded file]"))
            self.main_export_entries.append(
                ExportEntry(role="user", content=text_input if text_input else "[Uploaded file]")
            )

            # Add "Executor is working on it" message
            main_history.append(ChatMessage(role="assistant", content="Executor is working on it 👉"))
            yield inner_history, main_history

            # Keep track of the ChatMessage for the code block currently executing, so its
            # status can be flipped from pending -> done once the matching observation arrives.
            code_execution_messages = []

            for event in stream_agent_events(self, text_input, files, history_messages, thread_id):
                export_target = self.main_export_entries if event.channel == "main" else self.inner_export_entries
                export_target.append(ui_event_to_export_entry(event))

                if event.type == "status":
                    inner_history.append(ChatMessage(role="assistant", content=event.content))
                    yield inner_history, main_history

                elif event.type == "reasoning":
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content=event.content,
                            metadata={"title": event.title, "log": "Agent's thinking process"},
                        )
                    )
                    yield inner_history, main_history

                elif event.type == "solution":
                    main_history.append(
                        ChatMessage(
                            role="assistant",
                            content=event.content,
                            metadata={"title": event.title, "log": "Final answer provided by the agent"},
                        )
                    )
                    self.main_history_copy += [{"role": "assistant", "content": event.content}]
                    yield inner_history, main_history

                elif event.type == "code":
                    code_msg = ChatMessage(
                        role="assistant",
                        content=f"##### Code: \n```{event.language}\n{event.content}\n```",
                        metadata={
                            "title": event.title,
                            "log": f"Executing {event.language.capitalize()} code block...",
                            "status": "pending",
                            "start_time": time(),
                        },
                    )
                    inner_history.append(code_msg)
                    code_execution_messages.append(code_msg)
                    yield inner_history, main_history

                elif event.type == "observation":
                    if code_execution_messages:
                        code_msg = code_execution_messages[-1]
                        code_msg.metadata.update(
                            {
                                "status": "done",
                                "duration": event.duration,
                                "log": f"Code execution completed in {event.duration:.2f}s",
                            }
                        )

                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content=f"##### Observation: \n```\n{event.content}\n```",
                            metadata={
                                "status": "done",
                                "duration": event.duration,
                                "log": "Observation from code execution",
                                "collapsed": event.collapsible,
                                "collapsible": event.collapsible,
                            },
                        )
                    )
                    yield inner_history, main_history

                elif event.type == "file":
                    if event.file_path is None:
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content="",
                                metadata={"title": event.title, "log": "Files generated by the agent"},
                            )
                        )
                    elif event.file_kind == "pdf":
                        inner_history.append(
                            ChatMessage(role="assistant", content=event.content, metadata={"title": event.title})
                        )
                    else:
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=gr.Image(event.file_path),
                                metadata={"title": event.title},
                            )
                        )
                    yield inner_history, main_history

                elif event.type == "summary":
                    main_history.append(
                        ChatMessage(role="assistant", content=event.content, metadata={"title": event.title})
                    )
                    self.main_history_copy += [{"role": "assistant", "content": event.content}]
                    yield inner_history, main_history

                elif event.type == "complete":
                    inner_history.append(
                        ChatMessage(role="assistant", content=event.content, metadata={"title": event.title})
                    )
                    yield inner_history, main_history

        def like(data: gr.LikeData):
            print("User liked the response")
            print(f"Index: {data.index}, Liked: {data.liked}")

        def on_model_change(selected):
            if not selected or ": " not in selected:
                return
            source, model = selected.split(": ", 1)
            try:
                self.llm = get_llm(model, source=source, config=default_config)
                print(f"Switched model to {source}: {model}")
            except Exception as e:
                print(f"Failed to switch model to {selected}: {e}")

        def export_panel_html(panel):
            entries = self.main_export_entries if panel == "main" else self.inner_export_entries
            panel_title = (
                "Biomentis A1 Agent — Main Output" if panel == "main" else "Biomentis Executor — Reasoning Log"
            )
            html_doc = render_transcript_html(entries, panel_title, get_llm_display_name(self.llm))
            try:
                path = save_html_export(html_doc, panel)
                gr.Info(f"Saved {panel} panel to {path}")
            except Exception as e:
                gr.Warning(f"Failed to export {panel} panel: {e}")

        # Create the Gradio interface
        with gr.Blocks() as demo:
            # Verification page (if enabled)
            verification_container = gr.Group(visible=require_verification)
            main_interface_container = gr.Group(visible=not require_verification)

            with verification_container:
                gr.Markdown("# Biomentis A1 Agent - Access Verification")
                gr.Markdown("Please enter your access code to continue.")
                access_code_input = gr.Textbox(label="Access Code", type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button("Verify Access")
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            # Main interface
            with main_interface_container:
                model_choices = [
                    f"{source}: {model}"
                    for source, models in list_available_providers().items()
                    for model in models
                ]
                if model_choices:
                    model_selector = gr.Dropdown(
                        label="Switch model (optional)",
                        choices=model_choices,
                        value=None,
                    )
                    model_selector.change(fn=on_model_change, inputs=[model_selector], outputs=[])

                with gr.Row():
                    with gr.Column(scale=1):
                        main_chatbot = gr.Chatbot(
                            label="Biomentis A1 Agent",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )
                        export_main_btn = gr.Button("📥 Export to HTML", size="sm")
                        export_main_btn.click(fn=lambda: export_panel_html("main"), inputs=[], outputs=[])
                    with gr.Column(scale=1):
                        innerloop_chatbot = gr.Chatbot(
                            label="Biomentis Executor",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )
                        export_inner_btn = gr.Button("📥 Export to HTML", size="sm")
                        export_inner_btn.click(fn=lambda: export_panel_html("inner"), inputs=[], outputs=[])

                with gr.Row():
                    prompt_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="Ask something or upload a file...",
                        show_label=False,
                    )

                # Bind submission
                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot],
                ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])
                main_chatbot.like(like)

        # Launch
        print(f"Launching Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)

    def launch_streamlit_demo(self, thread_id=42, require_verification=False, stream_fn=None):
        """Render a Streamlit UI for the A1 agent, sharing biomentis.ui_core with the Gradio demo.

        Must be called from within a script run via `streamlit run`. Because Streamlit
        re-executes the whole script on every interaction, construct the agent once via
        `st.cache_resource` in your entry script rather than re-instantiating A1 each rerun:

        Example (streamlit_app.py):
            >>> import streamlit as st
            >>> from biomentis.agent import A1
            >>>
            >>> @st.cache_resource
            ... def get_agent():
            ...     return A1(path="./data")
            >>>
            >>> get_agent().launch_streamlit_demo()

        Then run: streamlit run streamlit_app.py

        Args:
            thread_id: Thread ID for the conversation
            require_verification: If True, requires access code verification
            stream_fn: Optional callable `(agent, prompt, files, history, thread_id) -> Iterator[UIEvent]`
                used in place of the default `stream_agent_events`. The tutor layer passes
                a wrapper here that injects per-step instruction cards and a pause gate.
                When None, the run is byte-for-byte identical to a tutor-less run.
        """
        try:
            import streamlit as st
        except ImportError:
            raise ImportError("Streamlit is not installed. Please install it with: pip install streamlit") from None

        try:
            # Must be the first Streamlit command in the script; if the caller already
            # issued one (e.g. a spinner from st.cache_resource around agent construction),
            # Streamlit raises here — safe to ignore and keep whatever config is already set.
            st.set_page_config(page_title="Biomentis A1 Agent", layout="wide")
        except Exception:
            pass

        if not hasattr(self, "main_history_copy"):
            self.main_history_copy = []
        if "biomni_transcript" not in st.session_state:
            st.session_state.biomni_transcript = []  # list of {"panel": "main"|"inner", ...render fields}
        if "biomni_verified" not in st.session_state:
            st.session_state.biomni_verified = not require_verification

        if not st.session_state.biomni_verified:
            st.title("Biomentis A1 Agent - Access Verification")
            code = st.text_input("Access Code", type="password")
            if st.button("Verify Access"):
                if code == "Biomni2025":
                    st.session_state.biomni_verified = True
                    st.rerun()
                else:
                    st.error("Incorrect access code. Please check your access code.")
            return

        # Header logo is rendered by streamlit_app.py above the chat
        # panel (top of page, left-justified, 450px wide).
        st.title("Biomentis A1 Agent")

        # If the caller (e.g. streamlit_app.py) already wired up a model picker
        # and set agent.llm to the user's selection, skip the in-method picker
        # to avoid two dropdowns in the sidebar.
        if not st.session_state.get("biomni_agent_key"):
            provider_models = list_available_providers()
            model_choices = [f"{source}: {model}" for source, models in provider_models.items() for model in models]
            if model_choices:
                selected = st.sidebar.selectbox("Switch model (optional)", options=[None, *model_choices])
                if selected and selected != st.session_state.get("biomni_selected_model"):
                    source, model = selected.split(": ", 1)
                    try:
                        self.llm = get_llm(model, source=source, config=default_config)
                        st.session_state.biomni_selected_model = selected
                        st.sidebar.success(f"Switched model to {selected}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to switch model: {e}")

        # File attachments live next to the agent's prompt input (below the
        # transcript) instead of in the sidebar, so the boundary between
        # "this goes to the agent" and "this goes to the Tutor KB" is
        # visually obvious. The Tutor's KB is a long-lived grounding index;
        # these are per-prompt inputs the agent reads in its first step.
        uploaded_files: list = []

        # Two views of the same run, side by side in the old layout. They
        # have very different lengths and rarely need to be compared row
        # by row, so the user asked for tabs. The file-uploader / prompt
        # form below is a shared input and stays outside the tabs so it's
        # visible regardless of which view is active.
        #
        # While a run is in flight, swap the order so the Executor tab
        # is the active (first) tab — new users wouldn't otherwise know
        # the activity is happening in the other tab. `st.tabs` in
        # Streamlit 1.42 has no programmatic-switch API, so we
        # re-render the labels in the desired order each rerun; the
        # first label is always the active one. The tab labels are
        # identical either way, so this is a pure UX change.
        _in_run = bool(st.session_state.get("biomni_run_active"))
        _tab_labels = (
            ["Biomentis Executor", "Biomentis A1 Agent"]
            if _in_run
            else ["Biomentis A1 Agent", "Biomentis Executor"]
        )
        _first_tab, _second_tab = st.tabs(_tab_labels)
        if _in_run:
            tab_main, tab_inner = _second_tab, _first_tab
        else:
            tab_main, tab_inner = _first_tab, _second_tab
        # Alias the tab handles to the variable names the rest of the
        # method uses (`main_col` / `inner_col`). The dispatch in the
        # entry-rendering loop and the live-stream loop reads off each
        # entry's `panel` tag, so it works unchanged.
        main_col, inner_col = tab_main, tab_inner
        # Tab labels already convey the panel name; the per-column
        # subheader was redundant.

        # Live "elapsed" caption at the top of the Executor tab. Each
        # streaming event triggers a rerun, so the value updates on its
        # own — no autorefresh needed. Hidden when no run is active.
        if _in_run and st.session_state.get("biomni_run_started_at") is not None:
            _elapsed = time.monotonic() - st.session_state.biomni_run_started_at
            inner_col.caption(f"⏱  {_elapsed:0.1f}s elapsed")

        def export_entries(panel):
            return [
                ExportEntry(
                    role=e.get("role", "assistant"),
                    kind=e.get("kind", "text"),
                    title=e.get("title"),
                    content=e.get("content", ""),
                    language=e.get("language"),
                    file_path=e.get("file_path"),
                )
                for e in st.session_state.biomni_transcript
                if e["panel"] == panel
            ]

        if main_col.button("📥 Export to HTML", key="export_main_html"):
            html_doc = render_transcript_html(
                export_entries("main"), "Biomentis A1 Agent — Main Output", get_llm_display_name(self.llm)
            )
            try:
                path = save_html_export(html_doc, "main")
                main_col.success(f"Saved to {path}")
            except Exception as e:
                main_col.error(f"Failed to export: {e}")

        if inner_col.button("📥 Export to HTML", key="export_inner_html"):
            html_doc = render_transcript_html(
                export_entries("inner"), "Biomentis Executor — Reasoning Log", get_llm_display_name(self.llm)
            )
            try:
                path = save_html_export(html_doc, "inner")
                inner_col.success(f"Saved to {path}")
            except Exception as e:
                inner_col.error(f"Failed to export: {e}")

        def render_entry(container, entry):
            with container.chat_message(entry.get("role", "assistant")):
                if entry.get("title"):
                    st.markdown(f"**{entry['title']}**")
                if entry.get("kind") == "code":
                    st.code(entry["content"], language=entry.get("language") or "python")
                elif entry.get("kind") == "image":
                    st.image(entry["file_path"])
                elif entry.get("kind") == "observation":
                    with st.expander("Observation", expanded=not entry.get("collapsible", False)):
                        st.code(entry["content"])
                elif entry.get("kind") == "instruction":
                    # Tutor teaching card. Try the rich renderer first
                    # (provided by ui_tutor.py when the tutor is enabled);
                    # fall back to a plain markdown card if the renderer is
                    # unavailable so research mode is unaffected.
                    card = entry.get("card")
                    if card is not None and _RENDER_INSTRUCTION_CARD is not None:
                        # Make sure the card carries a step_id for the
                        # renderer's expander labels.
                        if getattr(card, "step_id", None) is None:
                            try:
                                card.step_id = entry.get("step_id")
                            except Exception:
                                pass
                        _RENDER_INSTRUCTION_CARD(container, card)
                    elif entry.get("content"):
                        with st.expander("🎓 Teaching note", expanded=True):
                            st.markdown(entry["content"])
                elif entry.get("kind") == "paused":
                    # The Continue button is wired by the tutor module if
                    # present; otherwise we render a no-op so the stream
                    # remains a transcript.
                    if _RENDER_PAUSE_GATE is not None:
                        _RENDER_PAUSE_GATE(container, entry.get("step_id"))
                    else:
                        st.markdown("⏸ _(paused)_")
                elif entry.get("kind") == "qa":
                    if entry.get("content"):
                        with st.expander("💬 Tutor Q&A", expanded=False):
                            st.markdown(entry["content"])
                elif entry.get("content"):
                    st.markdown(entry["content"])

        for entry in st.session_state.biomni_transcript:
            render_entry(main_col if entry["panel"] == "main" else inner_col, entry)

        # Multi-line prompt input. st.chat_input is hard-coded to a single
        # line by Streamlit, so we replace it with a text area + Send button
        # so users can paste long prompts without truncation. Submit is via
        # the Send button or Ctrl+Enter / Cmd+Enter inside the text area.
        # The file uploader is inside the form so attachments are submitted
        # alongside the prompt; `clear_on_submit=True` resets both the text
        # area and the uploader after a send.
        st.session_state.setdefault("biomni_prompt_draft", "")
        with st.form(key="biomni_prompt_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Attach files (optional)",
                accept_multiple_files=True,
                key="biomni_prompt_attachments",
                help=(
                    "Files the agent should read in its first step. "
                    "For long-lived course materials, use the Tutor's "
                    "Knowledge base expander in the sidebar instead."
                ),
            )
            prompt_text = st.text_area(
                "Ask something...",
                value=st.session_state.biomni_prompt_draft,
                key="biomni_prompt_textarea",
                height=140,
                label_visibility="visible",
                placeholder=(
                    "Type or paste a biomedical task here. Submit with the "
                    "Send button or Ctrl+Enter / Cmd+Enter."
                ),
            )
            submitted = st.form_submit_button("Send", use_container_width=False)
        prompt = prompt_text.strip() if submitted else ""
        st.session_state.biomni_prompt_draft = "" if submitted else prompt_text

        # Tutor-mode resume: if the user clicked Continue, or if a tutor
        # run is in progress, drive the wrapper even with an empty prompt.
        # We only do this when `stream_fn` is provided (tutor installed);
        # in research mode the prompt is the only trigger.
        _tutor_continue = False
        _tutor_run_prompt: str | None = None
        if stream_fn is not None:
            try:
                from biomentis.ui_tutor import (
                    consume_pending_continue,
                    get_current_run_prompt,
                )
                _tutor_continue = consume_pending_continue()
                _tutor_run_prompt = get_current_run_prompt()
            except Exception:
                pass

        if prompt or (_tutor_continue and _tutor_run_prompt is not None):
            # In tutor-resume mode, the prompt is the in-flight run's prompt.
            effective_prompt = prompt or _tutor_run_prompt or ""

            # Start the per-task timer the moment we begin. The matching
            # stop point is the `complete` event below. `monotonic()`
            # is wall-clock-immune (DST / NTP jumps don't produce
            # negative durations). The flag also drives the tab
            # reorder: while a run is in flight, the Executor tab
            # becomes the active (first) tab.
            st.session_state.biomni_run_started_at = time.monotonic()
            st.session_state.biomni_run_active = True

            files = []
            saved_dir = os.path.join(self.path, "streamlit_uploads")
            for uploaded in uploaded_files or []:
                os.makedirs(saved_dir, exist_ok=True)
                dest = os.path.join(saved_dir, uploaded.name)
                with open(dest, "wb") as f:
                    f.write(uploaded.getbuffer())
                files.append(dest)

            history_messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))

            # Only append a user-message to the history if this is a *new*
            # prompt, not a tutor-resume. Resume re-uses the same prompt.
            if prompt:
                self.main_history_copy += [{"role": "user", "content": prompt}]
                user_entry = {"panel": "main", "role": "user", "content": prompt}
                st.session_state.biomni_transcript.append(user_entry)
                render_entry(main_col, user_entry)

            code_execution_entries = []

            # Resolve the event source: a tutor wrapper if provided, otherwise
            # the default `stream_agent_events` (research mode). Picking this
            # at submit time lets the caller swap the wrapper per session.
            _stream = stream_fn if stream_fn is not None else stream_agent_events
            for event in _stream(self, effective_prompt, files, history_messages, thread_id):
                entry = None
                if event.type == "status":
                    entry = {"panel": "inner", "role": "assistant", "content": event.content}
                elif event.type == "reasoning":
                    entry = {"panel": "inner", "role": "assistant", "content": event.content, "title": event.title}
                elif event.type == "solution":
                    entry = {"panel": "main", "role": "assistant", "content": event.content, "title": event.title}
                    self.main_history_copy += [{"role": "assistant", "content": event.content}]
                elif event.type == "code":
                    entry = {
                        "panel": "inner",
                        "role": "assistant",
                        "kind": "code",
                        "content": event.content,
                        "language": event.language,
                        "title": event.title,
                    }
                    code_execution_entries.append(entry)
                elif event.type == "observation":
                    if code_execution_entries:
                        code_execution_entries[-1]["title"] = f"🛠️ Code (done in {event.duration:.2f}s)"
                    entry = {
                        "panel": "inner",
                        "role": "assistant",
                        "kind": "observation",
                        "content": event.content,
                        "collapsible": event.collapsible,
                    }
                elif event.type == "file":
                    if event.file_path is None:
                        entry = {"panel": "inner", "role": "assistant", "content": "", "title": event.title}
                    elif event.file_kind == "pdf":
                        entry = {
                            "panel": "inner",
                            "role": "assistant",
                            "content": event.content,
                            "title": event.title,
                        }
                    else:
                        entry = {
                            "panel": "inner",
                            "role": "assistant",
                            "kind": "image",
                            "file_path": event.file_path,
                            "title": event.title,
                        }
                elif event.type == "summary":
                    entry = {"panel": "main", "role": "assistant", "content": event.content, "title": event.title}
                    self.main_history_copy += [{"role": "assistant", "content": event.content}]
                elif event.type == "complete":
                    # End the per-task timer that the submit branch
                    # started. We append the duration to the existing
                    # "🔄 Complete" entry's content (title stays the
                    # same) so the user immediately sees how long the
                    # task took. Clearing the run-active flag swaps
                    # the tab order back to A1 Agent on the next
                    # rerun, completing the auto-switch cycle.
                    _started = st.session_state.get("biomni_run_started_at")
                    _duration = time.monotonic() - _started if _started is not None else 0.0
                    _content = f"{event.content}  ·  total time: {format_duration(_duration)}"
                    entry = {"panel": "inner", "role": "assistant", "content": _content, "title": event.title}
                    st.session_state.biomni_run_active = False
                    st.session_state.biomni_run_started_at = None
                elif event.type == "instruction":
                    # Tutor-layer teaching card. Lands in the inner panel right
                    # after the event it explains. `card` carries the structured
                    # fields (what/why/prereqs/look_for/citations/bloom/dok);
                    # the renderer in ui_tutor.py (or render_entry fallback)
                    # handles the visual layout.
                    entry = {
                        "panel": "inner",
                        "role": "assistant",
                        "kind": "instruction",
                        "content": event.content,
                        "title": event.title or "🎓 Teaching note",
                        "card": getattr(event, "card", None),
                    }
                elif event.type == "paused":
                    # The tutor wrapper yields a "paused" event to halt the
                    # stream. The render loop shows a Continue button when it
                    # sees this; the run resumes on the next rerun.
                    entry = {
                        "panel": "inner",
                        "role": "assistant",
                        "kind": "paused",
                        "content": event.content,
                        "title": event.title or "⏸ Paused",
                        "step_id": getattr(event, "step_id", None),
                    }
                elif event.type == "qa":
                    entry = {
                        "panel": "main",
                        "role": "assistant",
                        "kind": "qa",
                        "content": event.content,
                        "title": event.title or "💬 Tutor Q&A",
                    }
                elif event.type == "rubric_update":
                    entry = {
                        "panel": "inner",
                        "role": "assistant",
                        "kind": "text",
                        "content": event.content,
                        "title": event.title or "📋 Rubric updated",
                    }

                if entry is not None:
                    st.session_state.biomni_transcript.append(entry)
                    render_entry(main_col if entry["panel"] == "main" else inner_col, entry)

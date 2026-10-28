import base64
import hashlib
import io
import os
import re
import sys
from io import StringIO

# Create a persistent namespace that will be shared across all executions
_persistent_namespace = {}

# Global list to store captured plots
_captured_plots = []


def _error_report(exc: BaseException, command: str, printed: str) -> str:
    """The observation for a step that raised: the error, where, and what it printed first.

    It used to be only "Error: {exc}". A self-check the agent wrote
    (`raise RuntimeError("GUARD FAILED")` after printing its simulation
    results, 2026-10-28) came back as "Error: GUARD FAILED", without the
    results, the failing line or which check failed; a bare `assert` came back
    as "Error: " and nothing more. The first line still starts with "Error:",
    which other code looks for, and comes first so it survives truncation.
    """
    lineno = None
    if isinstance(exc, SyntaxError):
        lineno = exc.lineno
    else:
        # The agent's own code runs as "<string>"; its last frame is the line
        # that failed, even when the error was raised inside a library.
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<string>":
                lineno = tb.tb_lineno
            tb = tb.tb_next
    message = str(exc).strip()
    head = f"Error: {type(exc).__name__}" + (f": {message}" if message else "")
    lines = command.splitlines()
    if lineno and 0 < lineno <= len(lines):
        head += f"\n  at line {lineno}: {lines[lineno - 1].strip()}"
    if printed.strip():
        head += "\n\nOutput printed before the error:\n" + printed.rstrip("\n")
    return head


def run_python_repl(command: str) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """

    def execute_in_repl(command: str) -> str:
        """Helper function to execute the command in the persistent environment."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Use the persistent namespace
        global _persistent_namespace

        try:
            # Apply matplotlib monkey patches before execution
            _apply_matplotlib_patches()

            # Execute the command in the persistent namespace
            exec(command, _persistent_namespace)
            output = mystdout.getvalue()

        except Exception as e:
            output = _error_report(e, command, mystdout.getvalue())
        finally:
            sys.stdout = old_stdout
            # Capture figures the code left open (no plt.show / savefig);
            # they are shown in the step, or withheld, per the plot setting.
            # In `finally` so a step that raises still owns its partial
            # figure, instead of it surfacing under the next step.
            summary = _capture_matplotlib_plots()
        if summary:
            output = output.rstrip("\n") + "\nPlot captured.\n" + "\n".join(summary) + "\n"
        return output

    command = command.strip("```").strip()
    # The agent guesses module paths for tools; point them at the real ones
    # before executing so a misfiled import isn't a dead run.
    try:
        from biomentis.tool.tool_imports import fix_tool_imports

        command = fix_tool_imports(command)
    except Exception:
        pass
    return execute_in_repl(command)


def _capture_matplotlib_plots() -> list[str]:
    """Capture and close every open matplotlib figure.

    Returns a text summary of the figures captured (see `describe_figure`),
    so the observation tells the agent what it drew; it never sees the image.
    """
    global _captured_plots
    summary: list[str] = []
    try:
        import matplotlib.pyplot as plt

        from biomentis.plots import describe_figure

        # Check if there are any active figures
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)

                # Save figure to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)

                # Convert to base64
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plot_data = f"data:image/png;base64,{image_data}"

                # Add to captured plots if not already there
                if plot_data not in _captured_plots:
                    _captured_plots.append(plot_data)
                try:
                    summary.extend(describe_figure(fig))
                except Exception as e:
                    summary.append(f"(could not summarize the figure: {e})")

                # Close the figure to free memory
                plt.close(fig)

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not capture matplotlib plots: {e}")
    return summary


def _apply_matplotlib_patches():
    """Apply simple monkey patches to matplotlib functions to automatically capture plots."""
    try:
        import matplotlib.pyplot as plt

        # Only patch if matplotlib is available and not already patched
        if hasattr(plt, "_biomni_patched"):
            return

        # Store original functions
        original_show = plt.show
        original_savefig = plt.savefig

        def show_with_capture(*args, **kwargs):
            """Enhanced show function that captures plots before displaying them."""
            # Capture any plots before showing
            summary = _capture_matplotlib_plots()
            # Neutral on purpose: under plot mode "student" the figure is
            # captured for the record but never displayed.
            print("Plot captured.")
            for line in summary:
                print(line)
            # Call the original show function
            return original_show(*args, **kwargs)

        def savefig_with_capture(*args, **kwargs):
            """Enhanced savefig function that captures plots after saving them."""
            # Get the filename from args if provided
            filename = args[0] if args else kwargs.get("fname", "unknown")
            # Call the original savefig function
            result = original_savefig(*args, **kwargs)
            # Capture the plot after saving
            summary = _capture_matplotlib_plots()
            # Print a message to indicate plot was saved
            print(f"Plot saved to: {filename}")
            for line in summary:
                print(line)
            return result

        # Replace functions with enhanced versions
        plt.show = show_with_capture
        plt.savefig = savefig_with_capture

        # Mark as patched to avoid double-patching
        plt._biomni_patched = True

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not apply matplotlib patches: {e}")


def get_captured_plots():
    """Get all captured matplotlib plots."""
    global _captured_plots
    return _captured_plots.copy()


def clear_captured_plots():
    """Clear all captured matplotlib plots."""
    global _captured_plots
    _captured_plots = []


def _structure_summary(path: str) -> str:
    """One line naming what `path` holds: its title/name and formula, if known."""
    try:
        from biomentis.structure_info import structure_metadata

        rows = dict(structure_metadata(path))
    except Exception:
        rows = {}
    if "Error" in rows:
        rows = {}
    bits = [b for b in (rows.get("Title") or rows.get("Name") or rows.get("Molecule(s)"), rows.get("Formula")) if b]
    label = os.path.basename(path)
    return f"Showing {label}: {' — '.join(bits)}" if bits else f"Showing {label}"


def show_structure(source: str) -> str:
    """Display a molecule in the app's 3D structure viewer.

    `source` is a PDB ID (e.g. 6NK5, downloaded from RCSB), a path to a
    .pdb/.sdf/.smi file, or a SMILES string. The structure is saved under the
    current run's artifact directory and a marker line is printed that the UI
    turns into a viewer. Returns a one-line summary, or
    "Could not show structure: <reason>" (and prints no marker) on bad input.
    Either line is also printed, so the agent sees it in the observation even
    when it calls this as a statement and drops the return value.
    """
    from biomentis.structures import STRUCTURE_MARKER, current_run_id, resolve_structure_reason

    src = source.strip() if isinstance(source, str) else ""
    if not src:
        message = "Could not show structure: no source given"
        print(message)
        return message

    path, reason = resolve_structure_reason(src, current_run_id(), [], source_db="agent")
    if path is None:
        message = f"Could not show structure: {reason}"
        print(message)
        return message

    path = os.path.abspath(path)
    print(f"{STRUCTURE_MARKER}{path}")
    summary = _structure_summary(path)
    print(summary)
    return summary


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Parameters
    ----------
        function_name (str): Fully qualified function name (e.g., 'bioagentos.tool.support_tools.write_python_code')

    Returns
    -------
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


# def request_human_feedback(question, context, reason_for_uncertainty):
#     """
#     Request human feedback on a question.

#     Parameters:
#         question (str): The question that needs human feedback.
#         context (str): Context or details that help the human understand the situation.
#         reason_for_uncertainty (str): Explanation for why the LLM is uncertain about its answer.

#     Returns:
#         str: The feedback provided by the human.
#     """
#     print("Requesting human feedback...")
#     print(f"Question: {question}")
#     print(f"Context: {context}")
#     print(f"Reason for Uncertainty: {reason_for_uncertainty}")

#     # Capture human feedback
#     human_response = input("Please provide your feedback: ")

#     return human_response


def download_synapse_data(
    entity_ids: str | list[str],
    download_location: str = ".",
    follow_link: bool = False,
    recursive: bool = False,
    timeout: int = 300,
    entity_type: str = "dataset",
):
    """Download data from Synapse using entity IDs.

    Uses the synapse CLI to download files, folders, or projects from Synapse.
    Requires SYNAPSE_AUTH_TOKEN environment variable for authentication.
    Automatically installs synapseclient if not available.

    CRITICAL: Always check entity type from query_synapse() search results or user hints and pass the correct entity_type!
    The default entity_type="dataset" may not be appropriate for your entity.

    IMPORTANT: Multiple entity IDs are only supported for entity_type="file".
    For datasets, folders, and projects, only a single entity_id is supported.

    Parameters
    ----------
    entity_ids : str or list of str
        Synapse entity ID(s) to download.
        - For files: Can be a single ID string or list of ID strings
        - For datasets/folders/projects: Must be a single ID string only
    download_location : str, default "."
        Directory where files will be downloaded (current directory by default)
    follow_link : bool, default False
        Whether to follow links to download the linked entity
    recursive : bool, default False
        Whether to recursively download folders and their contents
        ONLY valid for entity_type="folder" - ignored for other types
    timeout : int, default 300
        Timeout in seconds for each download operation
    entity_type : str, default "dataset"
        Type of Synapse entity ("dataset", "file", "folder", "project")
        MUST match the actual entity type from search results or user hints!
        The default "dataset" should only be used for actual datasets.
        Check the 'node_type' field in search results to determine correct type.

    Returns
    -------
    dict
        Dictionary containing download results and any errors

    Notes
    -----
    Requires SYNAPSE_AUTH_TOKEN environment variable with your Synapse personal
    access token for authentication.

    AGENT USAGE GUIDANCE:
    1. Always check the 'node_type' field from query_synapse() search results or user hints
    2. Pass the correct entity_type parameter matching the node_type
    3. Do NOT rely on the default entity_type="dataset" unless confirmed
    4. For multiple downloads, ensure all entities are of type "file"
    5. Only use recursive=True with entity_type="folder"

    Examples
    --------
    # After searching with query_synapse(), check node_type and use appropriate entity_type:

    # If search result shows 'node_type': 'dataset'
    download_synapse_data("syn123456", entity_type="dataset")

    # If search result shows 'node_type': 'file'
    download_synapse_data("syn654321", entity_type="file")

    # If search result shows 'node_type': 'folder'
    download_synapse_data("syn789012", entity_type="folder", recursive=True)

    # Multiple files (only if all are 'node_type': 'file')
    download_synapse_data(["syn111", "syn222"], entity_type="file")
    """
    import os
    import subprocess

    # Check for required authentication token
    synapse_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not synapse_token:
        return {
            "success": False,
            "error": "SYNAPSE_AUTH_TOKEN environment variable is required for downloading",
            "suggestion": "Set SYNAPSE_AUTH_TOKEN with your Synapse personal access token",
        }

    # Check if synapse CLI is available
    try:
        subprocess.run(["synapse", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try to install synapseclient
            print("Installing synapseclient...")
            subprocess.run(["pip", "install", "synapseclient"], check=True)
            print("✓ synapseclient installed successfully")
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Failed to install synapseclient: {e}",
                "suggestion": "Please install manually: pip install synapseclient",
            }

    # Ensure entity_ids is a list
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    # Validate that multiple IDs are only used with file entity type
    if len(entity_ids) > 1 and entity_type != "file":
        return {
            "success": False,
            "error": f"Multiple entity IDs are only supported for entity_type='file'. "
            f"For entity_type='{entity_type}', only a single entity_id is supported.",
            "suggestion": "Use a single entity_id string instead of a list, or change entity_type to 'file'",
        }

    # Validate that recursive is only used with folder entity type
    if recursive and entity_type != "folder":
        return {
            "success": False,
            "error": f"recursive=True is only valid for entity_type='folder'. "
            f"For entity_type='{entity_type}', recursive should be False.",
            "suggestion": "Set recursive=False, or change entity_type to 'folder' if appropriate",
        }

    # Create download directory if it doesn't exist
    os.makedirs(download_location, exist_ok=True)

    results = []
    errors = []

    for entity_id in entity_ids:
        try:
            # Build synapse download command with authentication
            if entity_type == "dataset":
                # For datasets, use query syntax to download the actual files
                cmd = [
                    "synapse",
                    "-p",
                    synapse_token,
                    "get",
                    "-q",
                    f"select * from {entity_id}",
                    "--downloadLocation",
                    download_location,
                ]
            else:
                # For files, folders, projects, use direct ID
                cmd = ["synapse", "-p", synapse_token, "get", entity_id, "--downloadLocation", download_location]

            # Add recursive flag only for folders (validation above ensures recursive is only True for folders)
            if entity_type == "folder" and recursive:
                cmd.append("-r")

            if follow_link:
                cmd.append("--followLink")

            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)

            results.append(
                {
                    "entity_id": entity_id,
                    "success": True,
                    "stdout": result.stdout,
                    "download_location": download_location,
                }
            )

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to download {entity_id}: {e.stderr if e.stderr else str(e)}"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})
        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout for {entity_id} (>{timeout} seconds)"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})

    # Summary
    successful_downloads = [r for r in results if r["success"]]
    failed_downloads = [r for r in results if not r["success"]]

    return {
        "success": len(failed_downloads) == 0,
        "total_requested": len(entity_ids),
        "successful": len(successful_downloads),
        "failed": len(failed_downloads),
        "download_location": download_location,
        "results": results,
        "errors": errors if errors else None,
    }

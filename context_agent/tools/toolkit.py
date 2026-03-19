
"""
This file provides a 'toolkit' for an AI agent that needs to work
with a local codebase under a 5000-token context limit.

The tools are designed to be:
- Safe: file access is restricted to a repository root folder, and command
  execution is restricted to an allowlist.
- Context-friendly: results are bounded (max hits, max lines, truncation) so
  the agent does not accidentally dump huge outputs into the LLM context.

The four main tools are:
1) `list_dir` — inspect the directory structure of the repo.
2) `read_file` — read only a small line-range from a file (prevents huge context).
3) `rg_search` — search the repo with ripgrep and return file/line hits.
4) `run_cmd` — run allowed build/test commands and capture stdout/stderr.

These tools form the foundation of a 'codebase-aware' agent. The LLM can decide
what to look at next, while the tools provide the actual file content and command
outputs needed to answer questions accurately.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List
import os
import subprocess


# Data structures
# ---------------
@dataclass(frozen=True)
class SearchHit:
    """
    Represents one match found by ripgrep (rg).

    Attributes:
        path: Which file the match is in.
        line: Line number where the match occurred.
        text: The text content of the matched line.
    """
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class CommandResult:
    """
    Represents the result of running a command (e.g., cmake/ctest) locally.

    Attributes:
        command: The command that was executed as a list of tokens.
                 Example: ["cmake", "-S", ".", "-B", "build"]
        cwd: The directory where the command was run.
        returncode: Process exit code (0 usually means success).
        stdout: Captured standard output (normal program output).
        stderr: Captured standard error (warnings/errors often appear here).
    """
    command: List[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str


# Utility helpers
# ---------------
def _safe_resolve(base_dir: str | Path, target: str | Path) -> Path:
    """
    Safely resolve `target` as a path inside `base_dir`.

    Why this exists:
    - The agent should be allowed to read only within the repository folder.
    - A malicious or accidental path should be blocked.

    Args:
        base_dir: The allowed root folder (the repository root).
        target: A relative path within the repository (file or directory).

    Returns:
        A path that is guaranteed to be inside base_dir.

    Raises:
        ValueError: If the resolved path would escape base_dir.
    """
    base = Path(base_dir).resolve()
    full = (base / target).resolve()

    # Ensure `full` is inside `base`
    if base not in full.parents and full != base:
        raise ValueError(f"Path escapes base_dir: {full}")
    return full


def _clamp(n: int, lo: int, hi: int) -> int:
    """
    Clamp an integer into the inclusive range [lo, hi].

    Example:
        _clamp(-5, 1, 10) -> 1
        _clamp(20, 1, 10) -> 10

    This is used primarily to keep requested line ranges in-bounds when reading files.
    """
    return max(lo, min(hi, n))


# Tool 1: list_dir
# ----------------
def list_dir(repo_root: str | Path, rel_path: str = ".", max_entries: int = 200) -> List[str]:
    """
    List the contents of a directory inside the repository.

    This tool helps the agent understand the repo structure (what folders exist,
    where tests live, where headers are, etc.).

    Args:
        repo_root: Path to the repository root folder.
        rel_path: Directory path relative to repo_root (default: ".").
        max_entries: Maximum number of entries to return. If there are more entries,
                     the result is truncated to avoid overly large outputs.

    Returns:
        A list of directory entries (names only). Directories have a trailing "/".
        Example: ["include/", "tests/", "CMakeLists.txt", "... (truncated)"]

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If rel_path points to a file instead of a directory.
        ValueError: If rel_path would escape the repo root.
    """
    root = Path(repo_root).resolve()
    target = _safe_resolve(root, rel_path)

    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    entries = []
    for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = p.name + ("/" if p.is_dir() else "")
        entries.append(name)
        # Limit the output
        if len(entries) >= max_entries:
            entries.append("... (truncated)")
            break
    return entries


# Tool 2: read_file
# -----------------
def read_file(
    repo_root: str | Path,
    rel_path: str,
    start_line: int,
    end_line: int,
    max_lines: int = 400,
) -> Tuple[str, int, int]:
    """
    Read a specific line range from a file inside the repository.

    This tool allows you to fetch only the relevant parts of a file rather than loading
    the entire file into the LLM context.

    Args:
        repo_root: Path to the repository root folder.
        rel_path: File path relative to the repo root (e.g., "include/nlohmann/json.hpp").
        start_line: 1-indexed start line to read.
        end_line: 1-indexed end line to read.
        max_lines: Maximum number of lines to return even if the user requests more (prevents huge context dumps).

    Returns:
        A tuple: (text, actual_start_line, actual_end_line)
        - text is the file content for that range
        - actual_start_line/actual_end_line reflect any clamping/truncation applied

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If rel_path points to a directory.
        ValueError: If rel_path would escape the repo root.
    """
    root = Path(repo_root).resolve()
    full = _safe_resolve(root, rel_path)

    if not full.exists():
        raise FileNotFoundError(f"File not found: {full}")
    if not full.is_file():
        raise IsADirectoryError(f"Path is not a file: {full}")

    # Read lines
    with full.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    if total == 0:
        return ("", 0, 0)

    # Clamp the requested range
    start_line = _clamp(start_line, 1, total)
    end_line = _clamp(end_line, 1, total)

    if end_line < start_line:
        start_line, end_line = end_line, start_line

    # Enforce max_lines to avoid huge context
    if (end_line - start_line + 1) > max_lines:
        end_line = start_line + max_lines - 1

    # Convert to 0-index slices
    chunk = lines[start_line - 1 : end_line]
    text = "".join(chunk)
    return (text, start_line, end_line)


# Tool 3: rg_search
# -----------------
def rg_search(
    repo_root: str | Path,
    query: str,
    glob: Optional[str] = None,
    max_hits: int = 50,
) -> List[SearchHit]:
    """
    Search the repository using ripgrep (rg) and return the matches/hits.

    This is the agent’s search engine for the codebase. It allows the agent to
    locate candidate files/lines before it decides which snippets to read.

    Args:
        repo_root: Path to the repository root folder.
        query: A ripgrep regex pattern. (If you want literal matching, you can adapt
               this to add `--fixed-strings` or escape the query.)
        glob: Optional glob filter to restrict which files are searched.
              Examples:
                  "*.hpp"
                  "{*.hpp,*.cpp}"
                  "CMakeLists.txt"
        max_hits: Maximum number of matches/hits to return.

    Returns:
        A list of SearchHit objects. If no matches/hits are found, returns an empty list.

    Raises:
        RuntimeError: If ripgrep returns an unexpected error code.
    """
    root = Path(repo_root).resolve()

    cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
    if glob:
        cmd += ["--glob", glob]
    cmd += [query, str(root)]

    # Use subprocess and capture output
    proc = subprocess.run(cmd, capture_output=True, text=True)

    hits: List[SearchHit] = []
    if proc.returncode not in (0, 1):
        # 0 = matches found, 1 = no matches, other = error
        raise RuntimeError(f"rg failed (code {proc.returncode}): {proc.stderr.strip()}")

    if proc.returncode == 1:
        return []

    # Parse lines like: path:line:text
    for line in proc.stdout.splitlines():
        # Split into 3 parts only; file paths can contain ':' rarely, but usually fine in Linux.
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path_str, line_str, text = parts
        try:
            ln = int(line_str)
        except ValueError:
            continue

        # Store path relative to repo root if possible
        try:
            rel = str(Path(path_str).resolve().relative_to(root))
        except Exception:
            rel = path_str

        hits.append(SearchHit(path=rel, line=ln, text=text))
        if len(hits) >= max_hits:
            break

    return hits


# Tool 4: run_cmd
# ---------------
_ALLOWED_CMDS = {"cmake", "ctest", "make", "ninja", "python3"}

def run_cmd(
    repo_root: str | Path,
    command: Sequence[str],
    cwd_rel: str = ".",
    timeout_sec: int = 600,
) -> CommandResult:
    """
    Run an allowed command inside the repository and capture stdout/stderr.

    This is how the agent can build and test the project. For safety,
    only a small allowlist of executables is permitted.

    Args:
        repo_root: Path to the repository root folder.
        command: Command tokens to run, e.g. ["cmake", "-S", ".", "-B", "build"].
                 The first token must be an allowed executable.
        cwd_rel: Working directory relative to repo_root. Default "." (repo root).
        timeout_sec: Maximum allowed runtime (seconds). If exceeded, subprocess.run
                     raises a TimeoutExpired exception.

    Returns:
        A CommandResult containing command, cwd, returncode, stdout, and stderr.

    Raises:
        ValueError: If command is empty, not allowed, or cwd escapes the repo.
        subprocess.TimeoutExpired: If the command exceeds timeout_sec.
    """
    if not command:
        raise ValueError("Empty command")

    exe = Path(command[0]).name  # handles "/usr/bin/cmake" -> "cmake"
    if exe not in _ALLOWED_CMDS:
        raise ValueError(f"Command not allowed: {exe}. Allowed: {sorted(_ALLOWED_CMDS)}")

    root = Path(repo_root).resolve()
    cwd = _safe_resolve(root, cwd_rel)

    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
    )

    return CommandResult(
        command=list(command),
        cwd=str(cwd),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )

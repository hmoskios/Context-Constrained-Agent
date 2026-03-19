
"""
This file contrains the retrieval layer for a context-constrained codebase agent.

The purpose of this module is to answer the question:
    "Given a user query (or compiler output), which pieces of the repo
     should we load into the LLM context?"

Key ideas:
- We NEVER load entire files.
- We prefer "high-signal" snippets:
    * symbol definitions (class/function definitions)
    * error locations from build/test output
    * minimal context around matches (line windows)
- All outputs are bounded (max hits, max snippets, max lines) to avoid context blowups.

The retriever works by:
1. Searching for likely matches with ripgrep (`rg`)
2. Reading only small line windows around the best matches
3. Ranking and de-duplicating those snippets
4. Returning only a few snippets so the final LLM prompt stays small
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import re
from context_agent.tools.toolkit import SearchHit, rg_search, read_file


# Data structures
# ---------------
@dataclass(frozen=True)
class Snippet:
    """
    A small chunk of text extracted from a repo file.

    Attributes:
        path: Repo file path.
        start_line: 1-indexed start line of this snippet.
        end_line: 1-indexed end line of this snippet.
        text: The snippet content (typically a small window of lines).
        reason: Why we fetched this snippet (e.g., "symbol_def", "build_error", "keyword_match").
        score: A simple numeric score used for ranking snippets (higher = more important).
    """
    path: str
    start_line: int
    end_line: int
    text: str
    reason: str
    score: float


# Helpers: query parsing
# ----------------------
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

def extract_candidate_symbols(query: str) -> List[str]:
    """
    Extract "symbol-like" identifier tokens from a user query.

    Examples:
        "What does json_pointer do?" -> ["What", "does", "json_pointer", "do"]
    We then filter down to the likely symbols.

    Heuristic:
    - Keep tokens that contain '_' OR are CamelCase (join multiple words w/o spaces) OR look like identifiers.
    - Drop common stop words (e.g., "what", "does", "do", etc.).
    """
    stop = {
        "what", "does", "do", "and", "or", "the", "a", "an", "is", "are", "where",
        "defined", "class", "function", "file", "why", "build", "failing", "tests",
        "run", "explain", "responsible", "for", "how", "to", "in", "of"
    }

    tokens = _SYMBOL_RE.findall(query)
    out: List[str] = []
    for t in tokens:
        tl = t.lower()
        if tl in stop:
            continue
        # Prefer identifier-looking tokens
        if "_" in t or t[0].isupper() or (t.isidentifier() and len(t) >= 3):
            out.append(t)

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def build_definition_patterns(symbol: str) -> List[str]:
    """
    Build ripgrep patterns that are likely to find a definition of `symbol`.

    For C++ code, a symbol may appear in a class definition, struct definition,
    enum definition, or method declaration/definition.
    """
    escaped = re.escape(symbol)
    return [
        rf"\bclass\s+{escaped}\b",
        rf"\bstruct\s+{escaped}\b",
        rf"\benum\s+(class\s+)?{escaped}\b",
        rf"\b{escaped}\s*::\s*\w+\s*\(",   # member function definition
        rf"\b{escaped}\s*\(",              # function or constructor call/decl
        rf"\b{escaped}\b",
    ]


def path_score_bonus(path: str) -> float:
    """
    Assign a small ranking bonus based on the file path.

    This helps prefer more useful implementation/detail files over broader or
    generated/single-include files.

    Heuristics:
    - Prefer files under include/nlohmann/detail/
    - Slightly penalize single_include/ because those files are often very large
    - Slightly penalize forward-declaration files like *_fwd.hpp
    """
    bonus = 0.0

    if "include/nlohmann/detail/" in path:
        bonus += 3.0
    if path.startswith("single_include/"):
        bonus -= 2.0
    if path.endswith("_fwd.hpp"):
        bonus -= 2.0
    if path.endswith("json_pointer.hpp"):
        bonus += 2.0

    return bonus


# Helpers: build output parsing
# -----------------------------
_ERROR_LOC_RE = re.compile(r"(?P<path>(?:/|~)?[^\s:]+?\.(?:c|cc|cpp|cxx|h|hpp|hh|hxx)):(?P<line>\d+)(?::\d+)?")

def parse_error_locations(build_output: str, repo_root: str | Path) -> List[Tuple[str, int]]:
    """
    Parses compiler/test output and extracts the (path, line) locations where errors occur.

    Many compiler errors contain paths like:
        /home/user/project/file.cpp:123: error: ...
        include/foo.hpp:42:17: error: ...

    This function finds those locations and converts absolute paths to paths
    relative to the repository root when possible.

    Args:
        build_output: Combined stdout/stderr from a build or test run.
        repo_root: Repo root used to convert absolute paths to relative.

    Returns:
        A list of (rel_path, line_number) tuples.
    """
    root = Path(repo_root).resolve()
    locs: List[Tuple[str, int]] = []

    for m in _ERROR_LOC_RE.finditer(build_output):
        path_str = m.group("path")
        line_str = m.group("line")
        try:
            line_no = int(line_str)
        except ValueError:
            continue

        # Normalize to repo-relative when possible
        p = Path(path_str).expanduser()
        try:
            p_resolved = p.resolve()
        except Exception:
            p_resolved = p

        try:
            rel = str(p_resolved.relative_to(root))
        except Exception:
            # It might already be relative
            rel = path_str

        locs.append((rel, line_no))

    # De-duplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, int]] = []
    for item in locs:
        if item not in seen:
            uniq.append(item)
            seen.add(item)
    return uniq


# Core retrieval functions
# ------------------------
def snippet_from_hit(
    repo_root: str | Path,
    hit: SearchHit,
    before: int = 8,
    after: int = 40,
    max_lines: int = 220,
    reason: str = "keyword_match",
    score: float = 1.0,
) -> Snippet:
    """
    Convert a SearchHit into a bounded Snippet by reading a window of lines around the hit.

    Args:
        repo_root: Repo root path.
        hit: A SearchHit containing (path, line, text).
        before/after: How many lines of context to include before/after the hit line.
        max_lines: Safety cap to avoid large snippets.
        reason: Tag explaining why this snippet was fetched.
        score: Ranking score for this snippet.

    Returns:
        Snippet containing file path, line range, and text.
    """
    start = max(1, hit.line - before)
    end = hit.line + after
    text, actual_start, actual_end = read_file(
        repo_root=repo_root,
        rel_path=hit.path,
        start_line=start,
        end_line=end,
        max_lines=max_lines,
    )
    return Snippet(
        path=hit.path,
        start_line=actual_start,
        end_line=actual_end,
        text=text,
        reason=reason,
        score=score,
    )


def retrieve_for_query(
    repo_root: str | Path,
    user_query: str,
    *,
    max_snippets: int = 5,
    glob: str = "{*.hpp,*.h,*.cpp,*.cc,*.cxx,*.cmake,CMakeLists.txt}",
) -> List[Snippet]:
    """
    Retrieve a small set of relevant snippets for a natural-language user query.

    Strategy:
    1) Extract likely symbols from the query (e.g., json_pointer).
    2) Search for definition-like patterns first (class/struct/enum).
    3) Fall back to a plain keyword search if needed.
    4) Convert top hits into snippets (small windows of lines).
    5) Return at most `max_snippets`.

    This keeps context small while still finding high-signal code sections.

    Args:
        repo_root: Path to the repo root.
        user_query: Natural-language question from the user.
        max_snippets: Maximum number of snippets to return.
        glob: File globs to search. Defaults to headers/sources/CMake.

    Returns:
        A ranked list of Snippet objects.
    """
    symbols = extract_candidate_symbols(user_query)

    snippets: List[Snippet] = []

    # 1) Try definition patterns for extracted symbols
    for sym in symbols:
        patterns = build_definition_patterns(sym)
        for pat in patterns:
            hits = rg_search(repo_root, pat, glob=glob, max_hits=10)
            for i, h in enumerate(hits[:3]):
                # Definition hits are high value
                snippets.append(snippet_from_hit(
                    repo_root,
                    h,
                    reason=f"symbol_def:{sym}",
                    score=10.0 - i,  # slight preference for earlier hits
                ))
            if snippets:
                # Once we have something good, we can stop early for this symbol
                break

        if len(snippets) >= max_snippets:
            break

    # 2) If we still have nothing, do a broader keyword search using the raw query
    if not snippets:
        # Pick the longest candidate symbol if we have one; else search the whole query.
        if symbols:
            key = max(symbols, key=len)
        else:
            key = user_query.strip()

        hits = rg_search(repo_root, re.escape(key), glob=glob, max_hits=20)
        for i, h in enumerate(hits[:max_snippets]):
            snippets.append(snippet_from_hit(
                repo_root,
                h,
                reason="keyword_fallback",
                score=5.0 - i,
            ))

    # 3) Rank + deduplicate snippets by (path, start, end)
    return rank_and_dedupe(snippets, max_snippets=max_snippets)


def retrieve_for_build_output(
    repo_root: str | Path,
    build_output: str,
    *,
    max_snippets: int = 5,
) -> List[Snippet]:
    """
    Retrieve snippets driven by compiler/test output.

    Strategy:
    1) Extract file:line locations from build output.
    2) Read small windows around those lines.
    3) Mark these as high-priority snippets (reason="build_error").

    Args:
        repo_root: Repo root.
        build_output: stdout/stderr combined from build/test run.
        max_snippets: Maximum number of snippets to return.

    Returns:
        List of Snippets, typically centered on error lines.
    """
    locs = parse_error_locations(build_output, repo_root)
    snippets: List[Snippet] = []

    # Prefer first few error locations (often most relevant)
    for i, (path, line_no) in enumerate(locs[:max_snippets]):
        # Use SearchHit-like info for convenience
        fake_hit = SearchHit(path=path, line=line_no, text="")
        snippets.append(snippet_from_hit(
            repo_root,
            fake_hit,
            before=20,
            after=40,
            reason="build_error",
            score=100.0 - i,  # very high priority
        ))

    return rank_and_dedupe(snippets, max_snippets=max_snippets)


def snippets_overlap(a: Snippet, b: Snippet) -> bool:
    """
    Return True if two snippets come from the same file and their line ranges overlap.

    This helps us avoid returning near-duplicate snippets from the same region.
    """
    if a.path != b.path:
        return False
    return not (a.end_line < b.start_line or b.end_line < a.start_line)
    

def rank_and_dedupe(snippets: List[Snippet], *, max_snippets: int) -> List[Snippet]:
    """
    Simple cleanup: rank snippets by score and remove duplicates.

    Duplicate definition:
        same (path, start_line, end_line)

    Args:
        snippets: Candidate snippets.
        max_snippets: Return at most this many.

    Returns:
        Ranked, de-duplicated snippets.
    """
    # Sort by score (descending order), then by path for stability
    snippets_sorted = sorted(snippets, key=lambda s: (-s.score, s.path, s.start_line))

    kept: List[Snippet] = []
    for snippet in snippets_sorted:
        duplicate = False
        for existing in kept:
            if snippets_overlap(snippet, existing):
                duplicate = True
                break
        # Remove duplicates
        if duplicate:
            continue
        # Append the remaining snippets to kept
        kept.append(snippet)
        if len(kept) >= max_snippets:
            break
    return kept

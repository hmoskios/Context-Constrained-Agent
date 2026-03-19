
"""
This file handles context budgeting for the agent.

Since the agent to stay within a strict 5,000-token limit, this 
module provides simple utilities for:

1. Estimating token usage.
2. Trimming large text blocks.
3. Selecting which snippets fit into the final prompt.
4. Dropping lower-priority content when necessary.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from drift_agent.retrieval.retriever import Snippet


# Data structures
# ---------------
@dataclass(frozen=True)
class BudgetItem:
    """
    Represents one item that may be included in the final LLM context.

    Attributes:
        label: Name for the item such as "system_prompt", "user_query", or "snippet:json_pointer.hpp".
        text: The actual text content that may go into the prompt.
        priority: Numeric priority used to decide what to keep when the context is full. Higher numbers are more important.
        token_estimate: Approximate number of tokens this item uses.
    """
    label: str
    text: str
    priority: int
    token_estimate: int


# Main budgeting functions
# ------------------------
def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a piece of text.

    This implementation uses the approximation: tokens ≈ characters / 4.

    Why this is acceptable:
    - The assignment requires programmatic enforcement of the limit.
    - Exact tokenization is nice, but not strictly necessary.
    - This estimate is simple, fast, and easy to explain in the design document.

    Args:
        text: Input text.

    Returns:
        Approximate token count as an integer.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def make_budget_item(label: str, text: str, priority: int) -> BudgetItem:
    """
    Create a BudgetItem and automatically compute its token estimate.

    Args:
        label: Name of the item.
        text: Text content of the item.
        priority: Higher number means higher importance.

    Returns:
        A BudgetItem object.
    """
    return BudgetItem(
        label=label,
        text=text,
        priority=priority,
        token_estimate=estimate_tokens(text),
    )


def trim_text_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Trim text so its estimated token count stays within `max_tokens`.

    This function uses character count as a proxy for tokens. If the text is too
    large, it truncates from the end.

    Args:
        text: Input text to trim.
        max_tokens: Maximum allowed estimated tokens.

    Returns:
        Trimmed text. If truncation occurs, an explanatory marker is added.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * 4
    trimmed = text[:max_chars]

    return trimmed + "\n\n...[truncated to fit context budget]"


def trim_command_output(text: str, max_tokens: int = 800) -> str:
    """
    Trim command output to a manageable size for inclusion in the LLM context.

    For command output, the most relevant information is often near the end
    (especially errors and test failures). So this function keeps the last
    portion rather than the first.

    Args:
        text: Command stdout/stderr text.
        max_tokens: Maximum estimated tokens to keep.

    Returns:
        Trimmed command output, usually preserving the end of the log.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * 4
    trimmed = text[-max_chars:]

    return "[Showing last portion of command output]\n\n" + trimmed


def snippet_to_text(snippet: Snippet) -> str:
    """
    Convert a Snippet object into a formatted text block for the prompt.

    Including path and line numbers is important because:
    - It gives the LLM source grounding.
    - It makes the final answer easier to trace back to code.
    - It helps with debugging.

    Args:
        snippet: Snippet selected by the retriever.

    Returns:
        A formatted string representation of the snippet.
    """
    return (
        f"File: {snippet.path}\n"
        f"Lines: {snippet.start_line}-{snippet.end_line}\n"
        f"Reason: {snippet.reason}\n"
        f"Score: {snippet.score:.2f}\n\n"
        f"{snippet.text}"
    )


def total_tokens(items: Sequence[BudgetItem]) -> int:
    """
    Compute the total estimated token count for a sequence of BudgetItems.

    Args:
        items: Items to sum.

    Returns:
        Sum of token estimates.
    """
    return sum(item.token_estimate for item in items)


def select_snippets_to_fit(
    snippets: Sequence[Snippet],
    max_tokens: int,
) -> List[Snippet]:
    """
    Select as many snippets as possible without exceeding a token budget.

    Strategy:
    - Sort snippets by score (descending).
    - Add snippets one by one until the budget would be exceeded.
    - Stop when the next snippet would not fit.

    Args:
        snippets: Candidate snippets.
        max_tokens: Maximum token budget available for snippets.

    Returns:
        A list of snippets that fit within the budget.
    """
    sorted_snippets = sorted(snippets, key=lambda s: s.score, reverse=True)

    selected: List[Snippet] = []
    used_tokens = 0

    for snippet in sorted_snippets:
        snippet_text = snippet_to_text(snippet)
        snippet_tokens = estimate_tokens(snippet_text)
        if used_tokens + snippet_tokens > max_tokens:
            continue
        selected.append(snippet)
        used_tokens += snippet_tokens

    return selected


def build_budgeted_items(
    system_prompt: str,
    user_query: str,
    snippets: Sequence[Snippet],
    command_output: str | None = None,
    memory_summary: str | None = None,
    total_budget_tokens: int = 5000,
) -> List[BudgetItem]:
    """
    Build a final list of BudgetItems that fit within the total context budget.

    Priority strategy:
    - system prompt: highest priority
    - user query: highest priority
    - memory summary: medium-high priority
    - snippets: high priority, selected by snippet score
    - command output: medium priority, trimmed if needed

    Workflow:
    1. Add required items first
    2. Reserve room for snippets
    3. Add snippets that fit
    4. Add trimmed command output if space remains

    Args:
        system_prompt: Instruction prompt for the model.
        user_query: The user's current question.
        snippets: Retrieved snippets from the repo.
        command_output: Optional stdout/stderr from a command.
        memory_summary: Optional short summary of earlier steps or context.
        total_budget_tokens: Maximum total context budget.

    Returns:
        A list of BudgetItems that should be packed into the prompt.
    """
    items: List[BudgetItem] = []

    # Always include these
    system_item = make_budget_item("system_prompt", system_prompt, priority=100)
    user_item = make_budget_item("user_query", user_query, priority=100)

    items.extend([system_item, user_item])

    if memory_summary:
        memory_item = make_budget_item("memory_summary", memory_summary, priority=80)
        items.append(memory_item)
    used = total_tokens(items)
    remaining = max(0, total_budget_tokens - used)

    # Reserve part of the remaining budget for command output if it exists
    command_budget = 0
    if command_output:
        command_budget = min(800, remaining // 3)
    snippet_budget = max(0, remaining - command_budget)
    selected_snippets = select_snippets_to_fit(snippets, max_tokens=snippet_budget)
    for snippet in selected_snippets:
        snippet_item = make_budget_item(
            label=f"snippet:{snippet.path}",
            text=snippet_to_text(snippet),
            priority=90,
        )
        items.append(snippet_item)
    used = total_tokens(items)
    remaining = max(0, total_budget_tokens - used)

    if command_output and remaining > 0:
        trimmed_output = trim_command_output(command_output, max_tokens=min(remaining, 800))
        command_item = make_budget_item("command_output", trimmed_output, priority=70)
        items.append(command_item)

    return items

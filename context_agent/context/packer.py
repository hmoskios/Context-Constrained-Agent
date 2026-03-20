
"""
This module turns budgeted context items into the final prompt text sent to the LLM.

It's job is to:
- Accept the items selected by the context budgeter.
- Organize them into a readable prompt.
- Produce a final user-facing prompt string.

This separation is useful because:
- `budget.py` decides what fits.
- `packer.py` decides how to format it.
"""

from __future__ import annotations
from typing import Sequence
from context_agent.context.budget import BudgetItem, total_tokens


def format_items_for_prompt(items: Sequence[BudgetItem]) -> str:
    """
    Formats budgeted items into a structured prompt body.

    Each item is placed under a labeled section so the LLM can clearly distinguish:
    - The user's question
    - Retrieved code snippets
    - Command output
    - Memory/context summary

    Args:
        items:
            Context items already selected to fit within budget.

    Returns:
        A single formatted prompt string.
    """
    sections = []

    for item in items:
        sections.append(
            f"## {item.label}\n"
            f"{item.text}\n"
        )

    return "\n".join(sections)


def build_final_prompt(items: Sequence[BudgetItem]) -> str:
    """
    Builds the final prompt string to send to the LLM.

    This function adds a short instruction header and then appends all selected
    context sections.

    Args:
        items:
            Budgeted prompt items.

    Returns:
        Final prompt text.
    """
    prompt_body = format_items_for_prompt(items)
    token_count = total_tokens(items)

    return (
        "You are a codebase assistant working under a strict context budget.\n"
        "Answer the user's question using the provided repository snippets and command output.\n"
        "Be precise, grounded in the provided evidence, and mention file paths and line ranges when relevant.\n"
        f"Estimated total prompt tokens: {token_count}\n\n"
        f"{prompt_body}"
    )

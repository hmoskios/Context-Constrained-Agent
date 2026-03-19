
"""
This module defines the core data structures shared across the agent system.

It centralizes:

1) The BuildPlan object:
        A structured representation of how the agent should interact with the
        local codebase through command execution (e.g., cmake, make, ctest).
2) The AgentState object:
        A shared, mutable state object used by LangGraph to coordinate the agent's
        workflow. Each node reads from and writes to this state as it processes a query.

The AgentState is intentionally flexible (TypedDict with total=False) so that
nodes only need to populate the fields they are responsible for, without
requiring a fully populated state at every step.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, TypedDict
from context_agent.retrieval.retriever import Snippet


@dataclass(frozen=True)
class BuildPlan:
    """
    Represents a sequence of commands the agent should run for a build/debug query.

    Attributes:
        plan_name:
            Short label describing the plan.
        commands:
            A list of commands to run, in order.
            Each command is itself a list of strings.
        cwd_rels:
            A list of working directories, one per command.
        explanation:
            Human-readable explanation of why this plan was selected.
    """
    plan_name: str
    commands: List[List[str]]
    cwd_rels: List[str]
    explanation: str


class AgentState(TypedDict, total=False):
    """
    Shared state object for the LangGraph workflow.

    LangGraph nodes read values from this shared state and return partial
    updates that get merged back into it.

    Keys:
        repo_root: Path to the local repository being analyzed.
        user_query: The user's current natural-language question.
        query_type: Query classification label: "understanding" or "build".
        build_plan: Selected BuildPlan for build/debug queries.
        combined_output: Combined command execution log from build/test steps.
        snippets: Retrieved code snippets relevant to the current query.
        prompt: Final packed prompt to send to Gemini.
        answer: Final model response text.
    """
    repo_root: str
    user_query: str
    query_type: Literal["understanding", "build"]
    build_plan: BuildPlan
    combined_output: str
    snippets: List[Snippet]
    prompt: str
    answer: str

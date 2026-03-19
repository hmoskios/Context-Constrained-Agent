
"""
This module contains routing logic for the LangGraph workflow.

In LangGraph, conditional edges use router functions that inspect the
current graph state and decide which node should run next.

For this project, the main routing decision happens after query classification:
- Understanding query -> go to the understanding-retrieval branch
- Build/debug query -> go to the build/debug branch
"""

from __future__ import annotations
from context_agent.state import AgentState


def route_after_classification(state: AgentState) -> str:
    """
    Routes the graph after the query has been classified.

    This function checks the `query_type` field in the shared graph state
    and returns the name of the next node that should execute.

    Expected query types:
        - "understanding"
        - "build"

    Returns:
        The name of the next node:
            - "retrieve_understanding" for code-understanding queries
            - "choose_build_plan" for build/debug queries

    Raises:
        ValueError:
            If query_type is missing or has an unexpected value.
    """
    query_type = state.get("query_type")

    if query_type == "understanding":
        return "retrieve_understanding"

    if query_type == "build":
        return "choose_build_plan"

    raise ValueError(f"Unexpected query_type in graph state: {query_type!r}")

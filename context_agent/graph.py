
"""
This module builds and compiles the LangGraph workflow for the codebase agent.

It connects:
- Shared graph state from agent.py
- Node functions from nodes.py
- Routing logic from edges.py

Graph structure:

    START
      ↓
    classify_query
      ↓
    ┌───────────────────────────────┐
    │ if understanding query        │
    │   → retrieve_understanding    │
    │   → build_prompt              │
    │   → call_llm                  │
    └───────────────────────────────┘
    ┌───────────────────────────────┐
    │ if build/debug query          │
    │   → choose_build_plan         │
    │   → run_build_plan            │
    │   → retrieve_build_snippets   │
    │   → build_prompt              │
    │   → call_llm                  │
    └───────────────────────────────┘
      ↓
     END
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from context_agent.state import AgentState
from context_agent.edges import route_after_classification
from context_agent.nodes import (
    classify_query_node,
    retrieve_understanding_node,
    choose_build_plan_node,
    run_build_plan_node,
    retrieve_build_snippets_node,
    build_prompt_node,
    call_llm_node,
)


def build_agent_graph():
    """
    Builds and compiles the LangGraph workflow for the agent.

    Workflow:
    1. Classify the query.
    2. Route to either:
       - the understanding branch.
       - the build/debug branch.
    3. Build the final prompt.
    4. Call Gemini.
    5. Return the final graph.

    Returns:
        A compiled LangGraph application.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve_understanding", retrieve_understanding_node)
    graph.add_node("choose_build_plan", choose_build_plan_node)
    graph.add_node("run_build_plan", run_build_plan_node)
    graph.add_node("retrieve_build_snippets", retrieve_build_snippets_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("call_llm", call_llm_node)

    # Start the workflow by classifying the query
    graph.add_edge(START, "classify_query")

    # Route to the correct branch after classification
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "retrieve_understanding": "retrieve_understanding",
            "choose_build_plan": "choose_build_plan",
        },
    )

    # Understanding branch
    graph.add_edge("retrieve_understanding", "build_prompt")

    # Build/debug branch
    graph.add_edge("choose_build_plan", "run_build_plan")
    graph.add_edge("run_build_plan", "retrieve_build_snippets")
    graph.add_edge("retrieve_build_snippets", "build_prompt")

    # Shared final path
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", END)

    # Compile the graph object
    return graph.compile()


# Compile the graph once when the module is imported so it does not need to be rebuilt every query
AGENT_GRAPH = build_agent_graph()

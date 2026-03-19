
"""
This module contains the LangGraph node functions for the agent workflow.

Each node is responsible for one specific step in the overall process.
The nodes read values from the shared graph state and return partial updates.

Current node responsibilities:
- Classify the query.
- Retrieve snippets for understanding queries.
- Choose a build plan for build/debug queries.
- Run build/test commands.
- Retrieve snippets from build/test output.
- Build the final prompt.
- Call Gemini to produce the final answer.
"""

from __future__ import annotations
from context_agent.retrieval.retriever import (retrieve_for_query, retrieve_for_build_output)
from context_agent.context.budget import build_budgeted_items
from context_agent.context.packer import build_final_prompt
from context_agent.llm.gemini_client import ask_gemini
from context_agent.agent import (looks_like_build_query, choose_build_plan, run_build_plan)
from context_agent.state import AgentState


# Nodes #1: Query classification
# ------------------------------
def classify_query_node(state: AgentState) -> dict:
    """
    Determines whether the user's query is a code-understanding query
    or a build/debug query.

    This node does not execute any tools. It only classifies the query
    so the graph can route to the correct branch.

    Args:
        state:
            The current shared graph state. Must contain: user_query

    Returns:
        A partial state update with:
            query_type: "understanding" or "build"
    """
    user_query = state["user_query"]

    if looks_like_build_query(user_query):
        print("[Agent] Query classified as build/debug.")
        return {"query_type": "build"}
    else:
        print("[Agent] Query classified as code-understanding.")
        return {"query_type": "understanding"}


# Nodes #2: Handling understanding queries
# ----------------------------------------
def retrieve_understanding_node(state: AgentState) -> dict:
    """
    Retrieves relevant code snippets for a code-understanding query.

    This node is used for questions such as:
    - What does json_pointer do?
    - Where is a class defined?
    - Which files handle serialization?

    Args:
        state:
            The current shared graph state. Must contain:
            - repo_root
            - user_query

    Returns:
        A partial state update with:
            snippets: List of retrieved code snippets
    """
    repo_root = state["repo_root"]
    user_query = state["user_query"]

    print("[Agent] Handling code-understanding query...")
    print("[Agent] Retrieving relevant snippets...")

    snippets = retrieve_for_query(repo_root, user_query, max_snippets=5)

    print(f"[Agent] Retrieved {len(snippets)} snippets.")
    return {"snippets": snippets}


# Nodes #3-5: Handling build/debug queries
# ----------------------------------------
def choose_build_plan_node(state: AgentState) -> dict:
    """
    Chooses a BuildPlan for a build/debug query.

    This node interprets the user's request and decides which command
    sequence should be executed.

    Args:
        state:
            The current shared graph state. Must contain:
            - user_query

    Returns:
        A partial state update with:
            build_plan: Selected BuildPlan object
    """
    user_query = state["user_query"]

    print("[Agent] Handling build/debug query...")
    plan = choose_build_plan(user_query)
    return {"build_plan": plan}


def run_build_plan_node(state: AgentState) -> dict:
    """
    Executes the selected BuildPlan and collect the resulting command output.

    Args:
        state:
            The current shared graph state. Must contain:
            - repo_root
            - build_plan

    Returns:
        A partial state update with:
            combined_output: Combined text log of all executed commands
    """
    repo_root = state["repo_root"]
    plan = state["build_plan"]

    combined_output = run_build_plan(repo_root, plan)
    return {"combined_output": combined_output}


def retrieve_build_snippets_node(state: AgentState) -> dict:
    """
    Retrieves relevant code snippets from build/test output.

    This node parses command output for file/line references and then
    loads targeted snippets around those locations.

    Args:
        state:
            The current shared graph state. Must contain:
            - repo_root
            - combined_output

    Returns:
        A partial state update with:
            snippets: List of build-related code snippets
    """
    repo_root = state["repo_root"]
    combined_output = state["combined_output"]

    print("[Agent] Retrieving snippets from build/test output...")
    snippets = retrieve_for_build_output(repo_root, combined_output, max_snippets=5)

    print(f"[Agent] Retrieved {len(snippets)} build-related snippets.")
    return {"snippets": snippets}


# Node #6: Prompt building
# ------------------------
def build_prompt_node(state: AgentState) -> dict:
    """
    Builds the final context-limited prompt for Gemini.

    This node works for both query types:
    - Understanding queries
    - Build/debug queries

    It selects the correct system prompt and passes the relevant snippets,
    command output, and memory summary into the context budget manager.

    Args:
        state: The current shared graph state.

    Returns:
        A partial state update with:
            prompt: Final prompt string to send to Gemini
    """
    user_query = state["user_query"]
    query_type = state["query_type"]
    snippets = state.get("snippets", [])

    if query_type == "understanding":
        system_prompt = (
            "You are an AI code assistant. "
            "Answer the user's question using ONLY the provided code snippets. "
            "Be precise and reference file paths and line numbers."
        )

        items = build_budgeted_items(
            system_prompt=system_prompt,
            user_query=user_query,
            snippets=snippets,
            memory_summary=None,
            command_output=None,
            total_budget_tokens=5000,
        )

    else:
        build_plan = state["build_plan"]
        combined_output = state["combined_output"]

        system_prompt = (
            "You are an AI debugging assistant. "
            "Analyze the command output and any relevant code snippets. "
            "Explain what happened, whether anything failed, and why. "
            "Reference file paths and line numbers when relevant."
        )

        memory_summary = f"Build plan used: {build_plan.plan_name}. {build_plan.explanation}"

        items = build_budgeted_items(
            system_prompt=system_prompt,
            user_query=user_query,
            snippets=snippets,
            command_output=combined_output,
            memory_summary=memory_summary,
            total_budget_tokens=5000,
        )

    print("[Agent] Building final prompt for Gemini...")
    prompt = build_final_prompt(items)
    return {"prompt": prompt}


# Node #7: Call the LLM
# ---------------------
def call_llm_node(state: AgentState) -> dict:
    """
    Sends the final prompt to Gemini and return the model's answer.

    Args:
        state:
            The current shared graph state. Must contain:
            - prompt

    Returns:
        A partial state update with:
            answer: Final model response text
    """
    prompt = state["prompt"]

    print("[Agent] Sending prompt to Gemini...")
    answer = ask_gemini(prompt)
    return {"answer": answer}

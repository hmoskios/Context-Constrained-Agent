
"""
This module implements the core agent logic.

It connects all parts of the system:
- Retrieval (retriever.py)
- Tools (toolkit.py)
- Context management (budget.py, packer.py)
- LLM interaction (gemini_client.py)

The agent:
1. Receives a user query.
2. Decides what type of query it is.
3. Retrieves relevant context (code snippets or build output).
4. Builds a context-limited prompt.
5. Sends it to Gemini.
6. Returns the answer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, TypedDict, Literal
from context_agent.tools.toolkit import run_cmd, CommandResult
from context_agent.state import BuildPlan


# Query classification
# --------------------
def looks_like_build_query(user_query: str) -> bool:
    """
    Query classifier: simple heuristic to detect build/debug queries.

    We check for keywords related to compilation, testing, or errors.

    Args:
        user_query:
            The user's input string.

    Returns:
        True if it looks like a build/debug query, False otherwise.
    """
    keywords = [
        "build", "compile", "cmake", "make", "test", "ctest", 
        "error", "fail", "failure", "configure", "debug"
    ]

    query_lower = user_query.lower()
    return any(k in query_lower for k in keywords)


# Build intent classification
# ---------------------------
def choose_build_plan(user_query: str) -> BuildPlan:
    """
    Choose a build/test/debug command sequence based on the user's query.

    Strategy:
    - If the query asks to configure, run cmake configure.
    - If it asks to build/compile, run the build command.
    - If it asks to test, run ctest.
    - If it asks for build + tests, run both.
    - If it asks why the build is failing, run build and inspect output.
    """
    q = user_query.lower()
    wants_configure = any(word in q for word in ["configure", "cmake"])
    wants_build = any(word in q for word in ["build", "compile", "make"])
    wants_test = any(word in q for word in ["test", "ctest", "run tests"])
    asks_why_failing = any(phrase in q for phrase in [
        "why is the build failing",
        "why is build failing",
        "build failing",
        "why is it failing",
        "explain the error",
    ])

    if (wants_build and wants_test) or "build and run tests" in q:
        return BuildPlan(
            plan_name="configure_build_test",
            commands=[
                ["cmake", "-S", ".", "-B", "build"],
                ["cmake", "--build", "build", "-j"],
                ["ctest", "--test-dir", "build", "--output-on-failure"],
            ],
            cwd_rels=[".", ".", "."],
            explanation="User asked to build and run tests, so the agent will configure, build, and test.",
        )

    if asks_why_failing:
        return BuildPlan(
            plan_name="build_then_debug",
            commands=[
                ["cmake", "--build", "build", "-j"],
            ],
            cwd_rels=["."],
            explanation="User asked why the build is failing, so the agent will run the build and inspect any errors.",
        )

    if wants_configure and not wants_build and not wants_test:
        return BuildPlan(
            plan_name="configure_only",
            commands=[
                ["cmake", "-S", ".", "-B", "build"],
            ],
            cwd_rels=["."],
            explanation="User asked about configuring the build.",
        )

    if wants_build and not wants_test:
        return BuildPlan(
            plan_name="build_only",
            commands=[
                ["cmake", "--build", "build", "-j"],
            ],
            cwd_rels=["."],
            explanation="User asked to build or compile the project.",
        )

    if wants_test:
        return BuildPlan(
            plan_name="test_only",
            commands=[
                ["ctest", "--test-dir", "build", "--output-on-failure"],
            ],
            cwd_rels=["."],
            explanation="User asked to run tests.",
        )

    # Safe fallback for generic debug/build queries
    return BuildPlan(
        plan_name="default_build_check",
        commands=[
            ["cmake", "--build", "build", "-j"],
        ],
        cwd_rels=["."],
        explanation="Defaulting to a build command for this build/debug-style query.",
    )


# Command execution helpers
# -------------------------
def format_command_result(result: CommandResult) -> str:
    """
    Converts a CommandResult into a readable text block.

    This makes it easier to pass command output into the context manager and LLM.
    """
    return (
        f"Command: {' '.join(result.command)}\n"
        f"Working directory: {result.cwd}\n"
        f"Return code: {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


def run_build_plan(repo_root: str, plan: BuildPlan) -> str:
    """
    Runs all commands in a BuildPlan and return a combined text log.

    Commands are run in sequence. If one command fails, the function stops early
    and returns the output collected so far, because that failure is usually the
    most relevant thing to debug.
    """
    print(f"[Agent] Selected build plan: {plan.plan_name}")
    print(f"[Agent] {plan.explanation}")

    logs: List[str] = [f"Selected build plan: {plan.plan_name}", plan.explanation]

    for command, cwd_rel in zip(plan.commands, plan.cwd_rels):
        print(f"[Agent] Running command: {' '.join(command)} (cwd={cwd_rel})")

        result = run_cmd(
            repo_root=repo_root,
            command=command,
            cwd_rel=cwd_rel,
            timeout_sec=600,
        )

        print(f"[Agent] Command finished with return code: {result.returncode}")
        
        logs.append(format_command_result(result))
        
        if result.returncode != 0:
            logs.append("Stopping early because a command failed.")
            print("[Agent] Stopping early because a command failed.")
            break

    return "\n\n".join(logs)


# Main entry point
# ----------------
def handle_query(repo_root: str, user_query: str) -> str:
    """
    Main entry point for the agent.

    This function invokes the compiled LangGraph workflow using the provided
    repository path and user query, then returns the final answer.

    Importing the graph inside the function avoids a circular import:
    - graph.py imports nodes.py
    - nodes.py imports helper functions from agent.py
    - agent.py should therefore avoid importing graph.py at module load time

    Args:
        repo_root: Path to the repo.
        user_query: The user's question.

    Returns:
        Final model response string.
    """
    from context_agent.graph import AGENT_GRAPH
    final_state = AGENT_GRAPH.invoke({"repo_root": repo_root, "user_query": user_query})
    return final_state["answer"]

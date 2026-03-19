
"""
This module provides a command-line interface (CLI) for the agent.

It allows you to:
1. Enter the path to a local code repository.
2. Type natural-language questions about that repository.
3. Get back answers from the agent.

The purpose of this file is to make the system easy to run, test, and 
demo without needing a web app or more advanced interface.
"""

from __future__ import annotations
from context_agent.agent import handle_query


def prompt_for_repo_path() -> str:
    """
    Asks the user for the repository path.

    Returns:
        The repository path as a string.
    """
    repo_path = input("Enter the path to the local repository: ").strip()
    return repo_path


def run_cli() -> None:
    """
    Runs the interactive command-line loop.

    Workflow:
    1. Ask for repo path once.
    2. Repeatedly ask the user for queries.
    3. Send each query to the agent.
    4. Print the result.
    5. Exit when the user types 'quit' or 'exit'.

    This function does not return anything. It just runs the app loop.
    """
    print("=== Codebase Agent CLI ===")
    print("Type 'quit' or 'exit' to stop.\n")

    repo_root = prompt_for_repo_path()

    while True:
        print()
        user_query = input("Ask a question about the repo: ").strip()

        if user_query.lower() in {"quit", "exit"}:
            print("Exiting agent.")
            break

        if not user_query:
            print("Please enter a non-empty query.")
            continue

        try:
            response = handle_query(repo_root, user_query)

            print("\n=== Agent Response ===\n")
            print(response)

        except Exception as exc:
            print("\n=== Error ===\n")
            print(f"Something went wrong: {exc}")


if __name__ == "__main__":
    run_cli()

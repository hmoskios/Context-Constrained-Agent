
"""
Smoke-test the high-level agent query workflow.

This script calls `handle_query` against a sample repository and asks where
`json_pointer` is defined. It prints the final agent response so the caller can
inspect whether retrieval, reasoning, and answer formatting work together.
"""

from context_agent.agent import handle_query

REPO = "/home/hmoskios/json"
query = "What does json_pointer do and where is it defined?"
response = handle_query(REPO, query)

print("\n=== AGENT RESPONSE ===\n")
print(response)

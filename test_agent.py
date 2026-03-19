
from drift_agent.agent import handle_query

REPO = "/home/hmoskios/json"
query = "What does json_pointer do and where is it defined?"
response = handle_query(REPO, query)

print("\n=== AGENT RESPONSE ===\n")
print(response)

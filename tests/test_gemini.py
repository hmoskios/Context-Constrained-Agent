
"""
Smoke-test the Gemini LLM client wrapper.

This script sends a short explanatory prompt through `ask_gemini` and prints
the returned text. It is intended to confirm that credentials, request wiring,
and basic response handling are functioning.
"""

from context_agent.llm.gemini_client import ask_gemini

prompt = "Explain in 2 sentences what a JSON parser library does."
response = ask_gemini(prompt)

print("Gemini response:")
print(response)

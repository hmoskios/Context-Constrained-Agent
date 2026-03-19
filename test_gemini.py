
from drift_agent.llm.gemini_client import ask_gemini

prompt = "Explain in 2 sentences what a JSON parser library does."
response = ask_gemini(prompt)

print("Gemini response:")
print(response)

"""
This module provides a small wrapper around the Gemini API.

Why this file exists:
- It keeps all Gemini-related code in one place.
- It gives us a clean function for sending prompts and getting back text.

Main responsibilities:
1. Read the Gemini API key from an environment variable.
2. Create a reusable Gemini client.
3. Send a prompt to the model.
4. Return the model's text response.
5. Raise helpful errors when something goes wrong.
"""

from __future__ import annotations
import os
from google import genai
from google.genai import types


DEFAULT_MODEL_NAME = "gemini-2.0-flash"


# Helper functions
# ----------------
def get_api_key() -> str:
    """
    Read the Gemini API key from the environment.

    We expect the key to be stored in the environment variable:
        GEMINI_API_KEY

    Why use an environment variable?
    - It keeps secrets out of the code.
    - It is standard practice for API credentials.
    - It makes the project easier to share safely.

    Returns:
        The API key as a string.

    Raises:
        EnvironmentError:
            If GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Please export your Gemini API key before running the agent."
        )
    return api_key


def get_client() -> genai.Client:
    """
    Configure the Gemini SDK by creating and returning a Google Gen AI client.

    This function should be called before making requests to the model.

    Returns:
        A configured genai.Client instance.
    """
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


def extract_response_text(response) -> str:
    """
    Extract text from a Gemini API response object.

    In most cases, `response.text` will already contain the model output.
    However, since API response objects can be messy, this helper handles 
    response parsing. It pulls the text from the object, strips whitespace, 
    and raises an error if no text is found.

    Args:
        response:
            Raw response object returned by the Gemini SDK.

    Returns:
        The model's text output as a string.

    Raises:
        ValueError:
            If no text could be extracted from the response.
    """
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    raise ValueError("Gemini response did not contain any text output.")


# Main function
# -------------
def ask_gemini(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.2,
    max_output_tokens: int = 1000,
) -> str:
    """
    Send a prompt to Gemini and return the generated text.

    This is the main function the rest of the project will call.

    Args:
        prompt:
            The full prompt string to send to Gemini.
        model_name:
            Gemini model name to use.
        temperature:
            Controls how random/creative the output is.
            Lower values are more deterministic.
        max_output_tokens:
            Maximum length of the generated response.

    Returns:
        The model's response text.

    Raises:
        RuntimeError:
            If the Gemini API call fails for any reason.
        ValueError:
            If the response does not contain usable text.
    """
    try:
        client = get_client()

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        return extract_response_text(response)

    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

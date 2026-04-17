"""Client wrapper for LLM interactions."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_llm_client():
    """Return an initialized LLM client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)

def call_llm(prompt: str) -> str:
    """Call the OpenAI API and return a text response."""
    client = get_llm_client()
    max_retries = 3
    retry_delay_seconds = 2

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == max_retries:
                return f'{{"error": "LLM call failed after {max_retries} attempts: {exc}"}}'
            time.sleep(retry_delay_seconds)

    return '{"error": "LLM call failed unexpectedly."}'

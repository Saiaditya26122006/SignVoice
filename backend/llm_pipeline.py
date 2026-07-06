"""
SignVoice - LLM pipeline.

Takes a list of detected sign-language words and asks Llama 3 (via Groq) to
form a natural first-person sentence in both English and Hindi.
"""

import json
import os

from dotenv import load_dotenv
from groq import APIError, APITimeoutError, Groq


load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 15


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)


SYSTEM_PROMPT = (
    "You are speaking in the first person as a deaf person who communicates "
    "through Indian Sign Language. You receive a small list of keywords that "
    "were signed, and you must turn them into ONE natural, grammatical "
    "sentence in the first person (using 'I', 'me', 'my'). Keep it short and "
    "clear. Do not add information that was not implied by the keywords.\n\n"
    "You must respond with ONLY a JSON object in this exact shape and nothing "
    "else:\n"
    '{"english": "<english sentence>", "hindi": "<hindi sentence in roman '
    'script>"}'
)


def _build_user_prompt(words):
    return (
        f"Signed keywords: {words}\n"
        "Return the JSON object now."
    )


def _parse_response(content):
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model did not return JSON: {content!r}")

    payload = json.loads(content[start : end + 1])
    if "english" not in payload or "hindi" not in payload:
        raise ValueError(f"JSON missing required keys: {payload}")
    return {"english": payload["english"], "hindi": payload["hindi"]}


def form_sentence(words):
    """Turn a list of signed keywords into an English + Hindi sentence."""
    if not words:
        raise ValueError("words must be a non-empty list")

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(words)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
    except APITimeoutError as exc:
        raise TimeoutError(f"Groq API timed out: {exc}") from exc
    except APIError as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc

    if not response.choices:
        raise RuntimeError("Groq returned no choices")

    content = response.choices[0].message.content or ""
    try:
        return _parse_response(content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid response from Groq: {exc}") from exc


if __name__ == "__main__":
    test_cases = [
        ["water", "want"],
        ["help", "need", "doctor"],
        ["pain", "chest"],
    ]

    for words in test_cases:
        print(f"\nInput words : {words}")
        try:
            result = form_sentence(words)
            print(f"  English   : {result['english']}")
            print(f"  Hindi     : {result['hindi']}")
        except (TimeoutError, RuntimeError, ValueError) as exc:
            print(f"  ERROR     : {exc}")

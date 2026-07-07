"""
SignVoice - TTS pipeline.

Turns the English + Hindi sentences from llm_pipeline into spoken audio
using Sarvam AI:
  * Hindi   -> output_hindi.wav
  * English -> output_english.wav
"""

import base64
import os

from dotenv import load_dotenv
from sarvamai import SarvamAI


load_dotenv()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
HINDI_OUTPUT_PATH = os.path.join(BACKEND_DIR, "output_hindi.wav")
ENGLISH_OUTPUT_PATH = os.path.join(BACKEND_DIR, "output_english.wav")

SARVAM_MODEL = "bulbul:v2"
SARVAM_SPEAKER = "anushka"
SARVAM_HINDI_LANGUAGE_CODE = "hi-IN"
SARVAM_ENGLISH_LANGUAGE_CODE = "en-IN"


def _get_sarvam_client():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it to your .env file."
        )
    return SarvamAI(api_subscription_key=api_key)


def _synthesize_and_save(text, language_code, output_path):
    client = _get_sarvam_client()
    response = client.text_to_speech.convert(
        text=text,
        target_language_code=language_code,
        speaker=SARVAM_SPEAKER,
        model=SARVAM_MODEL,
    )

    audios = getattr(response, "audios", None) or []
    if not audios:
        raise RuntimeError("Sarvam AI returned no audio")

    audio_bytes = b"".join(base64.b64decode(chunk) for chunk in audios)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


def speak_hindi(text):
    """Synthesize Hindi speech via Sarvam AI and save to output_hindi.wav."""
    if not text:
        raise ValueError("text must be a non-empty string")
    return _synthesize_and_save(
        text, SARVAM_HINDI_LANGUAGE_CODE, HINDI_OUTPUT_PATH
    )


def speak_english(text):
    """Synthesize English speech via Sarvam AI and save to output_english.wav."""
    if not text:
        raise ValueError("text must be a non-empty string")
    return _synthesize_and_save(
        text, SARVAM_ENGLISH_LANGUAGE_CODE, ENGLISH_OUTPUT_PATH
    )


def speak_sentence(english, hindi, language="both"):
    """Route the LLM output to the right TTS engine(s).

    language: 'hindi', 'english', or 'both'.
    Returns the list of file paths that were written.
    """
    lang = language.lower()
    if lang not in {"hindi", "english", "both"}:
        raise ValueError("language must be 'hindi', 'english', or 'both'")

    saved = []
    if lang in {"english", "both"}:
        saved.append(speak_english(english))
    if lang in {"hindi", "both"}:
        saved.append(speak_hindi(hindi))
    return saved


if __name__ == "__main__":
    from llm_pipeline import form_sentence

    words = ["water", "want"]
    print(f"Input words : {words}")

    result = form_sentence(words)
    print(f"  English   : {result['english']}")
    print(f"  Hindi     : {result['hindi']}")

    saved_files = speak_sentence(
        result["english"], result["hindi"], language="both"
    )
    print("\nSaved audio files:")
    for path in saved_files:
        print(f"  {path}")

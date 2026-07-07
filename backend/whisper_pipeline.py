"""
SignVoice - Whisper (speech-to-text) pipeline.

Sends an audio file to Groq's hosted Whisper Large v3 and returns the
transcribed text.
"""

import os

from dotenv import load_dotenv
from groq import APIError, APITimeoutError, Groq


load_dotenv()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL = "whisper-large-v3"
REQUEST_TIMEOUT_SECONDS = 30


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)


def transcribe_audio(file_path):
    """Transcribe an audio file with Groq Whisper and return the text."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    client = _get_client()

    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model=WHISPER_MODEL,
            )
    except APITimeoutError as exc:
        raise TimeoutError(f"Groq Whisper API timed out: {exc}") from exc
    except APIError as exc:
        raise RuntimeError(f"Groq Whisper API error: {exc}") from exc

    text = (getattr(transcription, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Groq Whisper returned empty transcription")
    return text


def _record_test_audio(output_path, duration_seconds=5, sample_rate=16000):
    """Record `duration_seconds` of mono microphone audio to a WAV file."""
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write

    print(f"Recording {duration_seconds}s of audio... speak now.")
    recording = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    wav_write(output_path, sample_rate, recording)
    print(f"Saved recording to {output_path}")
    return output_path


if __name__ == "__main__":
    test_audio_path = os.path.join(BACKEND_DIR, "test_audio.wav")

    _record_test_audio(test_audio_path, duration_seconds=5)

    try:
        text = transcribe_audio(test_audio_path)
        print(f"\nTranscription: {text}")
    except (FileNotFoundError, TimeoutError, RuntimeError) as exc:
        print(f"ERROR: {exc}")

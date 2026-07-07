"""
SignVoice - end-to-end pipeline orchestrator.

Forward path (sign -> speech):
    words -> form_sentence (LLM) -> speak_sentence (TTS) -> audio files

Reverse path (speech -> text):
    audio file -> transcribe_audio (Whisper) -> text
"""

from llm_pipeline import form_sentence
from tts_pipeline import speak_sentence


class PipelineStageError(RuntimeError):
    """Raised when a specific stage of the pipeline fails."""

    def __init__(self, stage, original):
        super().__init__(f"[{stage}] {original}")
        self.stage = stage
        self.original = original


def run_forward_pipeline(words, language="both"):
    """Run words -> sentence -> speech.

    Returns a dict with the English sentence, Hindi sentence, and the list of
    audio file paths that were written.
    """
    if not words:
        raise ValueError("words must be a non-empty list")

    try:
        sentence = form_sentence(words)
    except (TimeoutError, RuntimeError, ValueError) as exc:
        raise PipelineStageError("LLM", exc) from exc

    english = sentence["english"]
    hindi = sentence["hindi"]

    try:
        saved_files = speak_sentence(english, hindi, language=language)
    except (RuntimeError, ValueError) as exc:
        raise PipelineStageError("TTS", exc) from exc

    return {
        "english": english,
        "hindi": hindi,
        "audio_files": saved_files,
    }


def run_reverse_pipeline(audio_file_path):
    """Run audio -> text via Whisper. Returns the transcribed string."""
    from whisper_pipeline import transcribe_audio

    try:
        return transcribe_audio(audio_file_path)
    except FileNotFoundError as exc:
        raise PipelineStageError("Whisper/input", exc) from exc
    except (TimeoutError, RuntimeError) as exc:
        raise PipelineStageError("Whisper", exc) from exc


if __name__ == "__main__":
    words = ["water", "want"]
    print(f"Input words : {words}\n")

    try:
        result = run_forward_pipeline(words, language="both")
    except PipelineStageError as exc:
        print(f"Forward pipeline failed: {exc}")
    else:
        print(f"English      : {result['english']}")
        print(f"Hindi        : {result['hindi']}")
        print("Audio files  :")
        for path in result["audio_files"]:
            print(f"  {path}")

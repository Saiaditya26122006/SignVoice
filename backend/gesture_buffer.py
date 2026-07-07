"""
SignVoice - gesture buffer.

Collects high-confidence sign predictions in real time and flags when the
signer has paused long enough to be considered "done" — that's when we send
the buffered words to the LLM + TTS pipeline.
"""

import time


CONFIDENCE_THRESHOLD = 0.70
TRIGGER_IDLE_SECONDS = 1.5


class GestureBuffer:
    def __init__(
        self,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        trigger_idle_seconds=TRIGGER_IDLE_SECONDS,
    ):
        self._confidence_threshold = confidence_threshold
        self._trigger_idle_seconds = trigger_idle_seconds
        self._words = []
        self._last_added_at = None

    def add_word(self, word, confidence):
        if confidence < self._confidence_threshold:
            return
        self._words.append(word)
        self._last_added_at = time.monotonic()

    def should_trigger(self):
        if not self._words or self._last_added_at is None:
            return False
        return (
            time.monotonic() - self._last_added_at
            >= self._trigger_idle_seconds
        )

    def get_words(self):
        return list(self._words)

    def clear(self):
        self._words = []
        self._last_added_at = None


if __name__ == "__main__":
    buffer = GestureBuffer()

    detections = [
        ("water", 0.92),
        ("hello", 0.55),   # below threshold, should be ignored
        ("want", 0.81),
        ("please", 0.40),  # below threshold, should be ignored
    ]

    print("Feeding simulated detections:")
    for word, confidence in detections:
        buffer.add_word(word, confidence)
        kept = confidence >= CONFIDENCE_THRESHOLD
        print(f"  {word:>8}  conf={confidence:.2f}  {'KEPT' if kept else 'DROPPED'}")

    print(f"\nBuffer immediately after feeding: {buffer.get_words()}")
    print(f"should_trigger() now             : {buffer.should_trigger()}")

    print(f"\nWaiting 2 seconds for idle timeout...")
    time.sleep(2.0)

    print(f"should_trigger() after 2s wait   : {buffer.should_trigger()}")
    print(f"Buffered words                   : {buffer.get_words()}")

    if buffer.should_trigger():
        from pipeline import PipelineStageError, run_forward_pipeline

        words = buffer.get_words()
        buffer.clear()

        try:
            result = run_forward_pipeline(words, language="both")
        except PipelineStageError as exc:
            print(f"\nPipeline failed: {exc}")
        else:
            print(f"\nEnglish      : {result['english']}")
            print(f"Hindi        : {result['hindi']}")
            print("Audio files  :")
            for path in result["audio_files"]:
                print(f"  {path}")

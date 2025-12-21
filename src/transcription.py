from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from beartype import beartype
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.types import SpeechToTextChunkResponseModel

# Load environment variables
load_dotenv()


@beartype
def transcribe_audio_with_timestamps(
    audio_path: Path,
    output_dir: Path,
) -> Path:
    """Transcribe audio using ElevenLabs Scribe with word timestamps.

    Args:
        audio_path: Path to the input audio file (MP3)
        output_dir: Directory to save the output JSON

    Returns:
        Path to the created JSON file with word timestamps
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set in environment")

    # 10 hours timeout for long audio files
    client = ElevenLabs(api_key=api_key, timeout=36000.0)

    print(f"Transcribing {audio_path} with word timestamps (ElevenLabs Scribe)...")

    with audio_path.open("rb") as audio_file:
        transcription = client.speech_to_text.convert(
            model_id="scribe_v1",
            file=audio_file,
            timestamps_granularity="word",
        )

    # Handle different response types - we expect SpeechToTextChunkResponseModel
    if not isinstance(transcription, SpeechToTextChunkResponseModel):
        raise RuntimeError(
            f"Unexpected response type: {type(transcription).__name__}. "
            "Expected SpeechToTextChunkResponseModel."
        )

    # Prepare the output data
    output_data: dict[str, Any] = {
        "text": transcription.text,
        "language": transcription.language_code,
    }

    # Calculate duration from last word's end time
    if transcription.words:
        last_word_end = max(
            (w.end for w in transcription.words if w.end is not None),
            default=None,
        )
        output_data["duration"] = last_word_end

    # Include word timestamps - filter to actual words only
    if transcription.words:
        output_data["words"] = [
            {
                "word": word.text,
                "start": word.start,
                "end": word.end,
            }
            for word in transcription.words
            if word.type == "word"  # Only include actual words, not spacing/events
        ]

    output_filename = audio_path.stem + "_transcription.json"
    output_path = output_dir / output_filename

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Transcription saved to: {output_path}")
    return output_path


@beartype
def load_transcription_words(
    transcription_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[dict[str, Any]]:
    """Load words from transcription JSON file with optional time filtering.

    Args:
        transcription_path: Path to the transcription JSON file
        start_time: Optional start time in seconds (filter words starting from this time)
        end_time: Optional end time in seconds (filter words ending before this time)

    Returns:
        List of word dictionaries with word, start, end fields
    """
    with transcription_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    words: list[dict[str, Any]] = data.get("words", [])

    # Filter by time range if specified
    if start_time is not None or end_time is not None:
        filtered_words: list[dict[str, Any]] = []
        for word in words:
            word_start = word.get("start", 0.0)
            word_end = word.get("end", 0.0)

            # Skip words that end before start_time
            if start_time is not None and word_end < start_time:
                continue

            # Skip words that start after end_time
            if end_time is not None and word_start > end_time:
                continue

            filtered_words.append(word)

        words = filtered_words

    # Add index to each word for tracking (re-index after filtering)
    for i, word in enumerate(words):
        word["id"] = i

    return words


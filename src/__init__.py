"""AutoEditor - Automatic video editing based on script matching."""

from __future__ import annotations

from .audio import find_audio_end_by_loudness, trim_attempt_end_times
from .fcpxml import (
    frames_to_fcpxml_time,
    generate_fcpxml,
    seconds_to_fcpxml_time,
)
from .files import (
    find_existing_last_attempts,
    find_existing_transcription,
    get_output_dir,
    load_last_attempts,
)
from .llm import (
    create_llm_prompt,
    create_refinement_prompt,
    find_last_attempts,
    process_last_attempts,
    query_llm,
    refine_word_selection,
    validate_word_ids,
)
from .main import main
from .render import render_result_video
from .script import load_script_sentences
from .transcription import load_transcription_words, transcribe_audio_with_timestamps
from .video import convert_video_to_mp3, get_video_metadata, parse_timecode_to_seconds

__all__ = [
    # audio
    "find_audio_end_by_loudness",
    "trim_attempt_end_times",
    # fcpxml
    "frames_to_fcpxml_time",
    "generate_fcpxml",
    "seconds_to_fcpxml_time",
    # files
    "find_existing_last_attempts",
    "find_existing_transcription",
    "get_output_dir",
    "load_last_attempts",
    # llm
    "create_llm_prompt",
    "create_refinement_prompt",
    "find_last_attempts",
    "process_last_attempts",
    "query_llm",
    "refine_word_selection",
    "validate_word_ids",
    # main
    "main",
    # render
    "render_result_video",
    # script
    "load_script_sentences",
    # transcription
    "load_transcription_words",
    "transcribe_audio_with_timestamps",
    # video
    "convert_video_to_mp3",
    "get_video_metadata",
    "parse_timecode_to_seconds",
]

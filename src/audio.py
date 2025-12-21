from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from beartype import beartype
from pydub import AudioSegment  # type: ignore[import-untyped]


@beartype
def find_audio_end_by_loudness(
    audio_path: Path,
    start_time: float,
    end_time: float,
    silence_threshold_db: float = -40.0,
    window_ms: int = 20,
    min_silence_duration_ms: int = 100,
) -> float:
    """Find the actual end of speech by detecting when audio drops to near silence.

    Analyzes the audio from start_time to end_time, searching forward from the start
    to find the last non-silent window followed by sustained silence.
    This approach skips random clicks and noises at the ending.

    Typically called with the last word's time range (not the entire attempt) to
    avoid pauses between words from affecting the trimming.

    Args:
        audio_path: Path to the audio file (MP3 or other format)
        start_time: Start time in seconds (typically start of the last word)
        end_time: End time in seconds (the potentially too-late transcription end)
        silence_threshold_db: dBFS threshold below which audio is considered silence
        window_ms: Size of analysis window in milliseconds
        min_silence_duration_ms: Minimum consecutive silence to consider as "end"

    Returns:
        Adjusted end time in seconds, trimmed to actual speech end
    """
    # Load the full audio file (pydub has no type stubs)
    audio = cast(
        AudioSegment,
        AudioSegment.from_file(str(audio_path)),  # pyright: ignore[reportUnknownMemberType]
    )

    # Extract the segment we care about
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    # Clamp to audio bounds
    audio_len: int = len(audio)
    start_ms = max(0, start_ms)
    end_ms = min(audio_len, end_ms)

    if start_ms >= end_ms:
        return end_time

    segment = cast(AudioSegment, audio[start_ms:end_ms])

    # Analyze loudness in windows from start to end
    segment_duration_ms: int = len(segment)
    windows: list[tuple[int, float]] = []  # (position_from_start_ms, dBFS)

    # Analyze the segment in windows
    for pos in range(0, segment_duration_ms - window_ms + 1, window_ms):
        window = cast(AudioSegment, segment[pos:pos + window_ms])
        window_len: int = len(window)
        if window_len > 0:
            try:
                dbfs: float = window.dBFS
            except Exception:
                # Empty or silent window
                dbfs = -100.0
            windows.append((pos, dbfs))

    if not windows:
        return end_time

    # Search forward: find the last position where speech ends
    # (last non-silent window followed by enough silence)
    last_sound_end_pos: int | None = None
    consecutive_silence_ms = 0

    for pos, dbfs in windows:
        if dbfs >= silence_threshold_db:
            # Found sound - this could be the last speech position
            last_sound_end_pos = pos + window_ms
            consecutive_silence_ms = 0
        else:
            # Silent window
            consecutive_silence_ms += window_ms
            # If we had sound before and now have enough silence, we found the end
            if last_sound_end_pos is not None and consecutive_silence_ms >= min_silence_duration_ms:
                # Add a small buffer (50ms) for natural decay
                actual_end_ms = start_ms + min(last_sound_end_pos + 50, segment_duration_ms)
                return actual_end_ms / 1000.0

    # If we never found enough silence after sound, return original end time
    return end_time


@beartype
def trim_attempt_end_times(
    video_path: Path,
    attempts: list[dict[str, Any]],
    silence_threshold_db: float = -40.0,
) -> list[dict[str, Any]]:
    """Trim the end times of attempts based on audio loudness analysis.

    Args:
        video_path: Path to the source video file
        attempts: List of attempt dictionaries with start_time and end_time
        silence_threshold_db: dBFS threshold for silence detection

    Returns:
        Updated attempts list with trimmed end times
    """
    # First, extract audio from the video to a temp file for faster analysis
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_audio_path = Path(tmp.name)

    try:
        # Extract audio
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "128k",
            "-y",
            str(tmp_audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to extract audio: {result.stderr}", file=sys.stderr)
            return attempts

        print("Analyzing audio loudness to trim segment endings...")

        updated_attempts: list[dict[str, Any]] = []
        for attempt in attempts:
            end_time = attempt.get("end_time")
            # Use last_word_start to only analyze the last word, not the entire attempt
            # This avoids pauses between words from affecting the trimming
            last_word_start = attempt.get("last_word_start")

            if last_word_start is None or end_time is None:
                updated_attempts.append(attempt)
                continue

            # Find the actual end time based on loudness (analyzing only the last word)
            trimmed_end = find_audio_end_by_loudness(
                tmp_audio_path,
                last_word_start,
                end_time,
                silence_threshold_db=silence_threshold_db,
            )

            if trimmed_end < end_time:
                trim_amount = end_time - trimmed_end
                print(f"  Segment {attempt.get('sentence_idx', '?')}: "
                      f"trimmed {trim_amount:.2f}s (was {end_time:.2f}s, now {trimmed_end:.2f}s)")

            updated_attempt = attempt.copy()
            updated_attempt["end_time"] = trimmed_end
            updated_attempt["original_end_time"] = end_time
            updated_attempts.append(updated_attempt)

        return updated_attempts

    finally:
        # Clean up temp file
        if tmp_audio_path.exists():
            tmp_audio_path.unlink()


@beartype
def resolve_overlapping_segments(
    attempts: list[dict[str, Any]],
    pre_cut_buffer: float,
    post_cut_buffer: float,
) -> list[dict[str, Any]]:
    """Resolve overlapping segments caused by pre/post cut buffers.

    When consecutive clips would overlap due to buffers, this adjusts their
    clip times to meet at the midpoint between original end and start times,
    preventing the same video content from appearing twice.

    Args:
        attempts: List of attempt dictionaries with start_time and end_time
        pre_cut_buffer: Seconds subtracted from start for smoother transitions
        post_cut_buffer: Seconds added to end for smoother transitions

    Returns:
        Updated attempts list with 'clip_start' and 'clip_end' fields
        representing the actual times to use for cutting (overlaps resolved)
    """
    if not attempts:
        return []

    # Calculate initial clip times with buffers
    result: list[dict[str, Any]] = []
    for attempt in attempts:
        updated = attempt.copy()
        updated["clip_start"] = max(0.0, attempt["start_time"] - pre_cut_buffer)
        updated["clip_end"] = attempt["end_time"] + post_cut_buffer
        result.append(updated)

    # Resolve overlaps between consecutive clips
    for i in range(len(result) - 1):
        curr = result[i]
        next_seg = result[i + 1]

        if curr["clip_end"] > next_seg["clip_start"]:
            # Clips overlap - calculate midpoint between original times
            original_curr_end = curr["end_time"]
            original_next_start = next_seg["start_time"]

            # If original segments also overlap, use the boundary
            if original_curr_end >= original_next_start:
                # Original segments overlap - use the later of the two boundaries
                midpoint = max(original_curr_end, original_next_start)
            else:
                # Only buffers cause overlap - meet at midpoint
                midpoint = (original_curr_end + original_next_start) / 2.0

            overlap_amount = curr["clip_end"] - next_seg["clip_start"]
            print(
                f"  Resolving overlap between segments {i + 1} and {i + 2}: "
                f"{overlap_amount:.3f}s overlap, meeting at {midpoint:.3f}s"
            )

            curr["clip_end"] = midpoint
            next_seg["clip_start"] = midpoint

    return result


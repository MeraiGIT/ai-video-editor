from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from beartype import beartype

from .audio import resolve_overlapping_segments, trim_attempt_end_times


@beartype
def render_result_video(
    video_path: Path,
    attempts: list[dict[str, Any]],
    output_dir: Path,
    trim_by_loudness: bool = True,
    silence_threshold_db: float = -40.0,
    pre_cut_buffer: float = 0.1,
    post_cut_buffer: float = 0.1,
) -> Path:
    """Cut and concatenate video segments based on last attempts.

    Args:
        video_path: Path to the source video file
        attempts: List of attempt dictionaries with start_time and end_time
        output_dir: Directory to save the output video
        trim_by_loudness: Whether to trim segment ends based on audio loudness
        silence_threshold_db: dBFS threshold for silence detection
        pre_cut_buffer: Seconds to add before each cut for smoother transitions
        post_cut_buffer: Seconds to add after each cut for smoother transitions

    Returns:
        Path to the rendered result video
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Filter attempts that have valid timing (skip SKIPPED/NOT_FOUND)
    valid_attempts = [
        a for a in attempts
        if a.get("start_time") is not None and a.get("end_time") is not None
    ]

    if not valid_attempts:
        raise RuntimeError("No valid attempts with timing information found")

    # Sort by sentence_idx to maintain order
    valid_attempts.sort(key=lambda x: x.get("sentence_idx", 0))

    # Trim end times based on audio loudness if enabled
    if trim_by_loudness:
        valid_attempts = trim_attempt_end_times(
            video_path, valid_attempts, silence_threshold_db
        )

    # Resolve overlapping segments caused by pre/post cut buffers
    valid_attempts = resolve_overlapping_segments(
        valid_attempts, pre_cut_buffer, post_cut_buffer
    )

    print(f"Rendering {len(valid_attempts)} segments from source video...")

    # Create temporary directory for segment files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        segment_files: list[Path] = []

        # Cut each segment
        for i, attempt in enumerate(valid_attempts):
            # Use pre-calculated clip times (with overlaps resolved)
            clip_start = attempt["clip_start"]
            clip_end = attempt["clip_end"]
            duration = clip_end - clip_start

            segment_file = temp_path / f"segment_{i:04d}.mp4"
            segment_files.append(segment_file)

            print(f"  Cutting segment {i + 1}/{len(valid_attempts)}: "
                  f"{clip_start:.2f}s - {clip_end:.2f}s ({duration:.2f}s)")

            # Use ffmpeg to cut segment with re-encoding for clean cuts
            # Scale to 360p (height=360, width auto-calculated to preserve aspect ratio)
            cmd = [
                "ffmpeg",
                "-ss", str(clip_start),
                "-i", str(video_path),
                "-t", str(duration),
                "-vf", "scale=-2:360",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-y",
                str(segment_file),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}", file=sys.stderr)
                raise RuntimeError(f"FFmpeg cut failed for segment {i}")

        # Create concat file list
        concat_list_file = temp_path / "concat_list.txt"
        with concat_list_file.open("w", encoding="utf-8") as f:
            for segment_file in segment_files:
                # Use escaped path for ffmpeg concat
                f.write(f"file '{segment_file}'\n")

        # Concatenate all segments
        output_path = output_dir / "result.mp4"
        print(f"Concatenating {len(segment_files)} segments...")

        concat_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-c", "copy",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(concat_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}", file=sys.stderr)
            raise RuntimeError("FFmpeg concatenation failed")

    print(f"Result video saved to: {output_path}")

    # Calculate and display total duration using resolved clip times
    total_duration = sum(
        a["clip_end"] - a["clip_start"]
        for a in valid_attempts
    )
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.1f} minutes)")

    return output_path


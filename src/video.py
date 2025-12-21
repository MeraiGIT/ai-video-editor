from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from beartype import beartype


@beartype
def parse_timecode_to_seconds(timecode: str, frame_rate: float) -> float:
    """Parse SMPTE timecode string to seconds.

    Args:
        timecode: Timecode string like "01:00:00:00" (HH:MM:SS:FF)
        frame_rate: Frame rate to convert frames to seconds

    Returns:
        Time in seconds
    """
    parts = timecode.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0.0

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    frames = int(parts[3])

    total_seconds = hours * 3600 + minutes * 60 + seconds + frames / frame_rate
    return total_seconds


@beartype
def get_video_metadata(video_path: Path) -> dict[str, Any]:
    """Get video metadata using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with width, height, frame_rate, duration, audio_sample_rate,
        and start_timecode_seconds (offset from video's embedded timecode)
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # Find video stream
    video_stream: dict[str, Any] | None = None
    audio_stream: dict[str, Any] | None = None
    timecode_value: str | None = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
            # Check for timecode in video stream tags
            tags = stream.get("tags", {})
            if "timecode" in tags:
                timecode_value = tags["timecode"]
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError("No video stream found")

    # Parse frame rate (can be "30/1" or "30000/1001" format)
    frame_rate_str = video_stream.get("r_frame_rate", "30/1")
    frame_rate = Fraction(frame_rate_str)

    # Calculate start timecode in seconds
    start_tc_seconds = 0.0
    if timecode_value:
        start_tc_seconds = parse_timecode_to_seconds(
            timecode_value, float(frame_rate)
        )

    metadata: dict[str, Any] = {
        "width": int(video_stream.get("width", 1920)),
        "height": int(video_stream.get("height", 1080)),
        "frame_rate": frame_rate,
        "duration": float(data.get("format", {}).get("duration", 0)),
        "start_timecode_seconds": start_tc_seconds,
        "start_timecode": timecode_value,
    }

    if audio_stream:
        metadata["audio_sample_rate"] = int(
            audio_stream.get("sample_rate", 48000)
        )
        metadata["audio_channels"] = int(audio_stream.get("channels", 2))

    return metadata


@beartype
def convert_video_to_mp3(
    video_path: Path,
    output_dir: Path,
    bitrate: str = "192k",
) -> Path:
    """Convert video file to compressed MP3 audio.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the output MP3
        bitrate: Audio bitrate for compression (default: 192k)

    Returns:
        Path to the created MP3 file
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_filename = video_path.stem + ".mp3"
    output_path = output_dir / output_filename

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-ab", bitrate,
        "-ar", "44100",  # Sample rate
        "-y",  # Overwrite output
        str(output_path),
    ]

    print(f"Converting {video_path} to MP3...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"FFmpeg conversion failed with code {result.returncode}")

    print(f"Successfully saved to: {output_path}")
    return output_path



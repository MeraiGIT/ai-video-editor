from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from beartype import beartype
from pydub import AudioSegment  # type: ignore[import-untyped]
from scipy import signal  # type: ignore[import-untyped]


@dataclass
class SyncResult:
    """Result of audio synchronization."""

    path: Path
    offset_seconds: float  # Positive = this track starts later than reference
    correlation_strength: float  # 0-1, higher = better match


@beartype
def extract_audio_to_temp(
    media_path: Path,
    sample_rate: int = 8000,
) -> Path:
    """Extract audio from video/audio file to a temporary WAV file.

    Args:
        media_path: Path to the media file (video or audio)
        sample_rate: Target sample rate for analysis (lower = faster)

    Returns:
        Path to temporary WAV file
    """
    # Create temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    # Extract and resample audio using ffmpeg
    cmd = [
        "ffmpeg",
        "-i", str(media_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        "-y",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error extracting audio: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to extract audio from {media_path}")

    return tmp_path


def load_audio_samples(
    audio_path: Path,
) -> tuple[npt.NDArray[np.float64], int]:
    """Load audio file as numpy array of samples.

    Args:
        audio_path: Path to audio file (WAV preferred)

    Returns:
        Tuple of (samples as float array, sample rate)
    """
    # pydub has no type stubs - suppress unknown type errors
    audio: Any = AudioSegment.from_file(str(audio_path))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # Convert to mono if stereo
    if audio.channels > 1:  # pyright: ignore[reportUnknownMemberType]
        audio = audio.set_channels(1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # Get raw samples
    raw_samples: Any = audio.get_array_of_samples()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    samples: npt.NDArray[np.float64] = np.array(raw_samples, dtype=np.float64)

    # Normalize to -1.0 to 1.0 range
    max_val = float(np.max(np.abs(samples)))
    if max_val > 0:
        samples = samples / max_val

    frame_rate: int = int(audio.frame_rate)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return samples, frame_rate


@beartype
def find_audio_offset(
    reference_samples: npt.NDArray[np.float64],
    target_samples: npt.NDArray[np.float64],
    sample_rate: int,
    max_offset_seconds: float = 60.0,
) -> tuple[float, float]:
    """Find the time offset between two audio tracks using cross-correlation.

    Args:
        reference_samples: Audio samples from the reference track
        target_samples: Audio samples from the track to sync
        sample_rate: Sample rate of both tracks
        max_offset_seconds: Maximum offset to search (limits computation)

    Returns:
        Tuple of (offset_seconds, correlation_strength)
        Positive offset means target starts later than reference
    """
    # Limit search range to reduce computation
    max_offset_samples = int(max_offset_seconds * sample_rate)

    # Truncate signals if they're very long (use first ~2 minutes for sync)
    max_samples = sample_rate * 120  # 2 minutes
    ref = reference_samples[:max_samples]
    target = target_samples[:max_samples]

    # Perform cross-correlation using FFT method for speed
    # scipy.signal.correlate returns an ndarray
    correlation: npt.NDArray[Any] = signal.correlate(ref, target, mode="full", method="fft")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # The lag at the center of correlation array corresponds to zero offset
    # correlation[len(target)-1] corresponds to lag=0
    center = len(target) - 1

    # Limit search to max_offset range
    search_start = max(0, center - max_offset_samples)
    search_end = min(
        len(correlation),  # pyright: ignore[reportUnknownArgumentType]
        center + max_offset_samples,
    )

    # Find peak in the search range
    search_range: npt.NDArray[Any] = correlation[search_start:search_end]  # pyright: ignore[reportUnknownVariableType]
    peak_idx_in_range = int(
        np.argmax(np.abs(search_range))  # pyright: ignore[reportUnknownArgumentType]
    )
    peak_idx = search_start + peak_idx_in_range

    # Calculate offset in samples (positive = target is ahead of reference)
    offset_samples = center - peak_idx
    offset_seconds = float(offset_samples) / sample_rate

    # Calculate correlation strength (normalized)
    peak_value = float(
        np.abs(correlation[peak_idx])  # pyright: ignore[reportUnknownArgumentType]
    )
    ref_energy = float(np.sqrt(np.sum(ref**2)))
    target_energy = float(np.sqrt(np.sum(target**2)))

    if ref_energy > 0 and target_energy > 0:
        correlation_strength = peak_value / (ref_energy * target_energy) * len(ref)
        correlation_strength = min(1.0, correlation_strength)  # Cap at 1.0
    else:
        correlation_strength = 0.0

    return offset_seconds, correlation_strength


@beartype
def synchronize_media_files(
    video_paths: list[Path],
    audio_path: Path | None = None,
    sample_rate: int = 8000,
    max_offset_seconds: float = 60.0,
) -> list[SyncResult]:
    """Synchronize multiple video files (and optional audio) by their audio waveforms.

    The first video (or the separate audio if provided) is used as the reference.
    All other files are synchronized to this reference.

    Args:
        video_paths: List of video file paths to synchronize
        audio_path: Optional separate audio file to use as reference
        sample_rate: Sample rate for analysis (lower = faster, 8000 is good)
        max_offset_seconds: Maximum offset to search for

    Returns:
        List of SyncResult with offset for each video (and audio if provided)
    """
    if not video_paths:
        raise ValueError("At least one video path is required")

    print(f"Synchronizing {len(video_paths)} video(s)" +
          (f" with separate audio file" if audio_path else ""))

    temp_files: list[Path] = []
    results: list[SyncResult] = []

    try:
        print("  Extracting audio for analysis...")

        if audio_path is not None:
            # With separate audio: sync first video to audio, then other videos to first video
            # This creates a chain: audio -> video1 -> video2, etc.
            # More reliable than syncing all videos directly to a separate audio recorder

            # Audio is the reference (offset = 0)
            results.append(SyncResult(
                path=audio_path,
                offset_seconds=0.0,
                correlation_strength=1.0,
            ))

            # Extract audio from the separate audio file
            print(f"    Extracting reference audio: {audio_path.name}")
            audio_ref_temp = extract_audio_to_temp(audio_path, sample_rate)
            temp_files.append(audio_ref_temp)
            audio_samples, sr = load_audio_samples(audio_ref_temp)
            print(f"    Reference audio: {len(audio_samples) / sr:.1f}s at {sr}Hz")

            # Sync first video to audio file
            first_video = video_paths[0]
            print(f"    Syncing {first_video.name} to audio...")
            first_video_temp = extract_audio_to_temp(first_video, sample_rate)
            temp_files.append(first_video_temp)
            first_video_samples, _ = load_audio_samples(first_video_temp)

            first_video_offset, first_video_strength = find_audio_offset(
                audio_samples,
                first_video_samples,
                sample_rate,
                max_offset_seconds,
            )
            print(f"      Offset: {first_video_offset:+.3f}s (correlation: {first_video_strength:.2%})")

            results.append(SyncResult(
                path=first_video,
                offset_seconds=first_video_offset,
                correlation_strength=first_video_strength,
            ))

            # Sync additional videos to FIRST VIDEO (not to audio)
            # This is more reliable since camera mics are more similar to each other
            for video_path in video_paths[1:]:
                print(f"    Syncing {video_path.name} to {first_video.name}...")
                video_temp = extract_audio_to_temp(video_path, sample_rate)
                temp_files.append(video_temp)
                video_samples, _ = load_audio_samples(video_temp)

                # Sync this video to first video
                offset_to_first, strength = find_audio_offset(
                    first_video_samples,
                    video_samples,
                    sample_rate,
                    max_offset_seconds,
                )

                # Chain the offsets: video_to_audio = first_video_to_audio + video_to_first_video
                total_offset = first_video_offset + offset_to_first
                print(f"      Offset to {first_video.name}: {offset_to_first:+.3f}s (correlation: {strength:.2%})")
                print(f"      Total offset to audio: {total_offset:+.3f}s")

                results.append(SyncResult(
                    path=video_path,
                    offset_seconds=total_offset,
                    correlation_strength=strength,
                ))

        else:
            # No separate audio: use first video as reference
            print(f"    Using first video as reference: {video_paths[0].name}")
            ref_temp = extract_audio_to_temp(video_paths[0], sample_rate)
            temp_files.append(ref_temp)
            ref_samples, sr = load_audio_samples(ref_temp)
            print(f"    Reference audio: {len(ref_samples) / sr:.1f}s at {sr}Hz")

            # First video is reference (offset = 0)
            results.append(SyncResult(
                path=video_paths[0],
                offset_seconds=0.0,
                correlation_strength=1.0,
            ))

            # Sync other videos to first video
            for video_path in video_paths[1:]:
                print(f"    Analyzing: {video_path.name}...")
                video_temp = extract_audio_to_temp(video_path, sample_rate)
                temp_files.append(video_temp)
                video_samples, _ = load_audio_samples(video_temp)

                offset, strength = find_audio_offset(
                    ref_samples,
                    video_samples,
                    sample_rate,
                    max_offset_seconds,
                )

                results.append(SyncResult(
                    path=video_path,
                    offset_seconds=offset,
                    correlation_strength=strength,
                ))

                print(f"      Offset: {offset:+.3f}s (correlation: {strength:.2%})")

    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

    print("\nSynchronization complete:")
    for result in results:
        print(f"  {result.path.name}: {result.offset_seconds:+.3f}s")

    return results

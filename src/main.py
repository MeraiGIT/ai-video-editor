from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from beartype import beartype
from dotenv import load_dotenv

from .fcpxml import generate_fcpxml, generate_multicam_fcpxml
from .files import (
    find_existing_last_attempts,
    find_existing_transcription,
    get_output_dir,
    load_last_attempts,
)
from .llm import process_last_attempts
from .sync import synchronize_media_files
from .transcription import transcribe_audio_with_timestamps
from .video import convert_video_to_mp3

# Load environment variables
load_dotenv()


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert video to MP3 audio and transcribe with word timestamps"
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the input video file (for single-video mode)",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        default=None,
        help="Multiple video files for multicam mode (synced by audio waveform)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Separate audio file for multicam (used as sync reference and for transcription)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Path to the script .txt file (required for last-attempts analysis)",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="192k",
        help="Audio bitrate (default: 192k)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Reuse an existing output directory (skips transcription if JSON exists)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export FCPXML timeline for DaVinci Resolve from last_attempts.json",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable loudness-based end time trimming during export",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=None,
        help="Silence threshold in dBFS for trimming (default: -40.0 or SILENCE_THRESHOLD_DB env)",
    )
    parser.add_argument(
        "--pre-cut-buffer",
        type=float,
        default=None,
        help="Seconds to add before each cut (default: 0.1 or PRE_CUT_BUFFER env)",
    )
    parser.add_argument(
        "--post-cut-buffer",
        type=float,
        default=None,
        help="Seconds to add after each cut (default: 0.1 or POST_CUT_BUFFER env)",
    )
    parser.add_argument(
        "--original-audio",
        action="store_true",
        help="Use original audio file in timeline (default: use MP3 for better Resolve compatibility)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds for transcription processing (default: from beginning)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds for transcription processing (default: until end)",
    )

    args = parser.parse_args()

    # Determine mode: single video or multicam
    is_multicam = args.videos is not None or args.audio is not None

    if is_multicam:
        _run_multicam(args)
    else:
        if args.video is None:
            parser.error("video argument is required for single-video mode")
        _run_single_video(args)


@beartype
def _run_multicam(args: argparse.Namespace) -> None:
    """Run multicam workflow with multiple videos and optional separate audio."""
    video_paths: list[Path] = args.videos if args.videos else []
    audio_path: Path | None = args.audio

    if not video_paths:
        print("Error: --videos is required for multicam mode")
        sys.exit(1)

    # Validate files exist
    for vp in video_paths:
        if not vp.exists():
            print(f"Error: Video file not found: {vp}")
            sys.exit(1)
    if audio_path and not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    output_dir = get_output_dir(args.output_dir)

    # Synchronize videos by audio waveform
    print("\n--- Synchronizing video sources by audio waveform ---")
    sync_results = synchronize_media_files(video_paths, audio_path)

    # Check for existing transcription
    existing_transcription = find_existing_transcription(output_dir)
    if existing_transcription is not None:
        print(f"Found existing transcription: {existing_transcription}")
        transcription_path = existing_transcription
    else:
        # Use separate audio or first video for transcription
        source_for_transcription = audio_path if audio_path else video_paths[0]
        if audio_path:
            # Audio file - just copy or convert to mp3
            import shutil
            mp3_path = output_dir / "audio.mp3"
            if source_for_transcription.suffix.lower() == ".mp3":
                shutil.copy(source_for_transcription, mp3_path)
            else:
                mp3_path = convert_video_to_mp3(source_for_transcription, output_dir, args.bitrate)
        else:
            mp3_path = convert_video_to_mp3(source_for_transcription, output_dir, args.bitrate)
        transcription_path = transcribe_audio_with_timestamps(mp3_path, output_dir)

    # Check for existing last_attempts.json
    existing_last_attempts = find_existing_last_attempts(output_dir)

    if args.script is not None:
        if existing_last_attempts is not None:
            print(f"\nFound existing last_attempts.json: {existing_last_attempts}")
            print("Skipping last-attempts analysis.")
            last_attempts_path = existing_last_attempts
        else:
            print("\n--- Starting last-attempts analysis ---")
            last_attempts_path = process_last_attempts(
                transcription_path,
                args.script,
                output_dir,
                start_time=args.start_time,
                end_time=args.end_time,
            )
    else:
        if existing_last_attempts is not None:
            print(f"\nFound existing last_attempts.json: {existing_last_attempts}")
            last_attempts_path = existing_last_attempts
        else:
            print("\nNote: No script file provided. Skipping last-attempts analysis.")
            print("Use --script to provide a script file for analysis.")
            last_attempts_path = None

    # Export multicam FCPXML if requested
    if args.export:
        if last_attempts_path is None:
            print("\nError: Cannot export FCPXML without last_attempts.json")
            print("Run with --script first or provide --output-dir with existing data.")
            sys.exit(1)

        print("\n--- Generating multicam FCPXML timeline ---")
        attempts = load_last_attempts(last_attempts_path)

        silence_threshold = args.silence_threshold
        if silence_threshold is None:
            silence_threshold = float(os.getenv("SILENCE_THRESHOLD_DB", "-40.0"))

        pre_cut_buffer = args.pre_cut_buffer
        if pre_cut_buffer is None:
            pre_cut_buffer = float(os.getenv("PRE_CUT_BUFFER", "0.1"))

        post_cut_buffer = args.post_cut_buffer
        if post_cut_buffer is None:
            post_cut_buffer = float(os.getenv("POST_CUT_BUFFER", "0.1"))

        # Choose audio file for timeline
        audio_for_fcpxml = audio_path
        if audio_path and not args.original_audio:
            # Use MP3 version for better Resolve compatibility (default)
            mp3_in_output = output_dir / "audio.mp3"
            if mp3_in_output.exists():
                audio_for_fcpxml = mp3_in_output
                print(f"  Using MP3 for timeline (use --original-audio to use {audio_path.name})")

        generate_multicam_fcpxml(
            video_paths,
            sync_results,
            audio_for_fcpxml,
            attempts,
            output_dir,
            trim_by_loudness=not args.no_trim,
            silence_threshold_db=silence_threshold,
            pre_cut_buffer=pre_cut_buffer,
            post_cut_buffer=post_cut_buffer,
        )


@beartype
def _run_single_video(args: argparse.Namespace) -> None:
    """Run single video workflow (original behavior)."""
    video_path: Path = args.video
    output_dir = get_output_dir(args.output_dir)

    # Check for existing transcription
    existing_transcription = find_existing_transcription(output_dir)
    if existing_transcription is not None:
        print(f"Found existing transcription: {existing_transcription}")
        transcription_path = existing_transcription
    else:
        mp3_path = convert_video_to_mp3(video_path, output_dir, args.bitrate)
        transcription_path = transcribe_audio_with_timestamps(mp3_path, output_dir)

    # Check for existing last_attempts.json
    existing_last_attempts = find_existing_last_attempts(output_dir)

    # Process last attempts if script is provided and last_attempts.json doesn't exist
    if args.script is not None:
        if existing_last_attempts is not None:
            print(f"\nFound existing last_attempts.json: {existing_last_attempts}")
            print("Skipping last-attempts analysis.")
            last_attempts_path = existing_last_attempts
        else:
            print("\n--- Starting last-attempts analysis ---")
            last_attempts_path = process_last_attempts(
                transcription_path,
                args.script,
                output_dir,
                start_time=args.start_time,
                end_time=args.end_time,
            )
    else:
        if existing_last_attempts is not None:
            print(f"\nFound existing last_attempts.json: {existing_last_attempts}")
            last_attempts_path = existing_last_attempts
        else:
            print("\nNote: No script file provided. Skipping last-attempts analysis.")
            print("Use --script to provide a script file for analysis.")
            last_attempts_path = None

    # Export FCPXML timeline if requested
    if args.export:
        if last_attempts_path is None:
            print("\nError: Cannot export FCPXML without last_attempts.json")
            print("Run with --script first or provide --output-dir with existing data.")
            sys.exit(1)

        print("\n--- Generating FCPXML timeline ---")
        attempts = load_last_attempts(last_attempts_path)

        # Get silence threshold from CLI or env (default: -40.0 dBFS)
        silence_threshold = args.silence_threshold
        if silence_threshold is None:
            silence_threshold = float(os.getenv("SILENCE_THRESHOLD_DB", "-40.0"))

        # Get pre-cut buffer from CLI or env (default: 0.1 seconds)
        pre_cut_buffer = args.pre_cut_buffer
        if pre_cut_buffer is None:
            pre_cut_buffer = float(os.getenv("PRE_CUT_BUFFER", "0.1"))

        post_cut_buffer = args.post_cut_buffer
        if post_cut_buffer is None:
            post_cut_buffer = float(os.getenv("POST_CUT_BUFFER", "0.1"))

        generate_fcpxml(
            video_path,
            attempts,
            output_dir,
            trim_by_loudness=not args.no_trim,
            silence_threshold_db=silence_threshold,
            pre_cut_buffer=pre_cut_buffer,
            post_cut_buffer=post_cut_buffer,
        )


if __name__ == "__main__":
    main()


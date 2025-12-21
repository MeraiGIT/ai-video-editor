from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any
from xml.dom import minidom

from beartype import beartype

from .audio import resolve_overlapping_segments, trim_attempt_end_times
from .sync import SyncResult
from .video import get_video_metadata


@beartype
def seconds_to_fcpxml_time(seconds: float, frame_rate: Fraction) -> str:
    """Convert seconds to FCPXML rational time format.

    Args:
        seconds: Time in seconds
        frame_rate: Video frame rate as Fraction

    Returns:
        FCPXML time string like "1001/30000s" or "30s"
    """
    # Convert to frame-accurate time
    frames = round(seconds * float(frame_rate))
    return frames_to_fcpxml_time(frames, frame_rate)


@beartype
def frames_to_fcpxml_time(frames: int, frame_rate: Fraction) -> str:
    """Convert frame count to FCPXML rational time format.

    Args:
        frames: Number of frames
        frame_rate: Video frame rate as Fraction

    Returns:
        FCPXML time string like "1001/30000s" or "30s"
    """
    # FCPXML uses rational numbers for frame-accurate timing
    # Time = frames / frame_rate
    numerator = frames * frame_rate.denominator
    denominator = frame_rate.numerator

    # Simplify the fraction
    common = gcd(numerator, denominator)
    numerator //= common
    denominator //= common

    if denominator == 1:
        return f"{numerator}s"
    return f"{numerator}/{denominator}s"


# Default duration for placeholder clips (NOT_FOUND sentences)
PLACEHOLDER_DURATION_SECONDS = 2.0


@beartype
def generate_fcpxml(
    video_path: Path,
    attempts: list[dict[str, Any]],
    output_dir: Path,
    trim_by_loudness: bool = True,
    silence_threshold_db: float = -40.0,
    pre_cut_buffer: float = 0.1,
    post_cut_buffer: float = 0.1,
) -> Path:
    """Generate FCPXML file for importing timeline into DaVinci Resolve.

    Args:
        video_path: Path to the source video file
        attempts: List of attempt dictionaries with start_time and end_time
        output_dir: Directory to save the output FCPXML
        trim_by_loudness: Whether to trim segment ends based on audio loudness
        silence_threshold_db: dBFS threshold for silence detection
        pre_cut_buffer: Seconds to add before each cut for smoother transitions
        post_cut_buffer: Seconds to add after each cut for smoother transitions

    Returns:
        Path to the generated FCPXML file
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Separate valid attempts from NOT_FOUND ones
    valid_attempts = [
        a for a in attempts
        if a.get("start_time") is not None and a.get("end_time") is not None
    ]
    not_found_attempts = [
        a for a in attempts
        if a.get("status") == "NOT_FOUND" or (
            a.get("start_time") is None and a.get("end_time") is None
        )
    ]

    if not valid_attempts and not not_found_attempts:
        raise RuntimeError("No attempts found at all")

    # Sort valid attempts by sentence_idx to maintain order
    valid_attempts.sort(key=lambda x: x.get("sentence_idx", 0))

    # Trim end times based on audio loudness if enabled
    if trim_by_loudness and valid_attempts:
        valid_attempts = trim_attempt_end_times(
            video_path, valid_attempts, silence_threshold_db
        )

    # Resolve overlapping segments caused by pre/post cut buffers
    if valid_attempts:
        valid_attempts = resolve_overlapping_segments(
            valid_attempts, pre_cut_buffer, post_cut_buffer
        )

    # Merge valid and not_found attempts, sorted by sentence_idx
    all_attempts = valid_attempts + not_found_attempts
    all_attempts.sort(key=lambda x: x.get("sentence_idx", 0))

    found_count = len(valid_attempts)
    not_found_count = len(not_found_attempts)
    print(f"Generating FCPXML for {found_count} segments + {not_found_count} placeholders...")

    # Get video metadata
    metadata = get_video_metadata(video_path)
    frame_rate = metadata["frame_rate"]
    width = metadata["width"]
    height = metadata["height"]
    video_duration = metadata["duration"]
    audio_sample_rate = metadata.get("audio_sample_rate", 48000)

    # Get the source video's starting timecode offset (e.g., 01:00:00:00 = 3600s)
    tc_offset = metadata.get("start_timecode_seconds", 0.0)
    if tc_offset > 0:
        print(f"  Source video timecode starts at: {metadata.get('start_timecode', 'N/A')} ({tc_offset:.2f}s)")

    # Create FCPXML structure
    fcpxml = ET.Element("fcpxml", version="1.9")

    # Resources section
    resources = ET.SubElement(fcpxml, "resources")

    # Format definition
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat{height}p{int(float(frame_rate))}",
        frameDuration=seconds_to_fcpxml_time(1.0 / float(frame_rate), frame_rate),
        width=str(width),
        height=str(height),
    )

    # Asset definition (the source video)
    # The start attribute should match the source video's starting timecode
    video_uri = video_path.resolve().as_uri()
    asset = ET.SubElement(
        resources,
        "asset",
        id="r2",
        name=video_path.stem,
        src=video_uri,
        start=seconds_to_fcpxml_time(tc_offset, frame_rate),
        duration=seconds_to_fcpxml_time(video_duration, frame_rate),
        hasVideo="1",
        hasAudio="1",
        format="r1",
        audioSources="1",
        audioChannels=str(metadata.get("audio_channels", 2)),
        audioRate=str(audio_sample_rate),
    )

    # Media representation
    ET.SubElement(
        asset,
        "media-rep",
        kind="original-media",
        src=video_uri,
    )

    # Library > Event > Project structure
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AutoEditor Export")
    project = ET.SubElement(
        event,
        "project",
        name=f"{video_path.stem}_edited",
    )

    # Calculate total timeline duration in frames (to avoid floating point errors)
    total_duration_frames = 0
    placeholder_duration_frames = round(PLACEHOLDER_DURATION_SECONDS * float(frame_rate))
    for attempt in all_attempts:
        if attempt.get("start_time") is not None and attempt.get("end_time") is not None:
            # Valid attempt with pre-calculated clip times
            clip_start = attempt["clip_start"]
            clip_end = attempt["clip_end"]
            duration = clip_end - clip_start
            total_duration_frames += round(duration * float(frame_rate))
        else:
            # NOT_FOUND - use placeholder duration
            total_duration_frames += placeholder_duration_frames

    # Sequence (the timeline)
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=frames_to_fcpxml_time(total_duration_frames, frame_rate),
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate=f"{audio_sample_rate}/1",
    )

    # Spine (contains the clips)
    spine = ET.SubElement(sequence, "spine")

    # Add clips for each attempt
    # Use frame count for offset to avoid floating point rounding gaps
    timeline_offset_frames = 0
    segment_num = 0
    for attempt in all_attempts:
        is_not_found = attempt.get("start_time") is None or attempt.get("end_time") is None

        if is_not_found:
            # Add a gap (placeholder) for NOT_FOUND sentences
            sentence_text = attempt.get("sentence", "Unknown sentence")
            gap_name = f"[НЕ НАЙДЕНО] {sentence_text}"

            print(
                f"  Adding placeholder for NOT_FOUND: \"{sentence_text[:50]}...\""
            )

            ET.SubElement(
                spine,
                "gap",
                offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                name=gap_name,
                duration=frames_to_fcpxml_time(placeholder_duration_frames, frame_rate),
            )

            timeline_offset_frames += placeholder_duration_frames
        else:
            segment_num += 1
            # Use pre-calculated clip times (with overlaps resolved)
            clip_start = attempt["clip_start"]
            clip_end = attempt["clip_end"]
            duration = clip_end - clip_start
            duration_frames = round(duration * float(frame_rate))

            clip_name = f"Segment {segment_num}"
            if attempt.get("sentence"):
                sentence_preview = attempt["sentence"][:30]
                if len(attempt["sentence"]) > 30:
                    sentence_preview += "..."
                clip_name = f"{segment_num}: {sentence_preview}"

            print(
                f"  Adding segment {segment_num}/{found_count}: "
                f"{clip_start:.2f}s - {clip_end:.2f}s ({duration:.2f}s)"
            )

            # Asset clip on the timeline
            # The clip's start time must include the source video's timecode offset
            clip_source_start = clip_start + tc_offset
            ET.SubElement(
                spine,
                "asset-clip",
                ref="r2",
                offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                name=clip_name,
                start=seconds_to_fcpxml_time(clip_source_start, frame_rate),
                duration=frames_to_fcpxml_time(duration_frames, frame_rate),
                tcFormat="NDF",
                audioRole="dialogue",
            )

            timeline_offset_frames += duration_frames

    # Generate pretty XML
    xml_string = ET.tostring(fcpxml, encoding="unicode")

    # Add XML declaration and DOCTYPE
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype = '<!DOCTYPE fcpxml>\n'

    # Pretty print
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove the XML declaration from minidom (we'll add our own)
    lines = pretty_xml.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    pretty_body = "\n".join(lines)

    final_xml = xml_declaration + doctype + pretty_body

    # Save to file
    output_path = output_dir / "timeline.fcpxml"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(final_xml)

    print(f"\nFCPXML saved to: {output_path}")
    total_duration_seconds = total_duration_frames / float(frame_rate)
    print(f"Total timeline duration: {total_duration_seconds:.2f}s ({total_duration_seconds / 60:.1f} minutes)")
    print(f"\nTo use in DaVinci Resolve:")
    print(f"  1. Open DaVinci Resolve")
    print(f"  2. File > Import > Timeline...")
    print(f"  3. Select: {output_path}")
    print(f"  4. The source video will be linked automatically")

    return output_path


@beartype
def _format_xml(fcpxml: ET.Element) -> str:
    """Format FCPXML element to pretty-printed string with proper declaration."""
    xml_string = ET.tostring(fcpxml, encoding="unicode")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype = '<!DOCTYPE fcpxml>\n'

    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    lines = pretty_xml.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    pretty_body = "\n".join(lines)

    return xml_declaration + doctype + pretty_body


@beartype
def generate_multicam_fcpxml(
    video_paths: list[Path],
    sync_results: list[SyncResult],
    audio_path: Path | None,
    attempts: list[dict[str, Any]],
    output_dir: Path,
    trim_by_loudness: bool = True,
    silence_threshold_db: float = -40.0,
    pre_cut_buffer: float = 0.1,
    post_cut_buffer: float = 0.1,
) -> Path:
    """Generate FCPXML with multicam clips for multiple synchronized video sources.

    Creates parallel video tracks where each video is cut at the same timeline
    positions, allowing the user to choose camera angles in the editor.

    Args:
        video_paths: List of synchronized video file paths
        sync_results: Synchronization results with offsets for each source
        audio_path: Optional separate audio file (used as reference for timing)
        attempts: List of attempt dictionaries with start_time and end_time
        output_dir: Directory to save the output FCPXML
        trim_by_loudness: Whether to trim segment ends based on audio loudness
        silence_threshold_db: dBFS threshold for silence detection
        pre_cut_buffer: Seconds to add before each cut for smoother transitions
        post_cut_buffer: Seconds to add after each cut for smoother transitions

    Returns:
        Path to the generated FCPXML file
    """
    import json
    import subprocess

    for vp in video_paths:
        if not vp.exists():
            raise FileNotFoundError(f"Video file not found: {vp}")

    if audio_path and not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Separate valid attempts from NOT_FOUND ones
    valid_attempts = [
        a for a in attempts
        if a.get("start_time") is not None and a.get("end_time") is not None
    ]
    not_found_attempts = [
        a for a in attempts
        if a.get("status") == "NOT_FOUND" or (
            a.get("start_time") is None and a.get("end_time") is None
        )
    ]

    if not valid_attempts and not not_found_attempts:
        raise RuntimeError("No attempts found at all")

    valid_attempts.sort(key=lambda x: x.get("sentence_idx", 0))

    # Use first video (or audio if provided) for loudness trimming
    reference_path = audio_path if audio_path else video_paths[0]
    if trim_by_loudness and valid_attempts:
        valid_attempts = trim_attempt_end_times(
            reference_path, valid_attempts, silence_threshold_db
        )

    if valid_attempts:
        valid_attempts = resolve_overlapping_segments(
            valid_attempts, pre_cut_buffer, post_cut_buffer
        )

    # Merge valid and not_found attempts, sorted by sentence_idx
    all_attempts = valid_attempts + not_found_attempts
    all_attempts.sort(key=lambda x: x.get("sentence_idx", 0))

    found_count = len(valid_attempts)
    not_found_count = len(not_found_attempts)
    print(f"Generating multicam FCPXML for {found_count} segments + {not_found_count} placeholders "
          f"with {len(video_paths)} camera angles...")

    # Get metadata from first video as reference
    ref_metadata = get_video_metadata(video_paths[0])
    frame_rate = ref_metadata["frame_rate"]
    width = ref_metadata["width"]
    height = ref_metadata["height"]
    audio_sample_rate = ref_metadata.get("audio_sample_rate", 48000)

    # Build offset lookup from sync results
    offset_map: dict[Path, float] = {}
    for sr in sync_results:
        offset_map[sr.path] = sr.offset_seconds

    # Create FCPXML structure
    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")

    # Format definition
    format_id = "r1"
    ET.SubElement(
        resources,
        "format",
        id=format_id,
        name=f"FFVideoFormat{height}p{int(float(frame_rate))}",
        frameDuration=seconds_to_fcpxml_time(1.0 / float(frame_rate), frame_rate),
        width=str(width),
        height=str(height),
    )

    # Asset definitions for each video
    asset_ids: dict[Path, str] = {}
    video_metadata_cache: dict[Path, dict[str, Any]] = {}

    for i, video_path in enumerate(video_paths):
        v_metadata = get_video_metadata(video_path)
        video_metadata_cache[video_path] = v_metadata
        tc_offset = v_metadata.get("start_timecode_seconds", 0.0)
        video_uri = video_path.resolve().as_uri()
        asset_id = f"r{i + 2}"
        asset_ids[video_path] = asset_id

        asset = ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=video_path.stem,
            src=video_uri,
            start=seconds_to_fcpxml_time(tc_offset, frame_rate),
            duration=seconds_to_fcpxml_time(v_metadata["duration"], frame_rate),
            hasVideo="1",
            hasAudio="1",
            format=format_id,
            audioSources="1",
            audioChannels=str(v_metadata.get("audio_channels", 2)),
            audioRate=str(v_metadata.get("audio_sample_rate", audio_sample_rate)),
        )

        ET.SubElement(
            asset,
            "media-rep",
            kind="original-media",
            src=video_uri,
        )

    # Asset for separate audio if provided
    audio_asset_id: str | None = None
    if audio_path:
        audio_uri = audio_path.resolve().as_uri()
        audio_asset_id = f"r{len(video_paths) + 2}"

        # Get actual audio properties using ffprobe
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(audio_path)
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        audio_duration = 0.0
        audio_channels = 2
        audio_rate = audio_sample_rate

        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            audio_duration = float(probe_data.get("format", {}).get("duration", 0))
            # Get stream info for accurate channel/rate
            streams = probe_data.get("streams", [])
            for stream in streams:
                if stream.get("codec_type") == "audio":
                    audio_channels = int(stream.get("channels", 2))
                    audio_rate = int(stream.get("sample_rate", audio_sample_rate))
                    break

        audio_asset = ET.SubElement(
            resources,
            "asset",
            id=audio_asset_id,
            name=audio_path.name,  # Use full filename with extension
            src=audio_uri,
            start="0s",
            duration=seconds_to_fcpxml_time(audio_duration, frame_rate),
            hasVideo="0",
            hasAudio="1",
            format=format_id,
            audioSources="1",
            audioChannels=str(audio_channels),
            audioRate=str(audio_rate),
        )

        # Add media-rep to link the actual file (required for Resolve to find it)
        ET.SubElement(
            audio_asset,
            "media-rep",
            kind="original-media",
            src=audio_uri,
        )

    # Library > Event > Project structure
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AutoEditor Multicam Export")
    project = ET.SubElement(
        event,
        "project",
        name="multicam_edited",
    )

    # Calculate total timeline duration
    total_duration_frames = 0
    placeholder_duration_frames = round(PLACEHOLDER_DURATION_SECONDS * float(frame_rate))
    for attempt in all_attempts:
        if attempt.get("start_time") is not None and attempt.get("end_time") is not None:
            clip_start = attempt["clip_start"]
            clip_end = attempt["clip_end"]
            duration = clip_end - clip_start
            total_duration_frames += round(duration * float(frame_rate))
        else:
            # NOT_FOUND - use placeholder duration
            total_duration_frames += placeholder_duration_frames

    # Sequence
    sequence = ET.SubElement(
        project,
        "sequence",
        format=format_id,
        duration=frames_to_fcpxml_time(total_duration_frames, frame_rate),
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate=f"{audio_sample_rate}/1",
    )

    # Spine for main timeline
    spine = ET.SubElement(sequence, "spine")

    # Build clips for each segment - all tracks use consistent frame-based calculations
    # Structure: for each segment, add V1 (spine), then V2+ and Audio as connected clips
    # All timing values converted to frames ONCE to ensure perfect alignment

    timeline_offset_frames = 0
    segment_num = 0

    for attempt in all_attempts:
        is_not_found = attempt.get("start_time") is None or attempt.get("end_time") is None

        if is_not_found:
            # Add a gap (placeholder) for NOT_FOUND sentences
            sentence_text = attempt.get("sentence", "Unknown sentence")
            gap_name = f"[НЕ НАЙДЕНО] {sentence_text}"

            print(
                f"  Adding placeholder for NOT_FOUND: \"{sentence_text[:50]}...\""
            )

            ET.SubElement(
                spine,
                "gap",
                offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                name=gap_name,
                duration=frames_to_fcpxml_time(placeholder_duration_frames, frame_rate),
            )

            timeline_offset_frames += placeholder_duration_frames
        else:
            segment_num += 1
            clip_start = attempt["clip_start"]
            clip_end = attempt["clip_end"]
            duration = clip_end - clip_start

            # Convert to frames ONCE to ensure consistent rounding across all tracks
            clip_start_frames = round(clip_start * float(frame_rate))
            duration_frames = round(duration * float(frame_rate))

            clip_name = f"Segment {segment_num}"
            if attempt.get("sentence"):
                sentence_preview = attempt["sentence"][:30]
                if len(attempt["sentence"]) > 30:
                    sentence_preview += "..."
                clip_name = f"{segment_num}: {sentence_preview}"

            # V1 (main spine) - first video
            first_video = video_paths[0]
            first_metadata = video_metadata_cache[first_video]
            first_tc_offset = first_metadata.get("start_timecode_seconds", 0.0)
            first_sync_offset = offset_map.get(first_video, 0.0)

            # Calculate V1 source start in frames for consistency
            first_tc_offset_frames = round(first_tc_offset * float(frame_rate))
            first_sync_offset_frames = round(first_sync_offset * float(frame_rate))
            v1_start_frames = clip_start_frames + first_sync_offset_frames + first_tc_offset_frames
            v1_start_frames = max(0, v1_start_frames)

            ET.SubElement(
                spine,
                "asset-clip",
                ref=asset_ids[first_video],
                offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                name=f"{clip_name} - {first_video.stem}",
                start=frames_to_fcpxml_time(v1_start_frames, frame_rate),
                duration=frames_to_fcpxml_time(duration_frames, frame_rate),
                format=format_id,
                tcFormat="NDF",
            )

            # V2+ videos as connected clips on spine (same offset as V1)
            for j, video_path in enumerate(video_paths[1:], start=1):
                v_metadata = video_metadata_cache[video_path]
                v_tc_offset = v_metadata.get("start_timecode_seconds", 0.0)
                v_sync_offset = offset_map.get(video_path, 0.0)

                # Calculate source start in frames
                v_tc_offset_frames = round(v_tc_offset * float(frame_rate))
                v_sync_offset_frames = round(v_sync_offset * float(frame_rate))
                v_start_frames = clip_start_frames + v_sync_offset_frames + v_tc_offset_frames
                v_start_frames = max(0, v_start_frames)

                ET.SubElement(
                    spine,
                    "asset-clip",
                    ref=asset_ids[video_path],
                    lane=str(j),
                    offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                    name=f"{clip_name} - {video_path.stem}",
                    start=frames_to_fcpxml_time(v_start_frames, frame_rate),
                    duration=frames_to_fcpxml_time(duration_frames, frame_rate),
                    format=format_id,
                    tcFormat="NDF",
                )

            # Audio as connected clip on spine (same offset as V1)
            if audio_path and audio_asset_id:
                audio_sync_offset = offset_map.get(audio_path, 0.0)
                audio_sync_offset_frames = round(audio_sync_offset * float(frame_rate))
                audio_start_frames = clip_start_frames + audio_sync_offset_frames
                audio_start_frames = max(0, audio_start_frames)

                ET.SubElement(
                    spine,
                    "asset-clip",
                    ref=audio_asset_id,
                    lane="-1",
                    offset=frames_to_fcpxml_time(timeline_offset_frames, frame_rate),
                    name=f"{clip_name} - Audio",
                    start=frames_to_fcpxml_time(audio_start_frames, frame_rate),
                    duration=frames_to_fcpxml_time(duration_frames, frame_rate),
                    format=format_id,
                    tcFormat="NDF",
                )

            print(
                f"  Adding segment {segment_num}/{found_count}: "
                f"{clip_start:.2f}s - {clip_end:.2f}s ({duration:.2f}s)"
            )

            timeline_offset_frames += duration_frames

    final_xml = _format_xml(fcpxml)

    output_path = output_dir / "timeline_multicam.fcpxml"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(final_xml)

    print(f"\nMulticam FCPXML saved to: {output_path}")
    total_duration_seconds = total_duration_frames / float(frame_rate)
    print(f"Total timeline duration: {total_duration_seconds:.2f}s "
          f"({total_duration_seconds / 60:.1f} minutes)")
    print(f"\nCamera angles included:")
    for i, vp in enumerate(video_paths):
        offset = offset_map.get(vp, 0.0)
        lane_info = "V1 (main)" if i == 0 else f"V{i + 1}"
        print(f"  {i + 1}. {vp.name} ({lane_info}, sync offset: {offset:+.3f}s)")
    if audio_path:
        offset = offset_map.get(audio_path, 0.0)
        print(f"  Audio: {audio_path.name} (A1, sync offset: {offset:+.3f}s)")

    print(f"\nTo use in DaVinci Resolve:")
    print(f"  1. File > Import > Timeline...")
    print(f"  2. Select: {output_path}")
    print(f"  3. All cameras are stacked - V1 on top, use clip enable/disable to switch")
    print(f"  4. Or right-click clips to swap video sources")

    return output_path


from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype
from dotenv import load_dotenv
from nicegui import app, ui

from .fcpxml import generate_fcpxml, generate_multicam_fcpxml
from .files import (
    find_existing_last_attempts,
    find_existing_transcription,
    load_last_attempts,
)
from .llm import process_last_attempts
from .sync import synchronize_media_files
from .transcription import transcribe_audio_with_timestamps
from .video import convert_video_to_mp3

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        self.mode: str = "single"
        self.video_path: Path | None = None
        self.video_paths: list[Path] = []
        self.audio_path: Path | None = None
        self.script_path: Path | None = None
        self.output_dir: Path | None = None
        self.use_existing_output: bool = False
        self.bitrate: str = "192k"
        self.silence_threshold: float = float(os.getenv("SILENCE_THRESHOLD_DB", "-40.0"))
        self.pre_cut_buffer: float = float(os.getenv("PRE_CUT_BUFFER", "0.1"))
        self.post_cut_buffer: float = float(os.getenv("POST_CUT_BUFFER", "0.1"))
        self.no_trim: bool = False
        self.original_audio: bool = False
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.is_processing: bool = False
        self.logs: list[str] = []
        self.progress: float = 0.0


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@beartype
def log_message(message: str) -> None:
    """Add a log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.logs.append(f"[{timestamp}] {message}")


@beartype
def get_output_directory() -> Path:
    """Get or create output directory."""
    if state.output_dir and state.use_existing_output:
        return state.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    state.output_dir = output_dir
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────


@beartype
async def run_processing(
    log_area: Any,
    progress_bar: Any,
    status_label: ui.label,
    run_button: ui.button,
) -> None:
    """Run the main processing pipeline."""
    import shutil

    state.is_processing = True
    state.logs = []
    state.progress = 0.0
    run_button.disable()

    try:
        output_dir = get_output_directory()
        log_message(f"Output directory: {output_dir}")
        log_area.refresh()
        status_label.set_text("Processing...")

        if state.mode == "multicam":
            await _run_multicam_processing(output_dir, log_area, progress_bar, status_label)
        else:
            await _run_single_processing(output_dir, log_area, progress_bar, status_label)

        state.progress = 1.0
        progress_bar.refresh()
        status_label.set_text("✓ Complete!")
        log_message("Processing complete!")
        log_area.refresh()
        ui.notify("Processing complete!", type="positive")

    except Exception as e:
        log_message(f"Error: {e}")
        log_area.refresh()
        status_label.set_text(f"Error: {e}")
        ui.notify(f"Error: {e}", type="negative")

    finally:
        state.is_processing = False
        run_button.enable()


@beartype
async def _run_single_processing(
    output_dir: Path,
    log_area: Any,
    progress_bar: Any,
    status_label: ui.label,
) -> None:
    """Single video processing pipeline."""
    if state.video_path is None:
        raise ValueError("No video file selected")

    video_path = state.video_path

    existing_transcription = find_existing_transcription(output_dir)
    if existing_transcription is not None:
        log_message(f"Found existing transcription: {existing_transcription}")
        log_area.refresh()
        transcription_path = existing_transcription
    else:
        status_label.set_text("Converting video to audio...")
        log_message("Converting video to MP3...")
        log_area.refresh()
        state.progress = 0.1
        progress_bar.refresh()
        await asyncio.sleep(0.1)

        mp3_path = await asyncio.to_thread(
            convert_video_to_mp3, video_path, output_dir, state.bitrate
        )
        log_message(f"Audio saved: {mp3_path}")
        log_area.refresh()

        status_label.set_text("Transcribing audio...")
        log_message("Transcribing audio with Whisper...")
        log_area.refresh()
        state.progress = 0.3
        progress_bar.refresh()
        await asyncio.sleep(0.1)

        transcription_path = await asyncio.to_thread(
            transcribe_audio_with_timestamps, mp3_path, output_dir
        )
        log_message(f"Transcription saved: {transcription_path}")
        log_area.refresh()

    state.progress = 0.5
    progress_bar.refresh()

    if state.script_path is not None:
        existing_last_attempts = find_existing_last_attempts(output_dir)
        if existing_last_attempts is not None:
            log_message(f"Found existing last_attempts: {existing_last_attempts}")
            log_area.refresh()
            last_attempts_path = existing_last_attempts
        else:
            status_label.set_text("Analyzing script alignment...")
            log_message("Processing last attempts with LLM...")
            log_area.refresh()
            state.progress = 0.6
            progress_bar.refresh()
            await asyncio.sleep(0.1)

            last_attempts_path = await asyncio.to_thread(
                process_last_attempts,
                transcription_path,
                state.script_path,
                output_dir,
                state.start_time,
                state.end_time,
            )
            log_message(f"Last attempts saved: {last_attempts_path}")
            log_area.refresh()

        state.progress = 0.8
        progress_bar.refresh()

        status_label.set_text("Generating FCPXML timeline...")
        log_message("Generating FCPXML...")
        log_area.refresh()

        attempts = load_last_attempts(last_attempts_path)
        await asyncio.to_thread(
            generate_fcpxml,
            video_path,
            attempts,
            output_dir,
            not state.no_trim,
            state.silence_threshold,
            state.pre_cut_buffer,
            state.post_cut_buffer,
        )
        log_message("FCPXML timeline generated!")
        log_area.refresh()
    else:
        log_message("No script provided. Skipping LLM analysis.")
        log_area.refresh()


@beartype
async def _run_multicam_processing(
    output_dir: Path,
    log_area: Any,
    progress_bar: Any,
    status_label: ui.label,
) -> None:
    """Multicam processing pipeline."""
    import shutil

    if not state.video_paths:
        raise ValueError("No video files selected")

    video_paths = state.video_paths
    audio_path = state.audio_path

    status_label.set_text("Synchronizing video sources...")
    log_message("Synchronizing video sources by audio waveform...")
    log_area.refresh()
    state.progress = 0.1
    progress_bar.refresh()
    await asyncio.sleep(0.1)

    sync_results = await asyncio.to_thread(synchronize_media_files, video_paths, audio_path)
    log_message(f"Sync results: {sync_results}")
    log_area.refresh()

    state.progress = 0.25
    progress_bar.refresh()

    existing_transcription = find_existing_transcription(output_dir)
    if existing_transcription is not None:
        log_message(f"Found existing transcription: {existing_transcription}")
        log_area.refresh()
        transcription_path = existing_transcription
    else:
        source_for_transcription = audio_path if audio_path else video_paths[0]

        if audio_path:
            mp3_path = output_dir / "audio.mp3"
            if source_for_transcription.suffix.lower() == ".mp3":
                shutil.copy(source_for_transcription, mp3_path)
            else:
                status_label.set_text("Converting audio...")
                log_message("Converting audio to MP3...")
                log_area.refresh()
                mp3_path = await asyncio.to_thread(
                    convert_video_to_mp3, source_for_transcription, output_dir, state.bitrate
                )
        else:
            status_label.set_text("Extracting audio from video...")
            log_message("Extracting audio from first video...")
            log_area.refresh()
            mp3_path = await asyncio.to_thread(
                convert_video_to_mp3, source_for_transcription, output_dir, state.bitrate
            )

        status_label.set_text("Transcribing audio...")
        log_message("Transcribing audio with Whisper...")
        log_area.refresh()
        state.progress = 0.4
        progress_bar.refresh()
        await asyncio.sleep(0.1)

        transcription_path = await asyncio.to_thread(
            transcribe_audio_with_timestamps, mp3_path, output_dir
        )
        log_message(f"Transcription saved: {transcription_path}")
        log_area.refresh()

    state.progress = 0.55
    progress_bar.refresh()

    if state.script_path is not None:
        existing_last_attempts = find_existing_last_attempts(output_dir)
        if existing_last_attempts is not None:
            log_message(f"Found existing last_attempts: {existing_last_attempts}")
            log_area.refresh()
            last_attempts_path = existing_last_attempts
        else:
            status_label.set_text("Analyzing script alignment...")
            log_message("Processing last attempts with LLM...")
            log_area.refresh()
            state.progress = 0.65
            progress_bar.refresh()
            await asyncio.sleep(0.1)

            last_attempts_path = await asyncio.to_thread(
                process_last_attempts,
                transcription_path,
                state.script_path,
                output_dir,
                state.start_time,
                state.end_time,
            )
            log_message(f"Last attempts saved: {last_attempts_path}")
            log_area.refresh()

        state.progress = 0.8
        progress_bar.refresh()

        status_label.set_text("Generating multicam FCPXML...")
        log_message("Generating multicam FCPXML timeline...")
        log_area.refresh()

        attempts = load_last_attempts(last_attempts_path)

        audio_for_fcpxml = audio_path
        if audio_path and not state.original_audio:
            mp3_in_output = output_dir / "audio.mp3"
            if mp3_in_output.exists():
                audio_for_fcpxml = mp3_in_output

        await asyncio.to_thread(
            generate_multicam_fcpxml,
            video_paths,
            sync_results,
            audio_for_fcpxml,
            attempts,
            output_dir,
            not state.no_trim,
            state.silence_threshold,
            state.pre_cut_buffer,
            state.post_cut_buffer,
        )
        log_message("Multicam FCPXML timeline generated!")
        log_area.refresh()
    else:
        log_message("No script provided. Skipping LLM analysis and FCPXML generation.")
        log_area.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────


@ui.page("/")
def index_page() -> None:
    """Main page."""
    # Dark theme
    ui.dark_mode().enable()

    # Custom styles
    ui.add_css("""
        :root {
            --accent: #00d4aa;
            --accent-dark: #00b894;
        }
        body {
            background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%) !important;
            font-family: 'Segoe UI', system-ui, sans-serif !important;
        }
        .card-glass {
            background: rgba(26, 26, 26, 0.9) !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 16px !important;
        }
        .card-glass:hover {
            border-color: rgba(0, 212, 170, 0.3) !important;
        }
        .section-label {
            color: #00d4aa !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
        }
        .file-chip {
            background: #252525 !important;
            border: 1px solid #333 !important;
        }
        .log-box {
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            font-size: 0.8rem !important;
            background: #0d0d0d !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 8px !important;
            padding: 12px !important;
            max-height: 250px !important;
            overflow-y: auto !important;
        }
        .accent-btn {
            background: linear-gradient(135deg, #00d4aa, #00b894) !important;
            color: #000 !important;
            font-weight: 600 !important;
        }
        .picker-btn {
            background: #1e3a3a !important;
            border: 1px solid #00d4aa40 !important;
        }
        .picker-btn:hover {
            background: #254545 !important;
            border-color: #00d4aa !important;
        }
        .progress-track {
            height: 6px;
            background: #252525;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4aa, #00b894);
            transition: width 0.3s ease;
        }
    """)

    # Main container
    with ui.column().classes("w-full max-w-5xl mx-auto p-8 gap-6"):
        # Header
        with ui.row().classes("w-full items-center gap-4 mb-4"):
            ui.icon("bolt", size="xl", color="teal")
            with ui.column().classes("gap-0"):
                ui.label("AutoEditor").classes("text-3xl font-bold")
                ui.label("AI-powered video editing automation").classes("text-gray-500")

        # Mode selector and files
        @ui.refreshable
        def mode_and_files() -> None:
            # Mode toggle
            with ui.row().classes("w-full justify-center mb-4"):
                with ui.button_group():
                    ui.button(
                        "Single Video",
                        on_click=lambda: _set_mode("single"),
                        color="teal" if state.mode == "single" else "dark",
                    )
                    ui.button(
                        "Multicam",
                        on_click=lambda: _set_mode("multicam"),
                        color="teal" if state.mode == "multicam" else "dark",
                    )

            # File selection cards
            with ui.row().classes("w-full gap-4"):
                # Video files card
                with ui.card().classes("card-glass flex-1 p-5"):
                    ui.label("📹 Video Files").classes("section-label mb-3")

                    if state.mode == "single":
                        if state.video_path:
                            with ui.row().classes(
                                "items-center gap-2 file-chip px-3 py-2 rounded-lg w-full"
                            ):
                                ui.icon("videocam", color="teal", size="sm")
                                ui.label(state.video_path.name).classes("flex-1 truncate text-sm")
                                ui.button(icon="close", on_click=_clear_video).props(
                                    "flat dense round size=xs"
                                )
                        else:

                            async def pick_single_video() -> None:
                                result = await app.native.main_window.create_file_dialog()
                                if result:
                                    state.video_path = Path(result[0])
                                    mode_and_files.refresh()

                            ui.button(
                                "Select Video File",
                                icon="folder_open",
                                on_click=pick_single_video,
                            ).classes("picker-btn w-full")
                    else:
                        with ui.column().classes("w-full gap-2"):
                            for i, vp in enumerate(list(state.video_paths)):
                                with ui.row().classes(
                                    "items-center gap-2 file-chip px-3 py-2 rounded-lg w-full"
                                ):
                                    ui.icon("videocam", color="teal", size="sm")
                                    ui.label(vp.name).classes("flex-1 truncate text-sm")

                                    def make_remove(idx: int) -> Any:
                                        return lambda: _remove_video(idx)

                                    ui.button(icon="close", on_click=make_remove(i)).props(
                                        "flat dense round size=xs"
                                    )

                            async def pick_multi_videos() -> None:
                                result = await app.native.main_window.create_file_dialog(
                                    allow_multiple=True
                                )
                                if result:
                                    for p in result:
                                        state.video_paths.append(Path(p))
                                    mode_and_files.refresh()

                            ui.button(
                                "Add Video Files",
                                icon="add",
                                on_click=pick_multi_videos,
                            ).classes("picker-btn w-full")

                # Audio & Script card
                with ui.card().classes("card-glass flex-1 p-5"):
                    with ui.column().classes("w-full gap-4"):
                        if state.mode == "multicam":
                            ui.label("🎵 Audio Reference").classes("section-label")
                            if state.audio_path:
                                with ui.row().classes(
                                    "items-center gap-2 file-chip px-3 py-2 rounded-lg w-full"
                                ):
                                    ui.icon("audiotrack", color="orange", size="sm")
                                    ui.label(state.audio_path.name).classes(
                                        "flex-1 truncate text-sm"
                                    )
                                    ui.button(icon="close", on_click=_clear_audio).props(
                                        "flat dense round size=xs"
                                    )
                            else:

                                async def pick_audio() -> None:
                                    result = await app.native.main_window.create_file_dialog()
                                    if result:
                                        state.audio_path = Path(result[0])
                                        mode_and_files.refresh()

                                ui.button(
                                    "Select Audio File (optional)",
                                    icon="folder_open",
                                    on_click=pick_audio,
                                ).classes("picker-btn w-full")

                        ui.label("📝 Script File").classes("section-label")
                        if state.script_path:
                            with ui.row().classes(
                                "items-center gap-2 file-chip px-3 py-2 rounded-lg w-full"
                            ):
                                ui.icon("description", color="yellow", size="sm")
                                ui.label(state.script_path.name).classes("flex-1 truncate text-sm")
                                ui.button(icon="close", on_click=_clear_script).props(
                                    "flat dense round size=xs"
                                )
                        else:

                            async def pick_script() -> None:
                                result = await app.native.main_window.create_file_dialog()
                                if result:
                                    state.script_path = Path(result[0])
                                    mode_and_files.refresh()

                            ui.button(
                                "Select Script File (optional)",
                                icon="folder_open",
                                on_click=pick_script,
                            ).classes("picker-btn w-full")

        def _set_mode(mode: str) -> None:
            state.mode = mode
            mode_and_files.refresh()

        def _clear_video() -> None:
            state.video_path = None
            mode_and_files.refresh()

        def _remove_video(idx: int) -> None:
            if 0 <= idx < len(state.video_paths):
                state.video_paths.pop(idx)
                mode_and_files.refresh()

        def _clear_audio() -> None:
            state.audio_path = None
            mode_and_files.refresh()

        def _clear_script() -> None:
            state.script_path = None
            mode_and_files.refresh()

        mode_and_files()

        # Output directory
        with ui.card().classes("card-glass w-full p-5"):
            ui.label("📂 Output Directory").classes("section-label mb-3")

            @ui.refreshable
            def output_selector() -> None:
                if state.output_dir and state.use_existing_output:
                    with ui.row().classes(
                        "items-center gap-2 file-chip px-3 py-2 rounded-lg w-full"
                    ):
                        ui.icon("folder", color="blue", size="sm")
                        ui.label(str(state.output_dir)).classes("flex-1 truncate text-sm")

                        def clear_output() -> None:
                            state.output_dir = None
                            state.use_existing_output = False
                            output_selector.refresh()

                        ui.button(icon="close", on_click=clear_output).props(
                            "flat dense round size=xs"
                        )
                else:
                    with ui.row().classes("w-full gap-2 items-center"):

                        async def pick_output_dir() -> None:
                            try:
                                # Use macOS native folder picker via osascript
                                def select_folder() -> str | None:
                                    import subprocess
                                    
                                    # AppleScript to show folder picker
                                    script = '''
                                    tell application "Finder"
                                        activate
                                        set folderPath to choose folder with prompt "Select Output Folder"
                                        return POSIX path of folderPath
                                    end tell
                                    '''
                                    
                                    try:
                                        result = subprocess.run(
                                            ["osascript", "-e", script],
                                            capture_output=True,
                                            text=True,
                                            timeout=300  # 5 minute timeout
                                        )
                                        
                                        if result.returncode == 0:
                                            folder_path = result.stdout.strip()
                                            return folder_path if folder_path else None
                                        else:
                                            # User cancelled or error
                                            return None
                                    except subprocess.TimeoutExpired:
                                        return None
                                    except Exception:
                                        return None
                                
                                # Run folder picker in a separate thread
                                folder_path = await asyncio.to_thread(select_folder)
                                
                                if folder_path:
                                    folder = Path(folder_path)
                                    if folder.exists() and folder.is_dir():
                                        state.output_dir = folder
                                        state.use_existing_output = True
                                        output_selector.refresh()
                                        ui.notify(f"Selected folder: {state.output_dir}", type="positive")
                                    else:
                                        ui.notify(f"Invalid folder path: {folder_path}", type="negative")
                                # else: user cancelled, do nothing
                            except Exception as e:
                                ui.notify(f"Error selecting folder: {e}", type="negative")

                        ui.button(
                            "Select Output Folder",
                            icon="folder_open",
                            on_click=pick_output_dir,
                        ).classes("picker-btn")
                        ui.label("or leave empty for auto-generated").classes(
                            "text-gray-500 text-sm"
                        )

            output_selector()

        # Options
        with ui.card().classes("card-glass w-full p-5"):
            ui.label("⚙️ Options").classes("section-label mb-4")

            with ui.row().classes("w-full gap-6 flex-wrap"):
                with ui.column().classes("gap-2 min-w-40"):
                    ui.label("Audio").classes("text-sm text-gray-400 font-medium")
                    ui.input(
                        label="Bitrate",
                        value=state.bitrate,
                        on_change=lambda e: setattr(state, "bitrate", e.value),
                    )

                with ui.column().classes("gap-2 min-w-40"):
                    ui.label("Trimming").classes("text-sm text-gray-400 font-medium")
                    ui.number(
                        label="Silence (dB)",
                        value=state.silence_threshold,
                        format="%.1f",
                        on_change=lambda e: setattr(state, "silence_threshold", e.value or -40.0),
                    )
                    ui.switch(
                        "Disable trimming",
                        value=state.no_trim,
                        on_change=lambda e: setattr(state, "no_trim", e.value),
                    )

                with ui.column().classes("gap-2 min-w-40"):
                    ui.label("Buffers").classes("text-sm text-gray-400 font-medium")
                    ui.number(
                        label="Pre-cut (sec)",
                        value=state.pre_cut_buffer,
                        format="%.2f",
                        step=0.05,
                        on_change=lambda e: setattr(state, "pre_cut_buffer", e.value or 0.1),
                    )
                    ui.number(
                        label="Post-cut (sec)",
                        value=state.post_cut_buffer,
                        format="%.2f",
                        step=0.05,
                        on_change=lambda e: setattr(state, "post_cut_buffer", e.value or 0.1),
                    )

                with ui.column().classes("gap-2 min-w-40"):
                    ui.label("Time Range").classes("text-sm text-gray-400 font-medium")
                    ui.number(
                        label="Start (sec)",
                        value=state.start_time,
                        format="%.1f",
                        on_change=lambda e: setattr(state, "start_time", e.value),
                    )
                    ui.number(
                        label="End (sec)",
                        value=state.end_time,
                        format="%.1f",
                        on_change=lambda e: setattr(state, "end_time", e.value),
                    )

        # Processing section
        with ui.card().classes("card-glass w-full p-5"):
            ui.label("🚀 Processing").classes("section-label mb-4")

            @ui.refreshable
            def progress_bar() -> None:
                with ui.element("div").classes("progress-track w-full"):
                    ui.element("div").classes("progress-fill").style(
                        f"width: {state.progress * 100}%"
                    )

            progress_bar()

            with ui.row().classes("w-full items-center justify-between mt-4"):
                status_label = ui.label("Ready").classes("text-gray-400")
                run_btn = ui.button("▶ Start Processing").classes("accent-btn")

            @ui.refreshable
            def log_area() -> None:
                if state.logs:
                    with ui.element("div").classes("log-box w-full mt-4"):
                        for log_line in state.logs[-30:]:
                            ui.label(log_line).classes("text-gray-300 text-xs py-0.5")

            log_area()

            run_btn.on(
                "click",
                lambda: asyncio.create_task(
                    run_processing(log_area, progress_bar, status_label, run_btn)
                ),
            )


@beartype
def main() -> None:
    """Run the UI application in native mode with file picker support."""
    # Configure native window
    app.native.window_args["resizable"] = True
    app.native.settings["ALLOW_DOWNLOADS"] = True

    ui.run(
        title="AutoEditor",
        dark=True,
        reload=False,
        native=True,
        window_size=(1200, 900),
    )


if __name__ == "__main__":
    main()

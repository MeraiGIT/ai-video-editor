# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoEditor is an AI-powered video editing automation tool that automatically creates video timelines by matching transcribed speech to a written script. It uses ElevenLabs' Whisper transcription for word-level timestamps and OpenRouter LLMs to identify the "last complete attempt" of each script sentence, then generates FCPXML timelines for DaVinci Resolve.

### Problem Solved

Content creators often record multiple takes of each sentence/phrase when creating videos. After recording, they need to:
1. Listen through all the material
2. Manually identify the best/last complete take of each sentence
3. Cut and arrange clips in a video editor

This is extremely time-consuming and tedious. AutoEditor automates this entire workflow by:
- Transcribing all recorded material with precise word-level timestamps
- Using AI to intelligently find the "last complete attempt" of each script sentence
- Automatically generating a cut timeline ready for final review in DaVinci Resolve or Final Cut Pro

### Development Philosophy

This project was created using a "vibe-coding" approach with Cursor AI - an iterative, AI-assisted development workflow where:
- Ideas and requirements are described in natural language
- The AI agent writes the implementation
- Development progresses through iterations, fixes, and commits
- Minimal manual code manipulation - focus on high-level direction

Creator's demo video: [Вайб-кодинг в Cursor AI](https://www.youtube.com/watch?v=ZRqHzvA6OvY) (Russian)

**Key Development Phases** (from video timeline):
1. Audio extraction
2. Transcription with ElevenLabs
3. Finding last takes with LLM
4. Rendering logic
5. FCPXML generation
6. Multicam synchronized media support
7. Refactoring
8. GUI interface

## Development Commands

### Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv with Python 3.12)
uv sync

# Install ffmpeg (required for video/audio processing)
brew install ffmpeg  # macOS
```

### Running the Application
```bash
# Launch GUI (native window with file pickers)
uv run autoeditor-ui

# CLI - Single video mode
uv run autoeditor video.mp4 --script script.txt --export

# CLI - Multicam mode
uv run autoeditor --videos cam1.mp4 cam2.mp4 --script script.txt --export
```

### Type Checking
```bash
# Run mypy
uv run mypy .

# Run pyright
uv run pyright
```

## Architecture

### Core Pipeline

The application follows a sequential pipeline with two modes:

**Single Video Mode:**
1. Convert video → MP3 (`video.py`)
2. Transcribe MP3 → word timestamps (`transcription.py`)
3. Match transcription to script → last attempts (`llm.py`)
4. Generate FCPXML timeline (`fcpxml.py`)

**Multicam Mode:**
1. Sync multiple videos by audio waveform (`sync.py`)
2. Extract audio from reference source → MP3
3. Transcribe → word timestamps
4. Match to script → last attempts
5. Generate multicam FCPXML with sync offsets

### Key Components

**`llm.py` - LLM-based Script Matching**
- Core algorithm: batch-based progressive search through transcription
- Uses OpenRouter API (configured model, default: Claude 3.5 Sonnet)
- Implements two-phase approach:
  1. Find last complete attempt by looking for sentence boundaries (current + next sentence)
  2. Refine selection to trim false starts/expletives from beginning
- Handles speech recognition errors by matching semantic meaning, not exact text
- Maintains `llm_log.jsonl` for debugging LLM interactions
- Environment: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `WORD_BATCH_SIZE`

**`transcription.py` - ElevenLabs Integration**
- Uses ElevenLabs Scribe API for word-level timestamps
- Filters to only include "word" type items (excludes spacing/events)
- Supports time-range filtering via `start_time`/`end_time` parameters
- Environment: `ELEVENLABS_API_KEY`

**`sync.py` - Multicam Audio Synchronization**
- Cross-correlation algorithm using scipy
- Extracts low-sample-rate mono audio (8kHz) for fast processing
- Returns offset in seconds (positive = track starts later than reference)
- Correlation strength indicates match quality (0-1 scale)

**`fcpxml.py` - Timeline Generation**
- Generates frame-accurate FCPXML using rational time format
- Handles both single video and multicam modes
- Features:
  - Resolves overlapping segments
  - Trims attempt end times based on audio loudness analysis
  - Adds pre/post-cut buffers for smooth transitions
  - Inserts placeholder clips for NOT_FOUND sentences
- Environment: `SILENCE_THRESHOLD_DB`, `PRE_CUT_BUFFER`, `POST_CUT_BUFFER`

**`ui.py` - NiceGUI Interface**
- Native window mode with macOS file pickers (uses `pywebview`)
- State-driven UI with `@ui.refreshable` decorators for reactive updates
- Async processing pipeline with progress tracking
- Custom dark theme with accent colors
- Uses `asyncio.to_thread()` for CPU-intensive operations

### Data Flow

```
Video/Audio Files
    ↓
MP3 Conversion (ffmpeg)
    ↓
Transcription JSON
    {
      "text": "full text",
      "words": [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
    }
    ↓
Script TXT File
    ↓
LLM Analysis (batched word processing)
    ↓
Last Attempts JSON
    {
      "attempts": [{
        "sentence": "...",
        "first_word_id": 10,
        "last_word_id": 25,
        "start_time": 5.2,
        "end_time": 8.7
      }, ...]
    }
    ↓
FCPXML Timeline (frame-accurate cuts)
```

### Important Patterns

**Beartype Decorators**: All functions use `@beartype` for runtime type checking. Maintain strict type annotations.

**Path Handling**: Always use `pathlib.Path` objects, never strings. All file operations expect absolute paths.

**Environment Variables**: Load via `python-dotenv` at module level. Check for required keys and raise `RuntimeError` if missing.

**Error Handling**: Use explicit error messages that guide users to solutions (e.g., "ELEVENLABS_API_KEY not set in environment").

**Time Precision**:
- Transcription uses float seconds with millisecond precision
- FCPXML uses rational time (frames/frame_rate) for frame accuracy
- Always convert seconds → frames for timeline generation

**Output Directory Structure**:
```
output/
└── YYYYMMDD_HHMMSS/
    ├── source.mp3 or audio.mp3
    ├── source_transcription.json or audio_transcription.json
    ├── last_attempts.json
    ├── llm_log.jsonl
    └── timeline.fcpxml or timeline_multicam.fcpxml
```

## Configuration

Required API keys in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here

# Optional
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
SILENCE_THRESHOLD_DB=-40.0
PRE_CUT_BUFFER=0.1
POST_CUT_BUFFER=0.1
WORD_BATCH_SIZE=50
```

## Dependencies

- **Python 3.12+**: Uses modern type hints (e.g., `list[dict]`, `Path | None`)
- **ffmpeg**: External binary for video/audio conversion
- **uv**: Package manager that creates isolated virtual environments
- **beartype**: Runtime type checking decorator
- **nicegui**: Web UI framework with native window support
- **pywebview**: Native window and file picker integration (macOS)
- **elevenlabs**: Speech-to-text API client
- **openai**: Used for OpenRouter API (compatible interface)
- **scipy/numpy**: Audio signal processing for sync
- **pydub**: Audio manipulation wrapper

## Important Notes

**LLM Prompt Engineering**: The `llm.py` prompts are carefully tuned for:
- Handling speech recognition errors
- Distinguishing complete vs incomplete attempts
- Avoiding merging multiple takes into one range
- Finding sentence boundaries using context from the next sentence

**Multicam Sync Accuracy**: Audio sync uses cross-correlation on downsampled (8kHz) mono audio. This is fast but may have ±0.1s accuracy. For critical applications, consider increasing sample rate in `sync.py`.

**FCPXML Compatibility**: Generated timelines target DaVinci Resolve. Final Cut Pro compatibility is not guaranteed.

**Type Checking**: Project uses strict type checking (`mypy --strict`, `pyright strict`). All functions must have complete type annotations including return types.

## Typical Workflow

### For Content Creators

1. **Preparation**: Write your script as a text file (.txt), one sentence per line or paragraph
2. **Recording**: Record your video while reading the script - don't worry about mistakes, just keep re-recording sentences until you get them right
3. **Processing**:
   - Launch AutoEditor GUI: `uv run autoeditor-ui`
   - Select your video file(s)
   - Select your script file
   - Adjust timing parameters if needed (silence threshold, buffers)
   - Click "Start Processing"
4. **Review**: Open the generated FCPXML in DaVinci Resolve or Final Cut Pro
5. **Polish**: The timeline will contain the best takes automatically cut together - now you can fine-tune transitions, add B-roll, color grade, etc.

### For Multicam Setups

Use multicam mode when you have:
- Multiple camera angles of the same recording session
- A separate high-quality audio recording (e.g., from a dedicated microphone)

The tool will automatically synchronize all sources by audio waveform and generate a multicam timeline.

## Resources

- **OpenRouter** (for LLM access): https://openrouter.ai/
- **ElevenLabs** (for transcription): https://elevenlabs.io/
- **Source code & Cursor Rules**: Check creator's Telegram channel
- **Video demonstration**: https://www.youtube.com/watch?v=ZRqHzvA6OvY

# AutoEditor

AI-powered video editing automation tool that automatically creates video timelines based on script matching using Whisper transcription and LLM analysis.

## Features

- 🎬 **Single Video Mode**: Process a single video file with script alignment
- 🎥 **Multicam Mode**: Synchronize multiple video sources by audio waveform
- 🎤 **Audio Transcription**: Automatic transcription with word-level timestamps using Whisper
- 🤖 **LLM Analysis**: Intelligent script-to-transcription matching using OpenAI
- 📝 **FCPXML Export**: Generate Final Cut Pro XML timelines for DaVinci Resolve
- 🎨 **Beautiful UI**: Modern dark-themed web interface with native file pickers

## Requirements

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- macOS (for native UI file pickers)
- OpenAI API key (for LLM analysis)

## Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone or navigate to the project directory**:
   ```bash
   cd autoeditor
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Create `.env` file** in the project root:
   ```bash
   cp .env.example .env  # if you have an example file
   # or create manually:
   ```
   
   Add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ELEVENLABS_API_KEY=...
   
   # Optional: Customize defaults
   SILENCE_THRESHOLD_DB=-40.0
   PRE_CUT_BUFFER=0.1
   POST_CUT_BUFFER=0.1
   ```

## Usage

### GUI Mode (Recommended)

Launch the beautiful web-based UI:

```bash
uv run autoeditor-ui
```

The application will open in a native window at `http://127.0.0.1:8000` (or next available port).

**Features:**
- Select video files through native file picker
- Choose output directory
- Configure processing options
- Real-time progress tracking
- Live log output

### CLI Mode

#### Single Video Mode

```bash
# Basic usage
uv run autoeditor video.mp4 --script script.txt

# With custom output directory
uv run autoeditor video.mp4 --script script.txt --output-dir output/my_project

# Export FCPXML timeline
uv run autoeditor video.mp4 --script script.txt --export

# Custom options
uv run autoeditor video.mp4 \
  --script script.txt \
  --bitrate 256k \
  --silence-threshold -35.0 \
  --pre-cut-buffer 0.2 \
  --post-cut-buffer 0.2 \
  --export
```

#### Multicam Mode

```bash
# With multiple videos
uv run autoeditor --videos cam1.mp4 cam2.mp4 cam3.mp4 --script script.txt

# With separate audio reference
uv run autoeditor \
  --videos cam1.mp4 cam2.mp4 \
  --audio reference_audio.wav \
  --script script.txt \
  --export

# Export multicam timeline
uv run autoeditor \
  --videos cam1.mp4 cam2.mp4 \
  --audio reference_audio.wav \
  --script script.txt \
  --export \
  --output-dir output/multicam_project
```

## Command Line Options

### General Options

- `--script PATH`: Path to script text file (required for LLM analysis)
- `--output-dir PATH`: Reuse existing output directory (skips transcription if JSON exists)
- `--bitrate RATE`: Audio bitrate (default: `192k`)
- `--start-time SECONDS`: Start time for transcription processing
- `--end-time SECONDS`: End time for transcription processing

### Export Options

- `--export`: Export FCPXML timeline from last_attempts.json
- `--no-trim`: Disable loudness-based end time trimming
- `--silence-threshold DB`: Silence threshold in dBFS (default: `-40.0`)
- `--pre-cut-buffer SECONDS`: Seconds to add before each cut (default: `0.1`)
- `--post-cut-buffer SECONDS`: Seconds to add after each cut (default: `0.1`)
- `--original-audio`: Use original audio file in timeline (default: use MP3)

## Workflow

1. **Transcription**: Video/audio is converted to MP3 and transcribed with Whisper
2. **Analysis**: LLM matches transcription to script and identifies "last attempts" (best takes)
3. **Timeline Generation**: FCPXML timeline is created with cuts at identified moments
4. **Export**: Timeline can be imported into DaVinci Resolve or Final Cut Pro

## Output Structure

```
output/
└── YYYYMMDD_HHMMSS/
    ├── source.mp3 (or audio.mp3 for multicam)
    ├── source_transcription.json (or audio_transcription.json)
    ├── last_attempts.json
    ├── llm_log.jsonl
    └── timeline.fcpxml (or timeline_multicam.fcpxml)
```

## Development

### Type Checking

```bash
# Run mypy
uv run mypy .

# Run pyright
uv run pyright
```

### Project Structure

```
autoeditor/
├── src/
│   ├── main.py          # CLI entry point
│   ├── ui.py            # GUI entry point
│   ├── transcription.py  # Whisper transcription
│   ├── llm.py           # OpenAI LLM analysis
│   ├── sync.py          # Audio waveform synchronization
│   ├── fcpxml.py        # FCPXML generation
│   ├── video.py         # Video processing
│   └── files.py         # File management
├── output/              # Generated outputs
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Troubleshooting

### UI doesn't open file pickers
- Make sure you're running in native mode (default)
- Check that pywebview is properly installed: `uv sync`

### Transcription fails
- Verify your OpenAI API key is set in `.env`
- Check that video/audio file is accessible and valid

### Multicam sync issues
- Ensure all video files have audio tracks
- Try using a separate high-quality audio reference file

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]


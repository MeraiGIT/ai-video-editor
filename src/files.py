from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype


@beartype
def get_output_dir(existing_dir: Path | None = None) -> Path:
    """Create and return output directory with current timestamp, or use existing one.

    Args:
        existing_dir: Optional path to an existing output directory to reuse

    Returns:
        Path to the output directory
    """
    if existing_dir is not None:
        if not existing_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {existing_dir}")
        return existing_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@beartype
def find_existing_transcription(output_dir: Path) -> Path | None:
    """Find an existing transcription JSON file in the output directory.

    Args:
        output_dir: Directory to search for transcription files

    Returns:
        Path to the transcription file if found, None otherwise
    """
    json_files = list(output_dir.glob("*_transcription.json"))
    if json_files:
        return json_files[0]
    return None


@beartype
def find_existing_last_attempts(output_dir: Path) -> Path | None:
    """Find an existing last_attempts.json file in the output directory.

    Args:
        output_dir: Directory to search for last_attempts file

    Returns:
        Path to the last_attempts.json file if found, None otherwise
    """
    last_attempts_path = output_dir / "last_attempts.json"
    if last_attempts_path.exists():
        return last_attempts_path
    return None


@beartype
def load_last_attempts(last_attempts_path: Path) -> list[dict[str, Any]]:
    """Load attempts from last_attempts.json file.

    Args:
        last_attempts_path: Path to the last_attempts.json file

    Returns:
        List of attempt dictionaries with timing information
    """
    with last_attempts_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    attempts: list[dict[str, Any]] = data.get("attempts", [])
    return attempts


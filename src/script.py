from __future__ import annotations

import re
from pathlib import Path

from beartype import beartype


@beartype
def load_script_sentences(script_path: Path) -> list[str]:
    """Load and parse script file into individual sentences.

    Args:
        script_path: Path to the script .txt file

    Returns:
        List of non-empty sentences from the script
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with script_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Replace newlines with spaces to treat entire content as continuous text
    content = content.replace("\n", " ")

    # Split by sentence-ending punctuation (. ! ?) followed by space or end
    # Keep the punctuation with the sentence
    raw_sentences = re.split(r'(?<=[.!?])\s+', content)

    # Clean up and filter empty sentences
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    return sentences



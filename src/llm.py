from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype
from dotenv import load_dotenv
from openai import OpenAI

from .script import load_script_sentences
from .transcription import load_transcription_words

# Load environment variables
load_dotenv()


@beartype
def create_llm_prompt(
    words_batch: list[dict[str, Any]],
    sentence1: str,
    sentence2: str | None,
) -> str:
    """Create prompt for LLM to analyze word batch.

    Args:
        words_batch: List of word dictionaries with id, word, start, end
        sentence1: First sentence (current sentence we're matching)
        sentence2: Second sentence (next sentence) or None if last sentence

    Returns:
        Formatted prompt string
    """
    # Format with timestamps for better context
    words_text = " ".join([f"[{w['id']}]{w['word']}" for w in words_batch])

    if sentence2 is not None:
        prompt = f"""You are analyzing a video transcription to find the LAST COMPLETE reading of a sentence.

TRANSCRIPTION (format: [word_id]word):
{words_text}

SENTENCE 1 (find the LAST COMPLETE attempt):
"{sentence1}"

SENTENCE 2 (next sentence in script - helps identify the boundary):
"{sentence2}"

WHAT IS A "COMPLETE" ATTEMPT?
The speaker records multiple takes. A COMPLETE attempt is when they read the ENTIRE sentence from start to finish.
An INCOMPLETE attempt is when they stop mid-sentence, stutter, or restart.

YOUR TASK:
1. Find ALL attempts of SENTENCE 1 in the transcription
2. Identify which attempts are COMPLETE (read the whole sentence, including the ending)
3. Return the LAST (final) COMPLETE attempt
4. Also find where SENTENCE 2 starts (if present)

IMPORTANT:
- The transcription has speech recognition errors - match by meaning, not exact text
- The speaker may say words slightly differently (synonyms, minor variations)
- Focus on finding the attempt that covers the FULL meaning of the sentence
- If you see the speaker restart ("я сделал... я сделал..."), find the LAST full version

RESPONSE FORMAT:
Return JSON with these fields:
- "found": true if a complete attempt was found, false otherwise
- "first_word_id": The [word_id] of the FIRST word of the LAST COMPLETE attempt ONLY
- "last_word_id": The [word_id] of the LAST word of the LAST COMPLETE attempt ONLY
- "sentence2_starts_at": The [word_id] where SENTENCE 2 CONTENT actually begins, or null if not visible

CRITICAL - DO NOT MERGE ATTEMPTS:
- Return ONLY the word range of the SINGLE LAST COMPLETE attempt
- Do NOT include previous incomplete attempts in the range
- Do NOT include expletives, interruptions, or restarts before the last attempt

BAD EXAMPLE (merging attempts):
Transcription: [10]я [11]сделал [12]веб-- [13]блядь [14]Я [15]сделал [16]веб-сервис [17]готово
If [14]-[17] is complete, return first_word_id: 14, last_word_id: 17
Do NOT return first_word_id: 10 (that would include the incomplete attempt [10]-[12])

IMPORTANT: "sentence2_starts_at" should be where SENTENCE 2 actually starts being spoken.
Do NOT set it to just "the word after sentence 1 ends" - only set it if you see actual SENTENCE 2 content.

EXAMPLE:
If transcription is: [0]Привет [1]меня [2]зовут [3]Олег [4]Привет [5]меня [6]зовут [7]Олег [8]Сегодня [9]мы
And SENTENCE 1 is "Привет, меня зовут Олег." and SENTENCE 2 is "Сегодня мы..."
Then [4]-[7] is the LAST complete attempt, and [8] is where SENTENCE 2 starts.
Response: {{"found": true, "first_word_id": 4, "last_word_id": 7, "sentence2_starts_at": 8}}

RESPONSE (JSON only):
If COMPLETE attempt found:
{{"found": true, "first_word_id": <id>, "last_word_id": <id>, "sentence2_starts_at": <id or null>}}

If sentence NOT found in this batch (need more words to search):
{{"found": false, "reason": "EXPAND"}}"""
    else:
        # Last sentence case - no sentence 2 to look for
        prompt = f"""You are analyzing a video transcription to find the LAST COMPLETE reading of a sentence.

TRANSCRIPTION (format: [word_id]word):
{words_text}

SENTENCE TO FIND:
"{sentence1}"

WHAT IS A "COMPLETE" ATTEMPT?
The speaker records multiple takes. A COMPLETE attempt is when they read the ENTIRE sentence from start to finish.

YOUR TASK:
1. Find ALL attempts of this sentence in the transcription
2. Identify which attempts are COMPLETE (read the whole sentence)
3. Return the LAST (final) COMPLETE attempt

IMPORTANT:
- The transcription has speech recognition errors - match by meaning, not exact text
- Focus on finding the attempt that covers the FULL meaning of the sentence
- If you see restarts, find the LAST full version

RESPONSE FORMAT:
Return JSON with these fields:
- "found": true if a complete attempt was found, false otherwise
- "first_word_id": The [word_id] of the FIRST word of the LAST COMPLETE attempt ONLY
- "last_word_id": The [word_id] of the LAST word of the LAST COMPLETE attempt ONLY

CRITICAL - DO NOT MERGE ATTEMPTS:
- Return ONLY the word range of the SINGLE LAST COMPLETE attempt
- Do NOT include previous incomplete attempts in the range
- Do NOT include expletives, interruptions, or restarts before the last attempt

EXAMPLE:
If transcription is: [0]Спасибо [1]за [2]вним-- [3]блядь [4]Спасибо [5]за [6]внимание
And SENTENCE is "Спасибо за внимание."
Then [4]-[6] is the LAST complete attempt (not [0]-[6] which would merge attempts).
Response: {{"found": true, "first_word_id": 4, "last_word_id": 6}}

WHEN TO USE EACH RESPONSE:
- Use "found": true when you find a COMPLETE attempt of the sentence
- Use "NOT_FOUND" when the batch does NOT contain a complete attempt - need more words to search

RESPONSE (JSON only):
If found: {{"found": true, "first_word_id": <id>, "last_word_id": <id>}}
If not found in this batch: {{"found": false, "reason": "NOT_FOUND"}}"""

    return prompt


@beartype
def query_llm(
    prompt: str,
    model: str,
    api_key: str,
    log_file: Path | None = None,
    sentence_idx: int | None = None,
) -> dict[str, Any]:
    """Query OpenRouter LLM with the given prompt.

    Args:
        prompt: The prompt to send
        model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
        api_key: OpenRouter API key
        log_file: Optional path to log file for LLM interactions
        sentence_idx: Optional sentence index for logging context

    Returns:
        Parsed JSON response from the LLM
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty response")

    raw_response = content  # Save for logging

    # Parse JSON from response
    content = content.strip()

    # Handle potential markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last line (the ``` markers)
        content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        content = content.strip()

    # Try to find JSON object in the response
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    try:
        result: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON: {e}")
        print(f"  Raw response: {content[:500]}...")
        result = {"found": False, "reason": "PARSE_ERROR"}

    # Log the interaction
    if log_file is not None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sentence_idx": sentence_idx,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_response": result,
        }

        # Append to log file (JSON lines format)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return result


@beartype
def validate_word_ids(
    words: list[dict[str, Any]],
    first_id: int,
    last_id: int,
) -> bool:
    """Basic sanity check that word IDs are valid.

    Args:
        words: All transcription words
        first_id: First word ID of the found attempt
        last_id: Last word ID of the found attempt

    Returns:
        True if IDs are valid
    """
    if first_id < 0 or last_id < 0:
        return False
    if first_id >= len(words) or last_id >= len(words):
        return False
    if first_id > last_id:
        return False
    return True


@beartype
def create_refinement_prompt(
    selected_words: list[dict[str, Any]],
    sentence: str,
) -> str:
    """Create prompt for LLM to refine/trim the selected word range.

    Args:
        selected_words: List of word dictionaries with id, word, start, end
        sentence: The sentence we're matching

    Returns:
        Formatted prompt string for refinement
    """
    words_text = " ".join([f"[{w['id']}]{w['word']}" for w in selected_words])

    prompt = f"""You are refining a word selection from a video transcription.

SELECTED WORDS (format: [word_id]word):
{words_text}

TARGET SENTENCE:
"{sentence}"

PROBLEM:
The selection may include extra words at the BEGINNING that are NOT part of the last complete attempt.
These could be: incomplete attempts, expletives, restarts, or words from a previous attempt.

YOUR TASK:
1. Identify where the LAST COMPLETE attempt of the sentence actually STARTS
2. Remove any unnecessary words from the BEGINNING of the selection
3. Keep the END of the selection as-is (it's already correct)

EXAMPLES:
- Selection: [10]я [11]сделал [12]веб-- [13]блядь [14]Я [15]сделал [16]веб-сервис
  Sentence: "Я сделал веб-сервис"
  → The last complete attempt is [14]-[16], remove [10]-[13]
  Response: {{"trimmed": true, "new_first_word_id": 14}}

- Selection: [5]Привет [6]меня [7]зовут [8]Олег
  Sentence: "Привет, меня зовут Олег"
  → Already correct, no trimming needed
  Response: {{"trimmed": false}}

- Selection: [20]ну [21]типа [22]Это [23]важный [24]момент
  Sentence: "Это важный момент"
  → Remove filler words [20]-[21], keep [22]-[24]
  Response: {{"trimmed": true, "new_first_word_id": 22}}

RESPONSE FORMAT (JSON only):
If trimming needed: {{"trimmed": true, "new_first_word_id": <id>}}
If no trimming needed: {{"trimmed": false}}

RESPONSE:"""

    return prompt


@beartype
def refine_word_selection(
    words: list[dict[str, Any]],
    first_id: int,
    last_id: int,
    sentence: str,
    model: str,
    api_key: str,
    log_file: Path | None = None,
    sentence_idx: int | None = None,
) -> tuple[int, int]:
    """Refine the selected word range by trimming unnecessary words from the start.

    Args:
        words: All transcription words
        first_id: First word ID of the initial selection
        last_id: Last word ID of the selection
        sentence: The target sentence
        model: OpenRouter model identifier
        api_key: OpenRouter API key
        log_file: Optional path to log file for LLM interactions
        sentence_idx: Optional sentence index for logging context

    Returns:
        Tuple of (refined_first_id, last_id)
    """
    # Extract the selected words
    selected_words = [words[i] for i in range(first_id, min(last_id + 1, len(words)))]

    if len(selected_words) <= 3:
        # Too few words to trim, skip refinement
        return first_id, last_id

    prompt = create_refinement_prompt(selected_words, sentence)
    response = query_llm(prompt, model, api_key, log_file, sentence_idx)

    if response.get("trimmed") and "new_first_word_id" in response:
        new_first_id = response["new_first_word_id"]

        # Validate the new first_id
        if new_first_id >= first_id and new_first_id <= last_id:
            trimmed_count = new_first_id - first_id
            if trimmed_count > 0:
                print(f"  Refinement: trimmed {trimmed_count} words from start "
                      f"(was {first_id}, now {new_first_id})")
                return new_first_id, last_id
        else:
            print(f"  Refinement: invalid new_first_word_id {new_first_id}, keeping original")

    return first_id, last_id


@beartype
def find_last_attempts(
    words: list[dict[str, Any]],
    sentences: list[str],
    batch_size: int,
    model: str,
    api_key: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Find the last attempt boundaries for each sentence in the script.

    Args:
        words: List of transcription words with id, word, start, end
        sentences: List of script sentences
        batch_size: Number of words to process per batch (N)
        model: OpenRouter model identifier
        api_key: OpenRouter API key
        output_dir: Directory for log files

    Returns:
        List of dictionaries with sentence, first_word_id, last_word_id
    """
    # Create log file for LLM interactions
    log_file = output_dir / "llm_log.jsonl"
    print(f"LLM interactions will be logged to: {log_file}")

    results: list[dict[str, Any]] = []
    current_word_idx = 0
    sentence_idx = 0

    # Track last found attempt in case we need to accept it without sentence2
    last_found_attempt: dict[str, Any] | None = None

    while sentence_idx < len(sentences) and current_word_idx < len(words):
        sentence1 = sentences[sentence_idx]
        sentence2 = sentences[sentence_idx + 1] if sentence_idx + 1 < len(sentences) else None

        if sentence2 is not None:
            print(f"\nProcessing sentence {sentence_idx + 1}/{len(sentences)}: {sentence1[:50]}...")
        else:
            print(f"\nProcessing sentence {sentence_idx + 1}/{len(sentences)} (final): {sentence1[:50]}...")

        # Start with batch_size words
        batch_end = min(current_word_idx + batch_size, len(words))
        attempts = 0
        max_attempts = 10  # Safety limit
        found_valid = False
        exhausted_batch = False
        last_found_attempt = None

        while attempts < max_attempts and not found_valid and not exhausted_batch:
            attempts += 1
            batch = words[current_word_idx:batch_end]

            if not batch:
                print("  No more words to process")
                break

            print(f"  Attempt {attempts}: Processing words {current_word_idx}-{batch_end - 1} ({len(batch)} words)")

            prompt = create_llm_prompt(batch, sentence1, sentence2)
            response = query_llm(prompt, model, api_key, log_file, sentence_idx)

            if response.get("found"):
                first_id = response["first_word_id"]
                last_id = response["last_word_id"]
                sentence2_start = response.get("sentence2_starts_at")

                print(f"  LLM found: words {first_id}-{last_id}")
                if sentence2_start:
                    print(f"  Next sentence starts at: word {sentence2_start}")

                # Basic validation - just check IDs are valid
                if validate_word_ids(words, first_id, last_id):
                    # Save this attempt in case we need it later
                    last_found_attempt = {
                        "first_id": first_id,
                        "last_id": last_id,
                        "sentence2_start": sentence2_start,
                    }

                    # If sentence2 is not null but sentence2_starts_at is null,
                    # expand to find the TRUE last attempt
                    if sentence2 is not None and sentence2_start is None:
                        print(f"  Found attempt at {first_id}-{last_id}, but sentence2 not found yet")
                        new_batch_end = min(batch_end + batch_size, len(words))
                        if new_batch_end == batch_end:
                            # Can't expand further, accept last found attempt
                            exhausted_batch = True
                        else:
                            batch_end = new_batch_end
                            print(f"  Expanding batch to {batch_end - current_word_idx} words")
                    else:
                        # Either this is the last sentence (sentence2 is None)
                        # or we found where sentence2 starts - accept this result
                        found_valid = True
                else:
                    # Invalid IDs, try expanding
                    print("  ✗ Invalid word IDs, expanding batch...")
                    new_batch_end = min(batch_end + batch_size, len(words))
                    if new_batch_end == batch_end:
                        exhausted_batch = True
                    else:
                        batch_end = new_batch_end
            else:
                reason = response.get("reason", "unknown")
                print(f"  LLM returned found=false, reason: {reason}")

                # Expand batch by batch_size more words
                new_batch_end = min(batch_end + batch_size, len(words))
                if new_batch_end == batch_end:
                    exhausted_batch = True
                else:
                    batch_end = new_batch_end
                    print(f"  Expanding batch to {batch_end - current_word_idx} words")

        # Process results
        if found_valid and last_found_attempt is not None:
            # Found with sentence2 boundary
            first_id = last_found_attempt["first_id"]
            last_id = last_found_attempt["last_id"]
            sentence2_start = last_found_attempt["sentence2_start"]

            # Refine the selection to trim unnecessary words
            first_id, last_id = refine_word_selection(
                words, first_id, last_id, sentence1,
                model, api_key, log_file, sentence_idx
            )
            found_text = " ".join([words[i]["word"] for i in range(first_id, min(last_id + 1, len(words)))])
            print(f"  ✓ Found: \"{found_text[:100]}...\"")

            results.append({
                "sentence_idx": sentence_idx,
                "sentence": sentence1,
                "first_word_id": first_id,
                "last_word_id": last_id,
                "first_word": words[first_id]["word"] if first_id < len(words) else None,
                "last_word": words[last_id]["word"] if last_id < len(words) else None,
                "start_time": words[first_id]["start"] if first_id < len(words) else None,
                "end_time": words[last_id]["end"] if last_id < len(words) else None,
                "last_word_start": words[last_id]["start"] if last_id < len(words) else None,
            })

            # Move position based on sentence2_start if available, otherwise use last_id
            if sentence2_start and sentence2_start > last_id:
                current_word_idx = sentence2_start
            else:
                current_word_idx = last_id + 1

            sentence_idx += 1

        elif last_found_attempt is not None:
            # Exhausted batch but have a found attempt - accept it
            first_id = last_found_attempt["first_id"]
            last_id = last_found_attempt["last_id"]

            # Refine the selection to trim unnecessary words
            first_id, last_id = refine_word_selection(
                words, first_id, last_id, sentence1,
                model, api_key, log_file, sentence_idx
            )
            found_text = " ".join([words[i]["word"] for i in range(first_id, min(last_id + 1, len(words)))])
            print(f"  ✓ Accepting last found attempt: \"{found_text[:100]}...\"")

            results.append({
                "sentence_idx": sentence_idx,
                "sentence": sentence1,
                "first_word_id": first_id,
                "last_word_id": last_id,
                "first_word": words[first_id]["word"] if first_id < len(words) else None,
                "last_word": words[last_id]["word"] if last_id < len(words) else None,
                "start_time": words[first_id]["start"] if first_id < len(words) else None,
                "end_time": words[last_id]["end"] if last_id < len(words) else None,
                "last_word_start": words[last_id]["start"] if last_id < len(words) else None,
            })

            current_word_idx = last_id + 1
            sentence_idx += 1

        else:
            # Could not find sentence at all
            if attempts >= max_attempts:
                print("  Max attempts reached")
            print("  ✗ Recording as NOT FOUND and continuing...")
            results.append({
                "sentence_idx": sentence_idx,
                "sentence": sentence1,
                "first_word_id": None,
                "last_word_id": None,
                "first_word": None,
                "last_word": None,
                "start_time": None,
                "end_time": None,
                "last_word_start": None,
                "status": "NOT_FOUND",
            })
            sentence_idx += 1

    return results


@beartype
def process_last_attempts(
    transcription_path: Path,
    script_path: Path,
    output_dir: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Path:
    """Process transcription to find last attempts for each script sentence.

    Args:
        transcription_path: Path to the transcription JSON file
        script_path: Path to the script .txt file
        output_dir: Directory to save the output JSON
        start_time: Optional start time in seconds (filter transcription from this time)
        end_time: Optional end time in seconds (filter transcription until this time)

    Returns:
        Path to the created last_attempts.json file
    """
    # Load environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    batch_size = int(os.getenv("WORD_BATCH_SIZE", "50"))

    print(f"Using model: {model}")
    print(f"Batch size: {batch_size}")

    # Log time range if specified
    if start_time is not None or end_time is not None:
        start_str = f"{start_time:.1f}s" if start_time is not None else "beginning"
        end_str = f"{end_time:.1f}s" if end_time is not None else "end"
        print(f"Time range: {start_str} to {end_str}")

    # Load data with time filtering
    words = load_transcription_words(transcription_path, start_time, end_time)
    sentences = load_script_sentences(script_path)

    print(f"Loaded {len(words)} words from transcription")
    print(f"Loaded {len(sentences)} sentences from script")

    # Find last attempts
    results = find_last_attempts(words, sentences, batch_size, model, api_key, output_dir)

    # Save results
    output_path = output_dir / "last_attempts.json"
    output_data = {
        "model": model,
        "batch_size": batch_size,
        "total_sentences": len(sentences),
        "sentences_found": len(results),
        "attempts": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nLast attempts saved to: {output_path}")
    return output_path


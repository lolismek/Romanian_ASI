#!/usr/bin/env python3
"""
Load and parse RoEmoLex V3 emotion lexicon for RO-ASI benchmark.

RoEmoLex V3 contains ~11,051 Romanian words with emotion annotations:
- 8 basic Plutchik emotions: Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust
- Polarity: P (positive) / N (negative)
- POS tags
- SUMO ontology categories

Download files from: https://www.cs.ubbcluj.ro/~ria/romanian-nlp-resources/
Place in: data/roemolex/
- RoEmoLex_V3_expr.csv
- RoEmoLex_V3_pos.csv

This module provides fallback to the manual emotion_seed.py if RoEmoLex files are not available.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Emotion column names in RoEmoLex CSV (lowercase for lookup)
ROEMOLEX_EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

# Romanian to English emotion name mapping (for RoEmoLex V3 CSV columns)
ROMANIAN_EMOTION_MAP = {
    "furie": "anger",
    "anticipare": "anticipation",
    "dezgust": "disgust",
    "frica": "fear",
    "bucurie": "joy",
    "tristete": "sadness",
    "surpriza": "surprise",
    "incredere": "trust",
    # With diacritics
    "frică": "fear",
    "tristețe": "sadness",
    "surpriză": "surprise",
    "încredere": "trust",
}

# POS tags that work with "I feel" patterns
# ADJ: adjectives (mă simt fericit)
# NOUN: nouns (am frică)
# ADV: some adverbs (mă simt bine)
# Also include Romanian POS names from RoEmoLex
VALID_POS_TAGS = {
    "ADJ", "NOUN", "ADV", "adj", "noun", "adv", "A", "N", "R",
    "Adjective", "Noun", "Adverb",  # RoEmoLex format
    "Expresie",  # Expressions (multi-word)
}


def normalize_column_name(col: str) -> str:
    """Normalize column name for matching."""
    return col.strip().lower()


def parse_roemolex_csv(file_path: Path) -> List[Dict]:
    """
    Parse RoEmoLex CSV file.

    Expected columns: Word, POS, P, N, Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust, SUMO

    Returns list of word entries with their emotion annotations.
    """
    entries = []

    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to detect delimiter (could be comma or semicolon)
        sample = f.read(2048)
        f.seek(0)

        delimiter = ','
        if sample.count(';') > sample.count(','):
            delimiter = ';'

        reader = csv.DictReader(f, delimiter=delimiter)

        # Normalize column names
        if reader.fieldnames:
            col_map = {normalize_column_name(c): c for c in reader.fieldnames}
        else:
            return entries

        for row in reader:
            # Find word column (might be "word", "Word", "WORD", etc.)
            word = None
            for possible_col in ["word", "cuvant", "cuvânt", "expr", "expression"]:
                if possible_col in col_map:
                    word = row.get(col_map[possible_col], "").strip()
                    if word:
                        break

            if not word:
                # Try first column
                first_col = reader.fieldnames[0] if reader.fieldnames else None
                if first_col:
                    word = row.get(first_col, "").strip()

            if not word:
                continue

            # Get POS tag
            pos = None
            for possible_col in ["pos", "part of speech", "part_of_speech", "categorie"]:
                if possible_col in col_map:
                    pos = row.get(col_map[possible_col], "").strip()
                    if pos:
                        break

            # Get polarity
            polarity_positive = False
            polarity_negative = False

            for possible_col in ["p", "positive", "pozitiv", "pozitivitate"]:
                if possible_col in col_map:
                    val = row.get(col_map[possible_col], "").strip()
                    if val in ["1", "1.0", "true", "True", "TRUE"]:
                        polarity_positive = True
                        break

            for possible_col in ["n", "negative", "negativ", "negativitate"]:
                if possible_col in col_map:
                    val = row.get(col_map[possible_col], "").strip()
                    if val in ["1", "1.0", "true", "True", "TRUE"]:
                        polarity_negative = True
                        break

            # Get emotion annotations
            emotions = []

            # First try Romanian column names (RoEmoLex V3 format)
            for ro_name, en_name in ROMANIAN_EMOTION_MAP.items():
                if ro_name in col_map:
                    val = row.get(col_map[ro_name], "").strip()
                    if val in ["1", "1.0", "true", "True", "TRUE"]:
                        if en_name not in emotions:
                            emotions.append(en_name)

            # If no Romanian columns found, try English names
            if not emotions:
                for emotion in ROEMOLEX_EMOTIONS:
                    for possible_col in [emotion, emotion.capitalize(), emotion.upper()]:
                        if possible_col.lower() in col_map:
                            val = row.get(col_map[possible_col.lower()], "").strip()
                            if val in ["1", "1.0", "true", "True", "TRUE"]:
                                emotions.append(emotion)
                            break

            # Get SUMO category if available
            sumo = None
            for possible_col in ["sumo", "sumo_category"]:
                if possible_col in col_map:
                    sumo = row.get(col_map[possible_col], "").strip()
                    if sumo:
                        break

            entry = {
                "word": word,
                "pos": pos,
                "polarity_positive": polarity_positive,
                "polarity_negative": polarity_negative,
                "emotions": emotions,
                "sumo": sumo
            }
            entries.append(entry)

    return entries


def load_roemolex(roemolex_dir: Path) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Load RoEmoLex V3 from directory containing CSV files.

    Args:
        roemolex_dir: Directory containing RoEmoLex CSV files

    Returns:
        Tuple of (all_entries, word_to_emotions_dict)
    """
    all_entries = []
    word_to_emotions: Dict[str, Set[str]] = {}

    # Look for CSV files
    csv_files = list(roemolex_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {roemolex_dir}")
        return all_entries, {}

    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        entries = parse_roemolex_csv(csv_file)
        print(f"  Parsed {len(entries)} entries")

        for entry in entries:
            word = entry["word"].lower()

            # Add to word→emotions mapping
            if word not in word_to_emotions:
                word_to_emotions[word] = set()
            word_to_emotions[word].update(entry["emotions"])

        all_entries.extend(entries)

    # Convert sets to lists for JSON serialization
    word_to_emotions_list = {w: sorted(list(e)) for w, e in word_to_emotions.items()}

    return all_entries, word_to_emotions_list


def filter_for_i_feel_patterns(entries: List[Dict], word_to_emotions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Filter entries to only include ADJECTIVES for "I feel" patterns.

    For ASI benchmark, we only want adjectives like:
    - mă simt fericit/fericită (I feel happy)
    - sunt trist (I am sad)

    Nouns are handled separately via EMOTION_NOUNS_ONLY in pattern_matcher.py
    to avoid false positives like "sunt elev" (I am a student).
    """
    filtered = {}

    # Track POS tags for each word
    word_pos: Dict[str, Set[str]] = {}
    for entry in entries:
        word = entry["word"].lower()
        if entry["pos"]:
            if word not in word_pos:
                word_pos[word] = set()
            word_pos[word].add(entry["pos"].upper())

    # Only valid POS for adjective-based patterns
    ADJECTIVE_POS_TAGS = {"ADJ", "ADJECTIVE", "A", "ADV", "ADVERB", "R"}

    for word, emotions in word_to_emotions.items():
        if not emotions:
            continue

        # Check if word has valid POS tag
        pos_tags = word_pos.get(word, set())

        # Only include if it's an adjective or adverb
        if pos_tags:
            if pos_tags & ADJECTIVE_POS_TAGS:
                filtered[word] = emotions
        # Skip words without POS info - too risky for false positives

    return filtered


def build_emotion_seed_from_roemolex(roemolex_dir: Path, output_path: Path) -> Dict:
    """
    Build emotion seed from RoEmoLex V3 files.

    Args:
        roemolex_dir: Directory containing RoEmoLex CSV files
        output_path: Path to save emotion_seed.json

    Returns:
        Emotion seed dictionary
    """
    all_entries, word_to_emotions = load_roemolex(roemolex_dir)

    if not all_entries:
        print("No entries loaded from RoEmoLex. Check file format.")
        return {}

    # Filter for words usable with "I feel" patterns
    filtered_words = filter_for_i_feel_patterns(all_entries, word_to_emotions)

    # Build seed dictionary
    seed = {
        "source": "roemolex_v3",
        "word_to_emotions": filtered_words,
        "all_words": sorted(filtered_words.keys()),
        "statistics": {
            "total_entries": len(all_entries),
            "total_words": len(word_to_emotions),
            "filtered_words": len(filtered_words),
            "emotions_coverage": {}
        }
    }

    # Calculate emotion coverage
    for emotion in ROEMOLEX_EMOTIONS:
        count = sum(1 for e in filtered_words.values() if emotion in e)
        seed["statistics"]["emotions_coverage"][emotion] = count

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"\nSaved emotion seed to {output_path}")
    print(f"  Total RoEmoLex entries: {seed['statistics']['total_entries']}")
    print(f"  Total unique words: {seed['statistics']['total_words']}")
    print(f"  Filtered for patterns: {seed['statistics']['filtered_words']}")
    print(f"  Emotion coverage: {seed['statistics']['emotions_coverage']}")

    return seed


def load_emotion_seed(seed_path: Path) -> Dict:
    """Load emotion seed from JSON file."""
    with open(seed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def use_manual_fallback(output_path: Path) -> Dict:
    """
    Use manually curated affective states as fallback.

    Imports from curated_affective_states.py module.
    """
    try:
        from . import curated_affective_states as curated
        seed = curated.build_curated_seed()
        print(f"Using curated affective states")
    except ImportError:
        # Fall back to old emotion_seed.py
        from . import emotion_seed as manual_seed
        seed = manual_seed.build_emotion_seed()
        seed["source"] = "manual_fallback"
        print(f"Using legacy manual fallback emotion seed")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {output_path}")
    print(f"  Total words: {seed['statistics'].get('total_words', seed['statistics'].get('total_word_forms', 0))}")

    return seed


def load_or_create_emotion_seed(
    roemolex_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    force_rebuild: bool = False
) -> Dict:
    """
    Load existing emotion seed or create from RoEmoLex/fallback.

    Args:
        roemolex_dir: Directory with RoEmoLex CSV files (optional)
        output_path: Path for emotion_seed.json
        force_rebuild: If True, rebuild even if output exists

    Returns:
        Emotion seed dictionary
    """
    # Set default paths
    base_path = Path(__file__).parent.parent.parent
    if roemolex_dir is None:
        roemolex_dir = base_path / "data" / "roemolex"
    if output_path is None:
        output_path = base_path / "data" / "emotion_seed.json"

    # Check if output exists
    if output_path.exists() and not force_rebuild:
        print(f"Loading existing emotion seed from {output_path}")
        return load_emotion_seed(output_path)

    # Try RoEmoLex first
    if roemolex_dir.exists() and list(roemolex_dir.glob("*.csv")):
        print(f"Building emotion seed from RoEmoLex at {roemolex_dir}")
        return build_emotion_seed_from_roemolex(roemolex_dir, output_path)

    # Fall back to manual seed
    print(f"RoEmoLex not found at {roemolex_dir}")
    return use_manual_fallback(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load RoEmoLex V3 emotion lexicon")
    parser.add_argument(
        "--roemolex-dir",
        type=Path,
        default=None,
        help="Directory containing RoEmoLex CSV files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for emotion_seed.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if output exists"
    )

    args = parser.parse_args()

    seed = load_or_create_emotion_seed(
        roemolex_dir=args.roemolex_dir,
        output_path=args.output,
        force_rebuild=args.force
    )

    # Print sample
    print("\nSample words by emotion:")
    word_to_emotions = seed.get("word_to_emotions", {})
    for emotion in ROEMOLEX_EMOTIONS:
        words = [w for w, e in word_to_emotions.items() if emotion in e][:5]
        print(f"  {emotion}: {', '.join(words)}")

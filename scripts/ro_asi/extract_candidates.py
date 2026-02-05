#!/usr/bin/env python3
"""
Extract ASI candidates from merged Romanian corpus.

Main extraction script that:
1. Loads emotion seed (from RoEmoLex or manual fallback)
2. Loads merged corpus
3. Runs pattern matching to find "I feel [state]" expressions
4. Outputs ASI candidates to JSONL file

Usage:
    python extract_candidates.py
    python extract_candidates.py --corpus data/merged_corpus.jsonl --output data/asi_candidates.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Generator, Any, Set
from collections import defaultdict
import hashlib
from datetime import datetime

from .load_roemolex import load_or_create_emotion_seed
from .pattern_matcher import PatternMatcher, PatternMatch


def load_merged_corpus(corpus_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Load merged corpus from JSONL file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def format_candidate(
    record: Dict[str, Any],
    match: PatternMatch
) -> Dict[str, Any]:
    """
    Format a matched record as ASI candidate.

    Output schema:
    {
        "id": str,                    # From merged corpus
        "text": str,                  # Full original text
        "matched_sentence": str,      # Sentence containing the pattern
        "pattern_used": str,          # Which pattern matched
        "pattern_category": str,      # primary/secondary
        "seed_word": str,             # The affective state word found
        "seed_word_normalized": str,  # Normalized version
        "emotion_category": list,     # Emotions from lexicon
        "source": str                 # Original dataset
    }
    """
    return {
        "id": record["id"],
        "text": record["text"],
        "matched_sentence": match.matched_text,
        "pattern_used": match.pattern_name,
        "pattern_category": match.pattern_category,
        "seed_word": match.seed_word,
        "seed_word_normalized": match.seed_word_normalized,
        "emotion_category": match.emotions,
        "source": record.get("source", "unknown"),
        "split": record.get("split", "unknown"),
        "original_labels": record.get("original_labels", {})
    }


def extract_candidates(
    corpus_path: Path,
    emotion_seed: Dict,
    output_path: Path,
    max_records: int = 0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract ASI candidates from corpus.

    Args:
        corpus_path: Path to merged_corpus.jsonl
        emotion_seed: Emotion seed dictionary
        output_path: Path for asi_candidates.jsonl
        max_records: Maximum records to process (0 = all)
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_processed": 0,
        "total_matches": 0,
        "unique_texts_matched": 0,
        "by_source": defaultdict(int),
        "by_pattern": defaultdict(int),
        "by_emotion": defaultdict(int),
        "unique_seed_words": set(),
        "started_at": datetime.now().isoformat(),
    }

    # Initialize pattern matcher
    word_to_emotions = emotion_seed.get("word_to_emotions", {})
    if not word_to_emotions:
        print("Error: No word_to_emotions in emotion seed")
        return stats

    # Get noun words from seed if available (curated seed has separate lists)
    noun_words = emotion_seed.get("nouns", None)

    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words)

    # Track unique texts to avoid duplicate candidates
    seen_text_hashes: Set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i, record in enumerate(load_merged_corpus(corpus_path)):
            if max_records > 0 and i >= max_records:
                break

            stats["total_processed"] += 1

            if verbose and stats["total_processed"] % 10000 == 0:
                print(f"Processed {stats['total_processed']} records, "
                      f"found {stats['total_matches']} matches...")

            text = record.get("text", "")
            if not text:
                continue

            # Check for duplicate text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen_text_hashes:
                continue

            # Find matches
            matches = matcher.find_matches(text, extract_sentences=True)

            if matches:
                seen_text_hashes.add(text_hash)
                stats["unique_texts_matched"] += 1

                for match in matches:
                    candidate = format_candidate(record, match)

                    # Write to output
                    out_f.write(json.dumps(candidate, ensure_ascii=False) + '\n')

                    # Update stats
                    stats["total_matches"] += 1
                    stats["by_source"][record.get("source", "unknown")] += 1
                    stats["by_pattern"][match.pattern_name] += 1
                    stats["unique_seed_words"].add(match.seed_word_normalized)

                    for emotion in match.emotions:
                        stats["by_emotion"][emotion] += 1

    # Convert set to count for JSON serialization
    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(list(stats["unique_seed_words"]))[:50]  # Sample
    stats["finished_at"] = datetime.now().isoformat()

    # Convert defaultdicts to regular dicts
    stats["by_source"] = dict(stats["by_source"])
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])

    return stats


def print_stats(stats: Dict[str, Any]):
    """Print extraction statistics."""
    print("\n" + "=" * 60)
    print("ASI Candidate Extraction Statistics")
    print("=" * 60)
    print(f"Total records processed: {stats['total_processed']}")
    print(f"Unique texts matched: {stats['unique_texts_matched']}")
    print(f"Total pattern matches: {stats['total_matches']}")
    print(f"Unique seed words found: {stats['unique_seed_words_count']}")

    print("\nMatches by source:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    print("\nMatches by pattern:")
    for pattern, count in sorted(stats['by_pattern'].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    print("\nMatches by emotion:")
    for emotion, count in sorted(stats['by_emotion'].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count}")

    if stats['unique_seed_words']:
        print(f"\nSample seed words matched: {', '.join(stats['unique_seed_words'][:20])}")


def sample_candidates(output_path: Path, n: int = 10):
    """Print sample candidates for verification."""
    print(f"\n{'=' * 60}")
    print(f"Sample Candidates (first {n})")
    print("=" * 60)

    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidate = json.loads(line)
            print(f"\n[{i+1}] ID: {candidate['id']}")
            print(f"    Source: {candidate['source']}")
            print(f"    Pattern: {candidate['pattern_used']} ({candidate['pattern_category']})")
            print(f"    Seed word: {candidate['seed_word']} → {candidate['emotion_category']}")
            print(f"    Matched: \"{candidate['matched_sentence'][:100]}...\""
                  if len(candidate['matched_sentence']) > 100
                  else f"    Matched: \"{candidate['matched_sentence']}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ASI candidates from Romanian corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to merged_corpus.jsonl"
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="Path to emotion_seed.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for asi_candidates.jsonl"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Max records to process (0 = all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample candidates to print"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Set default paths
    base_path = Path(__file__).parent.parent.parent
    corpus_path = args.corpus or base_path / "data" / "merged_corpus.jsonl"
    seed_path = args.seed or base_path / "data" / "emotion_seed.json"
    output_path = args.output or base_path / "data" / "asi_candidates.jsonl"

    # Check corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print("Run merge_datasets.py first to create the merged corpus.")
        return 1

    # Load or create emotion seed
    print(f"Loading emotion seed...")
    emotion_seed = load_or_create_emotion_seed(output_path=seed_path)

    print(f"\nCorpus: {corpus_path}")
    print(f"Output: {output_path}")

    # Run extraction
    stats = extract_candidates(
        corpus_path=corpus_path,
        emotion_seed=emotion_seed,
        output_path=output_path,
        max_records=args.max_records,
        verbose=not args.quiet
    )

    # Print statistics
    print_stats(stats)

    # Save stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    # Print samples
    if args.sample > 0 and stats['total_matches'] > 0:
        sample_candidates(output_path, args.sample)

    print(f"\nOutput saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())

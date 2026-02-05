#!/usr/bin/env python3
"""
Extract ASI candidates from the FULG dataset using HuggingFace streaming.

FULG dataset: 150B tokens, 289GB of Romanian web text.
Uses streaming to process without loading full dataset into memory.

Features:
- Streaming processing (never loads full dataset)
- Sentence-level context extraction (not full pages)
- Checkpoint/resume support for long runs
- Pre-filtering for efficiency (keyword + language score)
- Deduplication via MD5 hashes
- Soft domain categorization for analysis
- Compatible output format with existing ASI pipeline

Usage:
    python -m scripts.fulg.extract_candidates
    python -m scripts.fulg.extract_candidates --max-samples 100000 --max-records 5000000
    python -m scripts.fulg.extract_candidates --resume
    python -m scripts.fulg.extract_candidates --context-sentences 3
"""

import json
import re
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Generator, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

# Import from existing ro_asi modules
from scripts.ro_asi.load_roemolex import load_or_create_emotion_seed
from scripts.ro_asi.pattern_matcher import PatternMatcher, PatternMatch


@dataclass
class FulgExtractionConfig:
    """Configuration for FULG extraction."""
    max_extracted_samples: int = 50_000  # Stop after N extracted samples
    max_records_processed: int = 0       # 0 = unlimited
    batch_size: int = 1000               # Records per batch for progress reporting
    checkpoint_every: int = 10_000       # Save progress every N records
    min_text_length: int = 100           # Skip very short texts
    max_text_length: int = 100_000       # Skip very long texts (likely not personal content)
    min_language_score: float = 0.8      # Minimum Romanian confidence
    context_sentences: int = 2           # Sentences before/after match to include
    max_context_length: int = 1000       # Max chars for context window
    output_path: Path = field(default_factory=lambda: Path("data/fulg_asi_candidates.jsonl"))
    checkpoint_path: Path = field(default_factory=lambda: Path("data/fulg_extraction_checkpoint.json"))


# Trigger words for quick pre-filtering
# These are stems/roots that appear in "I feel" patterns
TRIGGER_WORDS = {
    "simt", "sunt", "eram", "fost", "mi-e", "mie",
    "imi", "îmi", "simteam", "simțeam", "simtit", "simțit",
    "suntem", "simtim", "simțim"
}

# Domain categorization for analysis (soft tagging, NO filtering)
# Categories: forum, social, blog, qa, news, wiki, other
DOMAIN_PATTERNS = {
    # Forums - Romanian and international
    "forum": [
        r"forum\.", r"\.forum\.", r"/forum",
        r"softpedia\.ro", r"sfatulmedicului\.ro", r"ciao\.ro",
        r"pcgarage\.ro", r"emag\.ro/forum",
        # International forums
        r"reddit\.com", r"reddit\.com/r/romania",
        r"quora\.com", r"stackexchange\.com", r"stackoverflow\.com",
        r"4chan\.org", r"boards\.",
    ],
    # Social media - all major platforms
    "social": [
        r"facebook\.com", r"fb\.com", r"twitter\.com", r"x\.com",
        r"instagram\.com", r"tiktok\.com", r"youtube\.com",
        r"linkedin\.com", r"pinterest\.com", r"tumblr\.com",
        r"vk\.com", r"ok\.ru", r"discord\.com",
        # Romanian social
        r"trilulilu\.ro", r"hi5\.com",
    ],
    # Blogs and personal sites
    "blog": [
        r"blogspot\.", r"wordpress\.com", r"wordpress\.org",
        r"medium\.com", r"substack\.com", r"blogger\.com",
        r"typepad\.com", r"livejournal\.com", r"wix\.com",
        r"blog\.", r"\.blog",
    ],
    # Q&A and discussion platforms
    "qa": [
        r"answers\.", r"ask\.", r"raspunsuri\.",
        r"intrebari\.", r"discutii\.", r"comunitate\.",
        r"yahoo\.com/answers", r"askfm\.com",
    ],
    # News sites (typically lower quality for personal expression)
    "news": [
        r"hotnews\.ro", r"digi24\.ro", r"mediafax\.ro",
        r"adevarul\.ro", r"libertatea\.ro", r"gandul\.ro",
        r"stirileprotv\.ro", r"observator\.tv", r"antena3\.ro",
        r"realitatea\.net", r"ziare\.com", r"news\.",
        r"bbc\.com", r"cnn\.com", r"reuters\.com",
    ],
    # Wiki/encyclopedic (typically not personal)
    "wiki": [
        r"wikipedia\.org", r"wikimedia\.org", r"wiktionary\.org",
        r"wiki\.", r"fandom\.com", r"wikia\.com",
    ],
    # Reviews and e-commerce (can have personal opinions)
    "reviews": [
        r"tripadvisor\.", r"booking\.com", r"yelp\.com",
        r"trustpilot\.com", r"amazon\.", r"emag\.ro",
        r"reviews\.", r"recenzii\.",
    ],
}


def categorize_domain(domain: str, url: str = "") -> str:
    """
    Categorize a domain into a source type.
    Returns category string for tagging (not filtering).
    """
    check_str = f"{domain} {url}".lower()

    for category, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, check_str):
                return category

    return "other"


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into sentences with their positions.

    Returns list of (start_pos, end_pos, sentence_text) tuples.
    """
    # Sentence-ending punctuation followed by space or end
    # Be careful with abbreviations (Dr., etc.) - simple heuristic
    sentence_pattern = r'[.!?]+(?:\s+|$)'

    sentences = []
    last_end = 0

    for match in re.finditer(sentence_pattern, text):
        end_pos = match.end()
        sentence = text[last_end:end_pos].strip()
        if sentence and len(sentence) > 10:  # Skip very short fragments
            sentences.append((last_end, end_pos, sentence))
        last_end = end_pos

    # Add any remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining and len(remaining) > 10:
            sentences.append((last_end, len(text), remaining))

    return sentences


def extract_context_window(
    text: str,
    match_start: int,
    match_end: int,
    num_sentences: int = 2,
    max_length: int = 1000
) -> Tuple[str, str, str]:
    """
    Extract a context window around a match using sentence boundaries.

    Args:
        text: Full text
        match_start: Start position of match in text
        match_end: End position of match in text
        num_sentences: Number of sentences before/after to include
        max_length: Maximum total context length

    Returns:
        Tuple of (context_before, matched_sentence, context_after)
    """
    sentences = split_into_sentences(text)

    if not sentences:
        # Fallback: just return a character window
        start = max(0, match_start - 200)
        end = min(len(text), match_end + 200)
        return ("", text[start:end].strip(), "")

    # Find which sentence contains the match
    match_sentence_idx = -1
    for i, (start, end, sent) in enumerate(sentences):
        if start <= match_start < end:
            match_sentence_idx = i
            break

    if match_sentence_idx == -1:
        # Match not found in any sentence, use character window
        start = max(0, match_start - 200)
        end = min(len(text), match_end + 200)
        return ("", text[start:end].strip(), "")

    # Get matched sentence
    matched_sentence = sentences[match_sentence_idx][2]

    # Get context before
    before_sentences = []
    for i in range(max(0, match_sentence_idx - num_sentences), match_sentence_idx):
        before_sentences.append(sentences[i][2])
    context_before = " ".join(before_sentences)

    # Get context after
    after_sentences = []
    for i in range(match_sentence_idx + 1, min(len(sentences), match_sentence_idx + num_sentences + 1)):
        after_sentences.append(sentences[i][2])
    context_after = " ".join(after_sentences)

    # Trim if too long
    total_len = len(context_before) + len(matched_sentence) + len(context_after)
    if total_len > max_length:
        # Prioritize matched sentence, then trim context
        available = max_length - len(matched_sentence)
        half = available // 2
        if len(context_before) > half:
            context_before = "..." + context_before[-(half-3):]
        if len(context_after) > half:
            context_after = context_after[:half-3] + "..."

    return (context_before, matched_sentence, context_after)


def stream_fulg_dataset() -> Generator[Dict[str, Any], None, None]:
    """
    Stream FULG dataset with HuggingFace datasets library.

    FULG schema:
        raw_content: str - Main text
        url: str
        title: str
        source_domain: str
        length: int
        language_score: float (0.74-1.0)
        digest: str - Unique identifier

    Yields normalized records compatible with our extraction pipeline.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. "
            "Install with: pip install datasets"
        )

    print("Loading FULG dataset with streaming...")
    ds = load_dataset("faur-ai/fulg", split="train", streaming=True)

    for record in ds:
        # Normalize to common schema
        digest = record.get("digest", "")
        yield {
            "text": record.get("raw_content", ""),
            "id": f"fulg_{digest[:12]}" if digest else f"fulg_{hash(record.get('url', ''))}"[:16],
            "source": "fulg",
            "url": record.get("url", ""),
            "title": record.get("title", ""),
            "source_domain": record.get("source_domain", ""),
            "language_score": record.get("language_score", 0),
            "length": record.get("length", 0),
        }


def should_process(record: Dict[str, Any], config: FulgExtractionConfig) -> Tuple[bool, str]:
    """
    Quick pre-filter to reduce pattern matching load.

    Filters:
    1. Minimum text length
    2. Maximum text length (skip very long pages)
    3. Minimum language score (Romanian confidence)
    4. Contains trigger words (quick keyword check)

    Returns:
        Tuple of (should_process, skip_reason)
    """
    text = record.get("text", "")

    # Length filters
    if len(text) < config.min_text_length:
        return False, "too_short"

    if len(text) > config.max_text_length:
        return False, "too_long"

    # Language score filter
    if record.get("language_score", 0) < config.min_language_score:
        return False, "low_language_score"

    # Quick keyword filter - check if any trigger word is present
    text_lower = text.lower()
    if not any(word in text_lower for word in TRIGGER_WORDS):
        return False, "no_trigger_words"

    return True, ""


def load_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load checkpoint from previous run.

    Returns:
        Tuple of (stats_dict, seen_hashes_set, start_offset)
    """
    if not checkpoint_path.exists():
        return (
            {
                "total_processed": 0,
                "total_matches": 0,
                "filtered_out": 0,
                "filter_reasons": defaultdict(int),
                "duplicates_skipped": 0,
                "by_source_domain": defaultdict(int),
                "by_source_category": defaultdict(int),
                "by_pattern": defaultdict(int),
                "by_emotion": defaultdict(int),
                "unique_seed_words": set(),
                "context_lengths": [],
                "started_at": datetime.now().isoformat(),
            },
            set(),
            0
        )

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)

    stats = {
        "total_processed": checkpoint.get("total_processed", 0),
        "total_matches": checkpoint.get("total_matches", 0),
        "filtered_out": checkpoint.get("filtered_out", 0),
        "filter_reasons": defaultdict(int, checkpoint.get("filter_reasons", {})),
        "duplicates_skipped": checkpoint.get("duplicates_skipped", 0),
        "by_source_domain": defaultdict(int, checkpoint.get("by_source_domain", {})),
        "by_source_category": defaultdict(int, checkpoint.get("by_source_category", {})),
        "by_pattern": defaultdict(int, checkpoint.get("by_pattern", {})),
        "by_emotion": defaultdict(int, checkpoint.get("by_emotion", {})),
        "unique_seed_words": set(checkpoint.get("unique_seed_words_sample", [])),
        "context_lengths": checkpoint.get("context_lengths", []),
        "started_at": checkpoint.get("started_at", datetime.now().isoformat()),
    }

    seen_hashes = set(checkpoint.get("seen_hashes", []))
    start_offset = checkpoint.get("records_offset", 0)

    print(f"Resuming from checkpoint at record {start_offset:,}")
    print(f"  Previous matches: {stats['total_matches']:,}")
    print(f"  Seen hashes: {len(seen_hashes):,}")

    return stats, seen_hashes, start_offset


def save_checkpoint(
    checkpoint_path: Path,
    stats: Dict[str, Any],
    seen_hashes: Set[str],
    current_offset: int
):
    """Save checkpoint for resuming later."""
    checkpoint = {
        "records_offset": current_offset,
        "total_processed": stats["total_processed"],
        "total_matches": stats["total_matches"],
        "filtered_out": stats["filtered_out"],
        "filter_reasons": dict(stats["filter_reasons"]),
        "duplicates_skipped": stats["duplicates_skipped"],
        "by_source_domain": dict(stats["by_source_domain"]),
        "by_source_category": dict(stats["by_source_category"]),
        "by_pattern": dict(stats["by_pattern"]),
        "by_emotion": dict(stats["by_emotion"]),
        "unique_seed_words_sample": sorted(list(stats["unique_seed_words"]))[:100],
        "context_lengths": stats["context_lengths"][-1000:],  # Keep last 1000
        "seen_hashes": list(seen_hashes),
        "started_at": stats["started_at"],
        "checkpoint_at": datetime.now().isoformat(),
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then rename (atomic write)
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    temp_path.rename(checkpoint_path)


def format_candidate(
    record: Dict[str, Any],
    match: PatternMatch,
    context_before: str,
    context_after: str,
    source_category: str,
    config: FulgExtractionConfig
) -> Dict[str, Any]:
    """
    Format a matched record as ASI candidate.

    Output schema with sentence-level context instead of full page text.
    """
    # Build full context string
    full_context = ""
    if context_before:
        full_context += context_before + " "
    full_context += match.matched_text
    if context_after:
        full_context += " " + context_after

    return {
        "id": record["id"],
        # New: focused context instead of full page
        "context": full_context.strip(),
        "context_before": context_before,
        "context_after": context_after,
        "matched_sentence": match.matched_text,
        "pattern_used": match.pattern_name,
        "pattern_category": match.pattern_category,
        "seed_word": match.seed_word,
        "seed_word_normalized": match.seed_word_normalized,
        "emotion_category": match.emotions,
        "source": "fulg",
        # Metadata
        "source_category": source_category,
        "source_domain": record.get("source_domain", ""),
        "url": record.get("url", ""),
        "title": record.get("title", ""),
        "full_text_length": len(record.get("text", "")),
    }


def extract_from_fulg(
    config: FulgExtractionConfig,
    resume: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main extraction loop for FULG dataset.

    Args:
        config: Extraction configuration
        resume: Whether to resume from checkpoint
        verbose: Print progress updates

    Returns:
        Final statistics dictionary
    """
    # Load emotion seed
    print("Loading emotion seed...")
    emotion_seed = load_or_create_emotion_seed()

    word_to_emotions = emotion_seed.get("word_to_emotions", {})
    if not word_to_emotions:
        raise ValueError("No word_to_emotions in emotion seed")

    # Get noun words from seed if available
    noun_words = emotion_seed.get("nouns", None)

    # Initialize pattern matcher
    matcher = PatternMatcher(word_to_emotions, noun_words=noun_words)

    # Load checkpoint if resuming
    if resume:
        stats, seen_hashes, start_offset = load_checkpoint(config.checkpoint_path)
    else:
        stats, seen_hashes, start_offset = load_checkpoint(Path("/nonexistent"))  # Fresh start

    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in append mode if resuming, otherwise write mode
    mode = 'a' if resume and config.output_path.exists() else 'w'

    print(f"\nStarting extraction:")
    print(f"  Output: {config.output_path}")
    print(f"  Max samples: {config.max_extracted_samples:,}")
    print(f"  Max records: {config.max_records_processed:,} (0 = unlimited)")
    print(f"  Checkpoint every: {config.checkpoint_every:,} records")
    print(f"  Text length: {config.min_text_length} - {config.max_text_length} chars")
    print(f"  Min language score: {config.min_language_score}")
    print(f"  Context sentences: {config.context_sentences} before/after")
    print()

    try:
        with open(config.output_path, mode, encoding='utf-8') as out_f:
            for i, record in enumerate(stream_fulg_dataset()):
                # Skip records before checkpoint offset
                if i < start_offset:
                    if i > 0 and i % 100_000 == 0 and verbose:
                        print(f"Skipping to checkpoint... {i:,}/{start_offset:,}")
                    continue

                # Check max records limit
                if config.max_records_processed > 0 and i >= config.max_records_processed:
                    print(f"\nReached max records limit: {config.max_records_processed:,}")
                    break

                # Check max samples limit
                if stats["total_matches"] >= config.max_extracted_samples:
                    print(f"\nReached max samples limit: {config.max_extracted_samples:,}")
                    break

                # Pre-filter
                should, skip_reason = should_process(record, config)
                if not should:
                    stats["filtered_out"] += 1
                    stats["filter_reasons"][skip_reason] += 1
                    continue

                # Deduplicate by text hash
                text_hash = hashlib.md5(record["text"].encode()).hexdigest()
                if text_hash in seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue

                # Pattern matching
                matches = matcher.find_matches(record["text"], extract_sentences=True)

                if matches:
                    seen_hashes.add(text_hash)

                    # Categorize domain (soft tagging)
                    source_category = categorize_domain(
                        record.get("source_domain", ""),
                        record.get("url", "")
                    )

                    for match in matches:
                        # Extract sentence-level context
                        context_before, matched_sent, context_after = extract_context_window(
                            record["text"],
                            match.start_pos,
                            match.end_pos,
                            num_sentences=config.context_sentences,
                            max_length=config.max_context_length
                        )

                        candidate = format_candidate(
                            record, match,
                            context_before, context_after,
                            source_category, config
                        )
                        out_f.write(json.dumps(candidate, ensure_ascii=False) + '\n')

                        # Update stats
                        stats["total_matches"] += 1
                        stats["by_source_domain"][record.get("source_domain", "unknown")] += 1
                        stats["by_source_category"][source_category] += 1
                        stats["by_pattern"][match.pattern_name] += 1
                        stats["unique_seed_words"].add(match.seed_word_normalized)
                        stats["context_lengths"].append(len(candidate["context"]))

                        for emotion in match.emotions:
                            stats["by_emotion"][emotion] += 1

                stats["total_processed"] += 1

                # Checkpoint
                if i > 0 and i % config.checkpoint_every == 0:
                    save_checkpoint(config.checkpoint_path, stats, seen_hashes, i)
                    out_f.flush()

                    if verbose:
                        match_rate = stats["total_matches"] / max(1, stats["total_processed"]) * 100
                        print(
                            f"Progress: {i:,} records | "
                            f"{stats['total_processed']:,} processed | "
                            f"{stats['total_matches']:,} matches ({match_rate:.2f}%) | "
                            f"{stats['filtered_out']:,} filtered"
                        )

                # Periodic progress for verbose mode
                elif verbose and stats["total_processed"] % config.batch_size == 0:
                    match_rate = stats["total_matches"] / max(1, stats["total_processed"]) * 100
                    print(
                        f"[{i:,}] Processed: {stats['total_processed']:,} | "
                        f"Matches: {stats['total_matches']:,} ({match_rate:.2f}%)"
                    )

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        save_checkpoint(config.checkpoint_path, stats, seen_hashes, i)
        print(f"Checkpoint saved. Resume with --resume flag.")
        raise

    # Final checkpoint
    save_checkpoint(config.checkpoint_path, stats, seen_hashes, i if 'i' in dir() else 0)

    # Finalize stats for return
    stats["finished_at"] = datetime.now().isoformat()
    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(list(stats["unique_seed_words"]))[:50]
    stats["by_source_domain"] = dict(stats["by_source_domain"])
    stats["by_source_category"] = dict(stats["by_source_category"])
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])
    stats["filter_reasons"] = dict(stats["filter_reasons"])

    # Context length stats
    if stats["context_lengths"]:
        lengths = stats["context_lengths"]
        stats["context_length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) // len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }
    stats.pop("context_lengths", None)  # Don't include raw list in final stats

    return stats


def print_stats(stats: Dict[str, Any]):
    """Print extraction statistics."""
    print("\n" + "=" * 70)
    print("FULG ASI Candidate Extraction Statistics")
    print("=" * 70)
    print(f"Total records processed: {stats['total_processed']:,}")
    print(f"Records filtered out: {stats['filtered_out']:,}")
    if stats.get('filter_reasons'):
        print("  Filter reasons:")
        for reason, count in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count:,}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']:,}")
    print(f"Total pattern matches: {stats['total_matches']:,}")
    print(f"Unique seed words found: {stats['unique_seed_words_count']}")

    if stats['total_processed'] > 0:
        match_rate = stats['total_matches'] / stats['total_processed'] * 100
        print(f"Match rate: {match_rate:.3f}%")

    if stats.get('context_length_stats'):
        ctx = stats['context_length_stats']
        print(f"\nContext lengths: min={ctx['min']}, max={ctx['max']}, avg={ctx['avg']}, median={ctx['median']}")

    print("\nMatches by source category:")
    for cat, count in sorted(stats.get('by_source_category', {}).items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")

    print("\nMatches by pattern (top 10):")
    sorted_patterns = sorted(stats['by_pattern'].items(), key=lambda x: -x[1])[:10]
    for pattern, count in sorted_patterns:
        print(f"  {pattern}: {count:,}")

    print("\nMatches by emotion:")
    for emotion, count in sorted(stats['by_emotion'].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count:,}")

    print("\nTop source domains (top 10):")
    sorted_domains = sorted(stats['by_source_domain'].items(), key=lambda x: -x[1])[:10]
    for domain, count in sorted_domains:
        print(f"  {domain}: {count:,}")

    if stats.get('unique_seed_words'):
        print(f"\nSample seed words: {', '.join(stats['unique_seed_words'][:20])}")


def sample_candidates(output_path: Path, n: int = 10):
    """Print sample candidates for verification."""
    print(f"\n{'=' * 70}")
    print(f"Sample Candidates (first {n})")
    print("=" * 70)

    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidate = json.loads(line)
            print(f"\n[{i+1}] ID: {candidate['id']}")
            print(f"    Category: {candidate.get('source_category', 'N/A')} | Domain: {candidate.get('source_domain', 'N/A')}")
            print(f"    Pattern: {candidate['pattern_used']} ({candidate['pattern_category']})")
            print(f"    Seed word: {candidate['seed_word']} -> {candidate['emotion_category']}")
            print(f"    Context ({len(candidate.get('context', '')):,} chars):")
            context = candidate.get('context', candidate.get('matched_sentence', ''))
            if len(context) > 200:
                print(f"      \"{context[:200]}...\"")
            else:
                print(f"      \"{context}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ASI candidates from FULG dataset (streaming)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Maximum samples to extract (default: 50000)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum records to process, 0 = unlimited (default: 0)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save checkpoint every N records (default: 10000)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length to process (default: 100)"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=100_000,
        help="Maximum text length to process (default: 100000)"
    )
    parser.add_argument(
        "--min-language-score",
        type=float,
        default=0.8,
        help="Minimum language score for Romanian (default: 0.8)"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=2,
        help="Number of sentences before/after match to include (default: 2)"
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=1000,
        help="Maximum context length in chars (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for candidates JSONL"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample candidates to print (default: 10)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Set default paths
    base_path = Path(__file__).parent.parent.parent
    output_path = args.output or base_path / "data" / "fulg_asi_candidates.jsonl"
    checkpoint_path = base_path / "data" / "fulg_extraction_checkpoint.json"

    # Create config
    config = FulgExtractionConfig(
        max_extracted_samples=args.max_samples,
        max_records_processed=args.max_records,
        checkpoint_every=args.checkpoint_every,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        min_language_score=args.min_language_score,
        context_sentences=args.context_sentences,
        max_context_length=args.max_context_length,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    print("FULG Dataset ASI Candidate Extraction")
    print("=" * 70)

    # Run extraction
    try:
        stats = extract_from_fulg(
            config=config,
            resume=args.resume,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\nExtraction interrupted. Use --resume to continue later.")
        return 1

    # Print statistics
    print_stats(stats)

    # Save final stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    # Print samples
    if args.sample > 0 and stats['total_matches'] > 0:
        sample_candidates(output_path, args.sample)

    print(f"\nOutput saved to: {output_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Extract ASI candidates from Filmot/YouTube transcripts.

This pipeline:
1. Searches filmot.com for Romanian subtitle content matching affective patterns
2. Downloads YouTube transcripts for discovered videos
3. Applies pattern matching to extract ASI candidates

Features:
- Two-phase extraction: search + transcript download + pattern matching
- Checkpoint/resume support for long runs
- Deduplication via MD5 hashes
- Rate limiting to avoid blocks
- Compatible output format with existing ASI pipeline

Usage:
    python -m scripts.filmot.extract_candidates
    python -m scripts.filmot.extract_candidates --max-samples 10000 --max-videos 5000
    python -m scripts.filmot.extract_candidates --resume
    python -m scripts.filmot.extract_candidates --phase search  # Only search filmot
    python -m scripts.filmot.extract_candidates --phase transcripts  # Only download transcripts
    python -m scripts.filmot.extract_candidates --phase extract  # Only extract from cached transcripts
"""

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Import from existing ro_asi modules
from scripts.ro_asi.load_roemolex import load_or_create_emotion_seed
from scripts.ro_asi.pattern_matcher import PatternMatch, PatternMatcher

from .config import FilmotExtractionConfig, generate_search_queries
from .searcher import FilmotSearcher, VideoMetadata, save_video_ids
from .transcript_fetcher import (
    TranscriptFetcher,
    TranscriptResult,
    load_video_ids_from_file,
)


def load_checkpoint(checkpoint_path: Path) -> tuple:
    """
    Load checkpoint from previous run.

    Returns:
        Tuple of (stats_dict, seen_hashes_set, processed_video_ids_set, completed_queries_set)
    """
    if not checkpoint_path.exists():
        return (
            {
                "videos_discovered": 0,
                "videos_processed": 0,
                "transcripts_fetched": 0,
                "transcripts_failed": 0,
                "total_matches": 0,
                "duplicates_skipped": 0,
                "by_pattern": defaultdict(int),
                "by_emotion": defaultdict(int),
                "by_channel": defaultdict(int),
                "unique_seed_words": set(),
                "started_at": datetime.now().isoformat(),
            },
            set(),
            set(),
            set(),
        )

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    stats = {
        "videos_discovered": checkpoint.get("videos_discovered", 0),
        "videos_processed": checkpoint.get("videos_processed", 0),
        "transcripts_fetched": checkpoint.get("transcripts_fetched", 0),
        "transcripts_failed": checkpoint.get("transcripts_failed", 0),
        "total_matches": checkpoint.get("total_matches", 0),
        "duplicates_skipped": checkpoint.get("duplicates_skipped", 0),
        "by_pattern": defaultdict(int, checkpoint.get("by_pattern", {})),
        "by_emotion": defaultdict(int, checkpoint.get("by_emotion", {})),
        "by_channel": defaultdict(int, checkpoint.get("by_channel", {})),
        "unique_seed_words": set(checkpoint.get("unique_seed_words_sample", [])),
        "started_at": checkpoint.get("started_at", datetime.now().isoformat()),
    }

    seen_hashes = set(checkpoint.get("seen_hashes", []))
    processed_video_ids = set(checkpoint.get("processed_video_ids", []))
    completed_queries = set(checkpoint.get("completed_queries", []))

    print(f"Resuming from checkpoint:")
    print(f"  Videos discovered: {stats['videos_discovered']:,}")
    print(f"  Videos processed: {stats['videos_processed']:,}")
    print(f"  Total matches: {stats['total_matches']:,}")
    print(f"  Completed queries: {len(completed_queries)}")

    return stats, seen_hashes, processed_video_ids, completed_queries


def save_checkpoint(
    checkpoint_path: Path,
    stats: Dict[str, Any],
    seen_hashes: Set[str],
    processed_video_ids: Set[str],
    completed_queries: Set[str],
):
    """Save checkpoint for resuming later."""
    checkpoint = {
        "videos_discovered": stats["videos_discovered"],
        "videos_processed": stats["videos_processed"],
        "transcripts_fetched": stats["transcripts_fetched"],
        "transcripts_failed": stats["transcripts_failed"],
        "total_matches": stats["total_matches"],
        "duplicates_skipped": stats["duplicates_skipped"],
        "by_pattern": dict(stats["by_pattern"]),
        "by_emotion": dict(stats["by_emotion"]),
        "by_channel": dict(stats["by_channel"]),
        "unique_seed_words_sample": sorted(list(stats["unique_seed_words"]))[:100],
        "started_at": stats["started_at"],
        "checkpoint_at": datetime.now().isoformat(),
        "seen_hashes": list(seen_hashes),
        "processed_video_ids": list(processed_video_ids),
        "completed_queries": list(completed_queries),
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    temp_path.rename(checkpoint_path)


def format_candidate(
    video_id: str,
    transcript_text: str,
    match: PatternMatch,
    video_metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Format a matched record as ASI candidate.

    Output schema compatible with existing asi_candidates.jsonl,
    with YouTube-specific fields added.
    """
    candidate = {
        "id": f"filmot_{video_id}_{match.start_pos}",
        "text": transcript_text,
        "matched_sentence": match.matched_text,
        "pattern_used": match.pattern_name,
        "pattern_category": match.pattern_category,
        "seed_word": match.seed_word,
        "seed_word_normalized": match.seed_word_normalized,
        "emotion_category": match.emotions,
        "source": "filmot",
        "video_id": video_id,
    }

    # Add video metadata if available
    if video_metadata:
        candidate["video_title"] = video_metadata.get("title", "")
        candidate["channel"] = video_metadata.get("channel", "")
        candidate["views"] = video_metadata.get("views", 0)
        candidate["duration_seconds"] = video_metadata.get("duration_seconds", 0)
        candidate["upload_date"] = video_metadata.get("upload_date")

    return candidate


def should_stop(stats: Dict[str, Any], config: FilmotExtractionConfig) -> bool:
    """Check if extraction should stop."""
    if config.max_extracted_samples > 0:
        if stats["total_matches"] >= config.max_extracted_samples:
            return True
    if config.max_videos_processed > 0:
        if stats["videos_processed"] >= config.max_videos_processed:
            return True
    return False


async def phase_search(
    config: FilmotExtractionConfig,
    stats: Dict[str, Any],
    processed_video_ids: Set[str],
    completed_queries: Set[str],
) -> List[VideoMetadata]:
    """
    Phase 1: Search filmot for videos with Romanian subtitles.

    Returns list of discovered video metadata.
    """
    queries = generate_search_queries()
    all_videos = []
    seen_video_ids = set(processed_video_ids)

    print(f"\nPhase 1: Searching filmot.com")
    print(f"  Total queries: {len(queries)}")
    print(f"  Already completed: {len(completed_queries)}")
    print(f"  Browser state: {config.browser_state_path}")
    print(f"  (Solve CAPTCHA once - session will be saved for future runs)")

    async with FilmotSearcher(config, browser_state_path=config.browser_state_path) as searcher:
        for query in queries:
            if query in completed_queries:
                print(f"  Skipping completed query: {query}")
                continue

            if should_stop(stats, config):
                print(f"\nReached stopping condition during search")
                break

            print(f"\n  Searching: {query}")
            videos_from_query = []

            try:
                async for video in searcher.search(query):
                    # Skip already seen videos
                    if video.video_id in seen_video_ids:
                        continue

                    seen_video_ids.add(video.video_id)
                    videos_from_query.append(video)
                    stats["videos_discovered"] += 1

                    # Check limit
                    if config.max_videos_processed > 0:
                        if stats["videos_discovered"] >= config.max_videos_processed * 2:
                            break

                print(f"    Found {len(videos_from_query)} new videos")
                all_videos.extend(videos_from_query)

                # Save incrementally
                if videos_from_query:
                    await save_video_ids(videos_from_query, config.video_ids_path)

                completed_queries.add(query)

                # Delay between queries
                await asyncio.sleep(config.delay_between_searches)

            except Exception as e:
                print(f"    Error searching: {e}")
                continue

    print(f"\nSearch complete: {len(all_videos)} videos discovered")
    return all_videos


def phase_transcripts(
    config: FilmotExtractionConfig,
    stats: Dict[str, Any],
    processed_video_ids: Set[str],
    video_ids: Optional[List[str]] = None,
) -> Dict[str, TranscriptResult]:
    """
    Phase 2: Download transcripts for discovered videos.

    Returns dict of video_id -> TranscriptResult.
    """
    # Load video IDs from file if not provided
    if video_ids is None:
        video_ids = load_video_ids_from_file(config.video_ids_path)

    if not video_ids:
        print("No video IDs to process")
        return {}

    # Filter out already processed
    pending_ids = [vid for vid in video_ids if vid not in processed_video_ids]

    print(f"\nPhase 2: Downloading transcripts")
    print(f"  Total video IDs: {len(video_ids)}")
    print(f"  Already processed: {len(processed_video_ids)}")
    print(f"  Pending: {len(pending_ids)}")

    fetcher = TranscriptFetcher(config)
    results = {}

    def progress_callback(index: int, total: int, result: TranscriptResult):
        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  [{index + 1}/{total}] {result.video_id}: {status}")

        if result.success:
            stats["transcripts_fetched"] += 1
        else:
            stats["transcripts_failed"] += 1

    for video_id, result in fetcher.fetch_batch(
        pending_ids,
        progress_callback=progress_callback,
        skip_existing=True,
    ):
        results[video_id] = result
        processed_video_ids.add(video_id)
        stats["videos_processed"] += 1

        # Check stopping condition
        if should_stop(stats, config):
            print(f"\nReached stopping condition during transcript download")
            break

    print(f"\nTranscripts downloaded: {stats['transcripts_fetched']}")
    print(f"Transcripts failed: {stats['transcripts_failed']}")

    return results


def phase_extract(
    config: FilmotExtractionConfig,
    stats: Dict[str, Any],
    seen_hashes: Set[str],
    matcher: PatternMatcher,
    transcripts: Optional[Dict[str, TranscriptResult]] = None,
    video_metadata: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> int:
    """
    Phase 3: Extract ASI candidates from transcripts.

    Returns number of matches found.
    """
    # Load transcripts from cache if not provided
    if transcripts is None:
        fetcher = TranscriptFetcher(config)
        cached_ids = fetcher.get_cached_video_ids()

        transcripts = {}
        for video_id in cached_ids:
            result = fetcher._load_from_cache(video_id)
            if result and result.success:
                transcripts[video_id] = result

    if not transcripts:
        print("No transcripts to process")
        return 0

    # Load video metadata if not provided
    if video_metadata is None:
        video_metadata = {}
        if config.video_ids_path.exists():
            with open(config.video_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        vid = record.get("video_id")
                        if vid:
                            video_metadata[vid] = record
                    except json.JSONDecodeError:
                        continue

    print(f"\nPhase 3: Extracting ASI candidates")
    print(f"  Transcripts to process: {len(transcripts)}")

    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    matches_this_phase = 0

    with open(config.output_path, "a", encoding="utf-8") as out_f:
        for i, (video_id, transcript) in enumerate(transcripts.items()):
            if should_stop(stats, config):
                print(f"\nReached stopping condition during extraction")
                break

            if not transcript.success:
                continue

            # Pattern matching
            matches = matcher.find_matches(
                transcript.text, extract_sentences=True, max_matches=50
            )

            if not matches:
                continue

            # Get metadata for this video
            metadata = video_metadata.get(video_id, {})

            for match in matches:
                # Deduplicate by matched text hash
                text_hash = hashlib.md5(match.matched_text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue

                seen_hashes.add(text_hash)

                # Format and write candidate
                candidate = format_candidate(
                    video_id, transcript.text, match, metadata
                )
                out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

                # Update stats
                stats["total_matches"] += 1
                matches_this_phase += 1
                stats["by_pattern"][match.pattern_name] += 1
                stats["unique_seed_words"].add(match.seed_word_normalized)

                for emotion in match.emotions:
                    stats["by_emotion"][emotion] += 1

                channel = metadata.get("channel", "unknown")
                stats["by_channel"][channel] += 1

            if verbose and (i + 1) % 100 == 0:
                print(
                    f"  Processed {i + 1}/{len(transcripts)} transcripts | "
                    f"Matches: {stats['total_matches']:,}"
                )

    print(f"\nExtraction complete: {matches_this_phase} new matches")
    return matches_this_phase


def extract_from_filmot(
    config: FilmotExtractionConfig,
    resume: bool = False,
    phase: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main extraction loop for Filmot/YouTube pipeline.

    Args:
        config: Extraction configuration
        resume: Whether to resume from checkpoint
        phase: Run specific phase only ('search', 'transcripts', 'extract')
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

    # Load checkpoint
    if resume:
        stats, seen_hashes, processed_video_ids, completed_queries = load_checkpoint(
            config.checkpoint_path
        )
    else:
        stats, seen_hashes, processed_video_ids, completed_queries = load_checkpoint(
            Path("/nonexistent")
        )

    print(f"\nFilmot ASI Candidate Extraction")
    print("=" * 70)
    print(f"Output: {config.output_path}")
    print(f"Max samples: {config.max_extracted_samples:,}")
    print(f"Max videos: {config.max_videos_processed:,} (0 = unlimited)")
    print()

    try:
        # Phase 1: Search (unless skipping)
        if phase is None or phase == "search":
            asyncio.run(
                phase_search(config, stats, processed_video_ids, completed_queries)
            )
            save_checkpoint(
                config.checkpoint_path,
                stats,
                seen_hashes,
                processed_video_ids,
                completed_queries,
            )

        # Phase 2: Download transcripts (unless skipping)
        if phase is None or phase == "transcripts":
            phase_transcripts(config, stats, processed_video_ids)
            save_checkpoint(
                config.checkpoint_path,
                stats,
                seen_hashes,
                processed_video_ids,
                completed_queries,
            )

        # Phase 3: Extract candidates
        if phase is None or phase == "extract":
            phase_extract(
                config,
                stats,
                seen_hashes,
                matcher,
                verbose=verbose,
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        save_checkpoint(
            config.checkpoint_path,
            stats,
            seen_hashes,
            processed_video_ids,
            completed_queries,
        )
        print(f"Checkpoint saved. Resume with --resume flag.")
        raise

    # Final checkpoint
    save_checkpoint(
        config.checkpoint_path,
        stats,
        seen_hashes,
        processed_video_ids,
        completed_queries,
    )

    # Finalize stats
    stats["finished_at"] = datetime.now().isoformat()
    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(list(stats["unique_seed_words"]))[:50]
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])
    stats["by_channel"] = dict(stats["by_channel"])

    return stats


def print_stats(stats: Dict[str, Any]):
    """Print extraction statistics."""
    print("\n" + "=" * 70)
    print("Filmot ASI Candidate Extraction Statistics")
    print("=" * 70)
    print(f"Videos discovered: {stats['videos_discovered']:,}")
    print(f"Videos processed: {stats['videos_processed']:,}")
    print(f"Transcripts fetched: {stats['transcripts_fetched']:,}")
    print(f"Transcripts failed: {stats['transcripts_failed']:,}")
    print(f"Total pattern matches: {stats['total_matches']:,}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']:,}")
    print(f"Unique seed words found: {stats.get('unique_seed_words_count', 0)}")

    if stats["transcripts_fetched"] > 0:
        success_rate = stats["transcripts_fetched"] / (
            stats["transcripts_fetched"] + stats["transcripts_failed"]
        ) * 100
        print(f"Transcript success rate: {success_rate:.1f}%")

    if stats["videos_processed"] > 0:
        match_rate = stats["total_matches"] / stats["videos_processed"]
        print(f"Matches per video: {match_rate:.2f}")

    print("\nMatches by pattern (top 10):")
    sorted_patterns = sorted(stats["by_pattern"].items(), key=lambda x: -x[1])[:10]
    for pattern, count in sorted_patterns:
        print(f"  {pattern}: {count:,}")

    print("\nMatches by emotion:")
    for emotion, count in sorted(stats["by_emotion"].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count:,}")

    if stats.get("by_channel"):
        print("\nTop channels (top 10):")
        sorted_channels = sorted(stats["by_channel"].items(), key=lambda x: -x[1])[:10]
        for channel, count in sorted_channels:
            print(f"  {channel}: {count:,}")

    if stats.get("unique_seed_words"):
        print(f"\nSample seed words: {', '.join(stats['unique_seed_words'][:20])}")


def sample_candidates(output_path: Path, n: int = 10):
    """Print sample candidates for verification."""
    if not output_path.exists():
        print(f"\nNo output file found at {output_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"Sample Candidates (first {n})")
    print("=" * 70)

    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            candidate = json.loads(line)
            print(f"\n[{i + 1}] ID: {candidate['id']}")
            print(f"    Video: {candidate.get('video_id', 'N/A')}")
            print(f"    Channel: {candidate.get('channel', 'N/A')}")
            print(
                f"    Pattern: {candidate['pattern_used']} ({candidate['pattern_category']})"
            )
            print(f"    Seed word: {candidate['seed_word']} -> {candidate['emotion_category']}")
            matched = candidate["matched_sentence"]
            if len(matched) > 100:
                print(f'    Matched: "{matched[:100]}..."')
            else:
                print(f'    Matched: "{matched}"')


def main():
    parser = argparse.ArgumentParser(
        description="Extract ASI candidates from Filmot/YouTube transcripts"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Maximum samples to extract (default: 50000)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum videos to process, 0 = unlimited (default: 0)",
    )
    parser.add_argument(
        "--max-search-pages",
        type=int,
        default=50,
        help="Maximum pages per search query (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N videos (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for candidates JSONL",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--phase",
        choices=["search", "transcripts", "extract"],
        default=None,
        help="Run specific phase only",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no CAPTCHA solving)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample candidates to print (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Set up paths
    base_path = Path(__file__).parent.parent.parent
    output_path = args.output or base_path / "data" / "filmot_asi_candidates.jsonl"
    checkpoint_path = base_path / "data" / "filmot_checkpoint.json"
    video_ids_path = base_path / "data" / "filmot_video_ids.jsonl"
    transcripts_dir = base_path / "data" / "filmot_transcripts"
    browser_state_path = base_path / "data" / "filmot_browser_state.json"

    # Create config
    config = FilmotExtractionConfig(
        max_extracted_samples=args.max_samples,
        max_videos_processed=args.max_videos,
        max_search_pages=args.max_search_pages,
        checkpoint_every=args.checkpoint_every,
        headless=args.headless,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        video_ids_path=video_ids_path,
        transcripts_dir=transcripts_dir,
        browser_state_path=browser_state_path,
    )

    # Run extraction
    try:
        stats = extract_from_filmot(
            config=config,
            resume=args.resume,
            phase=args.phase,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\nExtraction interrupted. Use --resume to continue later.")
        return 1

    # Print statistics
    print_stats(stats)

    # Save final stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    # Print samples
    if args.sample > 0 and stats["total_matches"] > 0:
        sample_candidates(output_path, args.sample)

    print(f"\nOutput saved to: {output_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return 0


if __name__ == "__main__":
    exit(main())

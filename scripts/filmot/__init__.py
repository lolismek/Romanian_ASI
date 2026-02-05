"""
Filmot/YouTube Transcript Extraction Package.

This package provides extraction of affective state samples from YouTube
Romanian transcripts using filmot.com for subtitle search.

Modules:
    - config: Extraction configuration dataclass
    - searcher: Playwright-based filmot search
    - transcript_fetcher: YouTube transcript fetcher
    - extract_candidates: Main extraction pipeline

Usage:
    python -m scripts.filmot.extract_candidates
    python -m scripts.filmot.extract_candidates --max-samples 10000 --max-videos 5000
    python -m scripts.filmot.extract_candidates --resume
"""

__version__ = "0.1.0"

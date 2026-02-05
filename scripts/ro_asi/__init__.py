"""
Romanian Affective State Identification (RO-ASI) Benchmark Package.

This package provides tools for creating a Romanian ASI benchmark following
the MASIVE paper methodology:

1. load_roemolex: Load and parse RoEmoLex V3 emotion lexicon
2. merge_datasets: Merge Romanian NLP datasets into unified corpus
3. pattern_matcher: Romanian "I feel" pattern matching logic
4. extract_candidates: Main extraction script for ASI candidates

Modules:
    - emotion_seed: Manual emotion seed lexicon (fallback)
    - load_roemolex: RoEmoLex V3 CSV parser
    - merge_datasets: Dataset merger
    - pattern_matcher: Pattern matching utilities
    - extract_candidates: Candidate extraction pipeline
"""

__version__ = "0.1.0"
__author__ = "RO-ASI Benchmark Team"

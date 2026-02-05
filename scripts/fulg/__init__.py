"""
FULG Dataset Extraction Package.

This package provides streaming extraction of affective state samples from the
FULG dataset (150B tokens, 289GB) using HuggingFace streaming.

Modules:
    - extract_candidates: Streaming extraction pipeline for ASI candidates

Usage:
    python -m scripts.fulg.extract_candidates
    python -m scripts.fulg.extract_candidates --max-samples 100000 --max-records 5000000
    python -m scripts.fulg.extract_candidates --resume
"""

__version__ = "0.1.0"

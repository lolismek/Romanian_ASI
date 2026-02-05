"""
Configuration for Filmot extraction pipeline.

Defines extraction parameters, filters, rate limiting, and checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FilmotExtractionConfig:
    """Configuration for Filmot/YouTube extraction pipeline."""

    # === Stopping Conditions ===
    max_extracted_samples: int = 50_000  # Stop after N ASI matches
    max_videos_processed: int = 0  # 0 = unlimited
    max_search_pages: int = 50  # Per search query

    # === YouTube Filters ===
    min_views: int = 100
    max_views: int = 0  # 0 = no limit
    min_duration_seconds: int = 60  # Skip shorts
    max_duration_seconds: int = 3600  # 1 hour max
    upload_after: str = "2020-01-01"

    # === Transcript Filters ===
    min_transcript_chars: int = 500
    languages: List[str] = field(default_factory=lambda: ["ro"])

    # === Rate Limiting ===
    delay_between_searches: float = 5.0
    delay_between_pages: float = 2.0
    delay_between_transcripts: float = 1.0

    # === Browser Settings ===
    headless: bool = False  # Show browser for CAPTCHA solving
    captcha_timeout: int = 300  # 5 minutes to solve CAPTCHA

    # === Checkpointing ===
    checkpoint_every: int = 100  # videos
    checkpoint_path: Path = field(
        default_factory=lambda: Path("data/filmot_checkpoint.json")
    )
    output_path: Path = field(
        default_factory=lambda: Path("data/filmot_asi_candidates.jsonl")
    )
    video_ids_path: Path = field(
        default_factory=lambda: Path("data/filmot_video_ids.jsonl")
    )
    transcripts_dir: Path = field(
        default_factory=lambda: Path("data/filmot_transcripts")
    )
    browser_state_path: Path = field(
        default_factory=lambda: Path("data/filmot_browser_state.json")
    )


# Search queries for filmot - high-precision patterns
# These combine "I feel" verbs with common emotion adjectives
FILMOT_SEED_ADJECTIVES = [
    "fericit",
    "fericită",
    "trist",
    "tristă",
    "supărat",
    "supărată",
    "speriat",
    "speriată",
    "furios",
    "furioasă",
    "nervos",
    "nervoasă",
    "anxios",
    "anxioasă",
    "stresat",
    "stresată",
    "relaxat",
    "relaxată",
    "calm",
    "calmă",
    "entuziasmat",
    "entuziasmată",
    "dezamăgit",
    "dezamăgită",
    "confuz",
    "confuză",
    "singur",
    "singură",
    "obosit",
    "obosită",
    "mulțumit",
    "mulțumită",
    "recunoscător",
    "recunoscătoare",
    "mândru",
    "mândră",
]

FILMOT_SEED_NOUNS = [
    "frică",
    "teamă",
    "bucurie",
    "tristețe",
    "furie",
    "rușine",
    "vină",
]


def generate_search_queries() -> List[str]:
    """
    Generate search queries for filmot.

    Returns list of quoted phrases to search.
    """
    queries = []

    # High-precision queries: "mă simt [adjective]"
    for adj in FILMOT_SEED_ADJECTIVES[:20]:  # Top 20 adjectives
        queries.append(f'"mă simt {adj}"')

    # "sunt [adjective]" patterns
    for adj in FILMOT_SEED_ADJECTIVES[:10]:  # Top 10
        queries.append(f'"sunt {adj}"')

    # "m-am simțit [adjective]" patterns
    for adj in FILMOT_SEED_ADJECTIVES[:10]:
        queries.append(f'"m-am simțit {adj}"')

    # Noun patterns: "mi-e [noun]", "am [noun]"
    for noun in FILMOT_SEED_NOUNS:
        queries.append(f'"mi-e {noun}"')
        queries.append(f'"am {noun}"')

    # Bootstrapping queries (broader)
    queries.extend(
        [
            '"mă simt"',
            '"m-am simțit"',
            '"ne simțim"',
            '"mi-e frică"',
            '"mi-e teamă"',
        ]
    )

    return queries

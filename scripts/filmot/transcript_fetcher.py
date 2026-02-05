"""
YouTube transcript fetcher with rate limiting.

Uses youtube-transcript-api to fetch Romanian transcripts.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

from .config import FilmotExtractionConfig


@dataclass
class TranscriptResult:
    """Result of fetching a transcript."""

    video_id: str
    success: bool
    text: str  # Full concatenated text
    segments: List[Dict]  # List of {text, start, duration}
    language: str
    is_generated: bool  # Auto-generated vs manual
    error: Optional[str] = None


class TranscriptFetcher:
    """
    Fetch Romanian transcripts from YouTube.

    Uses youtube-transcript-api with rate limiting and caching.
    """

    def __init__(self, config: FilmotExtractionConfig):
        """
        Initialize the fetcher.

        Args:
            config: Extraction configuration
        """
        self.config = config

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api required. Install with: pip install youtube-transcript-api"
            )

        self._api = YouTubeTranscriptApi()

    def fetch(self, video_id: str) -> TranscriptResult:
        """
        Fetch transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            TranscriptResult with transcript data or error
        """
        try:
            # Try to get Romanian transcript
            transcript_list = self._api.list(video_id)

            # First try manually created Romanian transcripts
            transcript = None
            is_generated = False

            try:
                transcript = transcript_list.find_manually_created_transcript(
                    self.config.languages
                )
            except Exception:
                pass

            # Fall back to auto-generated
            if not transcript:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        self.config.languages
                    )
                    is_generated = True
                except Exception:
                    pass

            # If no Romanian, try to find any transcript and translate
            if not transcript:
                try:
                    # Get any available transcript
                    available = list(transcript_list)
                    if available:
                        # Try to translate to Romanian
                        transcript = available[0].translate("ro")
                        is_generated = True
                except Exception:
                    pass

            if not transcript:
                return TranscriptResult(
                    video_id=video_id,
                    success=False,
                    text="",
                    segments=[],
                    language="",
                    is_generated=False,
                    error="No Romanian transcript available",
                )

            # Fetch the transcript data
            segments = transcript.fetch()

            # Concatenate text
            text = " ".join(s.text for s in segments)

            # Check minimum length
            if len(text) < self.config.min_transcript_chars:
                return TranscriptResult(
                    video_id=video_id,
                    success=False,
                    text="",
                    segments=[],
                    language=transcript.language_code,
                    is_generated=is_generated,
                    error=f"Transcript too short ({len(text)} chars)",
                )

            return TranscriptResult(
                video_id=video_id,
                success=True,
                text=text,
                segments=[
                    {"text": s.text, "start": s.start, "duration": s.duration}
                    for s in segments
                ],
                language=transcript.language_code,
                is_generated=is_generated,
            )

        except Exception as e:
            return TranscriptResult(
                video_id=video_id,
                success=False,
                text="",
                segments=[],
                language="",
                is_generated=False,
                error=str(e),
            )

    def fetch_batch(
        self,
        video_ids: List[str],
        progress_callback: Optional[Callable[[int, int, TranscriptResult], None]] = None,
        skip_existing: bool = True,
    ) -> Generator[Tuple[str, TranscriptResult], None, None]:
        """
        Fetch transcripts for multiple videos with rate limiting.

        Args:
            video_ids: List of video IDs
            progress_callback: Optional callback(index, total, result)
            skip_existing: Skip videos that already have cached transcripts

        Yields:
            Tuples of (video_id, TranscriptResult)
        """
        total = len(video_ids)

        for i, video_id in enumerate(video_ids):
            # Check cache if skip_existing
            if skip_existing:
                cache_path = self._get_cache_path(video_id)
                if cache_path.exists():
                    # Load from cache
                    result = self._load_from_cache(video_id)
                    if result:
                        if progress_callback:
                            progress_callback(i, total, result)
                        yield video_id, result
                        continue

            # Fetch transcript
            result = self.fetch(video_id)

            # Cache successful results
            if result.success:
                self._save_to_cache(video_id, result)

            if progress_callback:
                progress_callback(i, total, result)

            yield video_id, result

            # Rate limiting
            time.sleep(self.config.delay_between_transcripts)

    def _get_cache_path(self, video_id: str) -> Path:
        """Get cache file path for a video."""
        cache_dir = self.config.transcripts_dir
        return cache_dir / f"{video_id}.json"

    def _save_to_cache(self, video_id: str, result: TranscriptResult):
        """Save transcript to cache."""
        cache_path = self._get_cache_path(video_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "video_id": result.video_id,
            "success": result.success,
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "is_generated": result.is_generated,
            "error": result.error,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_cache(self, video_id: str) -> Optional[TranscriptResult]:
        """Load transcript from cache."""
        cache_path = self._get_cache_path(video_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return TranscriptResult(
                video_id=data["video_id"],
                success=data["success"],
                text=data["text"],
                segments=data["segments"],
                language=data["language"],
                is_generated=data["is_generated"],
                error=data.get("error"),
            )
        except Exception:
            return None

    def get_cached_video_ids(self) -> List[str]:
        """Get list of video IDs with cached transcripts."""
        cache_dir = self.config.transcripts_dir
        if not cache_dir.exists():
            return []

        return [p.stem for p in cache_dir.glob("*.json")]


def load_video_ids_from_file(video_ids_path: Path) -> List[str]:
    """
    Load video IDs from JSONL file.

    Args:
        video_ids_path: Path to video_ids.jsonl

    Returns:
        List of video IDs
    """
    if not video_ids_path.exists():
        return []

    video_ids = []
    with open(video_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                video_id = record.get("video_id")
                if video_id:
                    video_ids.append(video_id)
            except json.JSONDecodeError:
                continue

    return video_ids

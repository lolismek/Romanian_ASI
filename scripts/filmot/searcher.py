"""
Playwright-based Filmot searcher with stealth.

Searches filmot.com for Romanian subtitle content and extracts video metadata.
Supports cookie persistence to avoid repeated CAPTCHA solving.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
from urllib.parse import quote_plus

from .config import FilmotExtractionConfig

# Default path for storing browser cookies/state
DEFAULT_BROWSER_STATE_PATH = Path("data/filmot_browser_state.json")


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video from filmot search."""

    video_id: str
    title: str
    channel: str
    views: int
    duration_seconds: int
    upload_date: Optional[str]
    matched_text: str  # The subtitle snippet that matched the search
    search_query: str


class FilmotSearcher:
    """
    Search filmot.com for Romanian subtitle content.

    Uses Playwright with stealth to avoid bot detection.
    Supports cookie persistence - solve CAPTCHA once, reuse session.
    """

    def __init__(
        self,
        config: FilmotExtractionConfig,
        browser_state_path: Optional[Path] = None,
    ):
        """
        Initialize the searcher.

        Args:
            config: Extraction configuration
            browser_state_path: Path to save/load browser cookies (enables session reuse)
        """
        self.config = config
        self.browser_state_path = browser_state_path or DEFAULT_BROWSER_STATE_PATH
        self.browser = None
        self.context = None
        self.page = None
        self._playwright = None
        self._stealth = None

    async def __aenter__(self):
        """Set up browser with stealth and optional session restoration."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright required. Install with: pip install playwright && playwright install chromium"
            )

        try:
            from playwright_stealth import Stealth
        except ImportError:
            raise ImportError(
                "playwright-stealth required. Install with: pip install playwright-stealth"
            )

        self._playwright = await async_playwright().start()
        self._stealth = Stealth()

        # Launch using system Chrome (not "Chrome for Testing")
        # This allows Google login and looks more legitimate
        self.browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            channel="chrome",  # Use system Chrome installation
        )

        # Check for saved browser state (cookies from previous session)
        storage_state = None
        if self.browser_state_path.exists():
            try:
                print(f"Loading saved browser session from {self.browser_state_path}")
                storage_state = str(self.browser_state_path)
            except Exception as e:
                print(f"Could not load browser state: {e}")

        # Create context with realistic settings and optional saved state
        context_opts = {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        if storage_state:
            context_opts["storage_state"] = storage_state

        self.context = await self.browser.new_context(**context_opts)
        self.page = await self.context.new_page()

        # Apply stealth mode
        await self._stealth.apply_stealth_async(self.page)

        return self

    async def __aexit__(self, *args):
        """Save browser state and clean up."""
        # Save cookies/state for next session
        if self.context:
            try:
                self.browser_state_path.parent.mkdir(parents=True, exist_ok=True)
                await self.context.storage_state(path=str(self.browser_state_path))
                print(f"Saved browser session to {self.browser_state_path}")
            except Exception as e:
                print(f"Could not save browser state: {e}")

        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def search(
        self, query: str, max_pages: int = None
    ) -> AsyncGenerator[VideoMetadata, None]:
        """
        Search filmot and yield video metadata.

        Args:
            query: Search query (e.g., '"mă simt fericit"')
            max_pages: Maximum pages to fetch (default from config)

        Yields:
            VideoMetadata objects for each result
        """
        if max_pages is None:
            max_pages = self.config.max_search_pages

        # URL encode the query
        encoded_query = quote_plus(query)

        # Filmot search URL with Romanian language filter
        base_url = f"https://filmot.com/search/{encoded_query}/1?lang=ro"

        print(f"Searching filmot for: {query}")
        await self.page.goto(base_url)

        # Wait for results or CAPTCHA
        # If CAPTCHA appears, user has time to solve it
        try:
            await self._wait_for_results_or_captcha()
        except Exception as e:
            print(f"Error waiting for results: {e}")
            return

        for page_num in range(1, max_pages + 1):
            # Extract results from current page
            results = await self._extract_results(query)

            for result in results:
                # Apply filters
                if self._passes_filters(result):
                    yield result

            # Check for next page
            if not await self._has_next_page():
                print(f"No more pages after page {page_num}")
                break

            # Delay between pages
            await asyncio.sleep(self.config.delay_between_pages)

            # Go to next page
            await self._goto_next_page()

            # Wait for new results
            try:
                await self._wait_for_results_or_captcha()
            except Exception as e:
                print(f"Error on page {page_num + 1}: {e}")
                break

    async def _wait_for_results_or_captcha(self):
        """
        Wait for search results to load or CAPTCHA to appear.

        If CAPTCHA appears, waits for user to solve it.
        """
        # Wait for either results or CAPTCHA
        # Filmot uses hCaptcha or similar

        # First, check if we're on a CAPTCHA page
        captcha_selectors = [
            "iframe[src*='hcaptcha']",
            "iframe[src*='recaptcha']",
            ".h-captcha",
            ".g-recaptcha",
            "#captcha",
        ]

        result_selectors = [
            ".search-result",
            ".video-result",
            ".result-item",
            "article",
            ".card",
        ]

        # Wait with longer timeout for CAPTCHA solving
        timeout = self.config.captcha_timeout * 1000  # ms

        for _ in range(int(timeout / 1000)):
            # Check for CAPTCHA
            for selector in captcha_selectors:
                if await self.page.query_selector(selector):
                    print("CAPTCHA detected. Please solve it in the browser...")
                    # Wait a bit and check again
                    await asyncio.sleep(1)
                    continue

            # Check for results
            for selector in result_selectors:
                element = await self.page.query_selector(selector)
                if element:
                    # Give page time to fully load
                    await asyncio.sleep(1)
                    return

            await asyncio.sleep(1)

        raise TimeoutError("Timed out waiting for search results")

    async def _extract_results(self, search_query: str) -> List[VideoMetadata]:
        """
        Extract video metadata from current search results page.

        Args:
            search_query: The search query used

        Returns:
            List of VideoMetadata objects
        """
        results = []

        # Get page HTML for parsing
        # Filmot's structure may vary, so we try multiple selectors
        content = await self.page.content()

        # Try to find video cards/results
        # Common patterns in filmot results
        selectors = [
            ".search-result",
            ".video-result",
            ".result-item",
            "article.card",
            ".video-card",
        ]

        elements = None
        for selector in selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                break

        if not elements:
            # Fallback: try to extract links to youtube
            links = await self.page.query_selector_all('a[href*="youtube.com/watch"]')
            if not links:
                links = await self.page.query_selector_all('a[href*="youtu.be"]')

            for link in links:
                href = await link.get_attribute("href")
                video_id = self._extract_video_id(href)
                if video_id:
                    # Get surrounding text as matched text
                    parent = await link.evaluate_handle("el => el.parentElement")
                    text = await parent.inner_text() if parent else ""

                    results.append(
                        VideoMetadata(
                            video_id=video_id,
                            title="",
                            channel="",
                            views=0,
                            duration_seconds=0,
                            upload_date=None,
                            matched_text=text[:500] if text else "",
                            search_query=search_query,
                        )
                    )
            return results

        # Parse structured results
        for element in elements:
            try:
                # Extract video ID from link
                link = await element.query_selector('a[href*="youtube"], a[href*="youtu.be"]')
                if not link:
                    continue

                href = await link.get_attribute("href")
                video_id = self._extract_video_id(href)
                if not video_id:
                    continue

                # Extract title
                title_el = await element.query_selector("h3, h4, .title, .video-title")
                title = await title_el.inner_text() if title_el else ""

                # Extract channel
                channel_el = await element.query_selector(".channel, .author, .channel-name")
                channel = await channel_el.inner_text() if channel_el else ""

                # Extract views (look for numbers with "views" nearby)
                text_content = await element.inner_text()
                views = self._parse_views(text_content)

                # Extract duration
                duration_el = await element.query_selector(".duration, .length, .time")
                duration_text = await duration_el.inner_text() if duration_el else ""
                duration_seconds = self._parse_duration(duration_text)

                # Extract upload date
                date_el = await element.query_selector(".date, .upload-date, time")
                upload_date = await date_el.inner_text() if date_el else None

                # Extract matched subtitle text
                subtitle_el = await element.query_selector(".subtitle, .caption, .matched-text, .snippet")
                matched_text = await subtitle_el.inner_text() if subtitle_el else text_content[:500]

                results.append(
                    VideoMetadata(
                        video_id=video_id,
                        title=title.strip(),
                        channel=channel.strip(),
                        views=views,
                        duration_seconds=duration_seconds,
                        upload_date=upload_date,
                        matched_text=matched_text.strip(),
                        search_query=search_query,
                    )
                )

            except Exception as e:
                print(f"Error parsing result element: {e}")
                continue

        return results

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        if not url:
            return None

        # youtube.com/watch?v=VIDEO_ID
        match = re.search(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)

        # youtu.be/VIDEO_ID
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)

        # /embed/VIDEO_ID
        match = re.search(r"/embed/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)

        return None

    def _parse_views(self, text: str) -> int:
        """Parse view count from text."""
        # Look for patterns like "123K views", "1.2M views", "1,234 views"
        patterns = [
            r"([\d,.]+)\s*[KkMm]?\s*(?:views|vizualizări|vizionări)",
            r"([\d,.]+)\s*[KkMm]",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(",", "").replace(".", "")
                try:
                    num = int(num_str)
                    # Check for K/M suffix
                    suffix = text[match.end(1) : match.end(1) + 1].upper()
                    if suffix == "K":
                        num *= 1000
                    elif suffix == "M":
                        num *= 1000000
                    return num
                except ValueError:
                    pass

        return 0

    def _parse_duration(self, text: str) -> int:
        """Parse duration from text like '5:23' or '1:23:45'."""
        if not text:
            return 0

        # Remove non-numeric/colon characters
        text = re.sub(r"[^\d:]", "", text)

        parts = text.split(":")
        try:
            if len(parts) == 3:  # H:M:S
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:  # M:S
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 1 and parts[0]:  # Just seconds
                return int(parts[0])
        except ValueError:
            pass

        return 0

    def _passes_filters(self, video: VideoMetadata) -> bool:
        """Check if video passes configured filters."""
        # View count filter
        if self.config.min_views > 0 and video.views < self.config.min_views:
            return False
        if self.config.max_views > 0 and video.views > self.config.max_views:
            return False

        # Duration filter
        if (
            self.config.min_duration_seconds > 0
            and video.duration_seconds < self.config.min_duration_seconds
            and video.duration_seconds > 0  # 0 means unknown
        ):
            return False
        if (
            self.config.max_duration_seconds > 0
            and video.duration_seconds > self.config.max_duration_seconds
        ):
            return False

        return True

    async def _has_next_page(self) -> bool:
        """Check if there's a next page of results."""
        # Look for next page link/button
        next_selectors = [
            'a[rel="next"]',
            ".pagination .next",
            'a:has-text("Next")',
            'a:has-text("Următoarea")',
            ".next-page",
            'a[aria-label="Next page"]',
        ]

        for selector in next_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    # Check if it's not disabled
                    is_disabled = await element.get_attribute("disabled")
                    if not is_disabled:
                        return True
            except Exception:
                continue

        return False

    async def _goto_next_page(self):
        """Navigate to the next page of results."""
        next_selectors = [
            'a[rel="next"]',
            ".pagination .next",
            'a:has-text("Next")',
            'a:has-text("Următoarea")',
            ".next-page",
        ]

        for selector in next_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    await element.click()
                    return
            except Exception:
                continue

        raise Exception("Could not find next page button")


async def save_video_ids(
    videos: List[VideoMetadata], output_path: Path, append: bool = True
):
    """
    Save video metadata to JSONL file.

    Args:
        videos: List of video metadata
        output_path: Output file path
        append: Whether to append or overwrite
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for video in videos:
            record = {
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel,
                "views": video.views,
                "duration_seconds": video.duration_seconds,
                "upload_date": video.upload_date,
                "matched_text": video.matched_text,
                "search_query": video.search_query,
                "scraped_at": datetime.now().isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

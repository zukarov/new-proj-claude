"""
Sentiment client: fetches recent news headlines via Google News RSS.

No API key required. Uses aiohttp (already a project dependency).
Results are cached for 15 minutes to avoid redundant HTTP calls within
a single scan cycle.
"""

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import aiohttp

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 900  # 15 minutes


@dataclass
class NewsItem:
    title: str
    source: str
    published: str


@dataclass
class MarketSentiment:
    query: str
    headlines: list[NewsItem]
    fetched_at: float = field(default_factory=time.time)

    def as_text(self) -> str:
        """Format headlines as a compact bulleted list for Claude's prompt."""
        if not self.headlines:
            return "No recent news found for this topic."
        lines = []
        for item in self.headlines:
            src = f" [{item.source}]" if item.source else ""
            # Google RSS dates look like "Wed, 12 Mar 2025 14:00:00 GMT" — show just date
            date = item.published[:16] if item.published else ""
            pub = f" ({date})" if date else ""
            lines.append(f"- {item.title}{src}{pub}")
        return "\n".join(lines)

    @property
    def has_news(self) -> bool:
        return len(self.headlines) > 0


class SentimentClient:
    """
    Lightweight news sentiment layer using Google News RSS.

    Design:
    - One HTTP request per unique market topic, cached 15 min.
    - Never raises — returns an empty MarketSentiment on any failure.
    - Thread-safe for use within a single asyncio event loop.
    """

    _GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search"
        "?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    def __init__(self) -> None:
        self._cache: dict[str, MarketSentiment] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_news_for_market(
        self,
        question: str,
        max_items: int = 8,
    ) -> MarketSentiment:
        """
        Return recent news headlines relevant to the market question.

        Checks an in-memory cache first (TTL = 15 min).
        Falls back to an empty result on any HTTP or parse error.
        """
        cache_key = self._cache_key(question)
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached.fetched_at) < _CACHE_TTL_SECONDS:
            return cached

        query = self._build_query(question)
        url = self._GOOGLE_NEWS_RSS.format(query=quote_plus(query))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"},
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        logger.debug("News RSS %d for query=%r", resp.status, query)
                        return self._empty(query, cache_key)
                    content = await resp.text()

            headlines = self._parse_rss(content, max_items)
            sentiment = MarketSentiment(query=query, headlines=headlines)
            self._cache[cache_key] = sentiment
            logger.debug("Fetched %d headlines for %r", len(headlines), query[:60])
            return sentiment

        except asyncio.TimeoutError:
            logger.debug("News fetch timeout for %r", query[:60])
        except Exception as exc:
            logger.debug("News fetch error for %r: %s", query[:60], exc)

        return self._empty(query, cache_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_key(self, question: str) -> str:
        return question[:80].lower().strip()

    def _empty(self, query: str, cache_key: str) -> MarketSentiment:
        s = MarketSentiment(query=query, headlines=[])
        self._cache[cache_key] = s
        return s

    def _build_query(self, question: str) -> str:
        """
        Extract a focused search query from a market question.

        Strategy: strip leading verb (Will / Does / Is / Are ...) and
        trailing punctuation, keep the rest up to 80 chars. Google News
        handles natural-language queries well.
        """
        q = question.rstrip("?!.").strip()
        words = q.split()
        if words and words[0].lower() in {
            "will", "does", "is", "are", "can", "has", "have",
            "did", "do", "would", "should", "could",
        }:
            words = words[1:]
        return " ".join(words)[:80].strip()

    def _parse_rss(self, xml_content: str, max_items: int) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            root = ET.fromstring(xml_content)
            channel = root.find("channel")
            if channel is None:
                return items
            for el in channel.findall("item")[:max_items]:
                title = (el.findtext("title") or "").strip()
                if not title:
                    continue
                # Google News often puts "Title - Source Name" in <title>
                source_el = el.find("source")
                source = (source_el.text or "").strip() if source_el is not None else ""
                if not source and " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source = parts[1].strip()
                pub_date = (el.findtext("pubDate") or "").strip()
                items.append(NewsItem(title=title, source=source, published=pub_date))
        except ET.ParseError as exc:
            logger.debug("RSS parse error: %s", exc)
        return items

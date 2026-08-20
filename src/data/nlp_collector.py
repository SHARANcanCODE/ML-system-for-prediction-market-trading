import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from src.data.client import GAMMA_URL
from src.utils.logger import get_logger

log = get_logger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

class CommentCollector:

    def __init__(self, output_dir: str = "data/raw/comments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(timeout=30)

    def collect_for_event(self, event_id: int | str) -> list[dict]:
        all_comments = []
        offset = 0
        while True:
            try:
                r = self._http.get(
                    f"{GAMMA_URL}/comments",
                    params={
                        "parent_entity_id": str(event_id),
                        "parent_entity_type": "Event",
                        "limit": "100",
                        "offset": str(offset),
                    },
                )
                r.raise_for_status()
                comments = r.json()
                if not comments:
                    break
                all_comments.extend(comments)
                offset += len(comments)
                time.sleep(0.05)
                if len(comments) < 100:
                    break
            except Exception as e:
                log.warning(f"Event {event_id} comments failed: {e}")
                break

        return all_comments

    def collect_for_events(
        self,
        event_ids: list[int | str],
    ) -> dict[str, list[dict]]:
        results = {}
        for i, eid in enumerate(event_ids):
            comments = self.collect_for_event(eid)
            results[str(eid)] = comments
            if (i + 1) % 10 == 0:
                log.info(f"[{i+1}/{len(event_ids)}] Collected comments")
            time.sleep(0.05)

        total = sum(len(v) for v in results.values())
        log.info(f"Collected {total} comments for {len(event_ids)} events")
        return results

    def save(self, comments: list[dict], filename: str | None = None):
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comments_{ts}.jsonl"
        path = self.output_dir / filename
        with open(path, "w") as f:
            for c in comments:
                f.write(json.dumps(c, default=str) + "\n")
        log.info(f"Saved {len(comments)} comments to {path}")
        return path

    def close(self):
        self._http.close()

class NewsCollector:

    def __init__(self, output_dir: str = "data/raw/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = self._http.get(url)
            r.raise_for_status()
        except Exception as e:
            log.warning(f"Google News search failed for '{query}': {e}")
            return []

        articles = []
        try:
            root = ET.fromstring(r.text)
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                source = item.findtext("source", "")

                articles.append({
                    "title": title,
                    "link": link,
                    "published": pub_date,
                    "source": source,
                    "query": query,
                    "collected_at": datetime.now().isoformat(),
                })
                if len(articles) >= max_results:
                    break
        except ET.ParseError as e:
            log.warning(f"RSS parse error: {e}")

        return articles

    def collect_for_markets(
        self,
        market_titles: list[str],
        max_per_market: int = 10,
    ) -> dict[str, list[dict]]:
        results = {}
        for title in market_titles:

            query = self._extract_query(title)
            articles = self.search(query, max_results=max_per_market)
            results[title] = articles
            time.sleep(1.0)

        total = sum(len(v) for v in results.values())
        log.info(f"Collected {total} articles for {len(market_titles)} markets")
        return results

    @staticmethod
    def _extract_query(title: str) -> str:
        skip = {"will", "the", "a", "an", "be", "is", "are", "was", "were",
                "by", "in", "on", "at", "to", "for", "of", "and", "or",
                "before", "after", "this", "that", "these", "those"}
        words = [w for w in title.split() if w.lower() not in skip]
        return " ".join(words[:6])

    def save(self, articles_by_market: dict[str, list[dict]], filename: str | None = None):
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_{ts}.jsonl"
        path = self.output_dir / filename
        with open(path, "w") as f:
            for market, articles in articles_by_market.items():
                for a in articles:
                    a["market_title"] = market
                    f.write(json.dumps(a, default=str) + "\n")
        total = sum(len(v) for v in articles_by_market.values())
        log.info(f"Saved {total} articles to {path}")
        return path

    def close(self):
        self._http.close()

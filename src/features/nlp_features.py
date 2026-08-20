from datetime import datetime
from typing import Any

import numpy as np

from src.features.sentiment import SentimentAnalyzer, SentimentResult
from src.utils.logger import get_logger

log = get_logger(__name__)

BULLISH_KEYWORDS = {
    "yes", "will", "bullish", "likely", "definitely", "certain",
    "confirmed", "guaranteed", "inevitable", "obvious", "100%",
    "easy", "lock", "sure", "slam dunk",
}
BEARISH_KEYWORDS = {
    "no", "won't", "bearish", "unlikely", "never", "impossible",
    "doubt", "overpriced", "bubble", "scam", "0%", "waste",
    "no chance", "not going",
}

def extract_comment_features(
    comments: list[dict],
    sentiment_results: list[SentimentResult] | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, float]:
    if not comments:
        return _empty_comment_features()

    texts = []
    for c in comments:
        text = c.get("body", "") or c.get("content", "") or c.get("text", "")
        if text and len(text.strip()) > 2:
            texts.append(text)

    if not texts:
        return _empty_comment_features()

    if sentiment_results is None and analyzer is not None:
        sentiment_results = analyzer.analyze_batch(texts)
    elif sentiment_results is None:
        sentiment_results = []

    all_text = " ".join(texts).lower()
    word_count = max(len(all_text.split()), 1)
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in all_text)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in all_text)
    keyword_total = max(bullish_count + bearish_count, 1)

    velocity = _compute_velocity(comments)

    if sentiment_results:
        sentiments = [r.sentiment for r in sentiment_results]
        features = {
            "nlp_comment_count": len(texts),
            "nlp_comment_sentiment": float(np.mean(sentiments)),
            "nlp_comment_sentiment_std": float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
            "nlp_comment_positive_ratio": sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            "nlp_comment_velocity": velocity,
            "nlp_bullish_keyword_ratio": bullish_count / keyword_total,
        }
    else:
        features = {
            "nlp_comment_count": len(texts),
            "nlp_comment_sentiment": 0.0,
            "nlp_comment_sentiment_std": 0.0,
            "nlp_comment_positive_ratio": 0.0,
            "nlp_comment_velocity": velocity,
            "nlp_bullish_keyword_ratio": bullish_count / keyword_total,
        }

    return features

def extract_news_features(
    articles: list[dict],
    sentiment_results: list[SentimentResult] | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, float]:
    if not articles:
        return _empty_news_features()

    texts = [a.get("title", "") for a in articles if a.get("title")]
    if not texts:
        return _empty_news_features()

    if sentiment_results is None and analyzer is not None:
        sentiment_results = analyzer.analyze_batch(texts)

    if sentiment_results:
        sentiments = [r.sentiment for r in sentiment_results]
        return {
            "nlp_news_count": len(texts),
            "nlp_news_sentiment": float(np.mean(sentiments)),
        }

    return {
        "nlp_news_count": len(texts),
        "nlp_news_sentiment": 0.0,
    }

def combine_nlp_features(
    comment_features: dict[str, float],
    news_features: dict[str, float],
) -> dict[str, float]:
    features = {}
    features.update(comment_features)
    features.update(news_features)

    cs = comment_features.get("nlp_comment_sentiment", 0.0)
    ns = news_features.get("nlp_news_sentiment", 0.0)
    features["nlp_sentiment_divergence"] = abs(cs - ns)

    comment_n = comment_features.get("nlp_comment_count", 0)
    news_n = news_features.get("nlp_news_count", 0)
    total = max(comment_n + news_n, 1)
    features["nlp_mention_frequency"] = total

    return features

def _compute_velocity(comments: list[dict]) -> float:
    timestamps = []
    for c in comments:
        ts = c.get("createdAt") or c.get("created_at") or c.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts)
                else:
                    continue
                timestamps.append(dt)
            except (ValueError, OSError):
                continue

    if len(timestamps) < 2:
        return 0.0

    time_span = (max(timestamps) - min(timestamps)).total_seconds()
    if time_span < 60:
        return 0.0
    hours = time_span / 3600
    return len(timestamps) / hours

def _empty_comment_features() -> dict[str, float]:
    return {
        "nlp_comment_count": 0,
        "nlp_comment_sentiment": 0.0,
        "nlp_comment_sentiment_std": 0.0,
        "nlp_comment_positive_ratio": 0.0,
        "nlp_comment_velocity": 0.0,
        "nlp_bullish_keyword_ratio": 0.0,
    }

def _empty_news_features() -> dict[str, float]:
    return {
        "nlp_news_count": 0,
        "nlp_news_sentiment": 0.0,
    }

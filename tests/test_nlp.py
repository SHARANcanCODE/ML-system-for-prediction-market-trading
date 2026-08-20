"""Tests for NLP modules: collector, sentiment, features."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.features.sentiment import SentimentResult, SentimentAnalyzer
from src.features.nlp_features import (
    extract_comment_features,
    extract_news_features,
    combine_nlp_features,
    _compute_velocity,
    BULLISH_KEYWORDS,
    BEARISH_KEYWORDS,
)
from src.data.nlp_collector import NewsCollector


class TestSentimentResult:
    def test_positive_sentiment(self):
        r = SentimentResult(text="Great!", label="positive", score=0.95, sentiment=0.95)
        assert r.sentiment > 0
        assert r.label == "positive"

    def test_negative_sentiment(self):
        r = SentimentResult(text="Bad!", label="negative", score=0.90, sentiment=-0.90)
        assert r.sentiment < 0

    def test_neutral_sentiment(self):
        r = SentimentResult(text="Ok", label="neutral", score=0.60, sentiment=0.0)
        assert r.sentiment == 0.0


class TestSentimentAnalyzerAggregate:
    def test_aggregate_mixed(self):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        results = [
            SentimentResult("a", "positive", 0.9, 0.9),
            SentimentResult("b", "negative", 0.8, -0.8),
            SentimentResult("c", "neutral", 0.7, 0.0),
        ]
        agg = analyzer.aggregate(results)
        assert agg["nlp_text_count"] == 3
        assert -1 <= agg["nlp_sentiment_mean"] <= 1
        assert agg["nlp_positive_ratio"] == pytest.approx(1 / 3)
        assert agg["nlp_negative_ratio"] == pytest.approx(1 / 3)
        assert agg["nlp_neutral_ratio"] == pytest.approx(1 / 3)
        assert agg["nlp_sentiment_strength"] > 0

    def test_aggregate_empty(self):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        agg = analyzer.aggregate([])
        assert agg["nlp_text_count"] == 0
        assert agg["nlp_sentiment_mean"] == 0.0

    def test_aggregate_all_positive(self):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        results = [SentimentResult("x", "positive", 0.9, 0.9) for _ in range(5)]
        agg = analyzer.aggregate(results)
        assert agg["nlp_positive_ratio"] == 1.0
        assert agg["nlp_sentiment_mean"] == pytest.approx(0.9)


class TestCommentFeatures:
    def test_basic_comment_features(self):
        comments = [
            {"content": "This will definitely happen", "createdAt": "2026-03-06T10:00:00Z"},
            {"content": "No chance at all", "createdAt": "2026-03-06T11:00:00Z"},
            {"content": "I think yes", "createdAt": "2026-03-06T12:00:00Z"},
        ]
        sentiments = [
            SentimentResult("a", "positive", 0.9, 0.9),
            SentimentResult("b", "negative", 0.8, -0.8),
            SentimentResult("c", "positive", 0.7, 0.7),
        ]
        features = extract_comment_features(comments, sentiment_results=sentiments)
        assert features["nlp_comment_count"] == 3
        assert features["nlp_comment_sentiment"] == pytest.approx((0.9 - 0.8 + 0.7) / 3, abs=0.01)
        assert features["nlp_comment_positive_ratio"] == pytest.approx(2 / 3)
        assert features["nlp_comment_velocity"] > 0

    def test_empty_comments(self):
        features = extract_comment_features([])
        assert features["nlp_comment_count"] == 0
        assert features["nlp_comment_sentiment"] == 0.0

    def test_comments_without_text(self):
        comments = [{"id": 1}, {"id": 2}]
        features = extract_comment_features(comments)
        assert features["nlp_comment_count"] == 0

    def test_bullish_keywords(self):
        comments = [
            {"content": "This will definitely happen, yes 100% guaranteed lock"},
        ]
        sentiments = [SentimentResult("a", "positive", 0.9, 0.9)]
        features = extract_comment_features(comments, sentiment_results=sentiments)
        assert features["nlp_bullish_keyword_ratio"] > 0.5


class TestNewsFeatures:
    def test_basic_news_features(self):
        articles = [
            {"title": "Bitcoin surges to new highs"},
            {"title": "Markets crash on uncertainty"},
        ]
        sentiments = [
            SentimentResult("a", "positive", 0.9, 0.9),
            SentimentResult("b", "negative", 0.8, -0.8),
        ]
        features = extract_news_features(articles, sentiment_results=sentiments)
        assert features["nlp_news_count"] == 2
        assert -1 <= features["nlp_news_sentiment"] <= 1

    def test_empty_news(self):
        features = extract_news_features([])
        assert features["nlp_news_count"] == 0
        assert features["nlp_news_sentiment"] == 0.0


class TestCombineFeatures:
    def test_combine_adds_divergence(self):
        comment_f = {
            "nlp_comment_count": 5,
            "nlp_comment_sentiment": 0.8,
            "nlp_comment_sentiment_std": 0.1,
            "nlp_comment_positive_ratio": 0.8,
            "nlp_comment_velocity": 2.0,
            "nlp_bullish_keyword_ratio": 0.7,
        }
        news_f = {
            "nlp_news_count": 3,
            "nlp_news_sentiment": -0.3,
        }
        combined = combine_nlp_features(comment_f, news_f)
        assert "nlp_sentiment_divergence" in combined
        assert combined["nlp_sentiment_divergence"] == pytest.approx(1.1)
        assert combined["nlp_mention_frequency"] == 8
        assert combined["nlp_comment_sentiment"] == 0.8
        assert combined["nlp_news_sentiment"] == -0.3


class TestCommentVelocity:
    def test_velocity_calculation(self):
        now = datetime(2026, 3, 6, 12, 0, 0)
        comments = [
            {"createdAt": (now - timedelta(hours=2)).isoformat()},
            {"createdAt": (now - timedelta(hours=1)).isoformat()},
            {"createdAt": now.isoformat()},
        ]
        velocity = _compute_velocity(comments)
        assert velocity == pytest.approx(1.5, abs=0.1)  # 3 comments / 2 hours

    def test_velocity_no_timestamps(self):
        comments = [{"id": 1}, {"id": 2}]
        assert _compute_velocity(comments) == 0.0

    def test_velocity_single_comment(self):
        comments = [{"createdAt": "2026-03-06T12:00:00Z"}]
        assert _compute_velocity(comments) == 0.0


class TestNewsCollectorQueryExtraction:
    def test_extract_query_removes_stopwords(self):
        query = NewsCollector._extract_query("Will Trump win the election in 2026?")
        assert "will" not in query.lower()
        assert "the" not in query.lower()
        assert "Trump" in query

    def test_extract_query_limits_words(self):
        long_title = "This is a very long market title with many words about something"
        query = NewsCollector._extract_query(long_title)
        assert len(query.split()) <= 6

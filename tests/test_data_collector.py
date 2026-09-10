"""Unit tests for the stock sentiment predictor data collection module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_collector import (
    DEFAULT_COMPANY_MAP,
    NewsCollector,
    RedditCollector,
    StockDataCollector,
    _FallbackDataFrame,
    collect_all_data,
)


class TestFallbackDataFrame(unittest.TestCase):
    """Tests for the lightweight fallback DataFrame."""

    def test_basic_operations(self):
        records = [
            {"title": "Test 1", "score": 10},
            {"title": "Test 2", "score": 20},
        ]
        df = _FallbackDataFrame(records)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.shape, (2, 2))
        self.assertIn("title", df.columns)
        self.assertEqual(df["score"], [10, 20])

        df["new_col"] = "constant"
        self.assertEqual(df["new_col"], ["constant", "constant"])

    def test_to_csv(self):
        records = [{"a": "val1", "b": 1}, {"a": "val2", "b": 2}]
        df = _FallbackDataFrame(records)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test.csv"
            df.to_csv(csv_path)
            self.assertTrue(csv_path.exists())
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("val1", content)
            self.assertIn("val2", content)


class TestStockDataCollector(unittest.TestCase):
    """Tests for StockDataCollector class."""

    def setUp(self):
        self.collector = StockDataCollector(
            tickers=["AAPL", "TSLA"],
            start_date="2023-01-01",
            end_date="2023-02-01",
            max_retries=2,
            retry_delay=0.01,
        )

    def test_init_defaults(self):
        default_collector = StockDataCollector()
        self.assertTrue(len(default_collector.tickers) > 0)
        self.assertIsNotNone(default_collector.start_date)
        self.assertIsNotNone(default_collector.end_date)

    @patch("src.data_collector.yf")
    def test_fetch_stock_data_success(self, mock_yf):
        # Create mock Ticker with history
        mock_ticker_instance = MagicMock()
        mock_df = _FallbackDataFrame(
            [
                {
                    "Date": "2023-01-03",
                    "Open": 150.0,
                    "High": 152.0,
                    "Low": 149.0,
                    "Close": 151.0,
                    "Adj Close": 151.0,
                    "Volume": 1000000,
                }
            ]
        )
        mock_ticker_instance.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker_instance

        df = self.collector.fetch_stock_data("AAPL")
        self.assertFalse(df.empty)
        self.assertIn("Ticker", df.columns)
        self.assertEqual(df["Ticker"], ["AAPL"])

    @patch("src.data_collector.yf", None)
    def test_fetch_stock_data_when_yf_missing(self):
        df = self.collector.fetch_stock_data("AAPL")
        self.assertTrue(df.empty)
        self.assertIn("Open", df.columns)

    @patch("src.data_collector.yf")
    def test_save_stock_data(self, mock_yf):
        mock_ticker = MagicMock()
        mock_df = _FallbackDataFrame(
            [{"Date": "2023-01-03", "Close": 100.0, "Open": 99.0, "Volume": 500}]
        )
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.collector.fetch_all_stocks()
            saved = self.collector.save_stock_data(tmp_dir)
            self.assertIn("AAPL", saved)
            self.assertIn("TSLA", saved)
            self.assertTrue(saved["AAPL"].exists())
            self.assertTrue(saved["TSLA"].exists())


class TestNewsCollector(unittest.TestCase):
    """Tests for NewsCollector class."""

    def setUp(self):
        self.collector = NewsCollector(
            api_key="valid_test_key",
            max_retries=2,
            retry_delay=0.01,
        )

    def test_init(self):
        self.assertEqual(self.collector.api_key, "valid_test_key")

    def test_fetch_news_placeholder_key(self):
        collector = NewsCollector(api_key="YOUR_NEWS_API_KEY_HERE")
        news = collector.fetch_news("Apple", "2023-01-01", "2023-01-10")
        self.assertEqual(news, [])

    @patch("src.data_collector.requests")
    def test_fetch_news_success_with_requests(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Apple hits record high",
                    "description": "Apple stock reached a new all-time high today.",
                    "source": {"name": "Reuters"},
                    "publishedAt": "2023-01-15T12:00:00Z",
                    "url": "https://example.com/apple",
                }
            ]
        }
        mock_requests.get.return_value = mock_response

        articles = self.collector.fetch_news("Apple", "2023-01-01", "2023-01-20")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Apple hits record high")
        self.assertEqual(articles[0]["source"], "Reuters")
        self.assertEqual(articles[0]["published_at"], "2023-01-15T12:00:00Z")

    @patch("src.data_collector.requests")
    def test_fetch_stock_news(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Tesla updates autonomy software",
                    "description": "Tesla rolling out update.",
                    "source": {"name": "Bloomberg"},
                    "publishedAt": "2023-01-10T08:00:00Z",
                    "url": "https://example.com/tesla",
                }
            ]
        }
        mock_requests.get.return_value = mock_response

        df = self.collector.fetch_stock_news("TSLA", "Tesla", days_back=10)
        self.assertFalse(df.empty)
        self.assertIn("ticker", df.columns)
        self.assertIn("company_name", df.columns)
        self.assertEqual(df["ticker"].tolist(), ["TSLA"])

    def test_save_news_data(self):
        df = _FallbackDataFrame(
            [{"title": "Test Title", "description": "Test Desc", "ticker": "AAPL"}]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "news.csv"
            saved_path = self.collector.save_news_data(df, file_path)
            self.assertTrue(saved_path.exists())


class TestRedditCollector(unittest.TestCase):
    """Tests for RedditCollector class."""

    def test_init_without_praw_or_placeholder_keys(self):
        collector = RedditCollector(
            client_id="YOUR_REDDIT_CLIENT_ID_HERE",
            client_secret="YOUR_REDDIT_CLIENT_SECRET_HERE",
            user_agent="test_agent",
        )
        self.assertIsNone(collector.reddit)
        posts = collector.fetch_subreddit_posts("stocks", "AAPL")
        self.assertTrue(posts.empty)

    @patch("src.data_collector.praw")
    def test_fetch_subreddit_posts_with_mock_praw(self, mock_praw):
        mock_reddit = MagicMock()
        mock_sub = MagicMock()
        mock_submission = MagicMock()
        mock_submission.title = "Bullish on Apple"
        mock_submission.selftext = "Strong quarterly results expected."
        mock_submission.score = 42
        mock_submission.created_utc = 1673740800.0
        mock_submission.num_comments = 15
        mock_submission.id = "abc123"
        mock_submission.url = "https://reddit.com/r/stocks/abc123"

        mock_sub.search.return_value = [mock_submission]
        mock_reddit.subreddit.return_value = mock_sub
        mock_praw.Reddit.return_value = mock_reddit

        collector = RedditCollector(
            client_id="real_client_id",
            client_secret="real_secret",
            user_agent="test_user_agent",
        )
        posts_df = collector.fetch_subreddit_posts("stocks", "AAPL")
        self.assertFalse(posts_df.empty)
        self.assertEqual(len(posts_df), 1)
        self.assertIn("score", posts_df.columns)
        self.assertEqual(posts_df["title"].tolist(), ["Bullish on Apple"])
        self.assertEqual(posts_df["score"].tolist(), [42])

    @patch("src.data_collector.praw")
    def test_fetch_stock_mentions(self, mock_praw):
        mock_reddit = MagicMock()
        mock_sub = MagicMock()
        mock_submission = MagicMock()
        mock_submission.title = "AAPL earnings incoming"
        mock_submission.selftext = "Thoughts on calls?"
        mock_submission.score = 55
        mock_submission.created_utc = 1673740800.0
        mock_submission.num_comments = 25
        mock_submission.id = "post_999"
        mock_submission.url = "https://reddit.com/r/stocks/post_999"

        mock_sub.search.return_value = [mock_submission]
        mock_reddit.subreddit.return_value = mock_sub
        mock_praw.Reddit.return_value = mock_reddit

        collector = RedditCollector(
            client_id="real_client_id",
            client_secret="real_secret",
            user_agent="test_agent",
        )
        mentions_df = collector.fetch_stock_mentions("AAPL", subreddits=["stocks"])
        self.assertIn("ticker", mentions_df.columns)
        self.assertEqual(mentions_df["ticker"].tolist(), ["AAPL"])
        self.assertEqual(len(mentions_df), 1)


class TestCollectAllData(unittest.TestCase):
    """Tests for the master collect_all_data orchestration function."""

    @patch("src.data_collector.StockDataCollector.fetch_all_stocks")
    @patch("src.data_collector.StockDataCollector.save_stock_data")
    @patch("src.data_collector.NewsCollector.fetch_stock_news")
    @patch("src.data_collector.NewsCollector.save_news_data")
    @patch("src.data_collector.RedditCollector.fetch_stock_mentions")
    @patch("src.data_collector.RedditCollector.save_reddit_data")
    def test_collect_all_data_flow(
        self,
        mock_save_reddit,
        mock_fetch_reddit,
        mock_save_news,
        mock_fetch_news,
        mock_save_stock,
        mock_fetch_stock,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw"
            dummy_stock_df = _FallbackDataFrame([{"Date": "2023-01-01", "Close": 100.0}])
            dummy_news_df = _FallbackDataFrame([{"title": "News 1", "ticker": "AAPL"}])
            dummy_reddit_df = _FallbackDataFrame([{"title": "Post 1", "ticker": "AAPL"}])

            mock_fetch_stock.return_value = {"AAPL": dummy_stock_df}
            mock_save_stock.return_value = {"AAPL": raw_path / "stocks" / "AAPL.csv"}
            mock_fetch_news.return_value = dummy_news_df
            mock_save_news.return_value = raw_path / "news" / "AAPL_news.csv"
            mock_fetch_reddit.return_value = dummy_reddit_df
            mock_save_reddit.return_value = raw_path / "reddit" / "AAPL_reddit.csv"

            config_override = {
                "DATA_RAW_DIR": raw_path,
                "DEFAULT_TICKERS": ["AAPL"],
                "START_DATE": "2023-01-01",
                "END_DATE": "2023-02-01",
                "NEWS_API_KEY": "dummy_news_key",
                "REDDIT_CLIENT_ID": "dummy_reddit_id",
                "REDDIT_CLIENT_SECRET": "dummy_reddit_secret",
                "REDDIT_USER_AGENT": "test_agent",
            }

            result = collect_all_data(tickers=["AAPL"], config=config_override)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["tickers"], ["AAPL"])
            self.assertIn("stocks", result)
            self.assertIn("news", result)
            self.assertIn("reddit", result)
            self.assertTrue(mock_fetch_stock.called)
            self.assertTrue(mock_save_stock.called)
            self.assertTrue(mock_fetch_news.called)
            self.assertTrue(mock_fetch_reddit.called)


if __name__ == "__main__":
    unittest.main()

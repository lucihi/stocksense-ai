"""Data collection module for stock prices, financial news, and Reddit sentiment.

This module orchestrates data ingestion from multiple external sources:
1. Historical OHLCV market data via Yahoo Finance (yfinance).
2. Recent company and market news via NewsAPI.
3. Social media discussions via Reddit (PRAW).

Includes robust error handling, exponential backoff retries, structured logging,
and automatic directory resolution.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("stock_sentiment.data_collector")
if not logger.handlers:
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Project Root & Configuration Resolution
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import (  # type: ignore
        DATA_RAW_DIR,
        DEFAULT_TICKERS,
        END_DATE,
        NEWS_API_KEY,
        REDDIT_CLIENT_ID,
        REDDIT_CLIENT_SECRET,
        REDDIT_USER_AGENT,
        START_DATE,
    )
except ImportError:
    try:
        import config  # type: ignore

        DATA_RAW_DIR = getattr(
            config,
            "DATA_RAW_DIR",
            getattr(config, "DATA_RAW", PROJECT_ROOT / "data" / "raw"),
        )
        DEFAULT_TICKERS = getattr(
            config, "DEFAULT_TICKERS", ["AAPL", "TSLA", "GOOGL"]
        )
        START_DATE = getattr(config, "START_DATE", "2022-01-01")
        END_DATE = getattr(config, "END_DATE", "2024-01-01")
        NEWS_API_KEY = getattr(config, "NEWS_API_KEY", "")
        REDDIT_CLIENT_ID = getattr(config, "REDDIT_CLIENT_ID", "")
        REDDIT_CLIENT_SECRET = getattr(config, "REDDIT_CLIENT_SECRET", "")
        REDDIT_USER_AGENT = getattr(
            config, "REDDIT_USER_AGENT", "stock-sentiment-predictor/1.0"
        )
    except ImportError:
        DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
        DEFAULT_TICKERS = ["AAPL", "TSLA", "GOOGL"]
        START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        END_DATE = datetime.now().strftime("%Y-%m-%d")
        NEWS_API_KEY = ""
        REDDIT_CLIENT_ID = ""
        REDDIT_CLIENT_SECRET = ""
        REDDIT_USER_AGENT = "stock-sentiment-predictor/1.0"

# ------------------------------------------------------------------------------
# Optional 3rd-Party Imports with Fallback Handling
# ------------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

try:
    import praw  # type: ignore
except ImportError:
    praw = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None


# ------------------------------------------------------------------------------
# Lightweight Fallback DataFrame (used when pandas is not installed)
# ------------------------------------------------------------------------------
class _FallbackDataFrame:
    """Lightweight 2D tabular data structure mimicking essential pandas DataFrame methods."""

    def __init__(
        self,
        data: list[dict[str, Any]] | dict[str, list[Any]] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self._rows: list[dict[str, Any]] = []
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                self._rows = [dict(r) for r in data]
                all_cols: list[str] = []
                for row in self._rows:
                    for k in row.keys():
                        if k not in all_cols:
                            all_cols.append(k)
                self.columns: list[str] = columns if columns is not None else all_cols
            elif data and isinstance(data[0], (list, tuple)):
                inferred_cols = columns or [f"col_{i}" for i in range(len(data[0]))]
                self.columns = list(inferred_cols)
                self._rows = [dict(zip(self.columns, r)) for r in data]
            else:
                self.columns = list(columns) if columns is not None else []
                self._rows = []
        elif isinstance(data, dict):
            keys = list(data.keys())
            self.columns = columns if columns is not None else keys
            num_rows = len(data[keys[0]]) if keys else 0
            self._rows = []
            for i in range(num_rows):
                self._rows.append({k: data[k][i] for k in keys})
        else:
            self.columns = list(columns) if columns is not None else []
            self._rows = []

    @property
    def empty(self) -> bool:
        """Returns True if DataFrame contains no rows."""
        return len(self._rows) == 0

    @property
    def shape(self) -> tuple[int, int]:
        """Returns tuple of (rows, columns)."""
        return (len(self._rows), len(self.columns))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> list[Any]:
        return [row.get(key) for row in self._rows]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.columns:
            self.columns.append(key)
        if isinstance(value, (list, tuple)):
            for i, val in enumerate(value):
                if i < len(self._rows):
                    self._rows[i][key] = val
        else:
            for row in self._rows:
                row[key] = value

    def to_csv(self, path_or_buf: str | Path, index: bool = False) -> None:
        """Saves tabular rows to a CSV file."""
        file_path = Path(path_or_buf)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({col: row.get(col, "") for col in self.columns})

    def drop_duplicates(
        self, subset: list[str] | None = None, keep: str = "first"
    ) -> _FallbackDataFrame:
        """Removes duplicate rows based on subset keys."""
        seen = set()
        unique_rows = []
        for row in self._rows:
            if subset:
                identifier = tuple(row.get(k) for k in subset)
            else:
                identifier = tuple(sorted(row.items()))
            if identifier not in seen:
                seen.add(identifier)
                unique_rows.append(row)
        return _FallbackDataFrame(data=unique_rows, columns=self.columns)

    def reset_index(self, drop: bool = False) -> _FallbackDataFrame:
        """No-op for fallback table compatibility."""
        return self

    def head(self, n: int = 5) -> _FallbackDataFrame:
        """Returns first n rows."""
        return _FallbackDataFrame(data=self._rows[:n], columns=self.columns)

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Exports data to a list of dicts."""
        return [dict(r) for r in self._rows]

    def __repr__(self) -> str:
        return f"<FallbackDataFrame shape={self.shape} columns={self.columns}>"


def _fallback_concat(
    dfs: list[_FallbackDataFrame], ignore_index: bool = True
) -> _FallbackDataFrame:
    """Concatenates multiple fallback DataFrames along rows."""
    combined_rows: list[dict[str, Any]] = []
    combined_cols: list[str] = []
    for df in dfs:
        for c in df.columns:
            if c not in combined_cols:
                combined_cols.append(c)
        combined_rows.extend(df._rows)
    return _FallbackDataFrame(data=combined_rows, columns=combined_cols)


if pd is None:

    class _FallbackPandasModule:
        DataFrame = _FallbackDataFrame
        concat = staticmethod(_fallback_concat)

        @staticmethod
        def to_datetime(val: Any, **kwargs: Any) -> Any:
            return str(val)

    pd = _FallbackPandasModule()  # type: ignore


# ------------------------------------------------------------------------------
# Default Ticker-to-Company Mapping
# ------------------------------------------------------------------------------
DEFAULT_COMPANY_MAP: dict[str, str] = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia",
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
}


# ==============================================================================
# 1. StockDataCollector
# ==============================================================================
class StockDataCollector:
    """Fetches and persists historical stock market OHLCV data via yfinance."""

    def __init__(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initializes the stock data collector.

        Args:
            tickers: List of stock ticker symbols (e.g. ['AAPL', 'TSLA']).
                     Defaults to config.DEFAULT_TICKERS.
            start_date: Historical start date string (YYYY-MM-DD). Defaults to config.START_DATE.
            end_date: Historical end date string (YYYY-MM-DD). Defaults to config.END_DATE.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Initial delay in seconds before exponential backoff retry.
        """
        self.tickers: list[str] = (
            list(tickers) if tickers is not None else list(DEFAULT_TICKERS)
        )
        self.start_date: str = (
            str(start_date) if start_date is not None else str(START_DATE)
        )
        self.end_date: str = (
            str(end_date) if end_date is not None else str(END_DATE)
        )
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self.data: dict[str, Any] = {}

    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """Fetches historical OHLCV data for a single ticker with retries.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            pd.DataFrame with OHLCV data, Date column, and Ticker identifier.
        """
        standard_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Ticker",
        ]

        if yf is None:
            logger.warning(
                "yfinance library is not installed. Returning empty DataFrame for '%s'.",
                ticker,
            )
            return pd.DataFrame(columns=standard_cols)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching OHLCV data for '%s' (%s to %s, attempt %d/%d)...",
                    ticker,
                    self.start_date,
                    self.end_date,
                    attempt,
                    self.max_retries,
                )
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=False,
                )

                # Fallback to yf.download if history returned empty
                if df is None or (hasattr(df, "empty") and df.empty):
                    logger.debug(
                        "Ticker.history empty for '%s', trying yf.download...",
                        ticker,
                    )
                    df = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                    )

                if df is not None and hasattr(df, "empty") and not df.empty:
                    # Flatten MultiIndex columns if generated by yf.download
                    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                        df.columns = df.columns.get_level_values(0)

                    # Reset DatetimeIndex to create a clean 'Date' column
                    if hasattr(df, "reset_index"):
                        df = df.reset_index()

                    # Standardize Date column
                    if "Date" not in df.columns:
                        for candidate in ["Datetime", "index"]:
                            if candidate in df.columns:
                                df.rename(
                                    columns={candidate: "Date"}, inplace=True
                                )
                                break

                    # Ensure timezone-naive Date strings for consistent CSV serialization
                    if "Date" in df.columns and hasattr(df["Date"], "dt"):
                        try:
                            df["Date"] = df["Date"].dt.tz_localize(None)
                        except (TypeError, AttributeError):
                            pass

                    # If Adj Close is missing, default to Close
                    if "Adj Close" not in df.columns and "Close" in df.columns:
                        df["Adj Close"] = df["Close"]

                    df["Ticker"] = ticker

                    logger.info(
                        "Successfully fetched %d records for '%s'.",
                        len(df),
                        ticker,
                    )
                    return df

                logger.warning(
                    "Attempt %d/%d: Empty dataset returned for '%s'.",
                    attempt,
                    self.max_retries,
                    ticker,
                )
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed fetching '%s': %s",
                    attempt,
                    self.max_retries,
                    ticker,
                    exc,
                )

            if attempt < self.max_retries:
                backoff_wait = self.retry_delay * (2 ** (attempt - 1))
                logger.debug(
                    "Backing off for %.1f seconds before retry...", backoff_wait
                )
                time.sleep(backoff_wait)

        logger.error(
            "Failed to fetch stock data for '%s' after %d attempts.",
            ticker,
            self.max_retries,
        )
        return pd.DataFrame(columns=standard_cols)

    def fetch_all_stocks(self) -> dict[str, pd.DataFrame]:
        """Fetches OHLCV data for all configured tickers.

        Returns:
            Dictionary mapping ticker symbols to their respective DataFrames.
        """
        logger.info(
            "Starting batch stock data collection for %d tickers: %s",
            len(self.tickers),
            self.tickers,
        )
        results: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            results[ticker] = df

        self.data = results
        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(
            "Batch stock collection completed: %d/%d tickers fetched successfully.",
            successful,
            len(self.tickers),
        )
        return results

    def save_stock_data(
        self, output_dir: str | Path | None = None
    ) -> dict[str, Path]:
        """Saves each ticker's DataFrame as a CSV file in the specified directory.

        Args:
            output_dir: Destination directory path. Defaults to DATA_RAW_DIR / 'stocks'.

        Returns:
            Dictionary mapping ticker symbols to their saved CSV file paths.
        """
        target_dir = (
            Path(output_dir) if output_dir else Path(DATA_RAW_DIR) / "stocks"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        if not self.data:
            logger.info("No cached stock data found. Fetching all stocks first...")
            self.fetch_all_stocks()

        saved_files: dict[str, Path] = {}
        for ticker, df in self.data.items():
            if df is not None and not df.empty:
                safe_ticker = ticker.replace(":", "_").replace("^", "")
                csv_path = target_dir / f"{safe_ticker}.csv"
                df.to_csv(csv_path, index=False)
                saved_files[ticker] = csv_path
                logger.info(
                    "Saved stock data for '%s' (%d rows) -> %s",
                    ticker,
                    len(df),
                    csv_path,
                )
            else:
                logger.warning(
                    "Skipping save for '%s': No data available.", ticker
                )

        return saved_files


# ==============================================================================
# 2. NewsCollector
# ==============================================================================
class NewsCollector:
    """Collects financial news articles via NewsAPI."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initializes the news collector.

        Args:
            api_key: NewsAPI authentication key. Defaults to config.NEWS_API_KEY.
            max_retries: Maximum number of retry attempts for network requests.
            retry_delay: Initial delay in seconds before exponential backoff retry.
        """
        self.api_key: str = str(api_key or NEWS_API_KEY or "").strip()
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self.base_url: str = "https://newsapi.org/v2/everything"

    def fetch_news(
        self,
        query: str,
        from_date: str,
        to_date: str,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Queries NewsAPI for articles matching the specified search parameters.

        Args:
            query: Search keyword or boolean expression (e.g. 'Apple OR AAPL').
            from_date: ISO 8601 or YYYY-MM-DD start date.
            to_date: ISO 8601 or YYYY-MM-DD end date.
            page_size: Maximum number of articles to return (capped at 100).

        Returns:
            List of dictionaries: [{title, description, source, published_at, url}, ...]
        """
        if (
            not self.api_key
            or self.api_key.startswith("YOUR_")
            or self.api_key == "demo"
        ):
            logger.warning(
                "NewsAPI key is not configured or is a placeholder. Skipping news fetch for '%s'.",
                query,
            )
            return []

        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "pageSize": min(page_size, 100),
            "sortBy": "relevancy",
            "language": "en",
        }
        headers = {
            "X-Api-Key": self.api_key,
            "User-Agent": "StockSentimentPredictor/1.0",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching news for '%s' (%s to %s, attempt %d/%d)...",
                    query,
                    from_date,
                    to_date,
                    attempt,
                    self.max_retries,
                )

                if requests is not None:
                    response = requests.get(
                        self.base_url,
                        params=params,
                        headers=headers,
                        timeout=15,
                    )
                    status_code = response.status_code
                    body_json = response.json()
                else:
                    encoded_params = urllib.parse.urlencode(params)
                    req_url = f"{self.base_url}?{encoded_params}"
                    req = urllib.request.Request(req_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        status_code = resp.status
                        body_json = json.loads(resp.read().decode("utf-8"))

                if status_code == 200:
                    raw_articles = body_json.get("articles", [])
                    extracted: list[dict[str, Any]] = []
                    for art in raw_articles:
                        src_data = art.get("source", {})
                        source_name = (
                            src_data.get("name", "")
                            if isinstance(src_data, dict)
                            else str(src_data or "")
                        )
                        extracted.append(
                            {
                                "title": art.get("title") or "",
                                "description": art.get("description") or "",
                                "source": source_name,
                                "published_at": art.get("publishedAt") or "",
                                "url": art.get("url") or "",
                            }
                        )
                    logger.info(
                        "Successfully fetched %d articles for '%s'.",
                        len(extracted),
                        query,
                    )
                    return extracted
                elif status_code == 401:
                    logger.error(
                        "NewsAPI returned 401 Unauthorized. Invalid API key provided."
                    )
                    return []
                elif status_code == 429:
                    logger.warning(
                        "NewsAPI rate limit reached (HTTP 429) on attempt %d/%d.",
                        attempt,
                        self.max_retries,
                    )
                else:
                    logger.warning(
                        "NewsAPI returned status %d on attempt %d/%d.",
                        status_code,
                        attempt,
                        self.max_retries,
                    )
            except Exception as exc:
                logger.warning(
                    "NewsAPI network request error on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                backoff_wait = self.retry_delay * (2 ** (attempt - 1))
                time.sleep(backoff_wait)

        logger.error(
            "Failed to fetch news for '%s' after %d attempts.",
            query,
            self.max_retries,
        )
        return []

    def fetch_stock_news(
        self,
        ticker: str,
        company_name: str = "",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Fetches news articles related to a specific stock ticker and company name.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            company_name: Common company name (e.g. 'Apple').
            days_back: Number of historical days to query. Default is 30.

        Returns:
            pd.DataFrame with columns: title, description, source, published_at, url, ticker, company_name.
        """
        now = datetime.now()
        from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        # Clean symbol (e.g. RELIANCE.NS -> RELIANCE)
        clean_ticker = ticker.split(".")[0]
        resolved_company = (
            company_name or DEFAULT_COMPANY_MAP.get(ticker, clean_ticker)
        )

        if resolved_company and resolved_company.lower() != clean_ticker.lower():
            query = f'"{resolved_company}" OR "{clean_ticker}"'
        else:
            query = f'"{clean_ticker}"'

        articles = self.fetch_news(
            query=query, from_date=from_date, to_date=to_date
        )

        standard_cols = [
            "title",
            "description",
            "source",
            "published_at",
            "url",
            "ticker",
            "company_name",
        ]

        if articles:
            df = pd.DataFrame(articles)
        else:
            df = pd.DataFrame(columns=standard_cols)

        df["ticker"] = ticker
        df["company_name"] = resolved_company
        return df

    def save_news_data(
        self, df: pd.DataFrame, output_path: str | Path
    ) -> Path:
        """Saves news DataFrame to a CSV file.

        Args:
            df: DataFrame of news records.
            output_path: Target file path.

        Returns:
            Path object pointing to the written CSV file.
        """
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(
            "Saved %d news records -> %s",
            len(df),
            file_path,
        )
        return file_path


# ==============================================================================
# 3. RedditCollector
# ==============================================================================
class RedditCollector:
    """Collects social media posts and sentiment discussions from Reddit via PRAW."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initializes the Reddit collector.

        Args:
            client_id: Reddit script application client ID. Defaults to config.REDDIT_CLIENT_ID.
            client_secret: Reddit application client secret. Defaults to config.REDDIT_CLIENT_SECRET.
            user_agent: User-Agent identifier string for Reddit API requests.
            max_retries: Maximum number of retries on network failures.
            retry_delay: Initial delay in seconds for exponential backoff.
        """
        self.client_id: str = str(client_id or REDDIT_CLIENT_ID or "").strip()
        self.client_secret: str = str(
            client_secret or REDDIT_CLIENT_SECRET or ""
        ).strip()
        self.user_agent: str = str(
            user_agent or REDDIT_USER_AGENT or "stock-sentiment-predictor/1.0"
        ).strip()
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay

        self.reddit: Any = None
        self._initialize_reddit_client()

    def _initialize_reddit_client(self) -> None:
        """Initializes PRAW Reddit instance if credentials are valid and PRAW is installed."""
        if praw is None:
            logger.warning(
                "PRAW library is not installed. RedditCollector is in no-op mode."
            )
            return

        if (
            not self.client_id
            or self.client_id.startswith("YOUR_")
            or not self.client_secret
            or self.client_secret.startswith("YOUR_")
        ):
            logger.warning(
                "Reddit API credentials are not set or are placeholders. RedditCollector is disabled."
            )
            return

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            # Enable read-only mode explicitly
            self.reddit.read_only = True
            logger.info("PRAW Reddit client initialized successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to initialize PRAW Reddit client: %s. RedditCollector is disabled.",
                exc,
            )
            self.reddit = None

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        query: str,
        limit: int = 100,
        time_filter: str = "month",
    ) -> pd.DataFrame:
        """Searches a specific subreddit for posts matching query.

        Args:
            subreddit: Subreddit name (e.g. 'stocks', 'wallstreetbets').
            query: Keyword search expression.
            limit: Maximum number of submissions to return. Default is 100.
            time_filter: Search time window ('all', 'day', 'hour', 'month', 'week', 'year').

        Returns:
            pd.DataFrame with columns: title, selftext, score, created_utc, num_comments.
        """
        cols = [
            "title",
            "selftext",
            "score",
            "created_utc",
            "num_comments",
            "id",
            "url",
            "subreddit",
        ]

        if self.reddit is None:
            logger.warning(
                "Reddit client is not active. Returning empty DataFrame for r/%s.",
                subreddit,
            )
            return pd.DataFrame(columns=cols)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Searching r/%s for '%s' (limit=%d, filter='%s', attempt %d/%d)...",
                    subreddit,
                    query,
                    limit,
                    time_filter,
                    attempt,
                    self.max_retries,
                )
                sub_instance = self.reddit.subreddit(subreddit)
                posts: list[dict[str, Any]] = []

                for submission in sub_instance.search(
                    query=query,
                    limit=limit,
                    time_filter=time_filter,
                ):
                    posts.append(
                        {
                            "title": getattr(submission, "title", "") or "",
                            "selftext": getattr(submission, "selftext", "")
                            or "",
                            "score": int(getattr(submission, "score", 0)),
                            "created_utc": float(
                                getattr(submission, "created_utc", 0.0)
                            ),
                            "num_comments": int(
                                getattr(submission, "num_comments", 0)
                            ),
                            "id": getattr(submission, "id", "") or "",
                            "url": getattr(submission, "url", "") or "",
                            "subreddit": subreddit,
                        }
                    )

                df = (
                    pd.DataFrame(posts, columns=cols)
                    if posts
                    else pd.DataFrame(columns=cols)
                )
                logger.info(
                    "Retrieved %d posts from r/%s for query '%s'.",
                    len(df),
                    subreddit,
                    query,
                )
                return df
            except Exception as exc:
                logger.warning(
                    "Error searching r/%s on attempt %d/%d: %s",
                    subreddit,
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                backoff_wait = self.retry_delay * (2 ** (attempt - 1))
                time.sleep(backoff_wait)

        logger.error(
            "Failed to fetch posts from r/%s after %d attempts.",
            subreddit,
            self.max_retries,
        )
        return pd.DataFrame(columns=cols)

    def fetch_stock_mentions(
        self,
        ticker: str,
        subreddits: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetches posts mentioning a stock ticker across financial subreddits.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            subreddits: Target subreddits list. Defaults to ['stocks', 'wallstreetbets', 'investing'].

        Returns:
            pd.DataFrame containing aggregated Reddit posts with a 'ticker' column.
        """
        target_subs = (
            subreddits
            if subreddits is not None
            else ["stocks", "wallstreetbets", "investing"]
        )

        clean_ticker = ticker.split(".")[0]
        query = f"${clean_ticker} OR {clean_ticker}"

        collected_dfs: list[pd.DataFrame] = []
        for sub in target_subs:
            df_sub = self.fetch_subreddit_posts(
                subreddit=sub,
                query=query,
                limit=100,
                time_filter="month",
            )
            if df_sub is not None and not df_sub.empty:
                collected_dfs.append(df_sub)

        cols = [
            "title",
            "selftext",
            "score",
            "created_utc",
            "num_comments",
            "id",
            "url",
            "subreddit",
            "ticker",
        ]

        if collected_dfs:
            combined_df = pd.concat(collected_dfs, ignore_index=True)
            if "id" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["id"])
        else:
            combined_df = pd.DataFrame(columns=cols)

        combined_df["ticker"] = ticker
        return combined_df

    def save_reddit_data(
        self, df: pd.DataFrame, output_path: str | Path
    ) -> Path:
        """Saves Reddit DataFrame to a CSV file.

        Args:
            df: DataFrame of Reddit posts.
            output_path: Target CSV file path.

        Returns:
            Path object pointing to the written CSV file.
        """
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(
            "Saved %d Reddit posts -> %s",
            len(df),
            file_path,
        )
        return file_path


# ==============================================================================
# 4. Master Orchestration Convenience Function
# ==============================================================================
def collect_all_data(
    tickers: list[str] | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Orchestrates comprehensive data collection across stock prices, news, and Reddit.

    Fetches OHLCV historical market data, company news articles, and Reddit
    community discussions for all specified tickers, then persists each dataset
    as CSV files under the raw data directory hierarchy.

    Args:
        tickers: Optional list of stock ticker symbols. Defaults to config.DEFAULT_TICKERS.
        config: Optional configuration module or container.

    Returns:
        Structured dictionary containing all collected DataFrames and persisted file paths:
            {
                "tickers": list[str],
                "raw_dir": str,
                "stocks": {"data": dict[str, pd.DataFrame], "files": dict[str, Path]},
                "news": {"data": dict[str, pd.DataFrame], "files": dict[str, Path]},
                "reddit": {"data": dict[str, pd.DataFrame], "files": dict[str, Path]},
                "status": str,
            }
    """
    logger.info("==================================================")
    logger.info("STARTING COMPREHENSIVE DATA INGESTION PIPELINE")
    logger.info("==================================================")

    # 1. Resolve configuration parameters
    selected_tickers = (
        list(tickers)
        if tickers is not None
        else list(getattr(config, "DEFAULT_TICKERS", DEFAULT_TICKERS))
    )
    start_date = str(getattr(config, "START_DATE", START_DATE))
    end_date = str(getattr(config, "END_DATE", END_DATE))
    news_key = str(getattr(config, "NEWS_API_KEY", NEWS_API_KEY))
    reddit_id = str(getattr(config, "REDDIT_CLIENT_ID", REDDIT_CLIENT_ID))
    reddit_secret = str(
        getattr(config, "REDDIT_CLIENT_SECRET", REDDIT_CLIENT_SECRET)
    )
    reddit_agent = str(getattr(config, "REDDIT_USER_AGENT", REDDIT_USER_AGENT))

    raw_dir_val = getattr(
        config, "DATA_RAW_DIR", getattr(config, "DATA_RAW", DATA_RAW_DIR)
    )
    raw_dir = Path(raw_dir_val)
    stocks_dir = raw_dir / "stocks"
    news_dir = raw_dir / "news"
    reddit_dir = raw_dir / "reddit"

    for d in [stocks_dir, news_dir, reddit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Collect Stock OHLCV Data
    logger.info("--- Step 1/3: Collecting Stock Market Data ---")
    stock_collector = StockDataCollector(
        tickers=selected_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    stock_data = stock_collector.fetch_all_stocks()
    stock_files = stock_collector.save_stock_data(output_dir=stocks_dir)

    # 3. Collect News Data
    logger.info("--- Step 2/3: Collecting Financial News ---")
    news_collector = NewsCollector(api_key=news_key)
    news_data: dict[str, pd.DataFrame] = {}
    news_files: dict[str, Path] = {}

    for ticker in selected_tickers:
        company = DEFAULT_COMPANY_MAP.get(ticker, ticker.split(".")[0])
        df_news = news_collector.fetch_stock_news(
            ticker=ticker, company_name=company, days_back=30
        )
        news_data[ticker] = df_news

        safe_ticker = ticker.replace(":", "_").replace("^", "")
        csv_file = news_dir / f"{safe_ticker}_news.csv"
        news_collector.save_news_data(df_news, csv_file)
        news_files[ticker] = csv_file

    # 4. Collect Reddit Mentions
    logger.info("--- Step 3/3: Collecting Reddit Social Discussions ---")
    reddit_collector = RedditCollector(
        client_id=reddit_id,
        client_secret=reddit_secret,
        user_agent=reddit_agent,
    )
    reddit_data: dict[str, pd.DataFrame] = {}
    reddit_files: dict[str, Path] = {}

    for ticker in selected_tickers:
        df_reddit = reddit_collector.fetch_stock_mentions(ticker=ticker)
        reddit_data[ticker] = df_reddit

        safe_ticker = ticker.replace(":", "_").replace("^", "")
        csv_file = reddit_dir / f"{safe_ticker}_reddit.csv"
        reddit_collector.save_reddit_data(df_reddit, csv_file)
        reddit_files[ticker] = csv_file

    logger.info("==================================================")
    logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Output Directory: %s", raw_dir)
    logger.info("==================================================")

    return {
        "tickers": selected_tickers,
        "raw_dir": str(raw_dir),
        "stocks": {
            "data": stock_data,
            "files": stock_files,
        },
        "news": {
            "data": news_data,
            "files": news_files,
        },
        "reddit": {
            "data": reddit_data,
            "files": reddit_files,
        },
        "status": "success",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest stock market, news, and Reddit sentiment data."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="List of stock tickers to fetch (e.g. AAPL TSLA).",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="Start date YYYY-MM-DD for stock data.",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="End date YYYY-MM-DD for stock data.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_RAW_DIR),
        help="Target directory for raw data.",
    )
    args = parser.parse_args()

    custom_config = {
        "DEFAULT_TICKERS": args.tickers,
        "START_DATE": args.start_date,
        "END_DATE": args.end_date,
        "DATA_RAW_DIR": Path(args.output_dir),
    }

    results = collect_all_data(tickers=args.tickers, config=custom_config)
    print(f"\nCollection Summary:")
    print(f"  Tickers Processed: {results['tickers']}")
    print(f"  Stock CSV Files: {list(results['stocks']['files'].keys())}")
    print(f"  News CSV Files: {list(results['news']['files'].keys())}")
    print(f"  Reddit CSV Files: {list(results['reddit']['files'].keys())}")

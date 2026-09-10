"""Central configuration module for Stock Price Prediction with Sentiment Analysis.

Defines API credentials, directory paths, default tickers, date range settings,
model hyperparameters, and sentiment analysis configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Attempt to load environment variables from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# Base Project Paths
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORT_DIR: Path = PROJECT_ROOT / "report"
LOGS_DIR: Path = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class PathConfig:
    """Directory paths for data, models, reports, and logs."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    data_raw: Path = DATA_RAW
    data_processed: Path = DATA_PROCESSED
    models_dir: Path = MODELS_DIR
    report_dir: Path = REPORT_DIR
    logs_dir: Path = LOGS_DIR


# ==============================================================================
# API Configuration
# ==============================================================================
@dataclass(frozen=True)
class APIConfig:
    """API credentials for external news and social media data sources.

    Place your actual keys in a .env file at project root or set them as
    environment variables. Defaults below act as placeholders.
    """

    # NewsAPI credentials (https://newsapi.org/)
    # Free tier provides up to 100 requests/day, 1-month historical news.
    news_api_key: str = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")
    )

    # Reddit API credentials via PRAW (https://www.reddit.com/prefs/apps)
    # Required to fetch sentiment discussions from subreddits (e.g., r/stocks, r/wallstreetbets).
    reddit_client_id: str = field(
        default_factory=lambda: os.getenv(
            "REDDIT_CLIENT_ID", "YOUR_REDDIT_CLIENT_ID_HERE"
        )
    )
    reddit_client_secret: str = field(
        default_factory=lambda: os.getenv(
            "REDDIT_CLIENT_SECRET", "YOUR_REDDIT_CLIENT_SECRET_HERE"
        )
    )
    reddit_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "REDDIT_USER_AGENT",
            "stock-sentiment-predictor:v1.0.0 (by /u/your_reddit_username)",
        )
    )


# ==============================================================================
# Data & Ticker Configuration
# ==============================================================================
@dataclass(frozen=True)
class DataConfig:
    """Settings for financial data collection and date ranges."""

    # Default stock tickers list (mix of US tech giants + Indian market leaders)
    default_tickers: list[str] = field(
        default_factory=lambda: [
            "AAPL",  # Apple Inc. (US - NASDAQ)
            "TSLA",  # Tesla Inc. (US - NASDAQ)
            "GOOGL",  # Alphabet Inc. (US - NASDAQ)
            "RELIANCE.NS",  # Reliance Industries Ltd (India - NSE)
            "TCS.NS",  # Tata Consultancy Services Ltd (India - NSE)
            "INFY.NS",  # Infosys Ltd (India - NSE)
        ]
    )

    # Date range settings (default: 2 years of historical stock data)
    history_years: int = 2
    history_days: int = 730

    # Data split ratios
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


# ==============================================================================
# Model Hyperparameters
# ==============================================================================
@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for deep learning (LSTM/GRU) and baseline models."""

    sequence_length: int = 30  # Sliding window of 30 trading days
    hidden_size: int = 128  # Number of hidden units per recurrent layer
    num_layers: int = 2  # Number of recurrent layers
    dropout: float = 0.2  # Dropout probability
    learning_rate: float = 0.001  # Adam optimizer learning rate
    batch_size: int = 32  # Mini-batch training size
    epochs: int = 100  # Maximum number of training epochs
    early_stopping_patience: int = 10  # Epochs without validation improvement before stop
    random_seed: int = 42  # Seed for reproducible train/val splits and model init


# ==============================================================================
# Sentiment Analysis Configuration
# ==============================================================================
@dataclass(frozen=True)
class SentimentConfig:
    """Settings for NLP sentiment scoring using FinBERT."""

    # Pretrained financial NLP model from Hugging Face Hub
    finbert_model: str = "ProsusAI/finbert"

    # Aggregation window for daily sentiment score alignment (e.g., '1D' for daily)
    sentiment_aggregation_window: str = "1D"

    # Max token length for FinBERT tokenizer
    max_token_length: int = 512

    # Inference batch size for transformer sentiment scoring
    batch_size: int = 16


# ==============================================================================
# Central Unified Config
# ==============================================================================
@dataclass(frozen=True)
class Config:
    """Master configuration container aggregating all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)


# Singleton instance for structured config access: `from config import config`
config: Config = Config()

# ==============================================================================
# Top-Level Constants for Direct Imports
# ==============================================================================
# Directory Paths
DATA_RAW = config.paths.data_raw
DATA_RAW_DIR = DATA_RAW
DATA_PROCESSED = config.paths.data_processed
MODELS_DIR = config.paths.models_dir
REPORT_DIR = config.paths.report_dir
LOGS_DIR = config.paths.logs_dir

# API Placeholders / Keys
NEWS_API_KEY = config.api.news_api_key
REDDIT_CLIENT_ID = config.api.reddit_client_id
REDDIT_CLIENT_SECRET = config.api.reddit_client_secret
REDDIT_USER_AGENT = config.api.reddit_user_agent

# Tickers & Date Range
DEFAULT_TICKERS = config.data.default_tickers
HISTORY_YEARS = config.data.history_years
HISTORY_DAYS = config.data.history_days
START_DATE = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Model Hyperparameters
SEQUENCE_LENGTH = config.model.sequence_length
HIDDEN_SIZE = config.model.hidden_size
NUM_LAYERS = config.model.num_layers
DROPOUT = config.model.dropout
LEARNING_RATE = config.model.learning_rate
BATCH_SIZE = config.model.batch_size
EPOCHS = config.model.epochs
EARLY_STOPPING_PATIENCE = config.model.early_stopping_patience
RANDOM_SEED = config.model.random_seed

# Sentiment Settings
FINBERT_MODEL = config.sentiment.finbert_model
SENTIMENT_AGGREGATION_WINDOW = config.sentiment.sentiment_aggregation_window

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATA_RAW",
    "DATA_RAW_DIR",
    "DATA_PROCESSED",
    "MODELS_DIR",
    "REPORT_DIR",
    "LOGS_DIR",
    "NEWS_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "DEFAULT_TICKERS",
    "HISTORY_YEARS",
    "HISTORY_DAYS",
    "START_DATE",
    "END_DATE",
    "SEQUENCE_LENGTH",
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "DROPOUT",
    "LEARNING_RATE",
    "BATCH_SIZE",
    "EPOCHS",
    "EARLY_STOPPING_PATIENCE",
    "RANDOM_SEED",
    "FINBERT_MODEL",
    "SENTIMENT_AGGREGATION_WINDOW",
    "PathConfig",
    "APIConfig",
    "DataConfig",
    "ModelConfig",
    "SentimentConfig",
    "Config",
    "config",
]

"""FinBERT Sentiment Analysis module for financial news and social text.

Provides the SentimentAnalyzer class powered by ProsusAI/finbert to extract
sentiment polarity and class probabilities, alongside daily aggregation and
rolling sentiment calculation utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Safe / optional imports for environments where packages are being installed
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

# Import configuration
try:
    from config import FINBERT_MODEL, config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from config import FINBERT_MODEL, config
    except ImportError:
        FINBERT_MODEL = "ProsusAI/finbert"
        config = None

from src.utils import setup_logger

logger = setup_logger(__name__)


class SentimentAnalyzer:
    """Financial sentiment analyzer using pretrained FinBERT from HuggingFace.

    Auto-detects hardware acceleration (CUDA, MPS, or CPU) and performs batched
    inference with torch.no_grad() for efficient sentiment scoring.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the FinBERT model and tokenizer.

        Args:
            model_name: HuggingFace model hub identifier. Defaults to 'ProsusAI/finbert'.
            device: Computing device ('cuda', 'mps', 'cpu'). If None, auto-detected.

        Raises:
            ImportError: If torch or transformers are not installed.
        """
        self.model_name = model_name

        # Auto-detect device
        if device is not None:
            self.device = str(device).lower()
        elif torch is not None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        logger.info(
            "Initializing SentimentAnalyzer with model '%s' on device '%s'",
            self.model_name,
            self.device,
        )

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            raise ImportError(
                "PyTorch and transformers are required to use SentimentAnalyzer. "
                "Install them via 'pip install torch transformers'."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Parse id2label mapping from model config
        raw_id2label = getattr(
            self.model.config,
            "id2label",
            {0: "positive", 1: "negative", 2: "neutral"},
        )
        self.id2label: dict[int, str] = {
            int(k): str(v).lower() for k, v in raw_id2label.items()
        }
        self.label2id: dict[str, int] = {v: k for k, v in self.id2label.items()}

        # Identify positive, negative, and neutral class indices
        self.pos_idx = next((k for k, v in self.id2label.items() if "pos" in v), 0)
        self.neg_idx = next((k for k, v in self.id2label.items() if "neg" in v), 1)
        self.neu_idx = next((k for k, v in self.id2label.items() if "neu" in v), 2)

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze sentiment of a single financial text headline or sentence.

        Handles empty, whitespace-only, or None inputs gracefully by returning
        a neutral sentiment score.

        Args:
            text: Text string to analyze.

        Returns:
            dict: Sentiment dictionary containing:
                - label (str): 'positive', 'negative', or 'neutral'
                - score (float): Sentiment polarity score in [-1.0, 1.0]
                - probabilities (dict): Class probabilities for positive, negative, neutral
        """
        if text is None or not isinstance(text, str) or not text.strip():
            logger.debug("Empty text input provided; returning neutral default.")
            return {
                "label": "neutral",
                "score": 0.0,
                "probabilities": {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                },
            }

        return self.analyze_batch([text], batch_size=1)[0]

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Perform batched sentiment inference using FinBERT.

        Empty or invalid strings within the batch are handled gracefully with
        neutral fallback scores while preserving input list ordering.

        Args:
            texts: List of text strings to analyze.
            batch_size: Number of texts per inference mini-batch. Defaults to 16.

        Returns:
            list[dict]: List of sentiment dictionaries matching analyze_text format.
        """
        if not texts:
            return []

        results: list[dict[str, Any]] = [None] * len(texts)  # type: ignore[list-item]
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for idx, text in enumerate(texts):
            if text is None or not isinstance(text, str) or not text.strip():
                results[idx] = {
                    "label": "neutral",
                    "score": 0.0,
                    "probabilities": {
                        "positive": 0.0,
                        "negative": 0.0,
                        "neutral": 1.0,
                    },
                }
            else:
                valid_indices.append(idx)
                valid_texts.append(text.strip())

        if not valid_texts:
            return results

        max_len = 512
        if config and hasattr(config, "sentiment") and hasattr(config.sentiment, "max_token_length"):
            max_len = config.sentiment.max_token_length

        for start in range(0, len(valid_texts), batch_size):
            chunk_texts = valid_texts[start : start + batch_size]
            chunk_indices = valid_indices[start : start + batch_size]

            inputs = self.tokenizer(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs_tensor = F.softmax(outputs.logits, dim=-1)
                probs_np = probs_tensor.detach().cpu().numpy()
                pred_indices = np.argmax(probs_np, axis=-1)

            for i, orig_idx in enumerate(chunk_indices):
                probs_row = probs_np[i]
                pred_id = int(pred_indices[i])
                label = self.id2label.get(pred_id, "neutral")

                pos_prob = float(probs_row[self.pos_idx])
                neg_prob = float(probs_row[self.neg_idx])
                neu_prob = float(probs_row[self.neu_idx])

                # Polarity score: positive prob minus negative prob in [-1.0, 1.0]
                score = float(pos_prob - neg_prob)

                results[orig_idx] = {
                    "label": label,
                    "score": round(score, 4),
                    "probabilities": {
                        "positive": round(pos_prob, 4),
                        "negative": round(neg_prob, 4),
                        "neutral": round(neu_prob, 4),
                    },
                }

        return results

    def analyze_dataframe(
        self,
        df: Any,
        text_column: str = "title",
        date_column: str = "date",
        batch_size: int = 16,
    ) -> Any:
        """Add sentiment analysis columns to an existing pandas DataFrame.

        Appends columns:
            - 'sentiment_score': Polarity score in [-1.0, 1.0]
            - 'sentiment_label': Class label ('positive', 'negative', 'neutral')
            - 'positive_prob': Probability of positive sentiment
            - 'negative_prob': Probability of negative sentiment
            - 'neutral_prob': Probability of neutral sentiment
            - 'sentiment': Alias for sentiment_score for backwards compatibility

        Args:
            df: Input pandas DataFrame.
            text_column: Column name containing text to analyze. Defaults to 'title'.
            date_column: Column name containing publication date. Defaults to 'date'.
            batch_size: Batch size for model inference. Defaults to 16.

        Returns:
            pd.DataFrame: Copy of input DataFrame with sentiment columns added.

        Raises:
            ImportError: If pandas is not available.
            ValueError: If df is None.
            KeyError: If text_column is not found in df.
        """
        if pd is None:
            raise ImportError(
                "pandas is required for analyze_dataframe. Install it via 'pip install pandas'."
            )
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        df_out = df.copy()

        sentiment_cols = [
            "sentiment_score",
            "sentiment_label",
            "positive_prob",
            "negative_prob",
            "neutral_prob",
            "sentiment",
        ]

        if df_out.empty:
            for col in sentiment_cols:
                dtype = str if col == "sentiment_label" else float
                df_out[col] = pd.Series([], dtype=dtype)
            return df_out

        if text_column not in df_out.columns:
            raise KeyError(
                f"Text column '{text_column}' not found in DataFrame. "
                f"Available columns: {list(df_out.columns)}"
            )

        texts = df_out[text_column].fillna("").astype(str).tolist()
        batch_results = self.analyze_batch(texts, batch_size=batch_size)

        scores = [res["score"] for res in batch_results]
        labels = [res["label"] for res in batch_results]
        pos_probs = [res["probabilities"]["positive"] for res in batch_results]
        neg_probs = [res["probabilities"]["negative"] for res in batch_results]
        neu_probs = [res["probabilities"]["neutral"] for res in batch_results]

        df_out["sentiment_score"] = scores
        df_out["sentiment_label"] = labels
        df_out["positive_prob"] = pos_probs
        df_out["negative_prob"] = neg_probs
        df_out["neutral_prob"] = neu_probs
        df_out["sentiment"] = scores  # Alias for downstream compatibility

        # Format date column if present
        if date_column in df_out.columns:
            try:
                df_out[date_column] = pd.to_datetime(df_out[date_column])
            except Exception as e:
                logger.debug("Could not parse date column '%s' to datetime: %s", date_column, e)

        return df_out


def aggregate_daily_sentiment(
    df: Any,
    date_column: str = "date",
) -> Any:
    """Group sentiment data by date and calculate daily summary statistics.

    Computes mean and median sentiment score, count of positive, negative, and
    neutral articles, and total articles per date. Determines the overall daily
    sentiment label:
      - 'positive' if mean sentiment > 0.05
      - 'negative' if mean sentiment < -0.05
      - 'neutral'  otherwise

    Args:
        df: Input DataFrame containing article sentiment scores and labels.
        date_column: Column name containing publication/news dates.

    Returns:
        pd.DataFrame: Daily aggregated sentiment metrics with columns:
            [date, mean_sentiment, median_sentiment, positive_count,
             negative_count, neutral_count, total_articles, sentiment_label]

    Raises:
        ImportError: If pandas is not available.
        ValueError: If df is None or lacks a sentiment score column.
        KeyError: If date_column is not found in df.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for aggregate_daily_sentiment. Install it via 'pip install pandas'."
        )
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    required_columns = [
        "date",
        "mean_sentiment",
        "median_sentiment",
        "positive_count",
        "negative_count",
        "neutral_count",
        "total_articles",
        "sentiment_label",
    ]

    if df.empty:
        return pd.DataFrame(columns=required_columns)

    if date_column not in df.columns:
        raise KeyError(
            f"Date column '{date_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Locate sentiment score column
    score_col: Optional[str] = None
    for candidate in ("sentiment_score", "score", "sentiment"):
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError(
            "DataFrame must contain a sentiment score column "
            "('sentiment_score', 'score', or 'sentiment')."
        )

    # Locate sentiment label column
    label_col: Optional[str] = None
    for candidate in ("sentiment_label", "label"):
        if candidate in df.columns:
            label_col = candidate
            break

    df_work = df.copy()

    # Normalize dates to YYYY-MM-DD strings for clean daily grouping
    try:
        parsed_dates = pd.to_datetime(df_work[date_column])
        if hasattr(parsed_dates.dt, "tz") and parsed_dates.dt.tz is not None:
            parsed_dates = parsed_dates.dt.tz_convert(None)
        df_work["_agg_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    except Exception:
        df_work["_agg_date"] = df_work[date_column].astype(str)

    rows: list[dict[str, Any]] = []

    for dt_val, group in df_work.groupby("_agg_date", sort=True):
        scores = pd.to_numeric(group[score_col], errors="coerce").dropna()
        if len(scores) > 0:
            mean_val = float(scores.mean())
            median_val = float(scores.median())
        else:
            mean_val = 0.0
            median_val = 0.0

        total_count = len(group)

        if label_col and label_col in group.columns:
            labels = group[label_col].astype(str).str.lower()
            pos_cnt = int((labels == "positive").sum())
            neg_cnt = int((labels == "negative").sum())
            neu_cnt = int((labels == "neutral").sum())
        else:
            pos_cnt = int((scores > 0.05).sum())
            neg_cnt = int((scores < -0.05).sum())
            neu_cnt = total_count - pos_cnt - neg_cnt

        # Threshold classification per specifications:
        # positive if mean > 0.05, negative if mean < -0.05, else neutral
        if mean_val > 0.05:
            daily_label = "positive"
        elif mean_val < -0.05:
            daily_label = "negative"
        else:
            daily_label = "neutral"

        rows.append(
            {
                "date": dt_val,
                "mean_sentiment": round(mean_val, 4),
                "median_sentiment": round(median_val, 4),
                "positive_count": pos_cnt,
                "negative_count": neg_cnt,
                "neutral_count": neu_cnt,
                "total_articles": total_count,
                "sentiment_label": daily_label,
            }
        )

    result_df = pd.DataFrame(rows, columns=required_columns)
    return result_df


def compute_rolling_sentiment(
    df: Any,
    windows: list[int] = [3, 7, 14],
    sentiment_column: Optional[str] = None,
    min_periods: int = 1,
) -> Any:
    """Add rolling mean sentiment columns for each specified window size.

    Args:
        df: Input DataFrame containing sentiment score data.
        windows: List of window sizes (in days/periods) for rolling mean calculation.
        sentiment_column: Specific column to compute rolling mean over. If None,
            automatically detects 'mean_sentiment', 'sentiment_score', 'sentiment', or 'score'.
        min_periods: Minimum observations in window required to produce a value.
            Defaults to 1 so initial rows are not lost as NaNs.

    Returns:
        pd.DataFrame: DataFrame copy with new columns 'sentiment_rolling_{w}' added.

    Raises:
        ImportError: If pandas is not available.
        ValueError: If df is None or no valid sentiment column is found.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for compute_rolling_sentiment. Install it via 'pip install pandas'."
        )
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    df_out = df.copy()

    if df_out.empty:
        for w in windows:
            df_out[f"sentiment_rolling_{w}"] = pd.Series([], dtype=float)
        return df_out

    # Locate target sentiment column
    target_col = sentiment_column
    if target_col is not None:
        if target_col not in df_out.columns:
            raise KeyError(
                f"Specified sentiment_column '{target_col}' not found in DataFrame. "
                f"Available columns: {list(df_out.columns)}"
            )
    else:
        for candidate in ("mean_sentiment", "sentiment_score", "sentiment", "score"):
            if candidate in df_out.columns:
                target_col = candidate
                break

        if target_col is None:
            raise ValueError(
                "No sentiment column found. Available columns: "
                f"{list(df_out.columns)}"
            )

    # Sort chronologically by date if a date column is present
    date_col = "date" if "date" in df_out.columns else ("Date" if "Date" in df_out.columns else None)
    if date_col:
        try:
            df_out = df_out.sort_values(by=date_col).reset_index(drop=True)
        except Exception as e:
            logger.debug("Could not sort by '%s': %s", date_col, e)

    # Add rolling mean columns for each window size
    for w in windows:
        col_name = f"sentiment_rolling_{w}"
        df_out[col_name] = (
            df_out[target_col]
            .rolling(window=w, min_periods=min_periods)
            .mean()
            .round(4)
        )

    return df_out


__all__ = [
    "SentimentAnalyzer",
    "aggregate_daily_sentiment",
    "compute_rolling_sentiment",
]

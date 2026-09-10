"""Feature engineering module for Stock Price Prediction with Sentiment Analysis.

Transforms raw historical OHLCV market data and sentiment scores into model-ready
features for both deep learning models (LSTM, GRU, Transformers) and gradient-boosted
tree baselines (XGBoost, LightGBM).

Includes:
1. Technical indicator calculation via `ta` (RSI, MACD, Bollinger Bands, SMA, EMA, ATR, OBV, Stochastic)
2. Lagged price returns, percentage changes, and rolling statistics
3. Forward-looking target generation (target price, directional movement, future returns)
4. Temporal alignment and merging with news/social sentiment signals
5. Sequence window generation and time-based train/test dataset preparation
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Tuple, Union

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils import setup_logger

    logger = setup_logger("stock_sentiment.feature_engineer")
except ImportError:
    logger = logging.getLogger("stock_sentiment.feature_engineer")
    if not logger.handlers:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Configuration Import
# ------------------------------------------------------------------------------
try:
    from config import SEQUENCE_LENGTH, config  # type: ignore
except ImportError:
    try:
        import config  # type: ignore

        SEQUENCE_LENGTH = getattr(
            getattr(config, "model", None), "sequence_length", 30
        )
    except ImportError:
        config = None
        SEQUENCE_LENGTH = 30

# ------------------------------------------------------------------------------
# Optional 3rd-Party Imports with Graceful Fallbacks
# ------------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore

try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore
except ImportError:
    # Lightweight pure-Python / numpy fallback if scikit-learn is not installed
    class MinMaxScaler:  # type: ignore
        """Fallback MinMaxScaler implementing scikit-learn compatible interface."""

        def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
            self.feature_range = feature_range
            self.data_min_ = None
            self.data_max_ = None
            self.data_range_ = None
            self.scale_ = None
            self.min_ = None
            self.n_features_in_ = None

        def fit(self, X: Any, y: Any = None) -> "MinMaxScaler":
            arr = np.asarray(X, dtype=float)
            self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
            self.data_min_ = np.nanmin(arr, axis=0)
            self.data_max_ = np.nanmax(arr, axis=0)
            self.data_range_ = self.data_max_ - self.data_min_
            safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range
            self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
            return self

        def transform(self, X: Any) -> Any:
            arr = np.asarray(X, dtype=float)
            if self.scale_ is None or self.min_ is None:
                raise RuntimeError("MinMaxScaler instance is not fitted yet.")
            return arr * self.scale_ + self.min_

        def fit_transform(self, X: Any, y: Any = None) -> Any:
            return self.fit(X, y).transform(X)

        def inverse_transform(self, X: Any) -> Any:
            arr = np.asarray(X, dtype=float)
            if self.scale_ is None or self.min_ is None:
                raise RuntimeError("MinMaxScaler instance is not fitted yet.")
            return (arr - self.min_) / self.scale_


try:
    from ta.momentum import RSIIndicator, StochasticOscillator  # type: ignore
    from ta.trend import EMAIndicator, MACD, SMAIndicator  # type: ignore
    from ta.volatility import AverageTrueRange, BollingerBands  # type: ignore
    from ta.volume import OnBalanceVolumeIndicator  # type: ignore

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


# ==============================================================================
# Feature Engineering Configuration Dataclass
# ==============================================================================
@dataclass
class FeatureConfig:
    """Configuration container for feature engineering pipelines."""

    target_column: str = "Close"
    forecast_horizon: int = 1
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    sequence_length: int = SEQUENCE_LENGTH
    test_size: float = 0.2
    drop_na: bool = True


# ==============================================================================
# Helper Functions
# ==============================================================================
def _check_dependencies() -> None:
    """Verify that essential libraries (pandas, numpy) are available."""
    if pd is None:
        raise ImportError(
            "pandas is required for feature engineering. Please install it using `pip install pandas`."
        )
    if np is None:
        raise ImportError(
            "numpy is required for feature engineering. Please install it using `pip install numpy`."
        )


def _resolve_column(df: pd.DataFrame, target: str) -> str:
    """Resolve a case-insensitive column name in DataFrame.

    Args:
        df: Input DataFrame.
        target: Desired column name (case-insensitive).

    Returns:
        str: Actual matching column name found in DataFrame.

    Raises:
        KeyError: If no matching column is found.
    """
    col_map = {str(col).lower(): str(col) for col in df.columns}
    target_lower = target.lower()
    if target_lower in col_map:
        return col_map[target_lower]
    raise KeyError(
        f"Column '{target}' not found in DataFrame. Available columns: {list(df.columns)}"
    )


# ==============================================================================
# 1. Technical Indicators
# ==============================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach technical indicators to stock price data.

    Calculates indicators using the `ta` library (with high-performance
    vectorized pandas/numpy fallbacks if `ta` is not installed):
    - RSI (14-period Relative Strength Index)
    - MACD (12, 26, 9) — `macd`, `macd_signal`, `macd_histogram`
    - Bollinger Bands (20-period, 2 std) — `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`
    - Simple Moving Averages — `SMA_10`, `SMA_20`, `SMA_50`
    - Exponential Moving Averages — `EMA_10`, `EMA_20`
    - ATR (14-period Average True Range)
    - OBV (On-Balance Volume)
    - Stochastic Oscillator (14-period %K and 3-period %D) — `stoch`, `stoch_signal`

    Args:
        df: Input DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Returns:
        pd.DataFrame: A copy of the DataFrame with all new technical indicator columns.

    Raises:
        ValueError: If any required OHLCV column is missing.
    """
    _check_dependencies()
    df_out = df.copy()

    # Identify OHLCV column names safely (case-insensitive matching)
    open_col = _resolve_column(df_out, "Open")
    high_col = _resolve_column(df_out, "High")
    low_col = _resolve_column(df_out, "Low")
    close_col = _resolve_column(df_out, "Close")
    volume_col = _resolve_column(df_out, "Volume")

    close_series = df_out[close_col].astype(float)
    high_series = df_out[high_col].astype(float)
    low_series = df_out[low_col].astype(float)
    volume_series = df_out[volume_col].astype(float)

    if TA_AVAILABLE:
        logger.debug("Calculating technical indicators using `ta` library...")
        # 1. RSI (14-period)
        rsi_ind = RSIIndicator(close=close_series, window=14)
        df_out["RSI"] = rsi_ind.rsi()

        # 2. MACD (12, 26, 9)
        macd_ind = MACD(
            close=close_series,
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )
        df_out["macd"] = macd_ind.macd()
        df_out["macd_signal"] = macd_ind.macd_signal()
        df_out["macd_histogram"] = macd_ind.macd_diff()

        # 3. Bollinger Bands (20-period)
        bb_ind = BollingerBands(close=close_series, window=20, window_dev=2)
        df_out["bb_upper"] = bb_ind.bollinger_hband()
        df_out["bb_middle"] = bb_ind.bollinger_mavg()
        df_out["bb_lower"] = bb_ind.bollinger_lband()
        df_out["bb_width"] = bb_ind.bollinger_wband()

        # 4. SMA_10, SMA_20, SMA_50
        df_out["SMA_10"] = SMAIndicator(close=close_series, window=10).sma_indicator()
        df_out["SMA_20"] = SMAIndicator(close=close_series, window=20).sma_indicator()
        df_out["SMA_50"] = SMAIndicator(close=close_series, window=50).sma_indicator()

        # 5. EMA_10, EMA_20
        df_out["EMA_10"] = EMAIndicator(close=close_series, window=10).ema_indicator()
        df_out["EMA_20"] = EMAIndicator(close=close_series, window=20).ema_indicator()

        # 6. ATR (14-period)
        atr_ind = AverageTrueRange(
            high=high_series, low=low_series, close=close_series, window=14
        )
        df_out["ATR"] = atr_ind.average_true_range()

        # 7. OBV (On-Balance Volume)
        obv_ind = OnBalanceVolumeIndicator(close=close_series, volume=volume_series)
        df_out["OBV"] = obv_ind.on_balance_volume()

        # 8. Stochastic Oscillator (14-period)
        stoch_ind = StochasticOscillator(
            high=high_series,
            low=low_series,
            close=close_series,
            window=14,
            smooth_window=3,
        )
        df_out["stoch"] = stoch_ind.stoch()
        df_out["stoch_signal"] = stoch_ind.stoch_signal()

    else:
        logger.warning(
            "`ta` library is not installed; falling back to vectorized numpy/pandas indicator implementations."
        )
        # 1. RSI (14-period)
        delta = close_series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df_out["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # 2. MACD (12, 26, 9)
        ema_fast = close_series.ewm(span=12, adjust=False).mean()
        ema_slow = close_series.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df_out["macd"] = macd
        df_out["macd_signal"] = macd_signal
        df_out["macd_histogram"] = macd - macd_signal

        # 3. Bollinger Bands (20-period)
        bb_middle = close_series.rolling(window=20).mean()
        bb_std = close_series.rolling(window=20).std(ddof=0)
        df_out["bb_upper"] = bb_middle + (2.0 * bb_std)
        df_out["bb_middle"] = bb_middle
        df_out["bb_lower"] = bb_middle - (2.0 * bb_std)
        df_out["bb_width"] = (
            (df_out["bb_upper"] - df_out["bb_lower"]) / bb_middle.replace(0.0, np.nan)
        ) * 100.0

        # 4. SMA_10, SMA_20, SMA_50
        df_out["SMA_10"] = close_series.rolling(window=10).mean()
        df_out["SMA_20"] = close_series.rolling(window=20).mean()
        df_out["SMA_50"] = close_series.rolling(window=50).mean()

        # 5. EMA_10, EMA_20
        df_out["EMA_10"] = close_series.ewm(span=10, adjust=False).mean()
        df_out["EMA_20"] = close_series.ewm(span=20, adjust=False).mean()

        # 6. ATR (14-period)
        prev_close = close_series.shift(1)
        tr1 = high_series - low_series
        tr2 = (high_series - prev_close).abs()
        tr3 = (low_series - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_out["ATR"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

        # 7. OBV (On-Balance Volume)
        obv_direction = np.sign(close_series.diff()).fillna(0.0)
        df_out["OBV"] = (obv_direction * volume_series).cumsum()

        # 8. Stochastic Oscillator (14-period)
        lowest_low = low_series.rolling(window=14).min()
        highest_high = high_series.rolling(window=14).max()
        hl_diff = (highest_high - lowest_low).replace(0.0, np.nan)
        stoch_k = 100.0 * ((close_series - lowest_low) / hl_diff)
        df_out["stoch"] = stoch_k
        df_out["stoch_signal"] = stoch_k.rolling(window=3).mean()

    # Helpful aliases for dashboard compatibility and naming conventions
    df_out["rsi"] = df_out["RSI"]
    df_out["atr"] = df_out["ATR"]
    df_out["obv"] = df_out["OBV"]
    df_out["stochastic_oscillator"] = df_out["stoch"]
    df_out["MACD"] = df_out["macd"]
    df_out["Signal_Line"] = df_out["macd_signal"]
    df_out["MACD_Hist"] = df_out["macd_histogram"]
    df_out["BB_Upper"] = df_out["bb_upper"]
    df_out["BB_Middle"] = df_out["bb_middle"]
    df_out["BB_Lower"] = df_out["bb_lower"]
    df_out["BB_Width"] = df_out["bb_width"]

    logger.info(
        "Added technical indicators (RSI, MACD, BBands, SMAs, EMAs, ATR, OBV, Stochastic) to DataFrame."
    )
    return df_out


# ==============================================================================
# 2. Lag Features
# ==============================================================================
def add_lag_features(
    df: pd.DataFrame,
    column: str = "Close",
    lags: list[int] = [1, 2, 3, 5, 7],
) -> pd.DataFrame:
    """Add lagged values, percentage changes, and rolling statistics.

    Computes:
    - Lagged values for each lag in `lags`: `{column}_lag_{lag}`
    - Percentage change for each lag in `lags`: `{column}_pct_change_{lag}`
    - Rolling mean and rolling standard deviation for windows [5, 10, 20]:
      `{column}_rolling_mean_{window}`, `{column}_rolling_std_{window}`

    Args:
        df: Input DataFrame containing the target column.
        column: The column to calculate lag and rolling features on (default: 'Close').
        lags: List of integer lag periods (default: [1, 2, 3, 5, 7]).

    Returns:
        pd.DataFrame: DataFrame with all new lag and rolling features.

    Raises:
        KeyError: If `column` is not present in DataFrame.
    """
    _check_dependencies()
    df_out = df.copy()
    actual_col = _resolve_column(df_out, column)
    series = df_out[actual_col].astype(float)

    # 1. Lagged values and percentage changes for each lag
    for lag in lags:
        df_out[f"{actual_col}_lag_{lag}"] = series.shift(lag)
        pct_change = series.pct_change(periods=lag)
        df_out[f"{actual_col}_pct_change_{lag}"] = pct_change
        # Alias for alternative naming convention
        df_out[f"{actual_col}_pct_change_lag_{lag}"] = pct_change

    # 2. Rolling mean and std for windows [5, 10, 20]
    rolling_windows = [5, 10, 20]
    for window in rolling_windows:
        df_out[f"{actual_col}_rolling_mean_{window}"] = series.rolling(
            window=window
        ).mean()
        df_out[f"{actual_col}_rolling_std_{window}"] = series.rolling(
            window=window
        ).std()

    logger.info(
        f"Added lag features for '{actual_col}': lags={lags}, rolling_windows={rolling_windows}."
    )
    return df_out


# ==============================================================================
# 3. Target Variables
# ==============================================================================
def add_target_variable(
    df: pd.DataFrame,
    column: str = "Close",
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    """Add target variables for supervised forecasting and classification.

    Generates forward-shifted labels:
    - `target_price`: Next day's (or horizon's) closing price (shifted by -forecast_horizon)
    - `target_direction`: 1 if next day's close > today's close, else 0
    - `target_pct_change`: Percentage change from today's close to next day's close

    Args:
        df: Input DataFrame containing the price column.
        column: Price column to calculate targets on (default: 'Close').
        forecast_horizon: Shift steps forward (default: 1 for next-day forecasting).

    Returns:
        pd.DataFrame: DataFrame with target columns added. Note that the last
                      `forecast_horizon` rows will contain NaNs for target variables.

    Raises:
        KeyError: If `column` is not found in DataFrame.
    """
    _check_dependencies()
    df_out = df.copy()
    actual_col = _resolve_column(df_out, column)
    current_price = df_out[actual_col].astype(float)

    # 1. Shifted target price
    target_price = current_price.shift(-forecast_horizon)
    df_out["target_price"] = target_price

    # 2. Direction: 1 if next day's close > today's close, else 0
    # On tail rows where target_price is NaN, target_direction is also NaN
    is_up = (target_price > current_price).astype(int)
    df_out["target_direction"] = is_up

    # 3. Target percentage change
    df_out["target_pct_change"] = (target_price - current_price) / current_price

    logger.info(
        f"Added target variables based on '{actual_col}' with forecast_horizon={forecast_horizon}."
    )
    return df_out


# ==============================================================================
# 4. Feature Merging (Stock + Sentiment)
# ==============================================================================
def merge_features(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    on: str = "Date",
) -> pd.DataFrame:
    """Left merge stock features with daily sentiment features on date.

    Forward-fills missing sentiment values to cover weekends, market holidays,
    or days without news/social sentiment signals.

    Args:
        stock_df: DataFrame containing stock market features with date column.
        sentiment_df: DataFrame containing sentiment features with date column.
        on: Date column name to join on (default: 'Date').

    Returns:
        pd.DataFrame: Merged DataFrame with forward-filled sentiment values.

    Raises:
        KeyError: If merge column `on` is missing from either DataFrame.
    """
    _check_dependencies()
    if stock_df.empty:
        logger.warning("stock_df is empty; returning empty merged DataFrame.")
        return stock_df.copy()

    if sentiment_df.empty:
        logger.warning("sentiment_df is empty; returning copy of stock_df.")
        return stock_df.copy()

    stock_col = _resolve_column(stock_df, on)
    sentiment_col = _resolve_column(sentiment_df, on)

    merged_stock = stock_df.copy()
    merged_sentiment = sentiment_df.copy()

    # Standardize date column formats for exact calendar alignment
    try:
        merged_stock[stock_col] = pd.to_datetime(merged_stock[stock_col]).dt.tz_localize(None).dt.normalize()
        merged_sentiment[sentiment_col] = pd.to_datetime(merged_sentiment[sentiment_col]).dt.tz_localize(None).dt.normalize()
    except Exception as exc:
        logger.debug(f"Date conversion skipped during merge_features: {exc}")

    # Determine sentiment feature columns to forward-fill
    sentiment_feature_cols = [
        col for col in merged_sentiment.columns if col != sentiment_col
    ]

    # Perform left merge
    merged = pd.merge(
        merged_stock,
        merged_sentiment,
        left_on=stock_col,
        right_on=sentiment_col,
        how="left",
    )

    # If right column had a different casing/name, clean up redundant column
    if stock_col != sentiment_col and sentiment_col in merged.columns:
        merged.drop(columns=[sentiment_col], inplace=True)

    # Forward-fill missing sentiment scores (e.g., weekends/holidays)
    if sentiment_feature_cols:
        merged[sentiment_feature_cols] = merged[sentiment_feature_cols].ffill()
        # Backward-fill or zero-fill any dates before the first sentiment record
        merged[sentiment_feature_cols] = (
            merged[sentiment_feature_cols].bfill().fillna(0.0)
        )

    logger.info(
        f"Merged stock data ({len(stock_df)} rows) with sentiment data ({len(sentiment_df)} rows) on '{on}'. "
        f"Forward-filled sentiment columns: {sentiment_feature_cols}."
    )
    return merged


# ==============================================================================
# 5. Dataset Preparation for LSTM/Models
# ==============================================================================
def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int = 30,
    test_size: float = 0.2,
    scaler: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, list[str]]:
    """Scale features and construct sequenced (X, y) pairs for LSTM/time-series models.

    Performs:
    1. Automatic NaN handling (drops incomplete rows from feature engineering).
    2. Scaler fitting strictly on the training partition to avoid lookahead bias / data leakage.
    3. Creation of 3D sequence arrays: `(samples, sequence_length, num_features)`.
    4. Chronological, time-based train/test splitting (no random shuffling).

    Args:
        df: Input DataFrame containing engineered features and target.
        feature_columns: List of feature column names to feed into the model.
        target_column: Column name of the forecasting target (e.g. 'target_price', 'target_direction').
        sequence_length: Sliding window length in trading days (default: 30).
        test_size: Ratio of observations allocated to test split (default: 0.2).
        scaler: Pre-instantiated or fitted scaler. If None, a new MinMaxScaler(0, 1) is fit on train.

    Returns:
        tuple containing:
        - X_train: Training sequences of shape `(n_train, sequence_length, num_features)`.
        - X_test: Testing sequences of shape `(n_test, sequence_length, num_features)`.
        - y_train: Training targets of shape `(n_train,)`.
        - y_test: Testing targets of shape `(n_test,)`.
        - scaler: Fitted scaler object.
        - feature_columns: Final list of feature column names used.

    Raises:
        ValueError: If DataFrame has fewer valid rows than sequence_length, or if columns are missing.
    """
    _check_dependencies()

    if not feature_columns:
        raise ValueError("feature_columns list cannot be empty.")

    # Validate column presence
    resolved_features = [_resolve_column(df, col) for col in feature_columns]
    resolved_target = _resolve_column(df, target_column)

    # Drop NaNs across required feature and target columns
    required_cols = list(resolved_features)
    if resolved_target not in required_cols:
        required_cols.append(resolved_target)

    df_clean = df.dropna(subset=required_cols).copy().reset_index(drop=True)
    n_rows = len(df_clean)

    if n_rows <= sequence_length:
        raise ValueError(
            f"Insufficient observations after dropping NaNs: {n_rows} rows available, "
            f"but sequence_length={sequence_length} requires at least {sequence_length + 1} rows."
        )

    # Determine time-based train/test split boundary on the underlying rows
    train_row_cutoff = int(n_rows * (1.0 - test_size))
    # Guarantee at least one sequence can be formed in train and test
    train_row_cutoff = max(sequence_length + 1, min(train_row_cutoff, n_rows - 1))

    # Initialize and fit scaler ONLY on the training split
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaler.fit(df_clean.iloc[:train_row_cutoff][resolved_features])
    elif not hasattr(scaler, "scale_") and hasattr(scaler, "fit"):
        scaler.fit(df_clean.iloc[:train_row_cutoff][resolved_features])

    # Transform all features using training scaler
    scaled_features = scaler.transform(df_clean[resolved_features])
    target_values = df_clean[resolved_target].values

    # Construct sequence windows (X, y)
    # Each sample X[i] contains sequence_length steps [i - sequence_length : i]
    # The corresponding target is target_values[i]
    X_list: list[np.ndarray] = []
    y_list: list[Any] = []

    for i in range(sequence_length, n_rows):
        X_list.append(scaled_features[i - sequence_length : i])
        y_list.append(target_values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    total_sequences = len(X)
    if total_sequences == 0:
        raise ValueError(
            f"Could not construct any sequences of length {sequence_length} from {n_rows} rows."
        )

    # Chronological split on sequenced dataset
    split_idx = int(total_sequences * (1.0 - test_size))
    split_idx = max(1, min(split_idx, total_sequences - 1))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    logger.info(
        f"Dataset prepared successfully: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}, "
        f"num_features={len(resolved_features)}."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        resolved_features,
    )


# ==============================================================================
# 6. High-Level Pipelines & Utilities
# ==============================================================================
def drop_nan_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows containing NaN values and log statistics.

    Args:
        df: Input DataFrame.
        subset: Optional list of columns to inspect for NaNs. If None, checks all columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaN rows removed.
    """
    _check_dependencies()
    initial_rows = len(df)
    clean_df = df.dropna(subset=subset).reset_index(drop=True)
    dropped = initial_rows - len(clean_df)
    logger.info(
        f"Dropped {dropped} rows with NaN values. Remaining rows: {len(clean_df)} / {initial_rows}."
    )
    return clean_df


def build_feature_pipeline(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    price_column: str = "Close",
    lags: list[int] = [1, 2, 3, 5, 7],
    forecast_horizon: int = 1,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Execute the complete end-to-end feature engineering pipeline.

    Chains:
    1. Technical indicators calculation (`add_technical_indicators`)
    2. Lag features and rolling statistics (`add_lag_features`)
    3. Target variables generation (`add_target_variable`)
    4. Optional sentiment data merging (`merge_features`)
    5. Incomplete row pruning (`drop_nan_rows`)

    Args:
        stock_df: Historical market DataFrame (Date, Open, High, Low, Close, Volume).
        sentiment_df: Optional sentiment scores DataFrame (Date, compound, positive, etc.).
        price_column: Base price column for targets and lags (default: 'Close').
        lags: List of lag intervals (default: [1, 2, 3, 5, 7]).
        forecast_horizon: Target prediction horizon in days (default: 1).
        drop_na: Whether to drop rows with NaN values after feature generation (default: True).

    Returns:
        pd.DataFrame: Fully engineered, model-ready DataFrame.
    """
    logger.info("Executing full feature engineering pipeline...")
    # 1. Technical indicators
    df_feat = add_technical_indicators(stock_df)

    # 2. Lag features
    df_feat = add_lag_features(df_feat, column=price_column, lags=lags)

    # 3. Target variables
    df_feat = add_target_variable(
        df_feat, column=price_column, forecast_horizon=forecast_horizon
    )

    # 4. Merge sentiment if provided
    if sentiment_df is not None and not sentiment_df.empty:
        df_feat = merge_features(df_feat, sentiment_df, on="Date")

    # 5. Drop NaNs
    if drop_na:
        df_feat = drop_nan_rows(df_feat)

    logger.info(
        f"Feature engineering pipeline completed: final shape is {df_feat.shape}."
    )
    return df_feat


__all__ = [
    "FeatureConfig",
    "add_technical_indicators",
    "add_lag_features",
    "add_target_variable",
    "merge_features",
    "prepare_dataset",
    "drop_nan_rows",
    "build_feature_pipeline",
]

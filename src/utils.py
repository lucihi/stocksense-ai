"""Utility functions for Stock Price Prediction with Sentiment Analysis.

Provides logging setup, directory management, DataFrame serialization/deserialization,
date range calculation, and a timing decorator for performance profiling.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

# Lazy/optional import of pandas so utils can be safely imported without crashing
# if dependencies are still being installed
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "stock_sentiment",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[Path, str]] = None,
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Avoids duplicating handlers if called multiple times with the same logger name.

    Args:
        name: Name of the logger (typically __name__ or module identifier).
        level: Logging level (e.g., logging.INFO, logging.DEBUG, or 'INFO').
        log_file: Optional path to a file where logs should also be written.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt=_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    # Avoid adding duplicate console handlers
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_path = Path(log_file)
        ensure_dir(file_path.parent)
        # Check if file handler for this path already exists
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == file_path.resolve()
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = setup_logger(__name__)


def ensure_dir(path: Union[Path, str]) -> Path:
    """Ensure a directory exists; creates it and any parent directories if needed.

    Args:
        path: Directory path as a string or Path object.

    Returns:
        Path: The resolved Path object for the created/existing directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_dataframe(
    df: Any,
    path: Union[Path, str],
    **kwargs: Any,
) -> Path:
    """Save a pandas DataFrame to disk based on its file extension.

    Supports .csv, .parquet, .pkl / .pickle, and .json formats.
    Automatically creates parent directories if they do not exist.

    Args:
        df: pandas DataFrame to save.
        path: Target file path (e.g., 'data/raw/AAPL.csv').
        **kwargs: Additional keyword arguments passed to the underlying pandas save method.

    Returns:
        Path: The Path object where the DataFrame was saved.

    Raises:
        ImportError: If pandas is not installed.
        ValueError: If the file extension is unsupported.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for save_dataframe. Install it via 'pip install pandas'."
        )

    file_path = Path(path)
    ensure_dir(file_path.parent)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        # Default index=True unless overridden
        index = kwargs.pop("index", True)
        df.to_csv(file_path, index=index, **kwargs)
    elif suffix == ".parquet":
        index = kwargs.pop("index", True)
        df.to_parquet(file_path, index=index, **kwargs)
    elif suffix in (".pkl", ".pickle"):
        df.to_pickle(file_path, **kwargs)
    elif suffix == ".json":
        df.to_json(file_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats: .csv, .parquet, .pkl, .pickle, .json"
        )

    logger.info(
        "Saved DataFrame [%d rows x %d cols] -> %s",
        len(df),
        len(df.columns),
        file_path,
    )
    return file_path


def load_dataframe(
    path: Union[Path, str],
    **kwargs: Any,
) -> Any:
    """Load a pandas DataFrame from disk based on its file extension.

    Supports .csv, .parquet, .pkl / .pickle, and .json formats.

    Args:
        path: Path to the data file.
        **kwargs: Additional keyword arguments passed to the underlying pandas read method.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ImportError: If pandas is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for load_dataframe. Install it via 'pip install pandas'."
        )

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataframe file not found at: {file_path.resolve()}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path, **kwargs)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path, **kwargs)
    elif suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(file_path, **kwargs)
    elif suffix == ".json":
        df = pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats: .csv, .parquet, .pkl, .pickle, .json"
        )

    logger.info(
        "Loaded DataFrame [%d rows x %d cols] <- %s",
        len(df),
        len(df.columns),
        file_path,
    )
    return df


def get_date_range(
    days_back: int = 730,
    end_date: Optional[Union[str, datetime, date]] = None,
) -> tuple[str, str]:
    """Calculate start and end date strings formatted as 'YYYY-MM-DD'.

    Args:
        days_back: Number of calendar days to look back from the end date.
        end_date: Ending date as a 'YYYY-MM-DD' string, datetime, or date object.
                  Defaults to the current date if not provided.

    Returns:
        tuple[str, str]: (start_date_str, end_date_str) in 'YYYY-MM-DD' format.
    """
    if end_date is None:
        target_end = date.today()
    elif isinstance(end_date, str):
        # Support both 'YYYY-MM-DD' and ISO format strings
        target_end = datetime.fromisoformat(end_date.split("T")[0]).date()
    elif isinstance(end_date, datetime):
        target_end = end_date.date()
    elif isinstance(end_date, date):
        target_end = end_date
    else:
        raise TypeError(
            f"Expected end_date to be None, str, datetime, or date; got {type(end_date).__name__}"
        )

    target_start = target_end - timedelta(days=days_back)
    return target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d")


def timer(func: F) -> F:
    """Decorator that measures and logs the execution time of a function.

    Uses high-resolution monotonic time and logs at INFO level using the
    decorated function's module logger.

    Args:
        func: The callable to profile.

    Returns:
        Callable: Wrapped function that logs its elapsed runtime.
    """
    func_logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            func_logger.info(
                "Execution of '%s' completed in %.4f seconds",
                func.__qualname__,
                elapsed,
            )

    return wrapper  # type: ignore[return-value]


__all__ = [
    "setup_logger",
    "ensure_dir",
    "save_dataframe",
    "load_dataframe",
    "get_date_range",
    "timer",
]

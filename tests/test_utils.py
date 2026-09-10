"""Unit tests for utility functions in src/utils.py."""

import logging
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.utils as utils_module
from src.utils import ensure_dir, get_date_range, load_dataframe, save_dataframe, setup_logger, timer


class TestUtils(unittest.TestCase):
    """Test suite for utility helper functions."""

    def test_setup_logger(self):
        """Test logger initialization and handler configuration."""
        logger_name = "test_logger_unique"
        logger = setup_logger(logger_name, level=logging.DEBUG)
        self.assertEqual(logger.name, logger_name)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertFalse(logger.propagate)

        # Ensure calling it again doesn't add duplicate stream handlers
        initial_handlers = len(logger.handlers)
        logger_again = setup_logger(logger_name, level=logging.DEBUG)
        self.assertEqual(len(logger_again.handlers), initial_handlers)

    def test_setup_logger_with_file(self):
        """Test logger with file handler."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test.log"
            logger = setup_logger("test_file_logger", log_file=log_file)
            logger.info("Test log entry")
            # Close file handlers so the file isn't locked
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler):
                    h.close()
            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("Test log entry", content)

    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_target = Path(tmp_dir) / "sub" / "folder" / "nested"
            self.assertFalse(test_target.exists())

            result = ensure_dir(test_target)
            self.assertTrue(test_target.exists())
            self.assertTrue(test_target.is_dir())
            self.assertEqual(result, test_target)

    def test_get_date_range_defaults(self):
        """Test get_date_range with explicit and default days."""
        start_str, end_str = get_date_range(days_back=10, end_date="2026-01-15")
        self.assertEqual(end_str, "2026-01-15")
        self.assertEqual(start_str, "2026-01-05")

        # Test date object
        end_d = date(2026, 6, 20)
        start_str2, end_str2 = get_date_range(days_back=20, end_date=end_d)
        self.assertEqual(end_str2, "2026-06-20")
        self.assertEqual(start_str2, "2026-05-31")

    def test_timer_decorator(self):
        """Test that the timer decorator executes wrapped function and preserves return value."""
        call_tracker = []

        @timer
        def sample_function(x: int, y: int) -> int:
            call_tracker.append((x, y))
            time.sleep(0.01)
            return x + y

        result = sample_function(10, 20)
        self.assertEqual(result, 30)
        self.assertEqual(call_tracker, [(10, 20)])
        self.assertEqual(sample_function.__name__, "sample_function")

    def test_save_and_load_dataframe_file_not_found(self):
        """Test load_dataframe raises FileNotFoundError if file does not exist."""
        mock_pd = MagicMock()
        with patch.object(utils_module, "pd", mock_pd):
            with self.assertRaises(FileNotFoundError):
                load_dataframe("/nonexistent/path/data.csv")

    def test_save_dataframe_unsupported_format(self):
        """Test save_dataframe raises ValueError on unsupported extension."""
        mock_pd = MagicMock()
        mock_df = MagicMock()
        with patch.object(utils_module, "pd", mock_pd):
            with self.assertRaises(ValueError):
                save_dataframe(mock_df, "test.docx")

    def test_save_and_load_dataframe_mocked(self):
        """Test save_dataframe and load_dataframe dispatching to proper pandas methods."""
        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_df.columns = ["a", "b"]

        with patch.object(utils_module, "pd", mock_pd):
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "data.csv"
                save_dataframe(mock_df, csv_path)
                mock_df.to_csv.assert_called_once_with(csv_path, index=True)

                # Now test load
                csv_path.touch()
                load_dataframe(csv_path)
                mock_pd.read_csv.assert_called_once_with(csv_path)


if __name__ == "__main__":
    unittest.main()

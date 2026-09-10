"""Unit tests for the central configuration module."""

import unittest
from pathlib import Path

import config


class TestConfig(unittest.TestCase):
    """Test suite for config module structure and default settings."""

    def test_paths_exist_or_valid(self):
        """Test that configured directory paths are Path instances and have expected relative hierarchy."""
        self.assertIsInstance(config.PROJECT_ROOT, Path)
        self.assertIsInstance(config.DATA_RAW, Path)
        self.assertIsInstance(config.DATA_PROCESSED, Path)
        self.assertIsInstance(config.MODELS_DIR, Path)

        self.assertEqual(config.DATA_RAW, config.PROJECT_ROOT / "data" / "raw")
        self.assertEqual(config.DATA_PROCESSED, config.PROJECT_ROOT / "data" / "processed")
        self.assertEqual(config.MODELS_DIR, config.PROJECT_ROOT / "models")

    def test_default_tickers(self):
        """Test default tickers list contains expected US and Indian stock symbols."""
        expected_subset = {"AAPL", "TSLA", "GOOGL", "RELIANCE.NS", "TCS.NS", "INFY.NS"}
        self.assertTrue(expected_subset.issubset(set(config.DEFAULT_TICKERS)))

    def test_model_hyperparameters(self):
        """Test default hyperparameters match project requirements."""
        self.assertEqual(config.SEQUENCE_LENGTH, 30)
        self.assertEqual(config.HIDDEN_SIZE, 128)
        self.assertEqual(config.NUM_LAYERS, 2)
        self.assertEqual(config.DROPOUT, 0.2)
        self.assertEqual(config.LEARNING_RATE, 0.001)
        self.assertEqual(config.BATCH_SIZE, 32)
        self.assertEqual(config.EPOCHS, 100)
        self.assertEqual(config.EARLY_STOPPING_PATIENCE, 10)

    def test_sentiment_config(self):
        """Test sentiment configuration values."""
        self.assertEqual(config.FINBERT_MODEL, "ProsusAI/finbert")
        self.assertEqual(config.SENTIMENT_AGGREGATION_WINDOW, "1D")

    def test_singleton_config(self):
        """Test access via structured config singleton."""
        self.assertEqual(config.config.model.sequence_length, 30)
        self.assertEqual(config.config.sentiment.finbert_model, "ProsusAI/finbert")


if __name__ == "__main__":
    unittest.main()

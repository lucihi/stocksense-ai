"""Unit tests for feature engineering module src/feature_engineer.py."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import src.feature_engineer as fe
from src.feature_engineer import (
    FeatureConfig,
    MinMaxScaler,
    _resolve_column,
    add_lag_features,
    add_target_variable,
    add_technical_indicators,
    build_feature_pipeline,
    drop_nan_rows,
    merge_features,
    prepare_dataset,
)


class TestFeatureEngineer(unittest.TestCase):
    """Test suite for feature engineering functions."""

    def test_feature_config_defaults(self):
        """Test default values of FeatureConfig dataclass."""
        cfg = FeatureConfig()
        self.assertEqual(cfg.target_column, "Close")
        self.assertEqual(cfg.forecast_horizon, 1)
        self.assertEqual(cfg.lags, [1, 2, 3, 5, 7])
        self.assertEqual(cfg.rolling_windows, [5, 10, 20])
        self.assertEqual(cfg.test_size, 0.2)
        self.assertTrue(cfg.drop_na)

    def test_resolve_column(self):
        """Test case-insensitive column resolution."""
        mock_df = MagicMock()
        mock_df.columns = ["date", "OPEN", "High", "low", "Close", "Volume"]

        self.assertEqual(_resolve_column(mock_df, "open"), "OPEN")
        self.assertEqual(_resolve_column(mock_df, "CLOSE"), "Close")
        self.assertEqual(_resolve_column(mock_df, "volume"), "Volume")

        with self.assertRaises(KeyError):
            _resolve_column(mock_df, "NonExistent")

    def test_check_dependencies_raises_when_missing(self):
        """Test that missing pandas or numpy raises ImportError."""
        with patch.object(fe, "pd", None):
            with self.assertRaises(ImportError):
                fe._check_dependencies()

        with patch.object(fe, "np", None):
            with self.assertRaises(ImportError):
                fe._check_dependencies()

    def test_fallback_minmax_scaler(self):
        """Test fallback MinMaxScaler scaling and inverse transform."""
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        scaler.fit(data)

        self.assertEqual(scaler.n_features_in_, 2)
        np.testing.assert_allclose(scaler.data_min_, [10.0, 100.0])
        np.testing.assert_allclose(scaler.data_max_, [30.0, 300.0])

        scaled = scaler.transform(data)
        np.testing.assert_allclose(scaled[0], [0.0, 0.0])
        np.testing.assert_allclose(scaled[1], [0.5, 0.5])
        np.testing.assert_allclose(scaled[2], [1.0, 1.0])

        inversed = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(inversed, data)

    def test_add_lag_features_missing_column(self):
        """Test that add_lag_features raises KeyError when column is missing."""
        mock_df = MagicMock()
        mock_df.columns = ["Open", "Volume"]
        with patch.object(fe, "pd", MagicMock()), patch.object(fe, "np", MagicMock()):
            with self.assertRaises(KeyError):
                add_lag_features(mock_df, column="Close")

    def test_add_target_variable_missing_column(self):
        """Test that add_target_variable raises KeyError when column is missing."""
        mock_df = MagicMock()
        mock_df.columns = ["Open", "Volume"]
        with patch.object(fe, "pd", MagicMock()), patch.object(fe, "np", MagicMock()):
            with self.assertRaises(KeyError):
                add_target_variable(mock_df, column="Close")

    def test_prepare_dataset_empty_features(self):
        """Test that prepare_dataset raises ValueError when feature_columns is empty."""
        mock_df = MagicMock()
        with patch.object(fe, "pd", MagicMock()), patch.object(fe, "np", MagicMock()):
            with self.assertRaises(ValueError):
                prepare_dataset(mock_df, feature_columns=[], target_column="target_price")

    def test_prepare_dataset_insufficient_rows(self):
        """Test prepare_dataset raises ValueError when rows <= sequence_length."""
        mock_df = MagicMock()
        mock_df.columns = ["feat1", "target_price"]
        mock_df.dropna.return_value.reset_index.return_value = ["row1", "row2"]

        with patch.object(fe, "pd", MagicMock()), patch.object(fe, "np", MagicMock()):
            with self.assertRaises(ValueError):
                prepare_dataset(
                    mock_df,
                    feature_columns=["feat1"],
                    target_column="target_price",
                    sequence_length=10,
                )

    def test_merge_features_empty_dfs(self):
        """Test merge_features gracefully returns copy when either df is empty."""
        mock_stock = MagicMock()
        mock_stock.empty = True
        mock_sentiment = MagicMock()
        mock_sentiment.empty = False

        with patch.object(fe, "pd", MagicMock()), patch.object(fe, "np", MagicMock()):
            res = merge_features(mock_stock, mock_sentiment)
            self.assertEqual(res, mock_stock.copy())

            mock_stock.empty = False
            mock_sentiment.empty = True
            res2 = merge_features(mock_stock, mock_sentiment)
            self.assertEqual(res2, mock_stock.copy())


if __name__ == "__main__":
    unittest.main()

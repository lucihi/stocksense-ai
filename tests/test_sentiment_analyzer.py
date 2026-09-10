"""Unit tests for FinBERT sentiment analyzer module."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import src.sentiment_analyzer as sa_module
from src.sentiment_analyzer import (
    SentimentAnalyzer,
    aggregate_daily_sentiment,
    compute_rolling_sentiment,
)


class DummySeries:
    """Lightweight Series simulator for testing when pandas is mocked."""

    def __init__(self, data=None, name=None, dtype=None):
        if data is None:
            self.values = []
        elif isinstance(data, (list, tuple)):
            self.values = list(data)
        elif isinstance(data, np.ndarray):
            self.values = data.tolist()
        elif isinstance(data, DummySeries):
            self.values = list(data.values)
        else:
            self.values = [data]
        self.name = name
        self.dtype = dtype or float

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, DummySeries)):
            mask = idx.values if isinstance(idx, DummySeries) else idx
            return DummySeries([v for v, m in zip(self.values, mask) if m], name=self.name)
        return self.values[idx]

    def __eq__(self, other):
        return DummySeries([v == other for v in self.values], name=self.name)

    def __gt__(self, other):
        return DummySeries([v > other for v in self.values], name=self.name)

    def __lt__(self, other):
        return DummySeries([v < other for v in self.values], name=self.name)

    def sum(self):
        return sum(1 for v in self.values if v)

    def mean(self):
        return float(np.mean(self.values)) if self.values else 0.0

    def median(self):
        return float(np.median(self.values)) if self.values else 0.0

    def dropna(self):
        return DummySeries([v for v in self.values if v is not None and not (isinstance(v, float) and np.isnan(v))])

    def astype(self, t):
        return self

    @property
    def str(self):
        class StrAccessor:
            def __init__(self, s):
                self.s = s

            def lower(self):
                return DummySeries([str(x).lower() for x in self.s.values])

        return StrAccessor(self)

    def tolist(self):
        return list(self.values)

    def fillna(self, val):
        return DummySeries([val if v is None else v for v in self.values])

    def rolling(self, window, min_periods=1):
        s = self

        class DummyRolling:
            def mean(self):
                res = []
                for i in range(len(s.values)):
                    start_i = max(0, i - window + 1)
                    sub = [v for v in s.values[start_i : i + 1] if v is not None]
                    if len(sub) >= min_periods:
                        res.append(float(np.mean(sub)))
                    else:
                        res.append(np.nan)
                return DummySeries(res)

        return DummyRolling()

    def round(self, decimals=4):
        return DummySeries([round(v, decimals) if v is not None else None for v in self.values])


class DummyDataFrame:
    """Lightweight DataFrame simulator for unit testing without third-party dependencies."""

    def __init__(self, data=None, columns=None):
        if data is None:
            self._data = {c: [] for c in (columns or [])}
        elif isinstance(data, dict):
            self._data = {k: list(v) if isinstance(v, (list, tuple)) else (v.tolist() if hasattr(v, "tolist") else [v]) for k, v in data.items()}
        elif isinstance(data, list):
            cols = columns or (list(data[0].keys()) if data else [])
            self._data = {c: [row.get(c) for row in data] for c in cols}
        else:
            self._data = {}
        self._columns = columns or list(self._data.keys())

    @property
    def columns(self):
        return list(self._data.keys())

    @columns.setter
    def columns(self, cols):
        self._columns = list(cols)

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        if not self._data:
            return 0
        first_key = next(iter(self._data))
        return len(self._data[first_key])

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        return DummySeries(self._data[key], name=key)

    def __setitem__(self, key, value):
        if isinstance(value, DummySeries):
            self._data[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            self._data[key] = list(value)
        elif hasattr(value, "tolist"):
            self._data[key] = value.tolist()
        else:
            self._data[key] = [value] * len(self)
        if key not in self._columns:
            self._columns.append(key)

    def copy(self):
        new_df = DummyDataFrame()
        new_df._data = {k: list(v) for k, v in self._data.items()}
        new_df._columns = list(self._columns)
        return new_df

    def groupby(self, col, sort=True):
        groups = {}
        dates = self._data[col]
        for idx, d in enumerate(dates):
            if d not in groups:
                groups[d] = []
            groups[d].append(idx)
        keys = sorted(groups.keys()) if sort else list(groups.keys())
        for k in keys:
            indices = groups[k]
            sub_dict = {c: [self._data[c][i] for i in indices] for c in self._data}
            yield k, DummyDataFrame(sub_dict)

    def sort_values(self, by, ascending=True):
        if by in self._data and len(self) > 0:
            vals = self._data[by]
            sorted_indices = sorted(range(len(vals)), key=lambda i: vals[i], reverse=not ascending)
            new_data = {c: [self._data[c][i] for i in sorted_indices] for c in self._data}
            return DummyDataFrame(new_data)
        return self.copy()

    def reset_index(self, drop=True):
        return self


class DummyDatetime:
    def __init__(self, series):
        self.series = series

    @property
    def tz(self):
        return None

    def strftime(self, fmt):
        return DummySeries([str(x)[:10] for x in self.series.values])


def dummy_to_datetime(s):
    ds = s if isinstance(s, DummySeries) else DummySeries(s)
    ds.dt = DummyDatetime(ds)
    return ds


def dummy_to_numeric(s, errors="coerce"):
    res = []
    for x in s:
        try:
            res.append(float(x))
        except (ValueError, TypeError):
            res.append(np.nan)
    return DummySeries(res)


class DummyMockPandas:
    """Mock pandas module implementing core DataFrame and Series APIs."""

    DataFrame = DummyDataFrame
    Series = DummySeries

    @staticmethod
    def to_datetime(s):
        return dummy_to_datetime(s)

    @staticmethod
    def to_numeric(s, errors="coerce"):
        return dummy_to_numeric(s, errors)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for SentimentAnalyzer and sentiment aggregation functions."""

    def setUp(self):
        """Set up mock torch and transformers environment for testing."""
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = False
        self.mock_torch.backends.mps.is_available.return_value = False

        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=lambda dev: "ids"),
            "attention_mask": MagicMock(to=lambda dev: "mask"),
        }

        # Mock model outputs (logits for positive, negative, neutral)
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}

        # Mock softmax probabilities: [pos=0.8, neg=0.1, neu=0.1]
        mock_logits = MagicMock()
        mock_output = MagicMock(logits=mock_logits)
        self.mock_model.return_value = mock_output

    def _create_analyzer(self, logits_np=None, device="cpu"):
        """Helper to create analyzer with mocked transformers and torch."""
        mock_probs = MagicMock()
        if logits_np is None:
            logits_np = np.array([[0.8, 0.1, 0.1]])

        mock_probs.detach().cpu().numpy.return_value = logits_np

        mock_F = MagicMock()
        mock_F.softmax.return_value = mock_probs

        with patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "F", mock_F), \
             patch.object(sa_module, "AutoTokenizer", MagicMock(from_pretrained=lambda m: self.mock_tokenizer)), \
             patch.object(sa_module, "AutoModelForSequenceClassification", MagicMock(from_pretrained=lambda m: self.mock_model)):
            analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert", device=device)
            analyzer._mock_F = mock_F
            return analyzer

    def test_device_auto_detection_cpu(self):
        """Test device auto-detection falls back to CPU when CUDA and MPS unavailable."""
        with patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "AutoTokenizer", MagicMock(from_pretrained=lambda m: self.mock_tokenizer)), \
             patch.object(sa_module, "AutoModelForSequenceClassification", MagicMock(from_pretrained=lambda m: self.mock_model)):
            analyzer = SentimentAnalyzer()
            self.assertEqual(analyzer.device, "cpu")

    def test_device_auto_detection_cuda(self):
        """Test device auto-detection selects CUDA when available."""
        mock_torch_cuda = MagicMock()
        mock_torch_cuda.cuda.is_available.return_value = True

        with patch.object(sa_module, "torch", mock_torch_cuda), \
             patch.object(sa_module, "AutoTokenizer", MagicMock(from_pretrained=lambda m: self.mock_tokenizer)), \
             patch.object(sa_module, "AutoModelForSequenceClassification", MagicMock(from_pretrained=lambda m: self.mock_model)):
            analyzer = SentimentAnalyzer()
            self.assertEqual(analyzer.device, "cuda")

    def test_device_auto_detection_mps(self):
        """Test device auto-detection selects MPS on Apple Silicon."""
        mock_torch_mps = MagicMock()
        mock_torch_mps.cuda.is_available.return_value = False
        mock_torch_mps.backends.mps.is_available.return_value = True

        with patch.object(sa_module, "torch", mock_torch_mps), \
             patch.object(sa_module, "AutoTokenizer", MagicMock(from_pretrained=lambda m: self.mock_tokenizer)), \
             patch.object(sa_module, "AutoModelForSequenceClassification", MagicMock(from_pretrained=lambda m: self.mock_model)):
            analyzer = SentimentAnalyzer()
            self.assertEqual(analyzer.device, "mps")

    def test_analyze_text_positive(self):
        """Test analyze_text with positive financial headline."""
        analyzer = self._create_analyzer(logits_np=np.array([[0.85, 0.05, 0.10]]))
        with patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "F", analyzer._mock_F):
            res = analyzer.analyze_text("Company reports record profits and beats revenue expectations.")
            self.assertEqual(res["label"], "positive")
            self.assertAlmostEqual(res["score"], 0.80, places=3)
            self.assertAlmostEqual(res["probabilities"]["positive"], 0.85, places=3)
            self.assertAlmostEqual(res["probabilities"]["negative"], 0.05, places=3)
            self.assertAlmostEqual(res["probabilities"]["neutral"], 0.10, places=3)

    def test_analyze_text_negative(self):
        """Test analyze_text with negative financial headline."""
        analyzer = self._create_analyzer(logits_np=np.array([[0.05, 0.85, 0.10]]))
        with patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "F", analyzer._mock_F):
            res = analyzer.analyze_text("Company faces federal probe and revenue collapse.")
            self.assertEqual(res["label"], "negative")
            self.assertAlmostEqual(res["score"], -0.80, places=3)
            self.assertAlmostEqual(res["probabilities"]["negative"], 0.85, places=3)

    def test_analyze_text_empty_input(self):
        """Test analyze_text handles empty and None inputs gracefully."""
        analyzer = self._create_analyzer()
        for empty_val in ("", "   ", None):
            res = analyzer.analyze_text(empty_val)
            self.assertEqual(res["label"], "neutral")
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["probabilities"]["neutral"], 1.0)
            self.assertEqual(res["probabilities"]["positive"], 0.0)
            self.assertEqual(res["probabilities"]["negative"], 0.0)

    def test_analyze_batch_empty_list(self):
        """Test analyze_batch returns empty list when given empty input."""
        analyzer = self._create_analyzer()
        self.assertEqual(analyzer.analyze_batch([]), [])

    def test_analyze_batch_mixed_inputs(self):
        """Test analyze_batch with mixed valid and empty strings."""
        logits = np.array([
            [0.90, 0.05, 0.05],  # Headline 1 (positive)
            [0.10, 0.80, 0.10],  # Headline 2 (negative)
        ])
        analyzer = self._create_analyzer(logits_np=logits)
        with patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "F", analyzer._mock_F):
            texts = ["Strong earnings growth", "", "Severe supply chain delays", None]
            results = analyzer.analyze_batch(texts, batch_size=2)
            self.assertEqual(len(results), 4)

            # 1st item: positive
            self.assertEqual(results[0]["label"], "positive")
            self.assertAlmostEqual(results[0]["score"], 0.85, places=3)

            # 2nd item: empty -> neutral fallback
            self.assertEqual(results[1]["label"], "neutral")
            self.assertEqual(results[1]["score"], 0.0)

            # 3rd item: negative
            self.assertEqual(results[2]["label"], "negative")
            self.assertAlmostEqual(results[2]["score"], -0.70, places=3)

            # 4th item: None -> neutral fallback
            self.assertEqual(results[3]["label"], "neutral")
            self.assertEqual(results[3]["score"], 0.0)

    def test_analyze_dataframe(self):
        """Test analyze_dataframe adds expected sentiment columns."""
        logits = np.array([
            [0.85, 0.05, 0.10],
            [0.05, 0.75, 0.20],
        ])
        analyzer = self._create_analyzer(logits_np=logits)

        with patch.object(sa_module, "pd", DummyMockPandas), \
             patch.object(sa_module, "torch", self.mock_torch), \
             patch.object(sa_module, "F", analyzer._mock_F):
            df = DummyDataFrame({
                "date": ["2026-01-01", "2026-01-02"],
                "title": ["Record profit beats forecast", "Revenue falls short amid slump"],
            })

            result_df = analyzer.analyze_dataframe(df, text_column="title", date_column="date")

            self.assertIn("sentiment_score", result_df.columns)
            self.assertIn("sentiment_label", result_df.columns)
            self.assertIn("positive_prob", result_df.columns)
            self.assertIn("negative_prob", result_df.columns)
            self.assertIn("neutral_prob", result_df.columns)
            self.assertIn("sentiment", result_df.columns)

            self.assertEqual(result_df["sentiment_label"].tolist(), ["positive", "negative"])
            self.assertAlmostEqual(result_df["sentiment_score"].tolist()[0], 0.80, places=3)
            self.assertAlmostEqual(result_df["sentiment_score"].tolist()[1], -0.70, places=3)

    def test_analyze_dataframe_empty(self):
        """Test analyze_dataframe handles empty dataframe."""
        analyzer = self._create_analyzer()
        with patch.object(sa_module, "pd", DummyMockPandas):
            empty_df = DummyDataFrame(columns=["title", "date"])
            res = analyzer.analyze_dataframe(empty_df)
            self.assertTrue(res.empty)
            self.assertIn("sentiment_score", res.columns)
            self.assertIn("sentiment_label", res.columns)

    def test_aggregate_daily_sentiment(self):
        """Test daily aggregation computes correct metrics and label thresholds."""
        with patch.object(sa_module, "pd", DummyMockPandas):
            df = DummyDataFrame({
                "date": [
                    "2026-01-01", "2026-01-01", "2026-01-01",
                    "2026-01-02", "2026-01-02",
                    "2026-01-03",
                ],
                "sentiment_score": [
                    0.8, 0.4, 0.6,   # Jan 1: positive (mean = 0.6 > 0.05)
                    -0.6, -0.4,       # Jan 2: negative (mean = -0.5 < -0.05)
                    0.02,             # Jan 3: neutral  (mean = 0.02 between -0.05 and 0.05)
                ],
                "sentiment_label": [
                    "positive", "positive", "positive",
                    "negative", "negative",
                    "neutral",
                ],
            })

            daily_df = aggregate_daily_sentiment(df, date_column="date")

            expected_cols = [
                "date", "mean_sentiment", "median_sentiment",
                "positive_count", "negative_count", "neutral_count",
                "total_articles", "sentiment_label"
            ]
            self.assertEqual(daily_df.columns, expected_cols)
            self.assertEqual(len(daily_df), 3)

            # Jan 1
            self.assertEqual(daily_df["date"].tolist()[0], "2026-01-01")
            self.assertAlmostEqual(daily_df["mean_sentiment"].tolist()[0], 0.60, places=3)
            self.assertAlmostEqual(daily_df["median_sentiment"].tolist()[0], 0.60, places=3)
            self.assertEqual(daily_df["positive_count"].tolist()[0], 3)
            self.assertEqual(daily_df["negative_count"].tolist()[0], 0)
            self.assertEqual(daily_df["neutral_count"].tolist()[0], 0)
            self.assertEqual(daily_df["total_articles"].tolist()[0], 3)
            self.assertEqual(daily_df["sentiment_label"].tolist()[0], "positive")

            # Jan 2
            self.assertEqual(daily_df["date"].tolist()[1], "2026-01-02")
            self.assertAlmostEqual(daily_df["mean_sentiment"].tolist()[1], -0.50, places=3)
            self.assertEqual(daily_df["negative_count"].tolist()[1], 2)
            self.assertEqual(daily_df["sentiment_label"].tolist()[1], "negative")

            # Jan 3
            self.assertEqual(daily_df["date"].tolist()[2], "2026-01-03")
            self.assertAlmostEqual(daily_df["mean_sentiment"].tolist()[2], 0.02, places=3)
            self.assertEqual(daily_df["neutral_count"].tolist()[2], 1)
            self.assertEqual(daily_df["sentiment_label"].tolist()[2], "neutral")

    def test_aggregate_daily_sentiment_empty(self):
        """Test aggregate_daily_sentiment returns empty dataframe with required columns."""
        with patch.object(sa_module, "pd", DummyMockPandas):
            empty_df = DummyDataFrame()
            res = aggregate_daily_sentiment(empty_df)
            self.assertTrue(res.empty)
            expected_cols = [
                "date", "mean_sentiment", "median_sentiment",
                "positive_count", "negative_count", "neutral_count",
                "total_articles", "sentiment_label"
            ]
            self.assertEqual(res.columns, expected_cols)

    def test_compute_rolling_sentiment(self):
        """Test compute_rolling_sentiment adds rolling mean columns."""
        with patch.object(sa_module, "pd", DummyMockPandas):
            daily_df = DummyDataFrame({
                "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                "mean_sentiment": [0.1, 0.3, 0.5, 0.7],
            })

            res = compute_rolling_sentiment(daily_df, windows=[3, 7])

            self.assertIn("sentiment_rolling_3", res.columns)
            self.assertIn("sentiment_rolling_7", res.columns)

            # Day 1: mean([0.1]) = 0.1
            # Day 2: mean([0.1, 0.3]) = 0.2
            # Day 3: mean([0.1, 0.3, 0.5]) = 0.3
            # Day 4: mean([0.3, 0.5, 0.7]) = 0.5
            rolling_3 = res["sentiment_rolling_3"].tolist()
            self.assertAlmostEqual(rolling_3[0], 0.1, places=3)
            self.assertAlmostEqual(rolling_3[1], 0.2, places=3)
            self.assertAlmostEqual(rolling_3[2], 0.3, places=3)
            self.assertAlmostEqual(rolling_3[3], 0.5, places=3)

    def test_compute_rolling_sentiment_empty(self):
        """Test compute_rolling_sentiment on empty dataframe."""
        with patch.object(sa_module, "pd", DummyMockPandas):
            empty_df = DummyDataFrame()
            res = compute_rolling_sentiment(empty_df, windows=[3, 7, 14])
            self.assertTrue(res.empty)
            self.assertIn("sentiment_rolling_3", res.columns)
            self.assertIn("sentiment_rolling_7", res.columns)
            self.assertIn("sentiment_rolling_14", res.columns)


if __name__ == "__main__":
    unittest.main()

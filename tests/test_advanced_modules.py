"""
Unit tests for the advanced modules:
- src.attention
- src.ensemble
- src.backtester
- src.risk_analysis
- src.explainability
- src.news_impact
"""

import unittest
import torch
import numpy as np
import pandas as pd

from src.attention import TemporalAttention, AttentionLSTM
from src.ensemble import EnsemblePredictor
from src.backtester import (
    generate_demo_backtest,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from src.risk_analysis import (
    generate_demo_risk_analysis,
)
from src.explainability import (
    generate_demo_explainability,
)
from src.news_impact import (
    generate_demo_news_impact,
)


class TestAttentionModules(unittest.TestCase):
    def test_temporal_attention_forward(self):
        batch_size, seq_len, hidden_size = 4, 30, 64
        attn = TemporalAttention(hidden_size=hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        context, weights = attn(x)
        self.assertEqual(context.shape, (batch_size, hidden_size))
        self.assertEqual(weights.shape, (batch_size, seq_len))
        sum_weights = weights.sum(dim=1)
        np.testing.assert_allclose(sum_weights.detach().numpy(), np.ones(batch_size), atol=1e-5)

    def test_attention_lstm_forward(self):
        batch_size, seq_len, input_size = 2, 10, 8
        model = AttentionLSTM(input_size=input_size, hidden_size=32, num_layers=1)
        x = torch.randn(batch_size, seq_len, input_size)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, 1))

        out_pred, weights = model.forward_with_attention(x)
        self.assertEqual(out_pred.shape, (batch_size, 1))
        self.assertEqual(weights.shape, (batch_size, seq_len))


class TestEnsembleModules(unittest.TestCase):
    def test_ensemble_predictor_uncertainty(self):
        class DummyModel:
            def __init__(self, val):
                self.val = val
            def predict(self, X):
                return np.full((len(X), 1), self.val)

        models = {
            "m1": DummyModel(100.0),
            "m2": DummyModel(110.0),
        }
        ensemble = EnsemblePredictor(models=models, weights={"m1": 0.5, "m2": 0.5})
        X = np.zeros((5, 10))
        mean_pred, uncertainty = ensemble.predict_with_uncertainty(X)
        self.assertEqual(mean_pred.shape, (5, 1))
        self.assertEqual(uncertainty.shape, (5, 1))
        self.assertAlmostEqual(mean_pred[0, 0], 105.0)
        self.assertGreater(uncertainty[0, 0], 0.0)


class TestBacktesterModules(unittest.TestCase):
    def test_generate_demo_backtest(self):
        results = generate_demo_backtest()
        self.assertIn("Model Strategy", results)
        self.assertIn("Buy & Hold", results)
        self.assertIn("Random", results)
        model_res = results["Model Strategy"]
        self.assertIsNotNone(model_res.metrics)
        self.assertGreater(len(model_res.equity_curve), 0)

    def test_calculate_sharpe_and_drawdown(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.005, 0.015])
        sharpe = calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

        equity = pd.Series([100, 105, 95, 110, 100, 120])
        max_dd, duration = calculate_max_drawdown(equity)
        self.assertGreater(max_dd, 0.0)
        self.assertGreaterEqual(duration, 0)


class TestRiskAnalysis(unittest.TestCase):
    def test_generate_demo_risk_analysis(self):
        simulator, paths, metrics = generate_demo_risk_analysis("AAPL")
        self.assertEqual(paths.ndim, 2)
        self.assertGreater(paths.shape[0], 10)
        self.assertIsNotNone(metrics.var_95)
        self.assertIsNotNone(metrics.cvar_95)


class TestExplainabilityAndNews(unittest.TestCase):
    def test_generate_demo_explainability(self):
        demo = generate_demo_explainability()
        self.assertIn("shap_values", demo)
        self.assertIn("importance_df", demo)
        self.assertIn("ablation_results", demo)
        self.assertIn("attention_weights", demo)

    def test_generate_demo_news_impact(self):
        demo = generate_demo_news_impact("AAPL")
        self.assertIn("stock_data", demo)
        self.assertIn("events", demo)
        self.assertIn("correlation", demo)


if __name__ == "__main__":
    unittest.main()

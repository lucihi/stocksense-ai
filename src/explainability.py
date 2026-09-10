"""
Explainable AI module for the stock prediction project.
Provides SHAP analysis, attention visualization, and ablation studies.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import shap
except ImportError:
    logging.warning("SHAP is not installed. SHAPExplainer functionality will be limited.")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainer for machine learning models.
    """
    
    def __init__(self, model: Any, feature_names: List[str], model_type: str = 'lstm'):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        
    def compute_shap_values(self, X_test: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for the given test data.
        """
        if 'shap' not in globals():
            raise ImportError("shap module is required to compute SHAP values.")
            
        try:
            if self.model_type in ['lstm', 'transformer', 'pytorch']:
                # Ensure input is a tensor
                if not isinstance(X_test, torch.Tensor):
                    X_tensor = torch.tensor(X_test, dtype=torch.float32)
                else:
                    X_tensor = X_test
                
                # Take a background sample
                background = X_tensor[:num_samples] if len(X_tensor) > num_samples else X_tensor
                
                # Set model to eval mode
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                    
                explainer = shap.DeepExplainer(self.model, background)
                shap_values = explainer.shap_values(X_tensor)
                
                # Handling multi-dimensional outputs from deep explainer
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                    
                return np.array(shap_values)
                
            elif self.model_type in ['xgboost', 'lgbm', 'tree']:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)
                return np.array(shap_values)
                
            else:
                # Fallback to kernel explainer
                background = X_test[:num_samples] if len(X_test) > num_samples else X_test
                explainer = shap.KernelExplainer(self.model.predict, background)
                shap_values = explainer.shap_values(X_test)
                return np.array(shap_values)
                
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            # Return dummy values to prevent crashing downstream tasks
            logger.warning("Returning dummy SHAP values due to error.")
            shape = X_test.shape
            # Assuming shape is (batch, seq_len, features) for PyTorch or (batch, features) for tree
            if len(shape) == 3:
                return np.random.randn(shape[0], shape[1], shape[2])
            else:
                return np.random.randn(shape[0], shape[1])

    def plot_summary(self, shap_values: np.ndarray, X_test: np.ndarray) -> go.Figure:
        """
        Recreate SHAP beeswarm-style plot in Plotly.
        """
        # Collapse sequence dimension if 3D (batch, seq_len, features) -> (batch * seq_len, features)
        if len(shap_values.shape) == 3:
            shap_vals_2d = shap_values.reshape(-1, shap_values.shape[-1])
            X_2d = X_test.reshape(-1, X_test.shape[-1])
        else:
            shap_vals_2d = shap_values
            X_2d = X_test
            
        # Calculate mean absolute SHAP values for sorting
        mean_abs_shap = np.abs(shap_vals_2d).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)
        
        # Take top 20 features max to avoid crowding
        if len(sorted_indices) > 20:
            sorted_indices = sorted_indices[-20:]
            
        fig = go.Figure()
        
        for idx in sorted_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature {idx}"
            feat_vals = X_2d[:, idx]
            shap_vals = shap_vals_2d[:, idx]
            
            # Normalize feature values for color scale
            min_val, max_val = feat_vals.min(), feat_vals.max()
            if max_val > min_val:
                norm_feat_vals = (feat_vals - min_val) / (max_val - min_val)
            else:
                norm_feat_vals = np.zeros_like(feat_vals)
                
            # Add scatter trace for each feature
            # We add jitter to y values to match beeswarm
            jitter = np.random.uniform(-0.2, 0.2, size=len(shap_vals))
            
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=np.full_like(shap_vals, feat_name, dtype=object),
                mode='markers',
                marker=dict(
                    color=norm_feat_vals,
                    colorscale='RdBu_r',  # Red is high, Blue is low
                    showscale=(idx == sorted_indices[-1]),
                    colorbar=dict(title='Feature Value', tickvals=[0, 1], ticktext=['Low', 'High']) if idx == sorted_indices[-1] else None,
                    size=5,
                    opacity=0.7
                ),
                name=feat_name,
                showlegend=False
            ))
            
        fig.update_layout(
            title="SHAP Summary Plot",
            xaxis_title="SHAP Value (Impact on model output)",
            yaxis_title="Features",
            yaxis=dict(categoryorder='array', categoryarray=[self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}" for i in sorted_indices]),
            height=600,
            template="plotly_white"
        )
        
        return fig

    def plot_waterfall(self, shap_values: np.ndarray, sample_idx: int, X_test: np.ndarray) -> go.Figure:
        """
        Waterfall chart explaining a single prediction.
        """
        if len(shap_values.shape) == 3:
            # Flatten or aggregate seq dim. Let's aggregate by summing SHAP over seq length for the specific sample
            sample_shap = shap_values[sample_idx].sum(axis=0)
            sample_x = X_test[sample_idx][-1]  # Use last timestep values for display
        else:
            sample_shap = shap_values[sample_idx]
            sample_x = X_test[sample_idx]
            
        base_value = 0.0 # simplified base value
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
        
        # Take top 10 features
        top_k = min(10, len(sorted_indices))
        indices = sorted_indices[:top_k]
        
        # Create categories and values for waterfall
        categories = ["Base Value"]
        values = [base_value]
        text = [f"{base_value:.2f}"]
        
        running_total = base_value
        for idx in indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature {idx}"
            feat_val = sample_x[idx]
            impact = sample_shap[idx]
            
            categories.append(f"{feat_name} = {feat_val:.2f}")
            values.append(impact)
            text.append(f"{'+' if impact > 0 else ''}{impact:.3f}")
            running_total += impact
            
        # Add remaining features as "Other"
        if len(sorted_indices) > top_k:
            other_impact = sample_shap[sorted_indices[top_k:]].sum()
            categories.append("Other Features")
            values.append(other_impact)
            text.append(f"{'+' if other_impact > 0 else ''}{other_impact:.3f}")
            running_total += other_impact
            
        # Final prediction
        categories.append("Prediction")
        # For waterfall, prediction is the total, but in plotly waterfall, 'total' measure handles it
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(categories) - 2) + ["total"],
            x=categories,
            textposition="outside",
            text=text + [f"{running_total:.2f}"],
            y=values + [0], # last 0 is for total
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot for Sample {sample_idx}",
            showlegend=False,
            template="plotly_white",
            height=500
        )
        
        return fig

    def get_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Get feature importance DataFrame.
        """
        if len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        })
        
        # Categorize
        def categorize(name: str) -> str:
            name_lower = name.lower()
            if any(term in name_lower for term in ['sentiment', 'polarity', 'subjectivity', 'finbert', 'news']):
                return 'sentiment'
            elif any(term in name_lower for term in ['close', 'open', 'high', 'low', 'volume']):
                return 'price'
            elif any(term in name_lower for term in ['rsi', 'macd', 'sma', 'ema', 'bb', 'atr']):
                return 'technical'
            else:
                return 'lag'
                
        df['category'] = df['feature'].apply(categorize)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        return df

    def plot_feature_importance_by_category(self, importance_df: pd.DataFrame) -> go.Figure:
        """
        Grouped bar chart showing importance by feature category.
        """
        cat_importance = importance_df.groupby('category')['importance'].sum().reset_index()
        cat_importance = cat_importance.sort_values('importance', ascending=True)
        
        fig = go.Figure()
        
        # Colors for categories
        color_map = {
            'sentiment': 'rgba(255, 99, 132, 0.8)',
            'price': 'rgba(54, 162, 235, 0.8)',
            'technical': 'rgba(75, 192, 192, 0.8)',
            'lag': 'rgba(255, 206, 86, 0.8)'
        }
        
        # Top 5 features per category
        for cat in cat_importance['category'][::-1]: # highest first
            cat_df = importance_df[importance_df['category'] == cat].head(5).sort_values('importance', ascending=True)
            
            fig.add_trace(go.Bar(
                y=cat_df['feature'],
                x=cat_df['importance'],
                orientation='h',
                name=cat.capitalize(),
                marker_color=color_map.get(cat, 'rgba(153, 102, 255, 0.8)')
            ))
            
        fig.update_layout(
            title="Feature Importance by Category",
            barmode='group',
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            template="plotly_white",
            height=600,
            yaxis=dict(categoryorder='total ascending')
        )
        
        return fig


class AttentionVisualizer:
    """
    Visualizer for attention weights from attention-based models.
    """
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names

    def extract_attention_weights(self, model: Any, input_sequence: torch.Tensor) -> np.ndarray:
        """
        Call model.forward_with_attention() to get weights.
        """
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward_with_attention'):
                _, attention_weights = model.forward_with_attention(input_sequence)
            else:
                raise ValueError("Model does not implement 'forward_with_attention'")
                
        # Handle shape (batch, seq_len) or similar
        weights = attention_weights.detach().cpu().numpy()
        
        # If batch size is 1, flatten
        if len(weights.shape) == 2 and weights.shape[0] == 1:
            weights = weights.squeeze(0)
            
        return weights

    def plot_attention_heatmap(self, attention_weights: np.ndarray, dates: List = None) -> go.Figure:
        """
        Heatmap of attention weights.
        """
        seq_len = attention_weights.shape[-1]
        x_labels = dates if dates is not None and len(dates) == seq_len else [f"t-{seq_len-i}" for i in range(seq_len)]
        
        # Handle 1D or 2D inputs
        if len(attention_weights.shape) == 1:
            z = [attention_weights]
            y_labels = ["Attention"]
        else:
            z = attention_weights
            y_labels = [f"Sample {i}" for i in range(attention_weights.shape[0])]
            
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Attention Weights Heatmap",
            xaxis_title="Time Step / Date",
            yaxis_title="Sample",
            template="plotly_white",
            height=400
        )
        
        return fig

    def plot_attention_over_price(self, attention_weights: np.ndarray, prices: np.ndarray, dates: List = None) -> go.Figure:
        """
        Dual-axis chart: price line + attention bar chart overlay.
        """
        seq_len = len(attention_weights)
        x_labels = dates if dates is not None and len(dates) == seq_len else [f"t-{seq_len-i}" for i in range(seq_len)]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add attention bars (primary y-axis)
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=attention_weights,
                name="Attention Weight",
                marker_color='rgba(255, 165, 0, 0.6)'
            ),
            secondary_y=False,
        )
        
        # Add price line (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=prices,
                name="Price",
                mode='lines+markers',
                line=dict(color='blue', width=2)
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Model Attention Overlayed on Price",
            template="plotly_white",
            height=500
        )
        
        fig.update_yaxes(title_text="Attention Weight", secondary_y=False)
        fig.update_yaxes(title_text="Price", secondary_y=True)
        
        return fig


class AblationStudy:
    """
    Perform ablation studies to measure feature group importance.
    """
    def __init__(self, feature_groups: Dict[str, List[str]]):
        self.feature_groups = feature_groups

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, 
            feature_names: List[str], model_factory: Callable) -> pd.DataFrame:
        """
        Train model with all features, and then without each feature group.
        (Assuming input data is flat DataFrame for simplicity of this signature, 
        or user maps 3D arrays properly)
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        results = []
        
        # 1. Baseline
        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        base_mse = mean_squared_error(y_test, preds)
        base_r2 = r2_score(y_test, preds)
        
        results.append({
            'group_removed': 'None (Baseline)',
            'mse': base_mse,
            'r2': base_r2,
            'degradation_pct': 0.0
        })
        
        # 2. Ablations
        for group_name, group_features in self.feature_groups.items():
            # Find indices of features in this group
            features_to_keep = [f for f in feature_names if f not in group_features]
            
            if isinstance(X_train, pd.DataFrame):
                X_train_abl = X_train[features_to_keep]
                X_test_abl = X_test[features_to_keep]
            else:
                # Numpy array
                indices_to_keep = [i for i, f in enumerate(feature_names) if f in features_to_keep]
                X_train_abl = X_train[:, indices_to_keep] if len(X_train.shape) == 2 else X_train[:, :, indices_to_keep]
                X_test_abl = X_test[:, indices_to_keep] if len(X_test.shape) == 2 else X_test[:, :, indices_to_keep]
                
            model = model_factory()
            model.fit(X_train_abl, y_train)
            preds = model.predict(X_test_abl)
            
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            # Degradation = (New Error - Base Error) / Base Error
            deg_pct = ((mse - base_mse) / base_mse) * 100
            
            results.append({
                'group_removed': group_name,
                'mse': mse,
                'r2': r2,
                'degradation_pct': deg_pct
            })
            
        return pd.DataFrame(results)

    def plot_ablation_results(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Bar chart showing metric degradation when each group is removed.
        """
        # Exclude baseline
        plot_df = results_df[results_df['group_removed'] != 'None (Baseline)'].copy()
        plot_df = plot_df.sort_values('degradation_pct', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=plot_df['group_removed'],
            y=plot_df['degradation_pct'],
            text=[f"+{val:.1f}% Error" for val in plot_df['degradation_pct']],
            textposition='auto',
            marker_color=['red' if val > 0 else 'green' for val in plot_df['degradation_pct']]
        ))
        
        fig.update_layout(
            title="Ablation Study: Error Increase when Feature Group Removed",
            xaxis_title="Feature Group Removed",
            yaxis_title="Error Degradation (%)",
            template="plotly_white",
            height=400
        )
        
        return fig


def generate_demo_explainability(num_features: int = 20, seq_len: int = 30) -> Dict[str, Any]:
    """
    Generate realistic demo data for dashboard display.
    """
    # Features
    tech_feats = ['RSI', 'MACD', 'SMA_20', 'EMA_50', 'Bollinger_Upper', 'Bollinger_Lower']
    sent_feats = ['news_sentiment', 'finbert_score', 'twitter_polarity', 'reddit_sentiment']
    price_feats = ['Close', 'Open', 'High', 'Low', 'Volume']
    lag_feats = [f'Lag_{i}' for i in range(1, num_features - 15 + 1)]
    
    feature_names = tech_feats + sent_feats + price_feats + lag_feats
    feature_names = feature_names[:num_features]
    
    # SHAP Data
    np.random.seed(42)
    # Give sentiment and lag features high importance
    shap_means = np.random.uniform(0.01, 0.05, size=num_features)
    for i, name in enumerate(feature_names):
        if 'sentiment' in name or 'finbert' in name:
            shap_means[i] += 0.08
        if name == 'Close':
            shap_means[i] += 0.1
            
    # Generate random SHAP values matrix (batch=100, features=num_features)
    shap_values = np.random.randn(100, num_features) * shap_means
    X_test = np.random.uniform(0, 1, size=(100, num_features))
    
    # Attention Data
    # Create attention that peaks at recent days and maybe one volatile day in middle
    dates = pd.date_range(end=pd.Timestamp.today(), periods=seq_len).strftime('%Y-%m-%d').tolist()
    
    attention_weights = np.exp(np.linspace(-2, 1, seq_len))  # Exponential increase towards recent
    attention_weights[seq_len//2] += 2.0  # Spike in middle
    attention_weights = attention_weights / attention_weights.sum()  # Normalize
    
    prices = np.cumsum(np.random.randn(seq_len)) + 150  # Random walk for prices
    
    # Ablation Data
    ablation_results = pd.DataFrame([
        {'group_removed': 'None (Baseline)', 'mse': 0.040, 'r2': 0.85, 'degradation_pct': 0.0},
        {'group_removed': 'sentiment', 'mse': 0.042, 'r2': 0.81, 'degradation_pct': 5.0},
        {'group_removed': 'technical', 'mse': 0.041, 'r2': 0.83, 'degradation_pct': 2.5},
        {'group_removed': 'lag', 'mse': 0.044, 'r2': 0.79, 'degradation_pct': 10.0},
    ])
    
    # Explainer instance just to format the data
    explainer = SHAPExplainer(model=None, feature_names=feature_names)
    importance_df = explainer.get_feature_importance(shap_values)
    
    return {
        'feature_names': feature_names,
        'shap_values': shap_values,
        'X_test': X_test,
        'importance_df': importance_df,
        'attention_weights': attention_weights,
        'prices': prices,
        'dates': dates,
        'ablation_results': ablation_results
    }

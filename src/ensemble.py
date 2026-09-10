import os
import glob
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False
import joblib

logger = logging.getLogger(__name__)

class ModelWrapper(ABC):
    """Abstract base class for wrapping different model types into a uniform interface."""
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output for given input X.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Model predictions.
        """
        pass
        
    @abstractmethod
    def load(self, path: str):
        """Load model from file.
        
        Args:
            path (str): Path to model file.
        """
        pass

class PyTorchModelWrapper(ModelWrapper):
    """Wrapper for PyTorch nn.Module models."""
    
    def __init__(self, model_class: type, model_kwargs: dict[str, Any], device: str = 'cpu'):
        """Initialize PyTorch model wrapper.
        
        Args:
            model_class (type): The PyTorch model class (e.g., StockLSTM).
            model_kwargs (dict): Keyword arguments for model instantiation.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model = model_class(**model_kwargs).to(self.device)
        self.model.eval()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the PyTorch model."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X_tensor)
            # Ensure output is a numpy array
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().numpy()
            
    def load(self, path: str):
        """Load PyTorch model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded PyTorch model from {path}")

class XGBoostModelWrapper(ModelWrapper):
    """Wrapper for XGBoost models."""
    
    def __init__(self):
        self.model = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the XGBoost model.
        
        Flattens 3D sequences (batch, seq_len, features) to 2D for XGBoost.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded.")
            
        # Flatten 3D input to 2D
        if X.ndim == 3:
            batch_size, seq_len, num_features = X.shape
            X_flat = X.reshape(batch_size, seq_len * num_features)
        else:
            X_flat = X
            
        dmatrix = xgb.DMatrix(X_flat)
        preds = self.model.predict(dmatrix)
        # Ensure shape matches expected (batch, 1) if necessary
        return preds.reshape(-1, 1)
        
    def load(self, path: str):
        """Load XGBoost model."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        logger.info(f"Loaded XGBoost model from {path}")

class EnsemblePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, models: dict[str, ModelWrapper], weights: dict[str, float] = None):
        """Initialize ensemble predictor.
        
        Args:
            models (dict[str, ModelWrapper]): Dictionary of named model wrappers.
            weights (dict[str, float], optional): Dictionary of model weights. 
                If None, equal weighting is used.
        """
        self.models = models
        if not self.models:
            raise ValueError("At least one model must be provided.")
            
        if weights is None:
            # Equal weighting
            n = len(models)
            self.weights = {name: 1.0 / n for name in models.keys()}
        else:
            self.weights = weights
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {name: w / total_weight for name, w in self.weights.items()}
            
    def get_model_predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Get individual predictions from each model.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            dict[str, np.ndarray]: Dictionary of model predictions.
        """
        predictions = {}
        for name, model in self.models.items():
            preds = model.predict(X)
            # Ensure shape is (batch, 1)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            predictions[name] = preds
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted average of all models.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Weighted ensemble prediction.
        """
        predictions = self.get_model_predictions(X)
        
        # Calculate weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, preds in predictions.items():
            ensemble_pred += self.weights[name] * preds
            
        return ensemble_pred
        
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict and return uncertainty based on model disagreement.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Mean prediction and uncertainty (std dev).
        """
        predictions = self.get_model_predictions(X)
        
        # Calculate weighted average
        mean_pred = self.predict(X)
        
        # Calculate uncertainty as standard deviation across model predictions
        all_preds = np.stack(list(predictions.values()), axis=-1) # (batch, 1, num_models)
        uncertainty = np.std(all_preds, axis=-1) # (batch, 1)
        
        return mean_pred, uncertainty
        
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        """Optimize ensemble weights to minimize validation RMSE.
        
        Args:
            X_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation target data.
            
        Returns:
            dict[str, float]: Optimized weights.
        """
        predictions = self.get_model_predictions(X_val)
        model_names = list(predictions.keys())
        pred_matrix = np.concatenate([predictions[name] for name in model_names], axis=1) # (batch, num_models)
        
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
            
        def objective(weights):
            """Objective function: RMSE of weighted predictions."""
            weighted_pred = np.dot(pred_matrix, weights).reshape(-1, 1)
            mse = np.mean((y_val - weighted_pred) ** 2)
            return np.sqrt(mse)
            
        # Initial weights: equal
        n_models = len(model_names)
        init_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n_models))
        
        # Optimize
        result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            logger.info("Weight optimization successful.")
            opt_weights = result.x
            self.weights = {name: float(w) for name, w in zip(model_names, opt_weights)}
            return self.weights
        else:
            logger.warning(f"Weight optimization failed: {result.message}. Keeping previous weights.")
            return self.weights

class LearnedWeightEnsemble(nn.Module):
    """PyTorch module that learns ensemble weights via gradient descent."""
    
    def __init__(self, num_models: int):
        """Initialize learned weight ensemble.
        
        Args:
            num_models (int): Number of models in the ensemble.
        """
        super().__init__()
        # Initialize logits to 0 (equal weights after softmax)
        self.weight_logits = nn.Parameter(torch.zeros(num_models))
        
    def forward(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass computing weighted sum of predictions.
        
        Args:
            predictions (dict[str, torch.Tensor]): Dictionary of model predictions.
            
        Returns:
            torch.Tensor: Weighted ensemble prediction.
        """
        # Calculate softmax weights
        weights = F.softmax(self.weight_logits, dim=0)
        
        # Stack predictions
        pred_list = list(predictions.values())
        # Ensure all tensors have shape (batch, 1) before stacking
        pred_list = [p.view(-1, 1) if p.ndim == 1 else p for p in pred_list]
        stacked_preds = torch.stack(pred_list, dim=-1) # (batch, 1, num_models)
        
        # Apply weights
        weighted_preds = (stacked_preds * weights).sum(dim=-1) # (batch, 1)
        return weighted_preds

def create_ensemble_from_dir(model_dir: str, config: Any) -> EnsemblePredictor:
    """Convenience function to load all saved models from a directory and create an ensemble.
    
    Args:
        model_dir (str): Directory containing saved models.
        config (Any): Project configuration containing model parameters.
        
    Returns:
        EnsemblePredictor: Instantiated and loaded ensemble predictor.
    """
    # Import model classes here to avoid circular imports
    from src.model import StockLSTM, StockTransformer
    
    models = {}
    
    # Load PyTorch models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example logic for discovering and loading models
    # This assumes models are saved with descriptive filenames like 'lstm.pth', 'transformer.pth', 'xgb.model'
    
    lstm_path = os.path.join(model_dir, 'lstm.pth')
    if os.path.exists(lstm_path):
        wrapper = PyTorchModelWrapper(StockLSTM, config.lstm_params, device=device)
        wrapper.load(lstm_path)
        models['lstm'] = wrapper
        
    transformer_path = os.path.join(model_dir, 'transformer.pth')
    if os.path.exists(transformer_path):
        wrapper = PyTorchModelWrapper(StockTransformer, config.transformer_params, device=device)
        wrapper.load(transformer_path)
        models['transformer'] = wrapper
        
    xgb_path = os.path.join(model_dir, 'xgb.model')
    if os.path.exists(xgb_path):
        wrapper = XGBoostModelWrapper()
        wrapper.load(xgb_path)
        models['xgboost'] = wrapper
        
    if not models:
        logger.warning(f"No models found in {model_dir}")
        
    return EnsemblePredictor(models=models)

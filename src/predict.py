"""
Inference pipeline for stock price prediction.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class StockPredictor:
    """Predictor class for running inference with trained models."""
    
    def __init__(self, model_path: str, scaler_path: str, config: Any):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.scaler = None
        
    def load(self) -> None:
        """Load model weights and scaler."""
        from src.model import StockLSTM
        
        # Load scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        else:
            logger.warning(f"Scaler not found at {self.scaler_path}. Predictions will not be inverse-transformed.")
            
        # Load model
        input_size = getattr(self.config, 'input_size', 5) # Provide default if missing
        self.model = StockLSTM(input_size=input_size)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict_next_day(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the stock price for the next day.
        
        Args:
            recent_data: DataFrame containing recent features (length should match sequence length)
            
        Returns:
            Dictionary with prediction details (predicted_price, predicted_direction, confidence, timestamp)
        """
        if self.model is None:
            self.load()
            
        # Ensure input data is a numpy array of shape (1, seq_len, features)
        x_input = recent_data.values
        if x_input.ndim == 2:
            x_input = np.expand_dims(x_input, axis=0)
            
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            output = self.model(x_tensor)
            pred_value = output.cpu().item()
            
        if self.scaler is not None and hasattr(self.scaler, 'inverse_transform'):
            try:
                pred_value = self.scaler.inverse_transform([[pred_value]])[0][0]
            except ValueError:
                pass
                
        # Determine direction
        last_price = recent_data.iloc[-1, 0] if not recent_data.empty else pred_value
        if self.scaler is not None and hasattr(self.scaler, 'inverse_transform'):
            try:
                last_price = self.scaler.inverse_transform([[last_price]])[0][0]
            except ValueError:
                pass
                
        diff = pred_value - last_price
        direction = 1 if diff > 0 else 0
        
        # Placeholder confidence (could be based on model uncertainty or ensemble variance)
        confidence = 0.75
        
        return {
            'predicted_price': float(pred_value),
            'predicted_direction': int(direction),
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }

    def predict_n_days(self, recent_data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Predict stock prices for the next n days via autoregression.
        
        Args:
            recent_data: Initial data sequence DataFrame
            n: Number of days to forecast
            
        Returns:
            DataFrame containing n-day forecasts
        """
        if self.model is None:
            self.load()
            
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(n):
            pred_dict = self.predict_next_day(current_data)
            pred_val = pred_dict['predicted_price']
            predictions.append(pred_val)
            
            # Autoregressive update: append new row and drop oldest
            new_row = current_data.iloc[-1].copy()
            # Assuming the 0th column is the price to be updated
            new_row.iloc[0] = pred_val 
            
            if self.scaler is not None and hasattr(self.scaler, 'transform'):
                try:
                    scaled_val = self.scaler.transform([[pred_val]])[0][0]
                    new_row.iloc[0] = scaled_val
                except ValueError:
                    pass
                
            current_data = pd.concat([current_data.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
            
        return pd.DataFrame({
            'day': range(1, n + 1),
            'predicted_price': predictions
        })

    def get_signal(self, prediction: Dict[str, Any]) -> str:
        """
        Determine trading signal based on prediction.
        
        Args:
            prediction: Output from predict_next_day
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        direction = prediction.get('predicted_direction', 0)
        confidence = prediction.get('confidence', 0.0)
        
        if confidence < 0.6:
            return 'HOLD'
            
        if direction == 1:
            return 'BUY'
        else:
            return 'SELL'

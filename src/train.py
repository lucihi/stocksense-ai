"""
Training pipeline for stock price prediction models.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)

class StockTrainer:
    """Trainer class for PyTorch models."""
    
    def __init__(self, model: nn.Module, device: str, learning_rate: float, model_dir: str):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int = 10) -> Dict[str, list]:
        """Train the model with early stopping."""
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(self.model_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best model after training
        best_path = os.path.join(self.model_dir, 'best_model.pth')
        if os.path.exists(best_path):
            self.load_model(best_path)
            
        return history

    def evaluate(self, test_loader: DataLoader, scaler: Any) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).cpu().numpy()
                predictions.extend(outputs)
                actuals.extend(batch_y.numpy())
                
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        if scaler is not None and hasattr(scaler, 'inverse_transform'):
            # Assuming scaler is fitted to 1D target
            try:
                predictions_inv = scaler.inverse_transform(predictions)
                actuals_inv = scaler.inverse_transform(actuals)
            except ValueError:
                predictions_inv = predictions
                actuals_inv = actuals
        else:
            predictions_inv = predictions
            actuals_inv = actuals

        rmse = np.sqrt(mean_squared_error(actuals_inv, predictions_inv))
        mae = mean_absolute_error(actuals_inv, predictions_inv)
        
        non_zero = actuals_inv != 0
        mape = np.mean(np.abs((actuals_inv[non_zero] - predictions_inv[non_zero]) / actuals_inv[non_zero])) * 100
        
        actual_diff = np.diff(actuals_inv.flatten())
        pred_diff = np.diff(predictions_inv.flatten())
        actual_direction = actual_diff > 0
        pred_direction = pred_diff > 0
        dir_acc = accuracy_score(actual_direction, pred_direction) if len(actual_direction) > 0 else 0.0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(dir_acc),
            'predictions': predictions_inv.flatten().tolist(),
            'actuals': actuals_inv.flatten().tolist()
        }

    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray, 
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train and test sets."""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                 torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_training_pipeline(config: Any) -> Dict[str, Any]:
    """
    End-to-end training and evaluation pipeline comparing LSTM to an XGBoost baseline.
    """
    logger.info("Starting training pipeline...")
    
    # Placeholder for loading config values and processed data
    # X_train, y_train, X_test, y_test, scaler = load_data(...)
    
    # Using mock data dimensions for pipeline completeness
    seq_len = getattr(config, 'seq_len', 10)
    features = getattr(config, 'features', 5)
    batch_size = getattr(config, 'batch_size', 32)
    epochs = getattr(config, 'epochs', 10)
    patience = getattr(config, 'patience', 5)
    model_dir = getattr(config, 'model_dir', 'models/')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock data
    X_train = np.random.rand(1000, seq_len, features)
    y_train = np.random.rand(1000)
    X_test = np.random.rand(200, seq_len, features)
    y_test = np.random.rand(200)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)
    
    # Train PyTorch Model (LSTM)
    from src.model import StockLSTM
    model = StockLSTM(input_size=features)
    trainer = StockTrainer(model, device, learning_rate=0.001, model_dir=model_dir)
    trainer.train(train_loader, test_loader, epochs=epochs, patience=patience)
    
    # Evaluate LSTM
    scaler = None # Mock scaler
    lstm_results = trainer.evaluate(test_loader, scaler=scaler)
    
    # Train Baseline XGBoost
    logger.info("Training XGBoost baseline...")
    # XGBoost requires 2D input (samples, features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(X_train_flat, y_train)
    xgb_preds = xgb_model.predict(X_test_flat)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    
    actual_diff = np.diff(y_test)
    pred_diff = np.diff(xgb_preds)
    xgb_dir_acc = accuracy_score(actual_diff > 0, pred_diff > 0) if len(actual_diff) > 0 else 0.0
    
    results = {
        'lstm': lstm_results,
        'xgboost': {
            'rmse': float(xgb_rmse),
            'mae': float(xgb_mae),
            'directional_accuracy': float(xgb_dir_acc)
        }
    }
    
    logger.info("Pipeline completed successfully.")
    return results

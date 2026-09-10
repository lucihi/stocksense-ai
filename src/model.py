"""
PyTorch models for stock price prediction.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class StockLSTM(nn.Module):
    """LSTM model for stock price prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1, bidirectional: bool = True):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.layer_norm = nn.LayerNorm(lstm_out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_out = lstm_out[:, -1, :]
        out = self.layer_norm(last_out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class StockTransformer(nn.Module):
    """Transformer model for stock price prediction."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, output_size: int = 1):
        super(StockTransformer, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        x = self.input_projection(x)
        
        # Adapting PE dynamically for batch_first=True
        seq_len = x.size(1)
        pe = self.pos_encoder.pe[:seq_len].squeeze(1).unsqueeze(0) # (1, seq_len, d_model)
        x = x + pe
        
        out = self.transformer_encoder(x)
        
        # Mean pooling over the sequence length
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

class DirectionClassifier(StockLSTM):
    """LSTM model for stock direction classification (binary)."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = True):
        super(DirectionClassifier, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            output_size=1, 
            bidirectional=bidirectional
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, 1) with values in [0, 1]
        """
        out = super().forward(x)
        return self.sigmoid(out)

class MultiHorizonHead(nn.Module):
    """Multiple prediction heads for different time horizons."""
    
    def __init__(self, input_size: int, horizons: list[int] = [1, 3, 5, 7]):
        super(MultiHorizonHead, self).__init__()
        self.horizons = horizons
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(input_size, 1) for h in horizons
        })
        
    def forward(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features of shape (batch, input_size)
            
        Returns:
            Dictionary mapping horizon (int) to predictions
        """
        return {h: self.heads[str(h)](features) for h in self.horizons}

class MultiHorizonLSTM(nn.Module):
    """LSTM model with multiple horizon predictions."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = True, horizons: list[int] = [1, 3, 5, 7]):
        super(MultiHorizonLSTM, self).__init__()
        self.lstm = StockLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.multi_head = MultiHorizonHead(lstm_out_size, horizons)
        
    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Dictionary mapping horizon (int) to predictions
        """
        # Get features from LSTM before the final fc layer
        lstm_out, _ = self.lstm.lstm(x)
        last_out = lstm_out[:, -1, :]
        features = self.lstm.layer_norm(last_out)
        features = self.lstm.dropout(features)
        
        return self.multi_head(features)

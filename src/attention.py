"""
Attention mechanisms for PyTorch models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TemporalAttention(nn.Module):
    """
    Bahdanau-style additive attention over LSTM hidden states.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super(TemporalAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes temporal attention.
        
        Args:
            lstm_outputs: Tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            context_vector: Tensor of shape (batch, hidden_size)
            attention_weights: Tensor of shape (batch, seq_len)
        """
        # lstm_outputs: (batch, seq_len, hidden_size)
        # energy: (batch, seq_len, attention_size) -> (batch, seq_len, 1) -> (batch, seq_len)
        energy = torch.tanh(self.W1(lstm_outputs))
        scores = self.W2(energy).squeeze(-1)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # context_vector: (batch, hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    """
    LSTM model with temporal attention mechanism for stock price prediction.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1, bidirectional: bool = True, 
                 attention_size: int = 64):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.attention = TemporalAttention(hidden_size=lstm_out_size, attention_size=attention_size)
        self.layer_norm = nn.LayerNorm(lstm_out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_size, output_size)
        
    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictions and attention weights.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            out: Predictions tensor of shape (batch, output_size)
            attention_weights: Tensor of shape (batch, seq_len)
        """
        lstm_out, _ = self.lstm(x)
        
        context_vector, attention_weights = self.attention(lstm_out)
        
        out = self.layer_norm(context_vector)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning only predictions.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            out: Predictions tensor of shape (batch, output_size)
        """
        out, _ = self.forward_with_attention(x)
        return out

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head version of temporal attention.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, attention_size: int = 64):
        super(MultiHeadTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            TemporalAttention(hidden_size, attention_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(hidden_size * num_heads, hidden_size)
        
    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes multi-head temporal attention.
        
        Args:
            lstm_outputs: Tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            context_vector: Tensor of shape (batch, hidden_size)
            attention_weights: Tensor of shape (batch, num_heads, seq_len)
        """
        context_vectors = []
        attention_weights_list = []
        
        for head in self.heads:
            c, a = head(lstm_outputs)
            context_vectors.append(c)
            attention_weights_list.append(a)
            
        # context_vectors: (batch, hidden_size * num_heads)
        concat_context = torch.cat(context_vectors, dim=-1)
        
        # projected context: (batch, hidden_size)
        context_vector = self.proj(concat_context)
        
        # attention_weights: (batch, num_heads, seq_len)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return context_vector, attention_weights

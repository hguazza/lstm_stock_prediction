"""Definição da arquitetura LSTM para predição de ações."""

import torch
import torch.nn as nn
from typing import Optional


class LSTMModel(nn.Module):
    """
    Modelo LSTM para predição de séries temporais de ações.

    A arquitetura consiste em:
    - Camada LSTM com múltiplas camadas e dropout
    - Camada Linear para output final
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Inicializa o modelo LSTM.

        Args:
            input_size: Número de features de entrada (padrão: 5)
            hidden_size: Número de unidades LSTM ocultas (padrão: 64)
            num_layers: Número de camadas LSTM empilhadas (padrão: 2)
            dropout: Taxa de dropout entre camadas LSTM (padrão: 0.2)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.

        Args:
            x: Tensor de entrada com shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor de saída com shape (batch_size, 1)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

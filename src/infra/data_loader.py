"""Data loaders e funções de processamento de dados para séries temporais."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def create_sequences(
    data: np.ndarray, window: int, target_idx: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências temporais para treinamento do modelo LSTM.

    Args:
        data: Array numpy com shape (n_samples, n_features) já normalizado
        window: Tamanho da janela temporal (look_back)
        target_idx: Índice da feature target (padrão: 3 para 'Close')

    Returns:
        Tupla (X, y) onde:
        - X: Array com shape (n_sequences, window, n_features)
        - y: Array com shape (n_sequences,) contendo apenas o target
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        X: Features com shape (n_samples, window, n_features)
        y: Targets com shape (n_samples,)
        test_size: Proporção dos dados para teste (padrão: 0.2)

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


class StockDataset(Dataset):
    """
    Dataset customizado para dados de ações.

    Converte arrays numpy em tensores PyTorch para uso com DataLoader.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Inicializa o dataset.

        Args:
            X: Features com shape (n_samples, window, n_features)
            y: Targets com shape (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        """Retorna o tamanho do dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna um item do dataset.

        Args:
            idx: Índice do item

        Returns:
            Tupla (X, y) como tensores PyTorch
        """
        return self.X[idx], self.y[idx]

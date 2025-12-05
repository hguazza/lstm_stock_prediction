"""Serviço de predição que carrega modelo e scaler para inferência."""

import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch

from src.core.config import settings
from src.infra.lstm_model import LSTMModel


class PredictorService:
    """
    Serviço para carregar modelo treinado e fazer predições.

    Carrega o modelo, scaler e configurações dos artefatos salvos.
    """

    def __init__(self) -> None:
        """Inicializa o serviço sem carregar o modelo."""
        self.model: Optional[LSTMModel] = None
        self.scaler: Optional[object] = None
        self.config: Optional[dict] = None

    def load_model(self, config_path: Optional[Path] = None) -> None:
        """
        Carrega o modelo a partir dos artefatos salvos.

        Args:
            config_path: Caminho para o arquivo config.json (opcional)
        """
        if config_path is None:
            config_path = settings.config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        model_weights_path = (
            settings.model_weights_path
            if settings.model_weights_path
            else config_path.parent / "model_weights.pt"
        )

        if not model_weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_weights_path}"
            )

        self.model = LSTMModel(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config.get("dropout", 0.2),
        )

        self.model.load_state_dict(
            torch.load(model_weights_path, map_location="cpu")
        )
        self.model.eval()

    def load_scaler(self, scaler_path: Optional[Path] = None) -> None:
        """
        Carrega o scaler usado no treinamento.

        Args:
            scaler_path: Caminho para o arquivo scaler.pkl (opcional)
        """
        if scaler_path is None:
            scaler_path = settings.scaler_path

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        self.scaler = joblib.load(scaler_path)

    def predict(self, sequence: np.ndarray) -> float:
        """
        Faz predição para uma sequência de dados.

        Args:
            sequence: Array numpy com shape (window, n_features) em escala original

        Returns:
            Valor predito em escala original
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Call load_scaler() first.")
        if self.config is None:
            raise ValueError("Config not loaded. Call load_model() first.")

        # Normalizar a sequência
        scaled_seq = self.scaler.transform(sequence)

        # Converter para tensor
        x = torch.tensor(
            scaled_seq.reshape(1, len(scaled_seq), self.config["input_size"]),
            dtype=torch.float32,
        )

        # Predição
        with torch.no_grad():
            pred = self.model(x).numpy()[0][0]

        # Inverter transformação apenas para a coluna Close
        target_idx = self.config.get("target_idx", 3)
        pred_full = np.zeros((1, self.config["input_size"]))
        pred_full[0, target_idx] = pred
        pred_original = self.scaler.inverse_transform(pred_full)[0, target_idx]

        return float(pred_original)

    def forecast_future(
        self, last_sequence: np.ndarray, steps: int
    ) -> List[float]:
        """
        Gera previsões futuras iterativamente.

        Args:
            last_sequence: Última sequência conhecida em escala original
            steps: Número de passos futuros a prever

        Returns:
            Lista de valores preditos em escala original
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Call load_scaler() first.")
        if self.config is None:
            raise ValueError("Config not loaded. Call load_model() first.")

        window = self.config["window"]
        target_idx = self.config.get("target_idx", 3)

        # Normalizar a sequência inicial
        seq = self.scaler.transform(last_sequence.copy())
        preds = []

        for _ in range(steps):
            x = torch.tensor(
                seq.reshape(1, window, self.config["input_size"]),
                dtype=torch.float32,
            )

            with torch.no_grad():
                y_pred = self.model(x).numpy()[0][0]

            preds.append(y_pred)

            # Atualizar sequência: remover primeira linha e adicionar nova
            new_row = seq[-1].copy()
            new_row[target_idx] = y_pred
            seq = np.append(seq[1:], [new_row], axis=0)

        # Inverter transformação
        preds_full = np.zeros((len(preds), self.config["input_size"]))
        preds_full[:, target_idx] = preds
        preds_original = self.scaler.inverse_transform(preds_full)[
            :, target_idx
        ]

        return preds_original.tolist()

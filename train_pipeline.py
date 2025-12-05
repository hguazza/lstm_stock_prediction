"""Pipeline completo de treinamento do modelo LSTM para predição de ações."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.infra.data_loader import StockDataset, create_sequences, train_test_split
from src.infra.lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações padrão extraídas do notebook
DEFAULT_CONFIG = {
    "symbol": "NVDA",
    "start_date": "2020-01-01",
    "end_date": "2025-12-01",
    "features": ["Open", "High", "Low", "Close", "Volume"],
    "target_idx": 3,  # Índice de 'Close'
    "window": 30,
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "test_size": 0.2,
}

# Grid search parameters
GRID_SEARCH_PARAMS = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "window": [30, 60],
}


def download_data(
    symbol: str, start_date: str, end_date: str, features: List[str]
) -> np.ndarray:
    """
    Baixa dados do Yahoo Finance.

    Args:
        symbol: Símbolo da ação
        start_date: Data inicial
        end_date: Data final
        features: Lista de features a extrair

    Returns:
        DataFrame com os dados
    """
    logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[features].dropna()
    logger.info(f"Downloaded {len(df)} rows")
    return df


def normalize_data(df: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normaliza os dados usando MinMaxScaler.

    Args:
        df: DataFrame com dados brutos

    Returns:
        Tupla (dados_normalizados, scaler)
    """
    logger.info("Normalizing data...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def train_model(
    model: LSTMModel,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str = "cpu",
) -> List[float]:
    """
    Treina o modelo LSTM.

    Args:
        model: Modelo LSTM
        train_loader: DataLoader com dados de treino
        epochs: Número de épocas
        learning_rate: Taxa de aprendizado
        device: Dispositivo para treino (cpu/cuda)

    Returns:
        Lista de losses por época
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

    return losses


def grid_search(
    scaled_data: np.ndarray,
    param_grid: Dict[str, List],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Tuple[Dict, float]:
    """
    Executa grid search para encontrar melhores hiperparâmetros.

    Args:
        scaled_data: Dados normalizados
        param_grid: Dicionário com parâmetros para testar
        epochs: Número de épocas para cada combinação
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado

    Returns:
        Tupla (melhores_parâmetros, melhor_loss)
    """
    from itertools import product

    logger.info("Starting grid search...")
    results = []

    for hidden_size, num_layers, window in product(
        param_grid["hidden_size"],
        param_grid["num_layers"],
        param_grid["window"],
    ):
        logger.info(
            f"Testing: hidden_size={hidden_size}, num_layers={num_layers}, window={window}"
        )

        X, y = create_sequences(scaled_data, window, target_idx=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        dataset = StockDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(
            input_size=5, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Treino rápido
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        results.append(
            (
                {"hidden_size": hidden_size, "num_layers": num_layers, "window": window},
                loss.item(),
            )
        )

    best_params, best_loss = min(results, key=lambda x: x[1])
    logger.info(f"Best parameters: {best_params}, Loss: {best_loss:.5f}")
    return best_params, best_loss


def save_artifacts(
    model: LSTMModel,
    scaler: MinMaxScaler,
    config: Dict,
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """
    Salva os artefatos do modelo treinado.

    Args:
        model: Modelo treinado
        scaler: Scaler usado na normalização
        config: Configuração do modelo
        artifacts_dir: Diretório para salvar artefatos
    """
    artifacts_dir.mkdir(exist_ok=True)

    # Salvar pesos do modelo
    model_weights_path = artifacts_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_weights_path)
    logger.info(f"Saved model weights to {model_weights_path}")

    # Salvar configuração
    config_path = artifacts_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Salvar scaler
    scaler_path = artifacts_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")


def main(
    symbol: str = DEFAULT_CONFIG["symbol"],
    start_date: str = DEFAULT_CONFIG["start_date"],
    end_date: str = DEFAULT_CONFIG["end_date"],
    do_grid_search: bool = True,
    epochs: int = DEFAULT_CONFIG["epochs"],
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """
    Pipeline principal de treinamento.

    Args:
        symbol: Símbolo da ação
        start_date: Data inicial
        end_date: Data final
        do_grid_search: Se deve executar grid search
        epochs: Número de épocas para treino final
        artifacts_dir: Diretório para salvar artefatos
    """
    # 1. Ingestão
    df = download_data(symbol, start_date, end_date, DEFAULT_CONFIG["features"])

    # 2. Processamento
    scaled, scaler = normalize_data(df)

    # 3. Grid Search (opcional)
    if do_grid_search:
        best_params, _ = grid_search(scaled, GRID_SEARCH_PARAMS)
        window = best_params["window"]
        hidden_size = best_params["hidden_size"]
        num_layers = best_params["num_layers"]
    else:
        window = DEFAULT_CONFIG["window"]
        hidden_size = DEFAULT_CONFIG["hidden_size"]
        num_layers = DEFAULT_CONFIG["num_layers"]

    # 4. Criar sequências com melhores parâmetros
    X, y = create_sequences(scaled, window, target_idx=DEFAULT_CONFIG["target_idx"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_CONFIG["test_size"]
    )

    # 5. Treino final
    logger.info("Starting final training...")
    dataset = StockDataset(X_train, y_train)
    loader = DataLoader(
        dataset, batch_size=DEFAULT_CONFIG["batch_size"], shuffle=True
    )

    model = LSTMModel(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=DEFAULT_CONFIG["dropout"],
    )

    losses = train_model(
        model,
        loader,
        epochs=epochs,
        learning_rate=DEFAULT_CONFIG["learning_rate"],
    )

    # 6. Persistência
    config = {
        "input_size": 5,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": DEFAULT_CONFIG["dropout"],
        "window": window,
        "features": DEFAULT_CONFIG["features"],
        "target_idx": DEFAULT_CONFIG["target_idx"],
    }

    save_artifacts(model, scaler, config, artifacts_dir)
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument("--symbol", type=str, default=DEFAULT_CONFIG["symbol"])
    parser.add_argument("--start-date", type=str, default=DEFAULT_CONFIG["start_date"])
    parser.add_argument("--end-date", type=str, default=DEFAULT_CONFIG["end_date"])
    parser.add_argument("--no-grid-search", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    args = parser.parse_args()

    main(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        do_grid_search=not args.no_grid_search,
        epochs=args.epochs,
        artifacts_dir=Path(args.artifacts_dir),
    )

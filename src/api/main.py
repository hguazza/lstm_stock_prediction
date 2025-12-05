"""Aplicação FastAPI principal com startup events."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.endpoints import router
from src.core.config import settings
from src.services.predictor import PredictorService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager para startup e shutdown events.

    Carrega o modelo e scaler na inicialização da aplicação.
    """
    # Startup
    logger.info("Loading model and scaler...")
    predictor = PredictorService()
    try:
        predictor.load_model()
        predictor.load_scaler()
        app.state.predictor = predictor
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model/scaler: {e}")
        app.state.predictor = None

    yield

    # Shutdown (opcional - limpeza se necessário)
    logger.info("Shutting down...")


app = FastAPI(
    title="LSTM Stock Prediction API",
    description="API para predição de preços de ações usando LSTM",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root() -> dict:
    """Endpoint raiz da API."""
    return {
        "message": "LSTM Stock Prediction API",
        "docs": "/docs",
        "health": "/health",
    }

"""Endpoints da API FastAPI."""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import PredictionRequest, PredictionResponse
from src.services.predictor import PredictorService

router = APIRouter()


def get_predictor(request: Request) -> PredictorService:
    """
    Dependency para obter o predictor do estado da aplicação.

    Args:
        request: Request object do FastAPI

    Returns:
        Instância do PredictorService

    Raises:
        HTTPException: Se o predictor não estiver inicializado
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service not initialized",
        )
    return predictor


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """
    Endpoint de health check.

    Returns:
        Status da API
    """
    return {"status": "healthy"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(
    request: PredictionRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> PredictionResponse:
    """
    Endpoint para fazer predição de preço de ação.

    Args:
        request: Dados da requisição com sequência temporal
        predictor: Instância do PredictorService (injetada via dependency)

    Returns:
        Predição e opcionalmente forecast futuro

    Raises:
        HTTPException: Se houver erro na predição
    """
    try:
        # Converter lista para numpy array
        sequence = np.array(request.sequence)

        # Validar dimensões
        if len(sequence.shape) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sequence must be 2D array (window, n_features)",
            )

        # Fazer predição
        prediction = predictor.predict(sequence)

        # Forecast futuro se solicitado
        forecast = None
        if request.steps is not None:
            forecast = predictor.forecast_future(sequence, request.steps)

        return PredictionResponse(prediction=prediction, forecast=forecast)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        ) from e

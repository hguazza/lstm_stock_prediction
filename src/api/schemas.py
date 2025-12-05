"""Pydantic schemas para validação de entrada e saída da API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Schema de requisição para predição.

    A sequência deve ter shape (window, n_features) onde:
    - window: tamanho da janela temporal (definido no treino)
    - n_features: número de features (5: Open, High, Low, Close, Volume)
    """

    sequence: List[List[float]] = Field(
        ...,
        description="Sequência temporal de dados com shape (window, n_features)",
        min_items=1,
    )

    steps: Optional[int] = Field(
        default=None,
        description="Número de passos futuros para forecast (opcional)",
        ge=1,
        le=100,
    )

    class Config:
        """Configuração do Pydantic."""

        json_schema_extra = {
            "example": {
                "sequence": [
                    [100.0, 105.0, 99.0, 103.0, 1000000.0],
                    [103.0, 108.0, 102.0, 106.0, 1100000.0],
                ],
                "steps": 5,
            }
        }


class PredictionResponse(BaseModel):
    """Schema de resposta da predição."""

    prediction: float = Field(..., description="Valor predito em escala original")
    forecast: Optional[List[float]] = Field(
        default=None, description="Lista de previsões futuras (se steps foi fornecido)"
    )

    class Config:
        """Configuração do Pydantic."""

        json_schema_extra = {
            "example": {
                "prediction": 150.25,
                "forecast": [151.0, 152.5, 153.0, 154.2, 155.1],
            }
        }

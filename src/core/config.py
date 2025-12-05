"""Configurações do projeto usando Pydantic Settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente."""

    # Diretórios
    artifacts_dir: Path = Field(default=Path("artifacts"), env="ARTIFACTS_DIR")
    model_weights_path: Optional[Path] = Field(
        default=None, env="MODEL_WEIGHTS_PATH"
    )
    config_path: Optional[Path] = Field(default=None, env="CONFIG_PATH")
    scaler_path: Optional[Path] = Field(default=None, env="SCALER_PATH")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def __init__(self, **kwargs):
        """Inicializa settings com paths padrão se não especificados."""
        super().__init__(**kwargs)
        if self.model_weights_path is None:
            self.model_weights_path = self.artifacts_dir / "model_weights.pt"
        if self.config_path is None:
            self.config_path = self.artifacts_dir / "config.json"
        if self.scaler_path is None:
            self.scaler_path = self.artifacts_dir / "scaler.pkl"


settings = Settings()

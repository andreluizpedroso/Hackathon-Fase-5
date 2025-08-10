from __future__ import annotations
from pathlib import Path
from loguru import logger
from .model_training import train_and_eval

if __name__ == "__main__":
    data_dir = Path("data")
    logger.info("Iniciando pipeline de treino…")
    metrics = train_and_eval(data_dir)
    logger.info(f"Concluído. Métricas: {metrics}")
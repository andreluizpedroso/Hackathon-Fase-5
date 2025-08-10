from __future__ import annotations
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from loguru import logger

from .preprocessing import make_supervised_dataset
from .feature_engineering import build_pipeline

# Deixe só os nomes dos arquivos aqui
ARTIFACTS = {
    "model": "model.joblib",
    "report": "last_report.txt",
}

def train_and_eval(data_dir: Path, artifacts_dir: Path = Path("artifacts")) -> Dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = make_supervised_dataset(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = build_pipeline()
    logger.info(f"Treinando em {len(X_train)} exemplos…")
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    f1 = f1_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, prob)
    except Exception:
        auc = float("nan")

    report_text = classification_report(y_test, pred)

    # Monte o caminho final juntando o diretório + nome do arquivo
    model_path = artifacts_dir / ARTIFACTS["model"]
    report_path = artifacts_dir / ARTIFACTS["report"]

    joblib.dump(pipe, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"F1: {f1:.3f}\nROC AUC: {auc:.3f}\n\n")
        f.write(report_text)

    logger.info(f"F1={f1:.3f} | ROC AUC={auc:.3f}")
    return {"f1": float(f1), "roc_auc": float(auc), "n_train": int(len(X_train)), "n_test": int(len(X_test))}

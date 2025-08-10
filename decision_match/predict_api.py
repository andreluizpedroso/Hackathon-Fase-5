from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
import joblib
from typing import Optional
from datetime import datetime

from .preprocessing import _extract_job_text, _extract_applicant_text, load_json
from .monitoring import log_prediction

ARTIFACT_PATH = Path("artifacts/model.joblib")
DATA_DIR = Path("data")

app = FastAPI(title="Decision Match API", version="1.1.0")

class PredictPayload(BaseModel):
    job_id: Optional[str] = None
    applicant_id: Optional[str] = None
    job_text: Optional[str] = None
    applicant_text: Optional[str] = None

_model = None

@app.on_event("startup")
def _load():
    global _model
    if not ARTIFACT_PATH.exists():
        raise RuntimeError("Modelo não encontrado. Treine a pipeline antes (python -m decision_match.main)")
    _model = joblib.load(ARTIFACT_PATH)

@app.get("/healthz")
def healthz():
    try:
        mtime = ARTIFACT_PATH.stat().st_mtime if ARTIFACT_PATH.exists() else None
    except Exception:
        mtime = None
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_mtime": datetime.utcfromtimestamp(mtime).isoformat() + "Z" if mtime else None,
        "version": app.version,
    }

@app.get("/examples")
def examples(n: int = 5):
    """Retorna alguns job_id e applicant_id válidos para teste rápido."""
    jobs = load_json(DATA_DIR / "vagas.json")
    apps = load_json(DATA_DIR / "applicants.json")
    job_ids = list(jobs.keys())[:max(0, n)]
    applicant_ids = list(apps.keys())[:max(0, n)]
    return {"job_ids": job_ids, "applicant_ids": applicant_ids}

def _get_texts_from_ids(job_id: str, applicant_id: str) -> tuple[str, str]:
    jobs = load_json(DATA_DIR / "vagas.json")
    apps = load_json(DATA_DIR / "applicants.json")
    job = jobs.get(str(job_id))
    app = apps.get(str(applicant_id))
    if job is None or app is None:
        raise HTTPException(status_code=404, detail="job_id ou applicant_id não encontrados nos dados")
    return _extract_job_text(job), _extract_applicant_text(app)

def _read_last_report() -> dict:
    report_path = ARTIFACT_PATH.parent / "last_report.txt"
    if not report_path.exists():
        return {"f1": None, "roc_auc": None, "raw": None}
    text = report_path.read_text(encoding="utf-8")
    f1 = None
    auc = None
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith("f1:"):
            try:
                f1 = float(line.split(":")[1].strip())
            except Exception:
                pass
        if s.startswith("roc auc:"):
            try:
                auc = float(line.split(":")[1].strip())
            except Exception:
                pass
    return {"f1": f1, "roc_auc": auc, "raw": text}

@app.get("/metrics")
def metrics():
    data = _read_last_report()
    return {"f1": data["f1"], "roc_auc": data["roc_auc"], "has_report": data["raw"] is not None}

@app.post("/predict")
def predict(payload: PredictPayload, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    if (payload.job_id and payload.applicant_id):
        jt, at = _get_texts_from_ids(payload.job_id, payload.applicant_id)
    elif (payload.job_text and payload.applicant_text):
        jt, at = payload.job_text, payload.applicant_text
    else:
        raise HTTPException(status_code=400, detail="Forneça (job_id & applicant_id) ou (job_text & applicant_text)")

    text = f"{jt} [SEP] {at}"
    proba = float(_model.predict_proba([text])[0, 1])
    label = int(proba >= threshold)

    # logging simples de auditoria
    try:
        log_prediction(text, proba)
    except Exception:
        pass

    return {"match_score": proba, "label": label, "threshold": threshold}

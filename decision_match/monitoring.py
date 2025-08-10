from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import joblib
from typing import List

LOG_PATH = Path("artifacts/predictions_log.jsonl")
MODEL_PATH = Path("artifacts/model.joblib")

# Loga previsões para auditoria e cálculo de drift simples (cobertura de vocabulário)

def log_prediction(input_text: str, score: float):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        rec = {"ts": datetime.utcnow().isoformat(), "score": score, "n_chars": len(input_text)}
        f.write(json.dumps(rec) + "\n")


def vocab_coverage(sample_texts: List[str]) -> float:
    """Retorna a fração de tokens do sample presentes no vocabulário do TF-IDF."""
    model = joblib.load(MODEL_PATH)
    tfidf = model.named_steps.get("tfidf")
    if tfidf is None:
        return 0.0
    vocab = set(tfidf.vocabulary_.keys())
    toks = 0
    in_vocab = 0
    for t in sample_texts:
        words = [w.lower() for w in t.split()]
        toks += len(words)
        in_vocab += sum(1 for w in words if w in vocab)
    return (in_vocab / max(toks, 1))
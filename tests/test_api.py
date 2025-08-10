from fastapi.testclient import TestClient
from decision_match.predict_api import app

client = TestClient(app)

def test_api_health_and_predict_by_text(monkeypatch):
    # Mock do startup: carregar um model leve em memória
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=50))
    ])
    X = ["a vaga pede java", "a vaga pede sap" ]
    y = [1, 0]
    pipe.fit(X, y)

    import decision_match.predict_api as mod
    mod._model = pipe  # injeta modelo fake

    payload = {"job_text": "pede java", "applicant_text": "java spring experiência"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "match_score" in r.json()
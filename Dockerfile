FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY decision_match ./decision_match
COPY data ./data
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "decision_match.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
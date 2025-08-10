# 🚀 Decision Match – FIAP Datathon

Solução de **Inteligência Artificial** para classificar o "match" entre uma vaga e um candidato utilizando dados históricos da empresa Decision.  
O sistema aprende com casos passados para prever a probabilidade de aprovação/contratação de um candidato para determinada vaga.

---

## 📌 1. Visão Geral do Projeto

### Problema de Negócio
A Decision gasta muito tempo para encontrar candidatos ideais.  
O processo manual de análise de currículos e vagas é lento e pode gerar decisões pouco padronizadas.

### Solução Proposta
Construir um **classificador de match vaga–candidato**:
- Recebe informações de uma vaga e de um candidato.
- Retorna um **score de probabilidade** de contratação.
- Permite consulta por **IDs** (usando base interna) ou por **textos** (informações brutas).
- Disponibilizado via **API** (FastAPI) para uso local ou deploy em nuvem.

---

## 🛠 2. Stack Tecnológica

- **Linguagem:** Python 3.11
- **Bibliotecas de ML:** scikit-learn, pandas, numpy
- **API:** FastAPI
- **Serialização:** joblib
- **Testes:** pytest
- **Empacotamento:** Docker
- **Monitoramento:** logging básico + checagem de vocabulário (drift simples)

---

## 📂 3. Estrutura do Projeto

```
decision-match/
├── data/                       # Arquivos JSON fornecidos
│   ├── applicants.json
│   ├── prospects.json
│   └── vagas.json
├── decision_match/
│   ├── preprocessing.py        # Leitura e limpeza dos dados
│   ├── feature_engineering.py  # Vetorização TF-IDF e modelo
│   ├── model_training.py       # Treino, avaliação e salvamento
│   ├── predict_api.py          # API FastAPI com /predict e endpoints extras
│   ├── monitoring.py           # Log de previsões e checagem de vocabulário
│   ├── main.py                 # Execução da pipeline de treino
├── tests/                      # Testes unitários
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔄 4. Etapas da Pipeline de Machine Learning

1. **Pré-processamento dos Dados** (`preprocessing.py`)
   - Leitura de `vagas.json`, `prospects.json`, `applicants.json`.
   - Criação de dataset supervisionado (vaga, candidato, rótulo).
   - Rótulos:
     - **1 (positivo):** "Contratado pela Decision", "Aprovado".
     - **0 (negativo):** "Não Aprovado pelo Cliente", "Não Aprovado pelo RH", "Não Aprovado pelo Requisitante".
     - Situações neutras são ignoradas.
   - Geração do texto consolidado da vaga e do candidato.

2. **Engenharia de Features** (`feature_engineering.py`)
   - Vetorização TF-IDF (português) com bigramas.
   - Stopwords PT-BR customizadas.
   - Modelo: Regressão Logística com `class_weight='balanced'`.

3. **Treinamento e Validação** (`model_training.py`)
   - Split 80/20 com estratificação.
   - Métricas: **F1-score** e **ROC AUC**.
   - Salva `artifacts/model.joblib` e `artifacts/last_report.txt`.

4. **API** (`predict_api.py`)
   - Endpoint `/predict`: recebe dados e retorna score de match.
   - Endpoints auxiliares:
     - `/examples` → IDs válidos para teste rápido.
     - `/healthz` → status do modelo.
     - `/metrics` → últimas métricas (F1/AUC).

5. **Monitoramento** (`monitoring.py`)
   - Log de cada previsão (`artifacts/predictions_log.jsonl`).
   - Checagem de cobertura de vocabulário para detectar drift.

---

## 📦 5. Instalação e Execução

### Local (Python)

```bash
# 1) Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2) Instalar dependências
pip install -r requirements.txt

# 3) Treinar modelo
python -m decision_match.main

# 4) Rodar API
uvicorn decision_match.predict_api:app --reload
# Abrir: http://127.0.0.1:8000/docs
```

### Docker

```bash
# Build da imagem
docker build -t decision-match .

# Rodar container
docker run -p 8000:8000 decision-match

# Testar
curl http://localhost:8000/healthz
```

---

## 🌐 6. Endpoints da API

### `GET /healthz`
Verifica se o modelo está carregado.
```bash
curl http://127.0.0.1:8000/healthz
```

### `GET /examples`
Retorna alguns IDs válidos para teste rápido.
```bash
curl http://127.0.0.1:8000/examples?n=3
```

### `GET /metrics`
Retorna F1-score e ROC AUC do último treino.
```bash
curl http://127.0.0.1:8000/metrics
```

### `POST /predict`
Recebe dados e retorna probabilidade de match.

**Entrada — modo 1 (por IDs):**
```json
{
  "job_id": "5185",
  "applicant_id": "31000"
}
```

**Entrada — modo 2 (por textos):**
```json
{
  "job_text": "Desenvolvedor Java com Spring e APIs",
  "applicant_text": "Experiência com Java, Spring Boot, REST"
}
```

**Saída:**
```json
{
  "match_score": 0.2639361085285857,
  "label": 0,
  "threshold": 0.5
}
```

- **`match_score`**: probabilidade (0 a 1) de match.
- **`label`**: classe prevista (1 = match, 0 = não match).
- **`threshold`**: valor de corte usado (padrão 0.5, pode alterar via query param).

---

## 🧪 7. Testes Unitários

Rodar:
```bash
pytest -q
```
Meta: ≥80% de cobertura.

---

## 📈 8. Monitoramento

- **Log de previsões**: `artifacts/predictions_log.jsonl`.
- **Cobertura de vocabulário**: função `vocab_coverage` em `monitoring.py`.

---

## 📊 9. Métricas do Modelo (exemplo)

Após treino inicial:
```
F1: 0.396
ROC AUC: 0.755
```

---

## 📹 10. Entrega

- Código no GitHub.
- Documentação completa.
- API funcional local ou em nuvem.
- Vídeo de até 5 min explicando a solução.

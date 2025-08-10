from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Dict, Tuple, List
import pandas as pd

POSITIVE = {"Contratado pela Decision", "Aprovado"}
NEGATIVE = {
    "Não Aprovado pelo Cliente",
    "Não Aprovado pelo RH",
    "Não Aprovado pelo Requisitante",
}

NEUTRAL = {"Prospect", "Inscrito", "Encaminhado ao Requisitante", "Documentação PJ", "Desistiu"}

def _clean_text(txt: str | None) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", str(txt)).strip()
    return txt

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_raw_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    apps = pd.DataFrame.from_dict(load_json(data_dir / "applicants.json"), orient="index")
    pro = pd.DataFrame.from_dict(load_json(data_dir / "prospects.json"), orient="index")
    jobs = pd.DataFrame.from_dict(load_json(data_dir / "vagas.json"), orient="index")
    return apps, pro, jobs

def _extract_job_text(job: dict) -> str:
    perfil = job.get("perfil_vaga", {})
    atividades = _clean_text(perfil.get("principais_atividades", ""))
    competencias = _clean_text(perfil.get("competencia_tecnicas_e_comportamentais", ""))
    areas = _clean_text(perfil.get("areas_atuacao", ""))
    nivel = _clean_text(perfil.get("nivel profissional", ""))
    return " \n ".join([atividades, competencias, areas, nivel])

def _extract_applicant_text(app: dict) -> str:
    infos_prof = app.get("informacoes_profissionais", {})
    form = app.get("formacao_e_idiomas", {})
    cv = _clean_text(app.get("cv_pt", ""))
    conhecimentos = _clean_text(infos_prof.get("conhecimentos_tecnicos", ""))
    area = _clean_text(infos_prof.get("area_atuacao", ""))
    titulo = _clean_text(infos_prof.get("titulo_profissional", ""))
    idiomas = _clean_text(" ".join([form.get("nivel_ingles", ""), form.get("nivel_espanhol", ""), form.get("outro_idioma", "")]))
    return " \n ".join([titulo, area, conhecimentos, idiomas, cv])

def make_supervised_dataset(data_dir: Path) -> pd.DataFrame:
    apps_raw, pro_raw, jobs_raw = build_raw_tables(data_dir)

    # Expand nested JSONs to rows
    rows: List[Dict] = []
    for job_id, rec in pro_raw.to_dict(orient="index").items():
        prospects = rec.get("prospects", []) or []
        for p in prospects:
            cand_code = str(p.get("codigo", "")).strip()
            status = _clean_text(p.get("situacao_candidado", ""))
            if not cand_code:
                continue
            label = None
            if status in POSITIVE:
                label = 1
            elif status in NEGATIVE:
                label = 0
            else:
                # skip neutral to evitar ruído
                continue
            rows.append({"job_id": str(job_id), "applicant_id": cand_code, "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Dataset supervisionado ficou vazio. Verifique mapeamento de rótulos.")

    # Attach texts
    job_texts = {str(jid): _extract_job_text(j) for jid, j in jobs_raw.to_dict(orient="index").items()}
    app_texts = {str(aid): _extract_applicant_text(a) for aid, a in apps_raw.to_dict(orient="index").items()}

    df["job_text"] = df["job_id"].map(job_texts).fillna("")
    df["applicant_text"] = df["applicant_id"].map(app_texts).fillna("")
    df["text"] = df["job_text"] + " [SEP] " + df["applicant_text"]

    # Filtro de linhas vazias
    df = df[(df["job_text"].str.len() > 5) & (df["applicant_text"].str.len() > 5)]
    if df.empty:
        raise ValueError("Após limpeza, não restaram pares com texto suficiente.")

    return df.reset_index(drop=True)
# tests/conftest.py
"""
Garante que o pacote `decision_match` seja importável durante os testes,
independente de onde o pytest for executado.
Também ajusta o diretório de trabalho para a raiz do projeto,
para que caminhos como Path("data") funcionem nos testes.
"""

import os
import sys
from pathlib import Path
import pytest


# Raiz do projeto: .../decision-match
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Coloca a raiz do projeto no sys.path (antes de outros caminhos)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True, scope="session")
def _set_cwd_to_project_root():
    """
    Força o diretório de trabalho para a raiz do projeto durante toda a sessão de testes.
    Isso garante que Path("data") e similares funcionem.
    """
    os.chdir(PROJECT_ROOT)
    yield

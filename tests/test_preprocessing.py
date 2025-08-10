from pathlib import Path
from decision_match.preprocessing import make_supervised_dataset

def test_make_supervised_dataset_not_empty(tmp_path: Path):
    # Usa os dados reais em data/
    df = make_supervised_dataset(Path("data"))
    assert not df.empty
    assert {"text", "label"}.issubset(df.columns)
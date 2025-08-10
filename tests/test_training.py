from pathlib import Path
from decision_match.model_training import train_and_eval

def test_train_and_eval_runs(tmp_path: Path):
    m = train_and_eval(Path("data"), artifacts_dir=tmp_path)
    assert m["n_train"] > 0 and m["n_test"] > 0
    assert 0.0 <= m["f1"] <= 1.0
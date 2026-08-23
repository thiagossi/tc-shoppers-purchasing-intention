import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from purchase_intent.pipeline import train


def _write_sample_processed_csv(path: Path) -> None:
    n_per_class = 10
    df = pd.DataFrame(
        {
            "feature_a": list(range(2 * n_per_class)),
            "feature_b": [0.1 * i for i in range(2 * n_per_class)],
            "Revenue": [0] * n_per_class + [1] * n_per_class,
        }
    )
    df.to_csv(path, index=False)


def _write_sample_params(path: Path) -> None:
    params = {
        "train": {
            "test_size": 0.5,
            "random_state": 42,
            "n_estimators": 10,
            "max_depth": 3,
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(params, f)


@pytest.fixture
def _patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    processed_path = tmp_path / "features.csv"
    params_path = tmp_path / "params.yaml"
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    _write_sample_processed_csv(processed_path)
    _write_sample_params(params_path)

    monkeypatch.setattr(train, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(train, "PARAMS_PATH", params_path)
    monkeypatch.setattr(train, "MODEL_PATH", model_path)
    monkeypatch.setattr(train, "METRICS_PATH", metrics_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    monkeypatch.chdir(tmp_path)

    return {"model": model_path, "metrics": metrics_path}


def test_run_training_saves_model_and_metrics(_patched_paths: dict[str, Path]) -> None:
    train.run_training()

    loaded_model = joblib.load(_patched_paths["model"])
    assert hasattr(loaded_model, "predict")

    metrics = json.loads(_patched_paths["metrics"].read_text())
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

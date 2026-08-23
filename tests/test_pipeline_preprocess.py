from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from purchase_intent.pipeline import preprocess


def _write_sample_raw_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Administrative": [0, 1, 2],
            "Administrative_Duration": [0.0, 10.5, 20.0],
            "Informational": [0, 0, 1],
            "Informational_Duration": [0.0, 0.0, 5.0],
            "ProductRelated": [1, 2, 3],
            "ProductRelated_Duration": [0.0, 64.0, 120.0],
            "BounceRates": [0.2, 0.0, 0.1],
            "ExitRates": [0.2, 0.1, 0.05],
            "PageValues": [0.0, 0.0, 12.3],
            "SpecialDay": [0.0, 0.0, 0.6],
            "Month": ["Feb", "Mar", "Feb"],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor"],
            "Weekend": [False, True, False],
            "OperatingSystems": [1, 2, 1],
            "Browser": [1, 2, 1],
            "Region": [1, 1, 3],
            "TrafficType": [1, 2, 3],
            "Revenue": [False, True, False],
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture
def _patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed" / "features.csv"
    preprocessor_path = tmp_path / "models" / "preprocessor.joblib"

    _write_sample_raw_csv(raw_path)

    monkeypatch.setattr(preprocess, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(preprocess, "PREPROCESSOR_PATH", preprocessor_path)

    return {"processed": processed_path, "preprocessor": preprocessor_path}


def test_run_preprocessing_saves_processed_data_and_preprocessor(
    _patched_paths: dict[str, Path],
) -> None:
    preprocess.run_preprocessing()

    processed_df = pd.read_csv(_patched_paths["processed"])
    assert "Revenue" in processed_df.columns
    assert len(processed_df) == 3

    loaded_preprocessor = joblib.load(_patched_paths["preprocessor"])
    assert isinstance(loaded_preprocessor, ColumnTransformer)

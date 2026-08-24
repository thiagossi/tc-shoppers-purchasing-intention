from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

from purchase_intent.data.loader import load_raw_data
from purchase_intent.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)

RAW_DATA_PATH = Path("data/raw/online_shoppers_intention.csv")
PROCESSED_DATA_PATH = Path("data/processed/features.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


def _fit_transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Ajusta o preprocessor aos dados e devolve as features transformadas, com nomes de coluna."""
    transformed = preprocessor.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    return pd.DataFrame(dense, columns=preprocessor.get_feature_names_out())


def _save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Salva o dataset processado em CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    """Salva o preprocessor treinado para reuso consistente na inferência."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def run_preprocessing() -> None:
    """Lê o CSV bruto e salva as features prontas pro treino."""
    df = load_raw_data(RAW_DATA_PATH)

    preprocessor = build_preprocessor()
    processed = _fit_transform_features(df, preprocessor)
    processed[TARGET] = df[TARGET].astype(int).to_numpy()

    _save_processed_data(processed, PROCESSED_DATA_PATH)
    _save_preprocessor(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    run_preprocessing()

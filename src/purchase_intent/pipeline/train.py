import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from purchase_intent.features.preprocessing import TARGET

MODEL_NAME = "shoppers-purchase-intent"
PRIMARY_METRIC = "f1"

PROCESSED_DATA_PATH = Path("data/processed/features.csv")
PARAMS_PATH = Path("configs/params.yaml")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("metrics.json")


def _load_params(path: Path) -> dict:
    """Lê os hiperparâmetros de treino do arquivo de configuração."""
    with open(path) as f:
        return yaml.safe_load(f)["train"]


def _load_processed_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega as features processadas e separa a coluna alvo."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def _build_model(params: dict) -> RandomForestClassifier:
    """Instancia o classificador com os hiperparâmetros configurados."""
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        class_weight="balanced",
    )


def _evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de avaliação no conjunto de teste."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _save_metrics(metrics: dict, path: Path) -> None:
    """Salva as métricas em JSON para rastreamento via DVC."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _register_and_promote_model(run_id: str, metrics: dict) -> None:
    """Registra o modelo no MLflow Model Registry e promove se superar o campeão atual."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    new_version = mlflow.register_model(model_uri, MODEL_NAME)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_metrics = client.get_run(champion.run_id).data.metrics
        is_better = metrics[PRIMARY_METRIC] >= champion_metrics[PRIMARY_METRIC]
    except MlflowException:
        is_better = True

    if is_better:
        client.set_registered_model_alias(MODEL_NAME, "champion", new_version.version)


def run_training() -> None:
    """Orquestra o treino: carrega dados, treina, avalia, registra no MLflow e salva artefatos."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    params = _load_params(PARAMS_PATH)
    X, y = _load_processed_data(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
    )

    mlflow.set_experiment("shoppers-purchasing-intention")
    with mlflow.start_run() as run:
        model = _build_model(params)
        model.fit(X_train, y_train)

        metrics = _evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        _save_metrics(metrics, METRICS_PATH)
        _register_and_promote_model(run.info.run_id, metrics)


if __name__ == "__main__":
    run_training()
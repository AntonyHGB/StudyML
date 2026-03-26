from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "heart-disease-audit"
RANDOM_STATE = 42
DATASET_FILE = "heart_disease_synth.csv"
FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "thalach"]

console = Console()


# Estrutura simples para transportar métricas e identificação de cada treino.
@dataclass
class ModelResult:
    name: str
    run_id: str
    accuracy: float
    f1: float
    recall: float
    precision: float


# Resolve a raiz do projeto a partir do arquivo atual.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Centraliza o caminho do diretório de dados.
def _data_dir() -> Path:
    return _project_root() / "data"


# Retorna o caminho completo do CSV utilizado no pipeline.
def _dataset_path() -> Path:
    return _data_dir() / DATASET_FILE


# Centraliza o caminho do diretório de tracking local do MLflow.
def _mlruns_dir() -> Path:
    return _project_root() / "mlruns"


# Configura MLflow local e garante experimento ativo.
def configure_mlflow() -> None:
    tracking_uri = _mlruns_dir().resolve().as_uri()
    _mlruns_dir().mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


# Gera dataset sintético com regras de risco para simular diagnóstico cardíaco.
def _generate_synthetic_dataset(n_samples: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(loc=54, scale=10, size=n_samples).round(), 29, 79).astype(int)
    sex = rng.integers(0, 2, size=n_samples)  # 0=feminino, 1=masculino
    cp = rng.integers(0, 4, size=n_samples)  # chest pain type
    trestbps = np.clip(rng.normal(loc=130, scale=15, size=n_samples).round(), 90, 200).astype(int)
    chol = np.clip(rng.normal(loc=245, scale=45, size=n_samples).round(), 120, 420).astype(int)
    thalach = np.clip(rng.normal(loc=150, scale=20, size=n_samples).round(), 80, 205).astype(int)

    # Combina fatores clínicos em um score contínuo de risco.
    risk_score = (
        0.05 * (age - 50)
        + 0.45 * sex
        + 0.25 * cp
        + 0.03 * (trestbps - 120)
        + 0.015 * (chol - 200)
        - 0.04 * (thalach - 150)
        + rng.normal(0, 1.2, n_samples)
    )
    probability = 1 / (1 + np.exp(-risk_score / 3.5))
    target = (probability > 0.5).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "target": target,
        }
    )


# Calcula métricas de classificação usadas na auditoria.
def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


# Treina modelo, registra parâmetros/métricas no MLflow e devolve resumo padronizado.
def _train_log_and_evaluate(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelResult:
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = _compute_metrics(y_test, preds)

        mlflow.log_params(
            {
                "model_name": model_name,
                "random_state": RANDOM_STATE,
                "test_size": 0.2,
                "n_features": len(FEATURE_COLUMNS),
            }
        )
        # Loga hiperparâmetros extras quando o modelo for regressão logística.
        if isinstance(model, LogisticRegression):
            mlflow.log_params(
                {
                    "solver": model.solver,
                    "max_iter": model.max_iter,
                    "penalty": model.penalty,
                }
            )
        # Loga estratégia quando o baseline for DummyClassifier.
        if isinstance(model, DummyClassifier):
            mlflow.log_param("strategy", model.strategy)

        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "fase_1_ML_1",
                "model_role": model_name,
                "version": "v1",
                "dataset": DATASET_FILE,
            }
        )
        mlflow.sklearn.log_model(model, artifact_path=f"model_{model_name}")

        run_id = mlflow.active_run().info.run_id

    return ModelResult(
        name=model_name,
        run_id=run_id,
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        recall=metrics["recall"],
        precision=metrics["precision"],
    )


# Exibe no terminal um resumo tabular do treinamento concluído.
def _print_setup_summary(results: list[ModelResult], dataset_path: Path) -> None:
    table = Table(title="Treinamento concluído — Heart Disease Audit", header_style="bold cyan")
    table.add_column("Modelo", style="white")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Run ID", style="dim")

    for item in results:
        table.add_row(
            item.name,
            f"{item.accuracy:.4f}",
            f"{item.f1:.4f}",
            f"{item.recall:.4f}",
            f"{item.precision:.4f}",
            item.run_id,
        )

    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]Dataset salvo em:[/bold green] {dataset_path}\n"
            f"[bold green]Experiment:[/bold green] {EXPERIMENT_NAME}",
            title="Setup finalizado",
            border_style="green",
        )
    )


# Pipeline principal: gera dados, treina baseline/challenger e registra tudo no MLflow.
def setup_pipeline() -> dict[str, Any]:
    configure_mlflow()
    _data_dir().mkdir(parents=True, exist_ok=True)

    df = _generate_synthetic_dataset(n_samples=1000, random_state=RANDOM_STATE)
    dataset_path = _dataset_path()
    df.to_csv(dataset_path, index=False)

    x = df[FEATURE_COLUMNS]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    challenger = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)

    baseline_result = _train_log_and_evaluate("baseline", baseline, x_train, x_test, y_train, y_test)
    challenger_result = _train_log_and_evaluate("challenger", challenger, x_train, x_test, y_train, y_test)

    results = [baseline_result, challenger_result]
    _print_setup_summary(results, dataset_path)

    return {
        "dataset_path": str(dataset_path),
        "experiment_name": EXPERIMENT_NAME,
        "results": [result.__dict__ for result in results],
    }


# Busca a run mais recente no MLflow para o papel de modelo informado.
def _fetch_latest_run_for_role(role: str) -> pd.Series:
    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experimento não encontrado. Execute 'python cli.py setup' primeiro.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_role = '{role}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"Run para o papel '{role}' não encontrada. Execute 'python cli.py setup' primeiro.")

    return runs.iloc[0]


# Compara baseline vs challenger priorizando recall para decisão clínica.
def compare_models() -> dict[str, Any]:
    baseline = _fetch_latest_run_for_role("baseline")
    challenger = _fetch_latest_run_for_role("challenger")

    baseline_recall = float(baseline["metrics.recall"])
    challenger_recall = float(challenger["metrics.recall"])
    recall_diff = abs(challenger_recall - baseline_recall)
    best_model = "challenger" if challenger_recall >= baseline_recall else "baseline"

    table = Table(title="Comparação de Modelos (prioridade: Recall)", header_style="bold magenta")
    table.add_column("Modelo")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")

    # Destaca visualmente o modelo vencedor na tabela.
    def style_row(model_name: str, is_best: bool) -> str:
        if is_best:
            return f"[bold green]{model_name}[/bold green]"
        return model_name

    table.add_row(
        style_row("baseline", best_model == "baseline"),
        f"{float(baseline['metrics.recall']):.4f}",
        f"{float(baseline['metrics.f1']):.4f}",
        f"{float(baseline['metrics.accuracy']):.4f}",
        f"{float(baseline['metrics.precision']):.4f}",
    )
    table.add_row(
        style_row("challenger", best_model == "challenger"),
        f"{float(challenger['metrics.recall']):.4f}",
        f"{float(challenger['metrics.f1']):.4f}",
        f"{float(challenger['metrics.accuracy']):.4f}",
        f"{float(challenger['metrics.precision']):.4f}",
    )

    console.print(table)

    # Emite alerta quando a diferença de recall é pequena para decisão robusta.
    if recall_diff < 0.05:
        console.print(
            Panel.fit(
                f"⚠️ Diferença de recall baixa: {recall_diff:.4f} (< 0.05)",
                title="Alerta Clínico",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"Δ recall = {recall_diff:.4f}",
                title="Comparação estável",
                border_style="green",
            )
        )

    return {
        "best_model_by_recall": best_model,
        "recall_difference": recall_diff,
        "baseline_recall": baseline_recall,
        "challenger_recall": challenger_recall,
        "warning_low_recall_delta": recall_diff < 0.05,
    }
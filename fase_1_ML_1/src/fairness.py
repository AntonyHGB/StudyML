from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "thalach"]
DATASET_FILE = "heart_disease_synth.csv"
EXPERIMENT_NAME = "heart-disease-audit"
RANDOM_STATE = 42
DISPARITY_THRESHOLD = 0.15

console = Console()

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _dataset_path() -> Path:
    return _project_root() / "data" / DATASET_FILE

def _mlruns_dir() -> Path:
    return _project_root() / "mlruns"

def _load_dataset() -> pd.DataFrame:
    path = _dataset_path()
    if not path.exists():
        raise RuntimeError("Dataset não encontrado. Execute 'python cli.py setup' primeiro.")
    return pd.read_csv(path)

def _configure_mlflow() -> None:
    tracking_uri = _mlruns_dir().resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

def _train_reference_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    y_pred = pd.Series(model.predict(x_test), index=y_test.index)

    return x_test, y_test.rename("y_true"), y_pred.rename("y_pred")

def _age_group(age: float) -> str:
    if age <= 40:
        return "jovem (<=40)"
    if age <= 60:
        return "adulto (41-60)"
    return "idoso (>60)"

def _manual_group_recall(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, group_df in df.groupby(group_col):
        positives = group_df[group_df["y_true"] == 1]
        tp = int(((positives["y_pred"] == 1)).sum())
        fn = int(((positives["y_pred"] == 0)).sum())
        denom = tp + fn
        recall = tp / denom if denom > 0 else 0.0

        out.append(
            {
                "group_type": group_col,
                "group_name": str(group_name),
                "recall": float(recall),
                "support_positives": int(denom),
                "tp": tp,
                "fn": fn,
            }
        )
    return out

def run_fairness_analysis() -> dict[str, Any]:
    _configure_mlflow()
    df = _load_dataset()

    x_test, y_test, y_pred = _train_reference_model(df)

    eval_df = x_test.copy()
    eval_df["y_true"] = y_test
    eval_df["y_pred"] = y_pred
    eval_df["age_group"] = eval_df["age"].apply(_age_group)
    eval_df["sex_group"] = eval_df["sex"].map({0: "feminino (0)", 1: "masculino (1)"}).fillna("desconhecido")

    age_results = _manual_group_recall(eval_df, "age_group")
    sex_results = _manual_group_recall(eval_df, "sex_group")
    all_results = age_results + sex_results

    recalls = [item["recall"] for item in all_results]
    disparity = max(recalls) - min(recalls) if recalls else 0.0
    gate_failed = disparity > DISPARITY_THRESHOLD

    table = Table(title="Fairness Audit — Recall por Grupo", header_style="bold cyan")
    table.add_column("Tipo de Grupo")
    table.add_column("Grupo")
    table.add_column("Recall", justify="right")
    table.add_column("Positivos", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FN", justify="right")

    for item in all_results:
        table.add_row(
            item["group_type"],
            item["group_name"],
            f"{item['recall']:.4f}",
            str(item["support_positives"]),
            str(item["tp"]),
            str(item["fn"]),
        )

    console.print(table)

    if gate_failed:
        console.print(
            Panel.fit(
                f"Disparidade de recall = {disparity:.4f} (> {DISPARITY_THRESHOLD:.2f})\n"
                "⛔ Governance Gate FAILED",
                title="Risco de Fairness",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"Disparidade de recall = {disparity:.4f} (<= {DISPARITY_THRESHOLD:.2f})\n"
                "✅ Governance Gate PASSED",
                title="Fairness dentro do limite",
                border_style="green",
            )
        )

    return {
        "disparity": float(disparity),
        "threshold": DISPARITY_THRESHOLD,
        "gate_failed": gate_failed,
        "results": all_results,
    }
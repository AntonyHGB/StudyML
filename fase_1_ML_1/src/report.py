from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from rich.console import Console
from rich.panel import Panel

from src.audit import EXPERIMENT_NAME, configure_mlflow
from src.drift import run_drift_analysis
from src.fairness import run_fairness_analysis

console = Console()

# Resolve a raiz do projeto para gerar o relatório no local correto.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

# Define o caminho do arquivo final REPORT.md.
def _report_path() -> Path:
    return _project_root() / "REPORT.md"

# Busca no MLflow as métricas mais recentes para o papel informado (baseline/challenger).
def _get_latest_metrics(model_role: str) -> dict[str, Any]:
    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experimento MLflow não encontrado. Execute 'python cli.py setup' primeiro.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_role = '{model_role}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"Run '{model_role}' não encontrada. Execute 'python cli.py setup' primeiro.")

    row = runs.iloc[0]
    return {
        "run_id": str(row["run_id"]),
        "accuracy": float(row["metrics.accuracy"]),
        "f1": float(row["metrics.f1"]),
        "recall": float(row["metrics.recall"]),
        "precision": float(row["metrics.precision"]),
    }

# Monta a tabela Markdown de métricas para inserir no relatório.
def _markdown_metrics_table(baseline: dict[str, Any], challenger: dict[str, Any]) -> str:
    return (
        "| Modelo | Accuracy | F1 | Recall | Precision | Run ID |\n"
        "|---|---:|---:|---:|---:|---|\n"
        f"| baseline | {baseline['accuracy']:.4f} | {baseline['f1']:.4f} | {baseline['recall']:.4f} | {baseline['precision']:.4f} | `{baseline['run_id']}` |\n"
        f"| challenger | {challenger['accuracy']:.4f} | {challenger['f1']:.4f} | {challenger['recall']:.4f} | {challenger['precision']:.4f} | `{challenger['run_id']}` |\n"
    )

# Orquestra geração do REPORT.md com métricas, drift, fairness e veredicto final.
def generate_report() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    baseline = _get_latest_metrics("baseline")
    challenger = _get_latest_metrics("challenger")

    drift_result = run_drift_analysis()
    fairness_result = run_fairness_analysis()

    # Escolhe melhor modelo com base em recall por prioridade clínica.
    better_model = "challenger" if challenger["recall"] >= baseline["recall"] else "baseline"

    # Aprova somente se não houver drift e o gate de fairness estiver aprovado.
    approved = (not drift_result["drift_detected"]) and (not fairness_result["gate_failed"])
    final_verdict = "✅ APROVADO para Staging" if approved else "❌ REPROVADO — ver alertas"

    drift_features = drift_result["drifted_features"]
    drift_features_text = ", ".join(drift_features) if drift_features else "nenhuma"

    report_md = (
        "# REPORT — Heart Disease Audit\n\n"
        f"**Timestamp:** {timestamp}\n\n"
        "## 1) Métricas dos Modelos (MLflow)\n\n"
        f"{_markdown_metrics_table(baseline, challenger)}\n"
        f"**Melhor modelo por recall (prioridade clínica):** `{better_model}`\n\n"
        "## 2) Data Drift\n\n"
        f"- **Veredicto:** {drift_result['verdict']}\n"
        f"- **Features com drift (p < 0.05):** {drift_features_text}\n\n"
        "## 3) Fairness\n\n"
        f"- **Disparidade de recall:** {fairness_result['disparity']:.4f}\n"
        f"- **Threshold de governança:** {fairness_result['threshold']:.2f}\n"
        f"- **Governance Gate:** {'⛔ FAILED' if fairness_result['gate_failed'] else '✅ PASSED'}\n\n"
        "### Recall por grupo\n\n"
        "| Tipo | Grupo | Recall | Positivos | TP | FN |\n"
        "|---|---|---:|---:|---:|---:|\n"
        + "".join(
            [
                f"| {item['group_type']} | {item['group_name']} | {item['recall']:.4f} | {item['support_positives']} | {item['tp']} | {item['fn']} |\n"
                for item in fairness_result["results"]
            ]
        )
        + "\n"
        "## 4) Veredicto Final\n\n"
        f"**{final_verdict}**\n"
    )

    report_path = _report_path()
    report_path.write_text(report_md, encoding="utf-8")

    # Exibe no terminal o caminho do relatório e o status final da auditoria.
    console.print(
        Panel.fit(
            f"[bold green]REPORT gerado:[/bold green] {report_path}\n"
            f"[bold white]{final_verdict}[/bold white]",
            title="Report",
            border_style="green" if approved else "red",
        )
    )

    return {
        "report_path": str(report_path),
        "approved": approved,
        "final_verdict": final_verdict,
        "drift": drift_result,
        "fairness": fairness_result,
    }
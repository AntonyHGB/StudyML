from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from scipy.stats import ks_2samp

FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "thalach"]
DATASET_FILE = "heart_disease_synth.csv"
P_VALUE_THRESHOLD = 0.05
RANDOM_STATE = 42

console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_path() -> Path:
    return _project_root() / "data" / DATASET_FILE


def _load_reference_dataset() -> pd.DataFrame:
    path = _dataset_path()
    if not path.exists():
        raise RuntimeError("Dataset não encontrado. Execute 'python cli.py setup' primeiro.")
    return pd.read_csv(path)


def _simulate_production_batch(reference_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    prod = reference_df.sample(n=len(reference_df), replace=True, random_state=random_state).reset_index(drop=True).copy()
    prod["age"] = np.clip((prod["age"] + 10 + rng.normal(0, 1.5, size=len(prod))).round(), 18, 95).astype(int)
    return prod


def run_drift_analysis() -> dict[str, Any]:
    reference = _load_reference_dataset()
    production = _simulate_production_batch(reference)

    results: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Executando KS test por feature...", total=len(FEATURE_COLUMNS))
        for feature in FEATURE_COLUMNS:
            stat, p_value = ks_2samp(reference[feature], production[feature])
            drifted = p_value < P_VALUE_THRESHOLD
            results.append(
                {
                    "feature": feature,
                    "ks_stat": float(stat),
                    "p_value": float(p_value),
                    "drifted": drifted,
                }
            )
            time.sleep(0.12)
            progress.update(task, advance=1)

    table = Table(title="Detecção de Data Drift (Kolmogorov-Smirnov)", header_style="bold blue")
    table.add_column("Feature", style="white")
    table.add_column("KS Stat", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Status", justify="center")

    drifted_features: list[str] = []
    for item in results:
        status = "[green]OK[/green]"
        if item["drifted"]:
            status = "[bold red]DRIFT[/bold red]"
            drifted_features.append(item["feature"])

        table.add_row(
            item["feature"],
            f"{item['ks_stat']:.4f}",
            f"{item['p_value']:.6f}",
            status,
        )

    console.print(table)

    if drifted_features:
        console.print(
            Panel.fit(
                f"Features com drift (p < 0.05): [bold red]{', '.join(drifted_features)}[/bold red]",
                title="🔴 DRIFT DETECTADO",
                border_style="red",
            )
        )
        verdict = "🔴 DRIFT DETECTADO"
    else:
        console.print(
            Panel.fit(
                "Nenhuma feature com evidência estatística de drift (p < 0.05).",
                title="🟢 ESTÁVEL",
                border_style="green",
            )
        )
        verdict = "🟢 ESTÁVEL"

    return {
        "verdict": verdict,
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "tests": results,
    }
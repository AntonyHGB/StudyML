from __future__ import annotations

from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel

from src.audit import compare_models, setup_pipeline
from src.drift import run_drift_analysis
from src.fairness import run_fairness_analysis
from src.report import generate_report

console = Console()


def run_full_pipeline() -> dict[str, Any]:
    stages: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("setup", setup_pipeline),
        ("compare", compare_models),
        ("drift", run_drift_analysis),
        ("fairness", run_fairness_analysis),
        ("report", generate_report),
    ]

    outputs: dict[str, Any] = {}
    for stage_name, handler in stages:
        console.print(
            Panel.fit(
                f"[bold cyan]Executando etapa:[/bold cyan] {stage_name}",
                title="Pipeline",
                border_style="cyan",
            )
        )
        outputs[stage_name] = handler()

    console.print(
        Panel.fit(
            "[bold green]Pipeline completo com sucesso.[/bold green]\n"
            "Etapas executadas: setup → compare → drift → fairness → report",
            title="Concluído",
            border_style="green",
        )
    )
    return outputs


def main() -> None:
    try:
        run_full_pipeline()
    except Exception as exc:  # pragma: no cover
        console.print(
            Panel.fit(
                f"[bold red]Falha na execução do pipeline:[/bold red]\n{exc}",
                title="Erro",
                border_style="red",
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
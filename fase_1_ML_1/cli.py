from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from src.audit import compare_models, setup_pipeline
from src.drift import run_drift_analysis
from src.fairness import run_fairness_analysis
from src.report import generate_report

console = Console()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="CLI de auditoria MLOps para Heart Disease (MLflow + Drift + Fairness).",
    )
    parser.add_argument(
        "command",
        choices=["setup", "compare", "drift", "fairness", "report"],
        help="Comando a executar.",
    )
    return parser

def run_command(command: str) -> dict:
    handlers = {
        "setup": setup_pipeline,
        "compare": compare_models,
        "drift": run_drift_analysis,
        "fairness": run_fairness_analysis,
        "report": generate_report,
    }
    return handlers[command]()

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_command(args.command)
    except Exception as exc:  # pragma: no cover
        console.print(
            Panel.fit(
                f"[bold red]Erro ao executar '{args.command}':[/bold red]\n{exc}",
                title="Falha na execução",
                border_style="red",
            )
        )
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
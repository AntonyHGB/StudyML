# Fase 1 — Auditoria de Modelos com MLOps (Heart Disease)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![MIT License](https://img.shields.io/badge/License-MIT-green)

Projeto de auditoria de modelos de classificação para Heart Disease com foco em práticas de MLOps:
- rastreabilidade de experimentos com MLflow,
- comparação orientada a métrica clínica (recall),
- detecção de data drift,
- verificação de fairness por grupos,
- emissão de relatório final de governança.

## Motivação (contexto MLOps)

Em cenários reais, treinar um modelo não é suficiente. É necessário:
- comparar versões de modelo de forma rastreável;
- monitorar mudanças na distribuição dos dados (drift);
- avaliar risco de viés entre grupos;
- consolidar evidências em um relatório para decisão de promoção.

Este projeto implementa esse fluxo de ponta a ponta em CLI.

---

## Pré-requisitos

- Python **3.11**
- `venv` já ativo

---

## Instalação

Na raiz do workspace:

python -m pip install -r fase_1_ML_1/requirements.txt

---

## Como executar

Todos os comandos abaixo devem ser rodados **de dentro de `fase_1_ML_1/`**:

cd fase_1_ML_1

### 1) Setup

python cli.py setup

O que faz:
- cria `data/heart_disease_synth.csv` (1000 linhas),
- treina `baseline` (DummyClassifier) e `challenger` (LogisticRegression),
- registra params, métricas e tags no MLflow (`heart-disease-audit`).

Exemplo de output (resumido):
- tabela com métricas `accuracy`, `f1`, `recall`, `precision`
- painel informando dataset salvo e experiment criado

### 2) Compare

python cli.py compare

O que faz:
- consulta runs mais recentes do baseline/challenger no MLflow,
- mostra tabela colorida com destaque em verde para melhor recall,
- emite alerta ⚠️ quando diferença de recall `< 0.05`.

Exemplo de output (resumido):
- tabela "Comparação de Modelos (prioridade: Recall)"
- painel "Alerta Clínico" ou "Comparação estável"

### 3) Drift

python cli.py drift

O que faz:
- simula batch de produção com `age + 10`,
- executa KS test por feature com barra de progresso animada,
- lista features com drift (`p-value < 0.05`),
- emite veredicto final.

Exemplo de output (resumido):
- progress bar "Executando KS test por feature..."
- tabela com `feature`, `KS Stat`, `p-value`, `Status`
- painel "🔴 DRIFT DETECTADO" ou "🟢 ESTÁVEL"

### 4) Fairness

python cli.py fairness

O que faz:
- segmenta por faixa etária (`jovem <=40`, `adulto 41-60`, `idoso >60`) e sexo,
- calcula recall manualmente por grupo (sem Fairlearn),
- falha gate se disparidade de recall `> 0.15`.

Exemplo de output (resumido):
- tabela "Fairness Audit — Recall por Grupo"
- painel "⛔ Governance Gate FAILED" ou "✅ Governance Gate PASSED"

### 5) Report

python cli.py report

O que faz:
- gera `REPORT.md` com timestamp,
- inclui métricas dos modelos, resultados de drift e fairness,
- define veredicto:
  - `✅ APROVADO para Staging`, ou
  - `❌ REPROVADO — ver alertas`.

Exemplo de output (resumido):
- painel informando caminho do report e veredicto final

---

## Estrutura de pastas

studyML/
- `README.md` (este arquivo)
- `.gitignore`
- `fase_1_ML_1/`
  - `cli.py` — entrypoint da CLI (apenas roteamento de comandos)
  - `requirements.txt` — dependências do projeto
  - `src/`
    - `__init__.py`
    - `audit.py` — setup MLflow, dataset sintético, treino e comparação
    - `drift.py` — simulação de drift + KS test
    - `fairness.py` — auditoria de fairness por grupos
    - `report.py` — geração do `REPORT.md`
  - `data/` — criado automaticamente por `python cli.py setup`
  - `mlruns/` — criado automaticamente pelo MLflow
  - `REPORT.md` — criado por `python cli.py report`

---

## Notas técnicas

- Splits com `random_state=42` e `stratify=y`.
- Todo output de terminal usa `rich`.
- Fairness implementado manualmente com `pandas` (sem Fairlearn).
- Experiment MLflow: `heart-disease-audit`.
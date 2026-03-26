# Fase 1 ML 2 - Paradigmas de Aprendizado de Máquina

Projeto básico em Python com exemplos práticos dos três paradigmas principais de Machine Learning:

- Aprendizado Supervisionado (classificação com Iris + Árvore de Decisão)
- Aprendizado Não Supervisionado (clusterização com K-Means)
- Aprendizado por Reforço (bandit com estratégia epsilon-greedy)

## Estrutura

- `main.py`: script principal com os 3 exemplos
- `requirements.txt`: dependências do projeto

## Como executar

1. (Opcional) criar e ativar ambiente virtual
2. Instalar dependências:

```bash
pip install -r requirements.txt
```

3. Executar:

```bash
python main.py
```

## Saída esperada (resumo)

- Acurácia da classificação no Iris (supervisionado)
- Quantidade de clusters e centroides (não supervisionado)
- Valores Q estimados e melhor ação (reforço)
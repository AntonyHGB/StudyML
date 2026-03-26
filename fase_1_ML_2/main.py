import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def exemplo_supervisionado() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"[Supervisionado] Acurácia no conjunto de teste: {acuracia:.2f}")


def exemplo_nao_supervisionado() -> None:
    np.random.seed(42)
    X = np.vstack(
        [
            np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
            np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2)),
        ]
    )

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    rotulos = kmeans.fit_predict(X)

    print(f"[Não supervisionado] Quantidade de clusters: {len(np.unique(rotulos))}")
    print(f"[Não supervisionado] Centroides: {kmeans.cluster_centers_}")


def exemplo_reforco(rodadas: int = 2000, epsilon: float = 0.1) -> None:
    probs = [0.2, 0.5, 0.7]
    q_valores = [0.0, 0.0, 0.0]
    contagens = [0, 0, 0]

    random.seed(42)
    np.random.seed(42)

    for _ in range(rodadas):
        if random.random() < epsilon:
            acao = random.randrange(len(q_valores))
        else:
            acao = int(np.argmax(q_valores))

        recompensa = 1 if random.random() < probs[acao] else 0
        contagens[acao] += 1
        q_valores[acao] += (recompensa - q_valores[acao]) / contagens[acao]

    q_arredondado = [round(q, 2) for q in q_valores]
    melhor_acao = int(np.argmax(q_valores))
    print(f"[Reforço] Valores Q estimados: {q_arredondado}")
    print(f"[Reforço] Melhor ação encontrada: {melhor_acao}")


if __name__ == "__main__":
    print("=== Fase 1 ML 2 | Paradigmas de Aprendizado de Máquina ===")
    exemplo_supervisionado()
    exemplo_nao_supervisionado()
    exemplo_reforco()
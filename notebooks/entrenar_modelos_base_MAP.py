"""Entrenamiento reproducible de modelos base para el reto MAP.

El script compara clasificadores lineales sobre una representación TF-IDF
construida exclusivamente con el conjunto de entrenamiento. Genera métricas,
figuras, predicciones de validación y un artefacto serializado del mejor modelo.
"""

from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
RESULTS_DIR = ROOT / "report" / "model_results"
ASSETS_DIR = ROOT / "report" / "assets" / "modelos"
MODELS_DIR = ROOT / "models"

for directory in (RESULTS_DIR, ASSETS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


COLORS = {
    "navy": "#081B34",
    "blue": "#134A9A",
    "cyan": "#00A6E0",
    "gold": "#F2B531",
    "green": "#1F9469",
    "red": "#CA4241",
    "muted": "#58687C",
    "pale": "#F2F7FC",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#C9D5E3",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def normalize_text(value: object) -> str:
    """Normaliza espacios sin eliminar símbolos matemáticos ni negaciones."""
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_target(frame: pd.DataFrame) -> pd.Series:
    """Construye la etiqueta compuesta que evalúa la competencia."""
    misconception = frame["Misconception"].fillna("").astype(str).str.strip()
    suffix = misconception.where(misconception.eq(""), ":" + misconception)
    return frame["Category"].astype(str).str.strip() + suffix


def build_text(frame: pd.DataFrame) -> pd.Series:
    """Concatena contexto, opción y explicación con marcadores explícitos."""
    question = frame["QuestionText"].map(normalize_text)
    answer = frame["MC_Answer"].map(normalize_text)
    explanation = frame["StudentExplanation"].map(normalize_text)
    return (
        "[PREGUNTA] "
        + question
        + " [OPCION] "
        + answer
        + " [EXPLICACION] "
        + explanation
    )


def map_at_3(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> float:
    """Calcula MAP@3 para un único objetivo relevante por observación."""
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    k = min(3, scores.shape[1])
    candidate = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidate, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    top_idx = np.take_along_axis(candidate, order, axis=1)
    top_labels = classes[top_idx]
    reciprocal = np.zeros(len(y_true), dtype=float)
    for rank in range(k):
        reciprocal[(reciprocal == 0) & (top_labels[:, rank] == y_true)] = 1.0 / (rank + 1)
    return float(reciprocal.mean())


def top3_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> float:
    """Proporción de observaciones cuya etiqueta aparece entre las tres primeras."""
    k = min(3, scores.shape[1])
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(classes[top_idx] == y_true[:, None], axis=1)))


def model_scores(model: object, matrix) -> np.ndarray:
    """Obtiene puntuaciones comparables para ordenar etiquetas."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix))
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix))
    raise TypeError(f"El modelo {type(model).__name__} no produce puntuaciones.")


def evaluate_model(name: str, model: object, x_train, y_train, x_val, y_val) -> tuple[dict, object, np.ndarray, np.ndarray]:
    """Ajusta y evalúa un clasificador bajo el mismo protocolo."""
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    scores = model_scores(model, x_val)
    predictions = model.predict(x_val)
    classes = np.asarray(model.classes_)
    metrics = {
        "modelo": name,
        "map_at_3": map_at_3(y_val, scores, classes),
        "top3_accuracy": top3_accuracy(y_val, scores, classes),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "tiempo_entrenamiento_s": float(fit_seconds),
        "n_clases_modelo": int(len(classes)),
    }
    return metrics, model, predictions, scores


def save_model_comparison(metrics: pd.DataFrame) -> None:
    """Crea una comparación legible para presentación."""
    display = metrics.sort_values("map_at_3", ascending=True).copy()
    labels = display["modelo"].str.replace(" (", "\n(", regex=False)
    y_pos = np.arange(len(display))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    bars = axes[0].barh(y_pos, display["map_at_3"], color=COLORS["blue"])
    axes[0].set_yticks(y_pos, labels)
    axes[0].set_xlim(0, max(0.8, display["map_at_3"].max() * 1.13))
    axes[0].set_xlabel("MAP@3")
    axes[0].set_title("Calidad del ranking top-3")
    axes[0].grid(axis="x", alpha=0.18)
    for bar, value in zip(bars, display["map_at_3"]):
        axes[0].text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontweight="bold")

    width = 0.36
    axes[1].barh(y_pos - width / 2, display["macro_f1"], height=width, label="Macro-F1", color=COLORS["gold"])
    axes[1].barh(y_pos + width / 2, display["accuracy"], height=width, label="Accuracy", color=COLORS["green"])
    axes[1].set_yticks(y_pos, labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Puntuación")
    axes[1].set_title("Equilibrio entre clases y acierto global")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", alpha=0.18)
    fig.suptitle("Comparación de modelos base · validación estratificada 80/20", color=COLORS["navy"], fontsize=18, fontweight="bold")
    fig.savefig(ASSETS_DIR / "comparacion_modelos_base.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_generalization_chart(known_metrics: dict, grouped_metrics: dict) -> None:
    """Contrasta preguntas conocidas con preguntas no vistas."""
    metric_keys = ["map_at_3", "macro_f1", "accuracy"]
    metric_labels = ["MAP@3", "Macro-F1", "Accuracy"]
    known = [known_metrics[key] for key in metric_keys]
    unseen = [grouped_metrics[key] for key in metric_keys]
    x = np.arange(len(metric_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    b1 = ax.bar(x - width / 2, known, width, label="Preguntas conocidas", color=COLORS["blue"])
    b2 = ax.bar(x + width / 2, unseen, width, label="Preguntas no vistas", color=COLORS["red"])
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntuación")
    ax.set_title("La generalización cae al excluir preguntas completas", color=COLORS["navy"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}", ha="center", fontweight="bold")
    ax.text(
        0.01,
        -0.16,
        f"Validación agrupada: {grouped_metrics['n_preguntas_validacion']} preguntas excluidas; "
        f"{grouped_metrics['porcentaje_etiquetas_no_vistas']:.1f}% de respuestas con etiqueta no observada en entrenamiento.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=10,
    )
    fig.savefig(ASSETS_DIR / "generalizacion_preguntas_no_vistas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_class_performance(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Resume el desempeño de las clases con mayor soporte de validación."""
    labels, support = np.unique(y_true, return_counts=True)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    frame = pd.DataFrame(
        {
            "etiqueta": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "soporte": support,
        }
    ).sort_values("soporte", ascending=False)
    frame.to_csv(RESULTS_DIR / "rendimiento_por_clase.csv", index=False, encoding="utf-8")

    plot_frame = frame.head(12).sort_values("f1", ascending=True)
    labels_short = plot_frame["etiqueta"].str.replace("False_Misconception:", "F_M:", regex=False).str.replace("True_Misconception:", "T_M:", regex=False)
    fig, ax = plt.subplots(figsize=(11.5, 7), constrained_layout=True)
    colors = [COLORS["green"] if value >= 0.6 else COLORS["gold"] if value >= 0.3 else COLORS["red"] for value in plot_frame["f1"]]
    bars = ax.barh(np.arange(len(plot_frame)), plot_frame["f1"], color=colors)
    ax.set_yticks(np.arange(len(plot_frame)), labels_short)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 por clase")
    ax.set_title("El desempeño varía incluso entre las clases más frecuentes", color=COLORS["navy"])
    ax.grid(axis="x", alpha=0.18)
    for bar, f1_value, count in zip(bars, plot_frame["f1"], plot_frame["soporte"]):
        ax.text(f1_value + 0.015, bar.get_y() + bar.get_height() / 2, f"{f1_value:.2f} · n={count}", va="center", fontsize=10)
    fig.savefig(ASSETS_DIR / "f1_clases_frecuentes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame


def main() -> None:
    started_all = time.perf_counter()
    raw = pd.read_csv(DATA_PATH)
    required = {
        "row_id",
        "QuestionId",
        "QuestionText",
        "MC_Answer",
        "StudentExplanation",
        "Category",
        "Misconception",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    raw = raw.copy()
    raw["target"] = build_target(raw)
    raw["model_text"] = build_text(raw)
    duplicate_columns = [
        "QuestionId",
        "QuestionText",
        "MC_Answer",
        "StudentExplanation",
        "target",
    ]
    data = raw.drop_duplicates(subset=duplicate_columns, keep="first").reset_index(drop=True)

    assert raw.shape[0] == 36_696
    assert raw.shape[1] == 9
    assert data["target"].nunique() == 65
    assert data["model_text"].str.len().gt(0).all()

    counts = data["target"].value_counts()
    singleton_labels = counts[counts < 2].index
    singleton_mask = data["target"].isin(singleton_labels)
    common_idx = data.index[~singleton_mask]
    forced_train_idx = data.index[singleton_mask]
    train_common_idx, val_idx = train_test_split(
        common_idx,
        test_size=0.20,
        random_state=SEED,
        stratify=data.loc[common_idx, "target"],
    )
    train_idx = np.concatenate([np.asarray(train_common_idx), np.asarray(forced_train_idx)])

    train = data.loc[train_idx].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val = data.loc[val_idx].reset_index(drop=True)
    x_train_text = train["model_text"]
    y_train = train["target"].to_numpy()
    x_val_text = val["model_text"]
    y_val = val["target"].to_numpy()

    vectorizer = FeatureUnion(
        [
            (
                "palabras",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=40_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_df=0.995,
                    max_features=40_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
        ],
        n_jobs=1,
    )

    vector_started = time.perf_counter()
    x_train = vectorizer.fit_transform(x_train_text)
    x_val = vectorizer.transform(x_val_text)
    vector_seconds = time.perf_counter() - vector_started
    assert x_train.shape[0] == len(train)
    assert x_val.shape[0] == len(val)

    models = {
        "Prior ingenuo": DummyClassifier(strategy="prior", random_state=SEED),
        "Complement Naive Bayes": ComplementNB(alpha=0.35),
        "Regresión logística (SGD)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=2_000,
            tol=1e-4,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVM lineal": LinearSVC(
            C=1.5,
            class_weight="balanced",
            random_state=SEED,
            max_iter=8_000,
        ),
    }

    metric_rows: list[dict] = []
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    score_matrices: dict[str, np.ndarray] = {}
    for name, model in models.items():
        metrics, trained, pred, scores = evaluate_model(name, model, x_train, y_train, x_val, y_val)
        metric_rows.append(metrics)
        fitted[name] = trained
        predictions[name] = pred
        score_matrices[name] = scores
        print(
            f"{name:<28} MAP@3={metrics['map_at_3']:.4f}  "
            f"macro-F1={metrics['macro_f1']:.4f}  accuracy={metrics['accuracy']:.4f}"
        )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["map_at_3", "macro_f1"], ascending=False
    ).reset_index(drop=True)
    best_name = str(metrics_frame.loc[0, "modelo"])
    best_model = fitted[best_name]
    best_predictions = predictions[best_name]
    best_known_metrics = metrics_frame.loc[0].to_dict()

    metrics_frame.to_csv(RESULTS_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8")
    val_output = val[["row_id", "QuestionId", "target"]].copy()
    val_output["prediccion"] = best_predictions
    best_scores = score_matrices[best_name]
    best_classes = np.asarray(best_model.classes_)
    top_idx = np.argsort(best_scores, axis=1)[:, -3:][:, ::-1]
    for rank in range(3):
        val_output[f"top_{rank + 1}"] = best_classes[top_idx[:, rank]]
    val_output.to_csv(RESULTS_DIR / "predicciones_validacion_mejor_modelo.csv", index=False, encoding="utf-8")

    class_frame = save_class_performance(y_val, best_predictions)
    save_model_comparison(metrics_frame)

    # Prueba de transferencia: se excluyen preguntas completas del entrenamiento.
    grouped_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    group_train_idx, group_val_idx = next(
        grouped_split.split(data["model_text"], data["target"], groups=data["QuestionId"])
    )
    group_train = data.iloc[group_train_idx].reset_index(drop=True)
    group_val = data.iloc[group_val_idx].reset_index(drop=True)

    group_vectorizer = FeatureUnion(vectorizer.transformer_list, n_jobs=1)
    gx_train = group_vectorizer.fit_transform(group_train["model_text"])
    gx_val = group_vectorizer.transform(group_val["model_text"])
    # Se reconstruye el algoritmo ganador con los mismos hiperparámetros.
    selected_estimator = models[best_name].__class__(**models[best_name].get_params())
    selected_estimator.fit(gx_train, group_train["target"])
    group_scores = model_scores(selected_estimator, gx_val)
    group_predictions = selected_estimator.predict(gx_val)
    group_classes = np.asarray(selected_estimator.classes_)
    unseen_mask = ~group_val["target"].isin(group_classes)
    grouped_metrics = {
        "modelo": best_name,
        "map_at_3": map_at_3(group_val["target"].to_numpy(), group_scores, group_classes),
        "top3_accuracy": top3_accuracy(group_val["target"].to_numpy(), group_scores, group_classes),
        "macro_f1": float(f1_score(group_val["target"], group_predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(group_val["target"], group_predictions)),
        "n_entrenamiento": int(len(group_train)),
        "n_validacion": int(len(group_val)),
        "n_preguntas_validacion": int(group_val["QuestionId"].nunique()),
        "preguntas_validacion": sorted(int(value) for value in group_val["QuestionId"].unique()),
        "porcentaje_etiquetas_no_vistas": float(unseen_mask.mean() * 100),
        "n_etiquetas_no_vistas": int(group_val.loc[unseen_mask, "target"].nunique()),
    }
    save_generalization_chart(best_known_metrics, grouped_metrics)

    model_artifact = {
        "vectorizer": vectorizer,
        "classifier": best_model,
        "target_format": "Category[:Misconception]",
        "input_format": "[PREGUNTA] ... [OPCION] ... [EXPLICACION] ...",
        "selected_model": best_name,
        "seed": SEED,
        "validation_metrics": best_known_metrics,
    }
    joblib.dump(model_artifact, MODELS_DIR / "mejor_modelo_base_MAP.joblib", compress=3)

    summary = {
        "fecha_ejecucion": pd.Timestamp.now(tz="America/Guatemala").isoformat(),
        "semilla": SEED,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "filas_originales": int(len(raw)),
        "filas_modelado_sin_duplicados_exactos": int(len(data)),
        "duplicados_exactos_retirados": int(len(raw) - len(data)),
        "n_clases": int(data["target"].nunique()),
        "n_entrenamiento": int(len(train)),
        "n_validacion": int(len(val)),
        "n_caracteristicas_tfidf": int(x_train.shape[1]),
        "densidad_tfidf_pct": float(x_train.nnz / (x_train.shape[0] * x_train.shape[1]) * 100),
        "tiempo_vectorizacion_s": float(vector_seconds),
        "mejor_modelo": best_name,
        "validacion_estratificada": best_known_metrics,
        "validacion_preguntas_no_vistas": grouped_metrics,
        "clases_validacion": int(class_frame.shape[0]),
        "tiempo_total_s": float(time.perf_counter() - started_all),
    }
    (RESULTS_DIR / "resumen_modelado.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nModelo seleccionado:", best_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

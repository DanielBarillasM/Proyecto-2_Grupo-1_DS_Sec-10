"""Genera versiones panorámicas de las métricas para la presentación Canva."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "report" / "model_results"
OUTPUT = ROOT / "report" / "assets" / "modelos"

NAVY = "#0B1F3A"
BLUE = "#2055A0"
GREEN = "#24966F"
GOLD = "#F3B52F"
RED = "#C94743"
GRID = "#D9E2EC"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=NAVY, labelsize=13)


def comparison_chart(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("map_at_3", ascending=True)
    names = ordered["modelo"].str.replace("Regresión logística (SGD)", "Regresión logística\n(SGD)")
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=160)
    fig.patch.set_facecolor("white")

    axes[0].barh(y, ordered["map_at_3"], color=BLUE, height=0.58)
    axes[0].set_yticks(y, names)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("MAP@3", color=NAVY, fontsize=14, fontweight="bold")
    axes[0].set_title("Calidad del ranking top-3", color=NAVY, fontsize=18, fontweight="bold", pad=12)
    style_axis(axes[0])
    for i, value in enumerate(ordered["map_at_3"]):
        axes[0].text(value + 0.015, i, f"{value:.3f}", va="center", color=NAVY, fontsize=13, fontweight="bold")

    h = 0.32
    axes[1].barh(y + h / 2, ordered["accuracy"], height=h, color=GREEN, label="Accuracy")
    axes[1].barh(y - h / 2, ordered["macro_f1"], height=h, color=GOLD, label="Macro-F1")
    axes[1].set_yticks(y, names)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Puntuación", color=NAVY, fontsize=14, fontweight="bold")
    axes[1].set_title("Acierto global y equilibrio", color=NAVY, fontsize=18, fontweight="bold", pad=12)
    style_axis(axes[1])
    axes[1].legend(loc="lower right", frameon=False, fontsize=13)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.16, wspace=0.48)
    fig.savefig(OUTPUT / "comparacion_modelos_presentacion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generalization_chart(summary: dict) -> None:
    known = summary["validacion_estratificada"]
    unseen = summary["validacion_preguntas_no_vistas"]
    labels = ["MAP@3", "Top-3 accuracy", "Macro-F1", "Accuracy"]
    known_values = [known["map_at_3"], known["top3_accuracy"], known["macro_f1"], known["accuracy"]]
    unseen_values = [unseen["map_at_3"], unseen["top3_accuracy"], unseen["macro_f1"], unseen["accuracy"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(16, 6), dpi=160)
    fig.patch.set_facecolor("white")
    b1 = ax.bar(x - width / 2, known_values, width, color=BLUE, label="Preguntas conocidas")
    b2 = ax.bar(x + width / 2, unseen_values, width, color=RED, label="Preguntas no vistas")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels, fontsize=15, fontweight="bold")
    ax.set_ylabel("Puntuación", color=NAVY, fontsize=14, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=NAVY, labelsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=13, ncol=2)
    ax.bar_label(b1, fmt="%.3f", padding=4, fontsize=12, color=NAVY, fontweight="bold")
    ax.bar_label(b2, fmt="%.3f", padding=4, fontsize=12, color=NAVY, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.17)
    fig.savefig(OUTPUT / "generalizacion_presentacion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def class_chart(per_class: pd.DataFrame) -> None:
    top = per_class.sort_values("soporte", ascending=False).head(10).sort_values("f1")
    labels = top["etiqueta"].str.replace("Misconception:", "M:", regex=False)
    labels = labels.str.slice(0, 42)
    colors = [GREEN if value >= 0.65 else GOLD if value >= 0.40 else RED for value in top["f1"]]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=160)
    fig.patch.set_facecolor("white")
    bars = ax.barh(labels, top["f1"], color=colors, height=0.62)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 por etiqueta", color=NAVY, fontsize=14, fontweight="bold")
    style_axis(ax)
    for bar, f1, support in zip(bars, top["f1"], top["soporte"]):
        ax.text(min(f1 + 0.018, 0.94), bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}  ·  n={support:,}", va="center", color=NAVY, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.16)
    fig.savefig(OUTPUT / "f1_clases_presentacion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "comparacion_modelos.csv")
    per_class = pd.read_csv(RESULTS / "rendimiento_por_clase.csv")
    summary = json.loads((RESULTS / "resumen_modelado.json").read_text(encoding="utf-8"))
    comparison_chart(metrics)
    generalization_chart(summary)
    class_chart(per_class)
    print("Gráficas panorámicas generadas en", OUTPUT)


if __name__ == "__main__":
    main()

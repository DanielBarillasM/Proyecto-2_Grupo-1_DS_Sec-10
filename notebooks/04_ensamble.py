"""Ensamble calibrado, abstención y figuras finales del reto MAP.

Requiere haber ejecutado antes:
    python notebooks/02_modelado.py
    python notebooks/03_transformer.py
    python notebooks/03_transformer.py --grupos

1. Calibra DeBERTa con escalado de temperatura sobre la partición de validación.
2. Combina las probabilidades de la regresión logística y de DeBERTa:
       p_E = α · p_DeBERTa + (1 − α) · p_RL,   α elegido en validación (MAP@3)
   y aplica la restricción jerárquica (veracidad y etiquetas observadas por pregunta).
3. Mide la calibración (ECE) y define un umbral de abstención.
4. Exporta métricas y figuras para el informe y la aplicación.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import map_modelos as mm  # noqa: E402

OUT = mm.ROOT / "results"
FIGS = mm.ROOT / "report" / "assets" / "modelos"
FIGS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.titlesize": 12})

PALETA = {
    "Prior (pregunta+opción)": "#9AA5B1",
    "TF-IDF + SVM lineal": "#28A9C7",
    "TF-IDF + Regresión logística jerárquica": "#2457A7",
    "DeBERTa-v3": "#F2B84B",
    "Ensamble calibrado": "#2F9B77",
}

df = mm.cargar_train()
CLASES = np.load(OUT / "clases.npy", allow_pickle=True)
_, _, idx_val, idx_te = mm.particiones(df)
val, te = df.iloc[idx_val].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
y_val, y_te = val["target"].values, te["target"].values
iy_val = np.searchsorted(CLASES, y_val)
iy_te = np.searchsorted(CLASES, y_te)

lin = json.loads((OUT / "resumen_lineales.json").read_text(encoding="utf-8"))
p_lr_val = np.load(OUT / "logistica_scores_val.npy")
p_lr_te = np.load(OUT / "logistica_scores_test.npy")
# Mientras no existan los resultados del Transformer, el ensamble se reduce a la
# regresión logística jerárquica (α = 0) y se omiten las figuras de DeBERTa.
HAY_DEB = (OUT / "deberta_logits_test.npy").exists()
if HAY_DEB:
    deb = json.loads((OUT / "deberta_resumen.json").read_text(encoding="utf-8"))
    z_val = np.load(OUT / "deberta_logits_val.npy")
    z_te = np.load(OUT / "deberta_logits_test.npy")
else:
    print("Sin resultados de DeBERTa: se usa solo la regresión logística.")
    z_val, z_te = np.log(p_lr_val.clip(1e-9)), np.log(p_lr_te.clip(1e-9))

# ---------------------------------------------------------------------------
# 1. Escalado de temperatura (minimiza la log-verosimilitud negativa en validación)
# ---------------------------------------------------------------------------
def nll(p, iy):
    return float(-np.log(p[np.arange(len(iy)), iy].clip(1e-12)).mean())


temps = np.linspace(0.5, 3.0, 51)
T = float(temps[np.argmin([nll(mm.softmax(z_val, t), iy_val) for t in temps])]) if HAY_DEB else 1.0
p_deb_val, p_deb_te = mm.softmax(z_val, T), mm.softmax(z_te, T)
print(f"Temperatura óptima: {T:.2f}")

# ---------------------------------------------------------------------------
# 2. Ensamble: α en validación + restricción jerárquica
# ---------------------------------------------------------------------------
logistica = joblib.load(mm.ROOT / "models" / "logistica.joblib")
mask_val = mm.alinear(logistica.mascara(val).astype(float), logistica.classes_, CLASES) > 0
mask_te = mm.alinear(logistica.mascara(te).astype(float), logistica.classes_, CLASES) > 0


def mezclar(p_deb, p_lr, a, mask):
    p = (a * p_deb + (1 - a) * p_lr) * mask
    return p / p.sum(axis=1, keepdims=True).clip(1e-12)


alfas = np.round(np.linspace(0, 1, 21), 2)
curva_alfa = [mm.map_at_3(y_val, mm.top_k(mezclar(p_deb_val, p_lr_val, a, mask_val), CLASES))
              for a in alfas]
ALFA = float(alfas[int(np.argmax(curva_alfa))]) if HAY_DEB else 0.0
p_ens_val = mezclar(p_deb_val, p_lr_val, ALFA, mask_val)
p_ens_te = mezclar(p_deb_te, p_lr_te, ALFA, mask_te)
print(f"α óptimo: {ALFA:.2f} (MAP@3 val = {max(curva_alfa):.4f})")

# ---------------------------------------------------------------------------
# 3. Métricas de prueba de todos los modelos
# ---------------------------------------------------------------------------
filas = [{k: v for k, v in r.items() if k != "Hiperparámetros"} for r in lin["prueba"]]
if HAY_DEB:
    m_deb = mm.metricas(y_te, mm.top_k(p_deb_te, CLASES))
    filas.append({"Modelo": "DeBERTa-v3", **m_deb,
                  "Entrenamiento (s)": deb["entrenamiento_s"], "Predicción (s)": deb["prediccion_s"]})
    t_lr = next(r for r in lin["prueba"] if "jerárquica" in r["Modelo"])
    m_ens = mm.metricas(y_te, mm.top_k(p_ens_te, CLASES))
    filas.append({"Modelo": "Ensamble calibrado", **m_ens,
                  "Entrenamiento (s)": deb["entrenamiento_s"] + t_lr["Entrenamiento (s)"],
                  "Predicción (s)": deb["prediccion_s"] + t_lr["Predicción (s)"]})
res = pd.DataFrame(filas)
res.to_csv(OUT / "metricas_prueba.csv", index=False)
print(res.round(4).to_string())

# Calibración (ECE) de los modelos probabilísticos
cal = {"TF-IDF + Regresión logística jerárquica": mm.ece(p_lr_te, iy_te)}
if HAY_DEB:
    cal.update({"DeBERTa-v3 (sin calibrar)": mm.ece(mm.softmax(z_te), iy_te),
                "DeBERTa-v3 (temperatura)": mm.ece(p_deb_te, iy_te),
                "Ensamble calibrado": mm.ece(p_ens_te, iy_te)})

# ---------------------------------------------------------------------------
# 4. Abstención: umbral sobre la confianza de la primera etiqueta
#    τ = menor umbral cuya exactitud en validación sobre los casos aceptados ≥ 95 %
# ---------------------------------------------------------------------------
def curva_abstencion(p, iy):
    conf = p.max(axis=1)
    acierto = p.argmax(axis=1) == iy
    umbrales = np.round(np.linspace(0, 0.99, 100), 2)
    return pd.DataFrame([{"umbral": u, "cobertura": (conf >= u).mean(),
                          "exactitud aceptados": acierto[conf >= u].mean() if (conf >= u).any() else np.nan}
                         for u in umbrales])


cv_val = curva_abstencion(p_ens_val, iy_val)
TAU = float(cv_val.loc[cv_val["exactitud aceptados"] >= 0.95, "umbral"].min())
conf_te = p_ens_te.max(axis=1)
acept = conf_te >= TAU
abst = {"umbral": TAU, "cobertura_prueba": float(acept.mean()),
        "exactitud_aceptados": float((p_ens_te.argmax(1) == iy_te)[acept].mean()),
        "exactitud_derivados": float((p_ens_te.argmax(1) == iy_te)[~acept].mean())}
print("Abstención:", abst)
cv_te = curva_abstencion(p_ens_te, iy_te)
cv_te.to_csv(OUT / "curva_abstencion.csv", index=False)

# Transferencia (validación B)
tb = pd.DataFrame(lin["transferencia"])
ruta_g = OUT / "deberta_grupos.json"
if ruta_g.exists():
    g = json.loads(ruta_g.read_text(encoding="utf-8"))
    tb = pd.concat([tb, pd.DataFrame([{
        "Modelo": "DeBERTa-v3 (pregunta + opción + explicación)",
        "MAP@3 preguntas no vistas": g["MAP@3 media"],
        "Accuracy Category preguntas no vistas": float(np.mean([p["Accuracy Category"] for p in g["pliegues"]])),
    }])], ignore_index=True)
tb.to_csv(OUT / "transferencia.csv", index=False)

# Desempeño por dimensión y por pregunta del ensamble
pred = mm.top_k(p_ens_te, CLASES)[:, 0]
dim = {
    "Veracidad (True/False)": float(np.mean([p.split("_")[0] == t.split("_")[0] for p, t in zip(pred, y_te)])),
    "Category (6 clases)": float(np.mean([p.split(":")[0] == t.split(":")[0] for p, t in zip(pred, y_te)])),
    "Etiqueta compuesta (65 clases)": float(np.mean(pred == y_te)),
}
mis = np.array([t.split(":")[1] != "NA" for t in y_te])
dim["Misconception (solo filas con concepción)"] = float(np.mean(
    [p.split(":")[1] == t.split(":")[1] for p, t in zip(pred[mis], y_te[mis])]))
por_preg = (pd.Series(mm.map_at_3_filas(y_te, mm.top_k(p_ens_te, CLASES)))
            .groupby(te["QuestionId"].values).mean())

resumen = {"temperatura": T, "alfa": ALFA, "curva_alfa": dict(zip(map(str, alfas), curva_alfa)),
           "ece": cal, "abstencion": abst, "por_dimension": dim,
           "map3_por_pregunta": {str(k): float(v) for k, v in por_preg.items()},
           "mejor_modelo": res.sort_values("MAP@3").iloc[-1]["Modelo"],
           "n_test": len(te), "n_val": len(val), "hay_deberta": HAY_DEB}
(OUT / "resumen_final.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
np.save(OUT / "ensamble_probs_test.npy", p_ens_te.astype(np.float32))
(mm.ROOT / "models" / "ensamble.json").write_text(json.dumps(
    {"temperatura": T, "alfa": ALFA, "umbral": TAU, "hay_deberta": HAY_DEB}))

# ---------------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------------
orden = res.sort_values("MAP@3")["Modelo"].tolist()
colores = [PALETA[m] for m in orden]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, met in zip(axes, ["MAP@3", "Accuracy", "Macro-F1"]):
    v = res.set_index("Modelo").loc[orden, met]
    ax.barh(orden, v, color=colores)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_title(met)
fig.suptitle(f"Desempeño en el conjunto de prueba ({len(te):,} explicaciones)".replace(",", " "),
             fontweight="bold")
plt.tight_layout(); fig.savefig(FIGS / "comparacion_metricas.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5))
desp = {"TF-IDF + Regresión logística jerárquica": (-10, -16), "Ensamble calibrado": (-10, 10),
        "DeBERTa-v3": (-10, -16)}
for _, r in res.iterrows():
    ax.scatter(r["Entrenamiento (s)"], r["MAP@3"], s=180, color=PALETA[r["Modelo"]], edgecolor="#1F2933", zorder=3)
    dx, dy = desp.get(r["Modelo"], (10, -4))
    ax.annotate(r["Modelo"], (r["Entrenamiento (s)"], r["MAP@3"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=9, ha="right" if dx < 0 else "left")
ax.set_xscale("log"); ax.set_xlabel("Tiempo de entrenamiento (s, escala logarítmica)")
ax.set_ylabel("MAP@3 en prueba"); ax.set_title("Eficiencia: calidad frente a costo de entrenamiento")
plt.tight_layout(); fig.savefig(FIGS / "desempeno_vs_tiempo.png", bbox_inches="tight"); plt.close(fig)

cats = ["True_Correct", "True_Neither", "True_Misconception", "False_Correct", "False_Neither", "False_Misconception"]
cm = confusion_matrix([s.split(":")[0] for s in y_te], [s.split(":")[0] for s in pred], labels=cats, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cats, yticklabels=cats, ax=ax,
            vmin=0, vmax=1, cbar_kws={"label": "Proporción de la fila"})
ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
ax.set_title("Matriz de confusión de Category — " + ("Ensamble calibrado" if HAY_DEB
             else "TF-IDF + Regresión logística jerárquica"))
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
fig.savefig(FIGS / "matriz_confusion_category.png", bbox_inches="tight"); plt.close(fig)

tr_counts = df.iloc[np.setdiff1d(np.arange(len(df)), idx_te)]["target"].value_counts()
etiquetas = np.unique(y_te)
fig, ax = plt.subplots(figsize=(8.5, 5))
for nombre, p in ((("TF-IDF + Regresión logística jerárquica", p_lr_te),) + ((("Ensamble calibrado", p_ens_te),) if HAY_DEB else ())):
    f1s = f1_score(y_te, mm.top_k(p, CLASES)[:, 0], labels=etiquetas, average=None, zero_division=0)
    ax.scatter(tr_counts.reindex(etiquetas).fillna(0).values + 1, f1s, s=38, alpha=.8,
               color=PALETA[nombre], edgecolor="#1F2933", label=nombre)
ax.set_xscale("log"); ax.axvline(50, ls="--", color="#9AA5B1", label="50 ejemplos")
ax.set_xlabel("Ejemplos de la clase en entrenamiento (log)"); ax.set_ylabel("F1 en prueba")
ax.set_title("F1 por clase frente a su frecuencia"); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(FIGS / "f1_vs_soporte.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
colores_b = ["#9AA5B1", "#7FB3D5", "#2457A7", "#F2B84B"][:len(tb)]
ax.barh(tb["Modelo"], tb["MAP@3 preguntas no vistas"], color=colores_b)
for i, v in enumerate(tb["MAP@3 preguntas no vistas"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1); ax.set_xlabel("MAP@3 (GroupKFold por QuestionId, 3 pliegues)")
ax.set_title("Validación B: generalización a preguntas no vistas")
plt.tight_layout(); fig.savefig(FIGS / "generalizacion_preguntas.png", bbox_inches="tight"); plt.close(fig)

ab = pd.DataFrame(lin["ablacion"])
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(ab["Entrada"], ab["MAP@3"], color=["#9AA5B1", "#7FB3D5", "#2457A7"])
for i, v in enumerate(ab["MAP@3"]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(max(0, ab["MAP@3"].min() - 0.1), 1); ax.set_ylabel("MAP@3 en prueba")
ax.set_title("Ablación del contexto de entrada (TF-IDF + regresión logística)")
plt.tight_layout(); fig.savefig(FIGS / "ablacion_contexto.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], ls="--", color="#9AA5B1", label="Calibración perfecta")
curvas_cal = [("TF-IDF + RL jerárquica", p_lr_te, "#2457A7")]
if HAY_DEB:
    curvas_cal += [("DeBERTa-v3 sin calibrar", mm.softmax(z_te), "#E07A5F"),
                   ("DeBERTa-v3 con temperatura", p_deb_te, "#F2B84B"),
                   ("Ensamble calibrado", p_ens_te, "#2F9B77")]
for nombre, p, color in curvas_cal:
    conf, ok = p.max(1), p.argmax(1) == iy_te
    bordes = np.linspace(0, 1, 11)
    xs, ys = [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append(conf[m].mean()); ys.append(ok[m].mean())
    ax.plot(xs, ys, marker="o", color=color, label=f"{nombre} (ECE {mm.ece(p, iy_te):.3f})")
ax.set_xlabel("Confianza media de la primera etiqueta"); ax.set_ylabel("Exactitud observada")
ax.set_title("Diagrama de confiabilidad (calibración)"); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(FIGS / "calibracion.png", bbox_inches="tight"); plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.plot(cv_te["umbral"], cv_te["exactitud aceptados"], color="#2457A7", label="Exactitud de los aceptados")
ax1.plot(cv_te["umbral"], cv_te["cobertura"], color="#F2B84B", label="Cobertura (casos aceptados)")
ax1.axvline(TAU, ls="--", color="#2F9B77", label=f"Umbral elegido τ = {TAU:.2f}")
ax1.set_xlabel("Umbral de confianza"); ax1.set_ylabel("Proporción"); ax1.set_ylim(0, 1.02)
ax1.set_title("Abstención: casos derivados al docente según el umbral"); ax1.legend(fontsize=8, loc="lower left")
plt.tight_layout(); fig.savefig(FIGS / "abstencion.png", bbox_inches="tight"); plt.close(fig)

hist = pd.DataFrame(deb["historia"]) if HAY_DEB else None
fig, ax = plt.subplots(figsize=(7, 4)) if HAY_DEB else (None, None)
if HAY_DEB:
    ax.plot(hist["época"], hist["MAP@3 val"], marker="o", color="#F2B84B", label="MAP@3 validación")
    ax2 = ax.twinx(); ax2.plot(hist["época"], hist["pérdida"], marker="s", color="#2457A7", label="Pérdida de entrenamiento")
    ax.set_xlabel("Época"); ax.set_ylabel("MAP@3 validación"); ax2.set_ylabel("Pérdida")
    ax.set_xticks(hist["época"]); ax.set_title("Curva de aprendizaje de DeBERTa-v3")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right"); ax2.grid(False)
    plt.tight_layout(); fig.savefig(FIGS / "curva_deberta.png", bbox_inches="tight"); plt.close(fig)

bq = pd.read_csv(OUT / "busqueda_hiperparametros.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, nombre, color in zip(axes, ["TF-IDF + SVM lineal", "TF-IDF + Regresión logística"], ["#28A9C7", "#2457A7"]):
    s2 = bq[bq["Modelo"] == nombre].copy()
    s2["hp"] = s2["Hiperparámetros"].apply(json.loads)
    for w, estilo in ((None, "-"), ("balanced", "--")):
        s3 = s2[[h["class_weight"] == w for h in s2["hp"]]]
        ax.errorbar([h["C"] for h in s3["hp"]], s3["MAP@3 CV (media)"], yerr=s3["MAP@3 CV (desv.)"],
                    marker="o", ls=estilo, color=color, capsize=3, label=f"class_weight={w}")
    ax.set_xscale("log"); ax.set_xlabel("C (inverso de la regularización)")
    ax.set_ylabel("MAP@3 (CV, 3 pliegues)"); ax.set_title(nombre); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(FIGS / "busqueda_hiperparametros.png", bbox_inches="tight"); plt.close(fig)

if HAY_DEB:
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(alfas, curva_alfa, marker="o", color="#2F9B77")
  ax.axvline(ALFA, ls="--", color="#9AA5B1")
  ax.set_xlabel("α (peso de DeBERTa en el ensamble)"); ax.set_ylabel("MAP@3 validación")
  ax.set_title("Selección del peso del ensamble")
  plt.tight_layout(); fig.savefig(FIGS / "peso_ensamble.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(cm, index=cats, columns=cats).to_csv(OUT / "confusion_category.csv")

# ---------------------------------------------------------------------------
# 6. Datos auxiliares para la aplicación (sin redistribuir los textos originales)
# ---------------------------------------------------------------------------
DATOS_APP = mm.ROOT / "app" / "datos"
DATOS_APP.mkdir(parents=True, exist_ok=True)
filas_q = []
for q, g in df.groupby("QuestionId"):
    ver = g.groupby("MC_Answer_clean")["Veracidad"].agg(lambda s: s.value_counts().index[0])
    filas_q.append({"QuestionId": q, "Pregunta": g["Question_clean"].iloc[0],
                    "Opciones": json.dumps(sorted(ver.index.tolist()), ensure_ascii=False),
                    "Correcta": ver[ver == "True"].index[0] if (ver == "True").any() else ""})
pd.DataFrame(filas_q).to_csv(DATOS_APP / "preguntas.csv", index=False)
e = df["Explanation_clean"]
(df.assign(n_palabras=e.str.split().str.len(), n_caracteres=e.str.len(), n_digitos=e.str.count(r"\d"),
           n_simbolos=e.str.count(r"[+\-*/=<>%]"),
           diversidad_lexica=e.str.split().apply(lambda w: len(set(w))) / e.str.split().str.len().clip(lower=1))
   [["QuestionId", "MC_Answer_clean", "Category", "Misconception", "target", "Veracidad", "Tipo",
     "n_palabras", "n_caracteres", "n_digitos", "n_simbolos", "diversidad_lexica"]]
   .to_csv(DATOS_APP / "variables_limpias.csv.gz", index=False))
print("Figuras y datos de la aplicación listos.")

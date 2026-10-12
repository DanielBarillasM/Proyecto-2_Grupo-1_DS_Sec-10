"""Modelos de línea base TF-IDF para el reto MAP.

1. Ajuste de hiperparámetros (búsqueda en malla, 3 pliegues agrupados) — opcional con --buscar.
2. Modelos finales sobre el entrenamiento completo y evaluación en prueba (validación A).
3. Ablación del contexto de entrada: texto solo / + pregunta / + pregunta y opción.
4. Validación B: GroupKFold por QuestionId (preguntas no vistas).
5. Probabilidades de validación y prueba para el ensamble (notebooks/04_ensamble.py).

Uso, desde la raíz del repositorio con los CSV en data/:
    python notebooks/02_modelado.py [--buscar]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import map_modelos as mm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--buscar", action="store_true", help="repite la búsqueda de hiperparámetros")
args = ap.parse_args()

OUT = mm.ROOT / "results"
MODELS = mm.ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = mm.cargar_train()
CLASES = np.array(sorted(df["target"].unique()))
idx_full, idx_sub, idx_val, idx_te = mm.particiones(df)
tr, sub, val, te = (df.iloc[i].reset_index(drop=True) for i in (idx_full, idx_sub, idx_val, idx_te))
print(f"Entrenamiento {len(tr):,} (de ellas {len(val):,} de validación) · prueba {len(te):,}")


def evaluar(modelo, d_tr, d_te):
    t0 = time.perf_counter()
    modelo.fit(d_tr, d_tr["target"].values)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = mm.alinear(modelo.scores(d_te), modelo.classes_, CLASES)
    t_pred = time.perf_counter() - t0
    m = mm.metricas(d_te["target"].values, mm.top_k(s, CLASES))
    m.update({"Entrenamiento (s)": t_fit, "Predicción (s)": t_pred})
    return m, s


# ---------------------------------------------------------------------------
# 1. Búsqueda de hiperparámetros (3 pliegues agrupados dentro del entrenamiento)
# ---------------------------------------------------------------------------
MALLAS = {
    "TF-IDF + SVM lineal": [{"C": c, "class_weight": w} for c in (0.1, 0.3, 1.0)
                            for w in (None, "balanced")],
    "TF-IDF + Regresión logística": [{"C": c, "class_weight": None} for c in (1.0, 4.0, 16.0)]
    + [{"C": 4.0, "class_weight": "balanced"}],
}
ruta_busqueda = OUT / "busqueda_hiperparametros.csv"
if args.buscar or not ruta_busqueda.exists():
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=mm.RANDOM_STATE)
    pliegues = list(cv.split(tr, tr["target"], groups=mm.grupos_duplicados(tr)))
    filas = []
    for nombre, malla in MALLAS.items():
        for hp in malla:
            puntajes = Parallel(n_jobs=3)(
                delayed(lambda a, b: evaluar(mm.crear(nombre, **hp), tr.iloc[a], tr.iloc[b])[0]["MAP@3"])(a, b)
                for a, b in pliegues)
            filas.append({"Modelo": nombre, "Hiperparámetros": json.dumps(hp),
                          "MAP@3 CV (media)": float(np.mean(puntajes)),
                          "MAP@3 CV (desv.)": float(np.std(puntajes))})
            print(f"  [CV] {nombre} {hp} -> {filas[-1]['MAP@3 CV (media)']:.4f}", flush=True)
    pd.DataFrame(filas).to_csv(ruta_busqueda, index=False)
busqueda = pd.read_csv(ruta_busqueda)
busqueda["Modelo"] = busqueda["Modelo"].replace({"SVM lineal": "TF-IDF + SVM lineal",
                                                 "Regresión logística": "TF-IDF + Regresión logística"})
busqueda = busqueda[busqueda["Modelo"].isin(MALLAS)]
busqueda.to_csv(ruta_busqueda, index=False)
mejores = {m: json.loads(busqueda[busqueda["Modelo"] == m]
                         .sort_values("MAP@3 CV (media)").iloc[-1]["Hiperparámetros"]) for m in MALLAS}
mejores["TF-IDF + Regresión logística jerárquica"] = mejores["TF-IDF + Regresión logística"]
mejores["Prior (pregunta+opción)"] = {}
print("Mejores hiperparámetros:", mejores)

# ---------------------------------------------------------------------------
# 2. Modelos finales — validación A
# ---------------------------------------------------------------------------
FINALES = ["Prior (pregunta+opción)", "TF-IDF + SVM lineal", "TF-IDF + Regresión logística jerárquica"]
resultados = []
for nombre in FINALES:
    modelo = mm.crear(nombre, **mejores[nombre])
    m, s = evaluar(modelo, tr, te)
    resultados.append({"Modelo": nombre, "Hiperparámetros": json.dumps(mejores[nombre]), **m})
    slug = {"Prior (pregunta+opción)": "prior", "TF-IDF + SVM lineal": "svm",
            "TF-IDF + Regresión logística jerárquica": "logistica"}[nombre]
    joblib.dump(modelo, MODELS / f"{slug}.joblib", compress=3)
    np.save(OUT / f"{slug}_scores_test.npy", s.astype(np.float32))
    print(f"[TEST] {nombre:<42} MAP@3={m['MAP@3']:.4f} Acc={m['Accuracy']:.4f} "
          f"MacroF1={m['Macro-F1']:.4f} fit={m['Entrenamiento (s)']:.0f}s", flush=True)
pd.DataFrame(resultados).to_csv(OUT / "metricas_lineales.csv", index=False)

# Probabilidades de validación para calibración y ensamble (modelo entrenado sin validación).
lr_sub = mm.crear("TF-IDF + Regresión logística jerárquica", **mejores["TF-IDF + Regresión logística"])
lr_sub.fit(sub, sub["target"].values)
np.save(OUT / "logistica_scores_val.npy",
        mm.alinear(lr_sub.scores(val), lr_sub.classes_, CLASES).astype(np.float32))
np.save(OUT / "clases.npy", CLASES)
del lr_sub

# ---------------------------------------------------------------------------
# 3. Ablación del contexto de entrada (regresión logística, validación A)
# ---------------------------------------------------------------------------
ablacion = []
for contexto, etiqueta in (("ninguno", "Texto solo"), ("pregunta", "Texto + pregunta"),
                           ("pregunta+opcion", "Texto + pregunta + opción")):
    m, _ = evaluar(mm.crear("TF-IDF + Regresión logística", contexto,
                            **mejores["TF-IDF + Regresión logística"]), tr, te)
    ablacion.append({"Entrada": etiqueta, **m})
    print(f"[ABL] {etiqueta:<28} MAP@3={m['MAP@3']:.4f}", flush=True)

# ---------------------------------------------------------------------------
# 4. Validación B — preguntas no vistas (GroupKFold por QuestionId, 3 pliegues)
# ---------------------------------------------------------------------------
grupos = []
CONFIG_B = [("Prior (pregunta+opción)", "pregunta+opcion", "Prior (pregunta+opción)"),
            ("TF-IDF + Regresión logística", "ninguno", "TF-IDF + RL (texto solo)"),
            ("TF-IDF + Regresión logística", "pregunta+opcion", "TF-IDF + RL (texto + pregunta + opción)")]
for nombre, contexto, etiqueta in CONFIG_B:
    vals, cats = [], []
    for a, b in GroupKFold(n_splits=3).split(df, groups=df["QuestionId"]):
        m, s = evaluar(mm.crear(nombre, contexto, **mejores.get(nombre, {})), df.iloc[a], df.iloc[b])
        vals.append(m["MAP@3"])
        pred = mm.top_k(s, CLASES)[:, 0]
        cats.append(accuracy_score(df["Category"].iloc[b], [p.split(":")[0] for p in pred]))
    grupos.append({"Modelo": etiqueta, "MAP@3 preguntas no vistas": float(np.mean(vals)),
                   "Accuracy Category preguntas no vistas": float(np.mean(cats))})
    print(f"[B] {etiqueta:<40} MAP@3={np.mean(vals):.4f}", flush=True)

resumen = {"n_train": len(tr), "n_sub": len(sub), "n_val": len(val), "n_test": len(te),
           "n_clases": len(CLASES), "clases_prueba": int(te["target"].nunique()),
           "mejores_hiperparametros": mejores, "prueba": resultados,
           "ablacion": ablacion, "transferencia": grupos}
(OUT / "resumen_lineales.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
print("Listo.")

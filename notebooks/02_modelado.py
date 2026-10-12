"""Modelado supervisado del reto MAP — Charting Student Math Misunderstandings.

Entrena y compara varios clasificadores sobre la etiqueta compuesta
`Category:Misconception`, ajusta hiperparámetros con validación cruzada
agrupada y exporta métricas, modelos y figuras para el informe.

Uso (desde la raíz del repositorio, con los CSV en data/):
    python notebooks/02_modelado.py
"""

import json
import re
import time
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = next(p for p in (ROOT / "data", ROOT / "data" / "raw") if (p / "train.csv").exists())
FIGS = ROOT / "report" / "assets" / "modelos"
OUT = ROOT / "results"
MODELS = ROOT / "models"
for carpeta in (FIGS, OUT, MODELS):
    carpeta.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.titlesize": 12})

# Paleta coherente con el informe y la presentación (azul UVG + acentos).
PALETA = {
    "Baseline (pregunta+opción)": "#9AA5B1",
    "Complement Naive Bayes": "#F2B84B",
    "SVM lineal": "#28A9C7",
    "Regresión logística": "#2457A7",
    "Regresión logística jerárquica": "#2F9B77",
}


# ---------------------------------------------------------------------------
# 1. Carga y preprocesamiento (mismas decisiones que el EDA)
# ---------------------------------------------------------------------------
def limpiar_latex(t: str) -> str:
    t = str(t)
    t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("\\(", " ").replace("\\)", " ")
    t = re.sub(r"[{}$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza una tabla con QuestionId, MC_Answer y StudentExplanation."""
    df = df.copy()
    df["MC_Answer_clean"] = df["MC_Answer"].apply(limpiar_latex)
    df["Explanation_clean"] = (
        df["StudentExplanation"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["QA"] = df["QuestionId"].astype(str) + "|" + df["MC_Answer_clean"]
    return df


train = pd.read_csv(DATA / "train.csv")
train["Misconception"] = train["Misconception"].fillna("NA")
train["target"] = train["Category"] + ":" + train["Misconception"]
train["Veracidad"] = train["Category"].str.split("_").str[0]
df = preprocesar(train)
print(f"Datos: {len(df):,} filas · {df['target'].nunique()} clases")


# ---------------------------------------------------------------------------
# 2. Partición entrenamiento / prueba
#    StratifiedGroupKFold: estratifica por etiqueta y mantiene juntos los
#    textos idénticos de una misma pregunta (evita fuga por duplicados).
# ---------------------------------------------------------------------------
grupos_dup = (df["QuestionId"].astype(str) + "|" + df["Explanation_clean"]).factorize()[0]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
idx_tr, idx_te = next(sgkf.split(df, df["target"], groups=grupos_dup))
tr, te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
grupos_tr = grupos_dup[idx_tr]
print(f"Entrenamiento: {len(tr):,} · Prueba: {len(te):,}")


# ---------------------------------------------------------------------------
# 3. Representación: TF-IDF de palabras + caracteres + contexto categórico
# ---------------------------------------------------------------------------
class Featurizer:
    """TF-IDF de palabras (1-2) y caracteres (2-5) + one-hot de pregunta y opción."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b|[+\-*/=<>%]")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                    sublinear_tf=True, max_features=120_000)
        self.ohe = OneHotEncoder(handle_unknown="ignore")

    def fit(self, d):
        self.word.fit(d["Explanation_clean"])
        self.char.fit(d["Explanation_clean"])
        self.ohe.fit(d[["QuestionId", "QA"]].astype(str))
        return self

    def transform(self, d):
        return sparse.hstack([
            self.word.transform(d["Explanation_clean"]),
            self.char.transform(d["Explanation_clean"]),
            self.ohe.transform(d[["QuestionId", "QA"]].astype(str)),
        ]).tocsr()


# ---------------------------------------------------------------------------
# 4. Métricas
# ---------------------------------------------------------------------------
def top_k(scores, classes, k=3):
    orden = np.argsort(-scores, axis=1)[:, :k]
    return classes[orden]


def map_at_3(y_true, top3):
    y_true = np.asarray(y_true)
    pesos = np.array([1.0, 1 / 2, 1 / 3])
    aciertos = top3 == y_true[:, None]
    return float((aciertos * pesos).sum(axis=1).mean())


def metricas(y_true, top3):
    y_pred = top3[:, 0]
    return {
        "MAP@3": map_at_3(y_true, top3),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Top-3 accuracy": float((top3 == np.asarray(y_true)[:, None]).any(axis=1).mean()),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# 5. Modelos
# ---------------------------------------------------------------------------
class BaselinePreguntaOpcion:
    """Sin texto: ordena las etiquetas por frecuencia dentro de (pregunta, opción)."""

    def fit(self, d, y):
        self.classes_ = np.array(sorted(pd.unique(y)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        tabla = pd.crosstab(d["QA"].values, np.asarray(y))
        self.tabla = {qa: np.array([fila.get(c, 0) for c in self.classes_], float)
                      for qa, fila in tabla.iterrows()}
        glob = pd.Series(y).value_counts()
        self.glob = np.array([glob.get(c, 0) for c in self.classes_], float)
        del idx
        return self

    def scores(self, d):
        return np.vstack([self.tabla.get(qa, self.glob) for qa in d["QA"]])


class ModeloTexto:
    """Featurizer + clasificador lineal de scikit-learn."""

    def __init__(self, estimador):
        self.est = estimador

    def fit(self, d, y):
        self.feat = Featurizer().fit(d)
        self.est.fit(self.feat.transform(d), y)
        self.classes_ = self.est.classes_
        return self

    def scores(self, d):
        X = self.feat.transform(d)
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(X)
        return self.est.decision_function(X)


class LogisticaJerarquica(ModeloTexto):
    """Regresión logística + restricción jerárquica.

    1) La veracidad se obtiene de la tabla (pregunta, opción) — casi determinista
       según el EDA (58/60 pares).
    2) Solo se permiten etiquetas compatibles con esa veracidad y que se hayan
       observado para la misma pregunta en entrenamiento (las concepciones
       erróneas están ligadas a la pregunta: 32/35 exclusivas).
    """

    def fit(self, d, y):
        super().fit(d, y)
        y = pd.Series(np.asarray(y))
        self.veracidad = (d.assign(v=y.str.split("_").str[0].values)
                          .groupby("QA")["v"].agg(lambda s: s.value_counts().index[0]).to_dict())
        self.permitidas = {}
        for q, sub in pd.DataFrame({"q": d["QuestionId"].values, "y": y.values}).groupby("q"):
            self.permitidas[q] = set(sub["y"])
        return self

    def scores(self, d):
        p = super().scores(d).copy()
        ver = np.array([c.split("_")[0] for c in self.classes_])
        for i, (q, qa) in enumerate(zip(d["QuestionId"], d["QA"])):
            mascara = np.ones(len(self.classes_), bool)
            if q in self.permitidas:
                mascara &= np.isin(self.classes_, list(self.permitidas[q]))
            if qa in self.veracidad:
                mascara &= ver == self.veracidad[qa]
            if mascara.any():
                p[i, ~mascara] = 0.0
        return p / p.sum(axis=1, keepdims=True).clip(1e-12)


def crear(nombre, **hp):
    if nombre == "Baseline (pregunta+opción)":
        return BaselinePreguntaOpcion()
    if nombre == "Complement Naive Bayes":
        return ModeloTexto(ComplementNB(alpha=hp.get("alpha", 0.1)))
    if nombre == "SVM lineal":
        return ModeloTexto(LinearSVC(C=hp.get("C", 0.3), class_weight=hp.get("class_weight"),
                                     max_iter=5000, random_state=RANDOM_STATE))
    lr = LogisticRegression(C=hp.get("C", 4.0), class_weight=hp.get("class_weight"),
                            max_iter=3000, random_state=RANDOM_STATE)
    if nombre == "Regresión logística":
        return ModeloTexto(lr)
    if nombre == "Regresión logística jerárquica":
        return LogisticaJerarquica(lr)
    raise ValueError(nombre)


def evaluar(modelo, d_tr, y_tr, d_te, y_te):
    t0 = time.perf_counter()
    modelo.fit(d_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = modelo.scores(d_te)
    t_pred = time.perf_counter() - t0
    top3 = top_k(s, np.asarray(modelo.classes_))
    m = metricas(y_te, top3)
    m.update({"Entrenamiento (s)": t_fit, "Predicción (s)": t_pred})
    return m, top3


# ---------------------------------------------------------------------------
# 6. Ajuste de hiperparámetros: búsqueda en malla, 3 pliegues agrupados
# ---------------------------------------------------------------------------
MALLAS = {
    "Complement Naive Bayes": [{"alpha": a} for a in (0.01, 0.03, 0.1, 0.3, 1.0)],
    "SVM lineal": [{"C": c, "class_weight": w} for c in (0.1, 0.3, 1.0) for w in (None, "balanced")],
    "Regresión logística": [{"C": c, "class_weight": None} for c in (1.0, 4.0, 16.0)]
    + [{"C": 4.0, "class_weight": "balanced"}],
}
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
pliegues = list(cv.split(tr, tr["target"], groups=grupos_tr))

busqueda, mejores = [], {}
for nombre, malla in MALLAS.items():
    for hp in malla:
        puntajes = Parallel(n_jobs=3)(
            delayed(lambda a, b: evaluar(crear(nombre, **hp), tr.iloc[a], tr["target"].iloc[a].values,
                                         tr.iloc[b], tr["target"].iloc[b].values)[0]["MAP@3"])(a, b)
            for a, b in pliegues)
        fila = {"Modelo": nombre, "Hiperparámetros": json.dumps(hp),
                "MAP@3 CV (media)": float(np.mean(puntajes)), "MAP@3 CV (desv.)": float(np.std(puntajes))}
        busqueda.append(fila)
        print(f"  [CV] {nombre:<26} {hp} -> {fila['MAP@3 CV (media)']:.4f}")
    mejor = max((f for f in busqueda if f["Modelo"] == nombre), key=lambda f: f["MAP@3 CV (media)"])
    mejores[nombre] = json.loads(mejor["Hiperparámetros"])
mejores["Regresión logística jerárquica"] = mejores["Regresión logística"]
mejores["Baseline (pregunta+opción)"] = {}
pd.DataFrame(busqueda).to_csv(OUT / "busqueda_hiperparametros.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Evaluación final en el conjunto de prueba retenido
# ---------------------------------------------------------------------------
resultados, top3s, modelos = [], {}, {}
for nombre in PALETA:
    modelo = crear(nombre, **mejores[nombre])
    m, top3 = evaluar(modelo, tr, tr["target"].values, te, te["target"].values)
    m = {"Modelo": nombre, "Hiperparámetros": json.dumps(mejores[nombre]), **m}
    resultados.append(m)
    top3s[nombre], modelos[nombre] = top3, modelo
    print(f"[TEST] {nombre:<32} MAP@3={m['MAP@3']:.4f}  Acc={m['Accuracy']:.4f}  "
          f"MacroF1={m['Macro-F1']:.4f}  fit={m['Entrenamiento (s)']:.1f}s")
res = pd.DataFrame(resultados)
res.to_csv(OUT / "metricas_prueba.csv", index=False)

mejor_nombre = res.sort_values("MAP@3", ascending=False).iloc[0]["Modelo"]
print(f"Mejor modelo: {mejor_nombre}")

# Métricas por dimensión jerárquica para el mejor modelo.
y_te = te["target"].values
pred = top3s[mejor_nombre][:, 0]
partes = lambda arr, i: np.array([re.split("[_:]", s)[i] for s in arr])  # noqa: E731
dim = {
    "Veracidad (True/False)": accuracy_score(partes(y_te, 0), partes(pred, 0)),
    "Category (6 clases)": accuracy_score([s.split(":")[0] for s in y_te],
                                          [s.split(":")[0] for s in pred]),
    "Etiqueta compuesta (65 clases)": accuracy_score(y_te, pred),
}
mis = np.array([s.split(":")[1] != "NA" for s in y_te])
dim["Misconception (solo filas con concepción)"] = accuracy_score(
    [s.split(":")[1] for s in y_te[mis]], [s.split(":")[1] for s in pred[mis]])


# ---------------------------------------------------------------------------
# 8. Transferencia a preguntas no vistas (GroupKFold por QuestionId)
# ---------------------------------------------------------------------------
transfer = []
for nombre in ["Baseline (pregunta+opción)", "Regresión logística"]:
    def pliegue_grupo(a, b):
        m, top3 = evaluar(crear(nombre, **mejores[nombre]), df.iloc[a], df["target"].iloc[a].values,
                          df.iloc[b], df["target"].iloc[b].values)
        cat_true = df["Category"].iloc[b].values
        return m["MAP@3"], accuracy_score(cat_true, [s.split(":")[0] for s in top3[:, 0]])

    pares = Parallel(n_jobs=3)(delayed(pliegue_grupo)(a, b) for a, b in
                               GroupKFold(n_splits=5).split(df, groups=df["QuestionId"]))
    vals, vals_cat = [p[0] for p in pares], [p[1] for p in pares]
    transfer.append({"Modelo": nombre, "MAP@3 preguntas no vistas": float(np.mean(vals)),
                     "Accuracy Category preguntas no vistas": float(np.mean(vals_cat))})
    print(f"[GROUP] {nombre:<26} MAP@3={np.mean(vals):.4f}")


# ---------------------------------------------------------------------------
# 9. Guardar modelos y resumen
# ---------------------------------------------------------------------------
for nombre, modelo in modelos.items():
    slug = re.sub(r"[^a-z0-9]+", "_", nombre.lower()
                  .translate(str.maketrans("áéíóúñ", "aeioun"))).strip("_")
    joblib.dump(modelo, MODELS / f"{slug}.joblib", compress=3)

por_pregunta = (pd.DataFrame({"QuestionId": te["QuestionId"], "y": y_te})
                .assign(acierto=[map_at_3([y], t[None, :]) for y, t in zip(y_te, top3s[mejor_nombre])])
                .groupby("QuestionId")["acierto"].mean())

resumen = {
    "n_train": int(len(tr)), "n_test": int(len(te)), "n_clases": int(df["target"].nunique()),
    "clases_prueba": int(te["target"].nunique()),
    "mejores_hiperparametros": mejores, "mejor_modelo": mejor_nombre,
    "prueba": resultados, "por_dimension": dim, "transferencia": transfer,
    "map3_por_pregunta": {str(k): float(v) for k, v in por_pregunta.items()},
}
(OUT / "resumen_modelado.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                            encoding="utf-8")


# ---------------------------------------------------------------------------
# 10. Figuras estáticas para el informe
# ---------------------------------------------------------------------------
orden = res.sort_values("MAP@3")["Modelo"].tolist()
colores = [PALETA[m] for m in orden]

# 10.1 Comparación de métricas
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, met in zip(axes, ["MAP@3", "Accuracy", "Macro-F1"]):
    vals = res.set_index("Modelo").loc[orden, met]
    ax.barh(orden, vals, color=colores)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_title(met)
fig.suptitle("Desempeño en el conjunto de prueba (20 %, 7,3 mil filas)", fontweight="bold")
plt.tight_layout()
fig.savefig(FIGS / "comparacion_metricas.png", bbox_inches="tight")
plt.close(fig)

# 10.2 Desempeño vs. tiempo de entrenamiento
fig, ax = plt.subplots(figsize=(8, 5))
for _, r in res.iterrows():
    ax.scatter(r["Entrenamiento (s)"], r["MAP@3"], s=180, color=PALETA[r["Modelo"]],
               edgecolor="#1F2933", zorder=3)
    desplazamiento = {"Regresión logística": (-10, -16), "Regresión logística jerárquica": (-10, 10)}
    dx, dy = desplazamiento.get(r["Modelo"], (10, -4))
    ax.annotate(r["Modelo"], (r["Entrenamiento (s)"], r["MAP@3"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=9, ha="right" if dx < 0 else "left")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel("Tiempo de entrenamiento (s, escala logarítmica)")
ax.set_ylabel("MAP@3 en prueba")
ax.set_title("Eficiencia: calidad frente a costo de entrenamiento")
plt.tight_layout()
fig.savefig(FIGS / "desempeno_vs_tiempo.png", bbox_inches="tight")
plt.close(fig)

# 10.3 Matriz de confusión de Category (mejor modelo)
cats = ["True_Correct", "True_Neither", "True_Misconception",
        "False_Correct", "False_Neither", "False_Misconception"]
cm = confusion_matrix([s.split(":")[0] for s in y_te], [s.split(":")[0] for s in pred],
                      labels=cats, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cats, yticklabels=cats,
            ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Proporción de la fila"})
ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
ax.set_title(f"Matriz de confusión de Category — {mejor_nombre}")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
fig.savefig(FIGS / "matriz_confusion_category.png", bbox_inches="tight")
plt.close(fig)

# 10.4 F1 por clase vs. soporte (efecto de la cola larga)
f1s = f1_score(y_te, pred, labels=np.unique(y_te), average=None, zero_division=0)
soporte = tr["target"].value_counts().reindex(np.unique(y_te)).fillna(0)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(soporte.values + 1, f1s, s=40, color="#2F9B77", alpha=.8, edgecolor="#1F2933")
ax.set_xscale("log")
ax.axvline(50, ls="--", color="#F2B84B", label="50 ejemplos")
ax.set_xlabel("Ejemplos de la clase en entrenamiento (log)")
ax.set_ylabel("F1 en prueba")
ax.set_title(f"F1 por clase frente a su frecuencia — {mejor_nombre}")
ax.legend(); plt.tight_layout()
fig.savefig(FIGS / "f1_vs_soporte.png", bbox_inches="tight")
plt.close(fig)

# 10.5 Validación estratificada vs. preguntas no vistas
tf = pd.DataFrame(transfer).set_index("Modelo")
comp = pd.DataFrame({
    "Preguntas conocidas\n(estratificada)": res.set_index("Modelo").loc[tf.index, "MAP@3"],
    "Preguntas no vistas\n(GroupKFold)": tf["MAP@3 preguntas no vistas"],
})
ax = comp.plot(kind="bar", figsize=(8, 4.8), color=["#2457A7", "#F2B84B"], rot=0, width=.7)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=9)
ax.set_ylabel("MAP@3"); ax.set_xlabel(""); ax.set_ylim(0, 1.2)
ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
ax.set_title("Generalización: preguntas conocidas vs. preguntas nuevas")
plt.tight_layout()
plt.savefig(FIGS / "generalizacion_preguntas.png", bbox_inches="tight")
plt.close()

# 10.6 Búsqueda de hiperparámetros
bq = pd.DataFrame(busqueda)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, nombre in zip(axes, MALLAS):
    sub = bq[bq["Modelo"] == nombre].copy()
    sub["hp"] = sub["Hiperparámetros"].apply(json.loads)
    if nombre == "Complement Naive Bayes":
        x = [h["alpha"] for h in sub["hp"]]
        ax.errorbar(x, sub["MAP@3 CV (media)"], yerr=sub["MAP@3 CV (desv.)"], marker="o",
                    color=PALETA[nombre], capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("alpha")
    else:
        for w, estilo in ((None, "-"), ("balanced", "--")):
            s2 = sub[[h["class_weight"] == w for h in sub["hp"]]]
            ax.errorbar([h["C"] for h in s2["hp"]], s2["MAP@3 CV (media)"], yerr=s2["MAP@3 CV (desv.)"],
                        marker="o", ls=estilo, color=PALETA[nombre], capsize=3,
                        label=f"class_weight={w}")
        ax.set_xscale("log"); ax.set_xlabel("C (inverso de la regularización)"); ax.legend(fontsize=8)
    ax.set_title(nombre); ax.set_ylabel("MAP@3 (CV, 3 pliegues)")
plt.tight_layout()
fig.savefig(FIGS / "busqueda_hiperparametros.png", bbox_inches="tight")
plt.close(fig)

print("Listo. Resultados en results/ y figuras en report/assets/modelos/")

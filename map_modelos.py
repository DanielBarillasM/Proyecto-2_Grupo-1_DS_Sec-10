"""Componentes compartidos del proyecto MAP.

Preprocesamiento, particiones, representación TF-IDF, modelos lineales y
métricas. Lo usan los scripts de entrenamiento (notebooks/) y la aplicación
Streamlit (app/), de modo que una explicación nueva se transforma exactamente
igual que los datos de entrenamiento y los modelos guardados con joblib pueden
cargarse desde cualquiera de ellos.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def ruta_datos() -> Path:
    for p in (ROOT / "data", ROOT / "data" / "raw"):
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("No se encontró data/train.csv (descárguelo con la API de Kaggle).")


# ---------------------------------------------------------------------------
# Preprocesamiento (mismas decisiones que el análisis exploratorio)
# ---------------------------------------------------------------------------
def limpiar_latex(t: str) -> str:
    """Convierte la notación LaTeX de las opciones a texto plano: \\frac{1}{3} -> 1/3."""
    t = str(t)
    t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("\\(", " ").replace("\\)", " ")
    t = re.sub(r"[{}$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza una tabla con QuestionId, QuestionText, MC_Answer y StudentExplanation."""
    df = df.copy()
    df["MC_Answer_clean"] = df["MC_Answer"].apply(limpiar_latex)
    df["Explanation_clean"] = (
        df["StudentExplanation"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["QA"] = df["QuestionId"].astype(str) + "|" + df["MC_Answer_clean"]
    if "QuestionText" in df:
        df["Question_clean"] = df["QuestionText"].apply(limpiar_latex)
    return df


def cargar_train() -> pd.DataFrame:
    train = pd.read_csv(ruta_datos() / "train.csv")
    train["Misconception"] = train["Misconception"].fillna("NA")
    train["target"] = train["Category"] + ":" + train["Misconception"]
    train["Veracidad"] = train["Category"].str.split("_").str[0]
    train["Tipo"] = train["Category"].str.split("_").str[1]
    return preprocesar(train)


def texto_transformer(df: pd.DataFrame) -> list:
    """Entrada contextual: pregunta + opción + explicación."""
    return ("Question: " + df["Question_clean"] + " Answer: " + df["MC_Answer_clean"]
            + " Explanation: " + df["StudentExplanation"].fillna("").astype(str)).tolist()


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def grupos_duplicados(df: pd.DataFrame) -> np.ndarray:
    """Agrupa explicaciones idénticas de una misma pregunta (evita fuga por duplicados)."""
    return (df["QuestionId"].astype(str) + "|" + df["Explanation_clean"]).factorize()[0]


def particiones(df: pd.DataFrame):
    """Devuelve índices (entrenamiento, validación, prueba).

    Prueba = 20 % (StratifiedGroupKFold de 5 pliegues, pliegue 0).
    Validación = 10 % del entrenamiento (StratifiedGroupKFold de 10 pliegues),
    usada para detener el Transformer, calibrar y pesar el ensamble.
    """
    g = grupos_duplicados(df)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    idx_tr, idx_te = next(sgkf.split(df, df["target"], groups=g))
    sub = df.iloc[idx_tr]
    sgkf2 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    a, b = next(sgkf2.split(sub, sub["target"], groups=g[idx_tr]))
    return idx_tr, idx_tr[a], idx_tr[b], idx_te  # train completo, train sin val, val, test


# ---------------------------------------------------------------------------
# Representación TF-IDF
# ---------------------------------------------------------------------------
class Featurizer:
    """TF-IDF de palabras (1-2) y caracteres (2-5) + contexto categórico opcional.

    contexto: "ninguno" (texto solo), "pregunta" (one-hot de QuestionId) o
    "pregunta+opcion" (one-hot de QuestionId y del par pregunta-opción).
    """

    def __init__(self, contexto="pregunta+opcion"):
        self.contexto = contexto
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b|[+\-*/=<>%]")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                    sublinear_tf=True, max_features=120_000)
        self.ohe = OneHotEncoder(handle_unknown="ignore")

    def _cols(self):
        return {"ninguno": [], "pregunta": ["QuestionId"],
                "pregunta+opcion": ["QuestionId", "QA"]}[self.contexto]

    def fit(self, d):
        self.word.fit(d["Explanation_clean"])
        self.char.fit(d["Explanation_clean"])
        if self._cols():
            self.ohe.fit(d[self._cols()].astype(str))
        return self

    def transform(self, d):
        bloques = [self.word.transform(d["Explanation_clean"]),
                   self.char.transform(d["Explanation_clean"])]
        if self._cols():
            bloques.append(self.ohe.transform(d[self._cols()].astype(str)))
        return sparse.hstack(bloques).tocsr()


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class BaselinePreguntaOpcion:
    """Sin texto: ordena las etiquetas por frecuencia dentro de (pregunta, opción)."""

    nombre_corto = "Prior"

    def fit(self, d, y):
        self.classes_ = np.array(sorted(pd.unique(y)))
        tabla = pd.crosstab(d["QA"].values, np.asarray(y))
        self.tabla = {qa: np.array([fila.get(c, 0) for c in self.classes_], float)
                      for qa, fila in tabla.iterrows()}
        glob = pd.Series(y).value_counts()
        self.glob = np.array([glob.get(c, 0) for c in self.classes_], float)
        return self

    def scores(self, d):
        s = np.vstack([self.tabla.get(qa, self.glob) for qa in d["QA"]])
        return s / s.sum(axis=1, keepdims=True).clip(1e-12)


class ModeloTexto:
    """Featurizer + clasificador lineal de scikit-learn."""

    def __init__(self, estimador, contexto="pregunta+opcion"):
        self.est = estimador
        self.contexto = contexto

    def fit(self, d, y):
        self.feat = Featurizer(self.contexto).fit(d)
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

    1) La veracidad se toma de la tabla (pregunta, opción), casi determinista.
    2) Solo se permiten etiquetas compatibles con esa veracidad y observadas
       para la misma pregunta en entrenamiento.
    """

    def fit(self, d, y):
        super().fit(d, y)
        y = pd.Series(np.asarray(y))
        self.veracidad = (d.assign(v=y.str.split("_").str[0].values)
                          .groupby("QA")["v"].agg(lambda s: s.value_counts().index[0]).to_dict())
        self.permitidas = {q: set(sub["y"]) for q, sub in
                           pd.DataFrame({"q": d["QuestionId"].values, "y": y.values}).groupby("q")}
        return self

    def mascara(self, d):
        ver = np.array([c.split("_")[0] for c in self.classes_])
        m = np.ones((len(d), len(self.classes_)), bool)
        for i, (q, qa) in enumerate(zip(d["QuestionId"], d["QA"])):
            if q in self.permitidas:
                m[i] &= np.isin(self.classes_, list(self.permitidas[q]))
            if qa in self.veracidad:
                m[i] &= ver == self.veracidad[qa]
            if not m[i].any():
                m[i] = True
        return m

    def scores(self, d):
        p = super().scores(d) * self.mascara(d)
        return p / p.sum(axis=1, keepdims=True).clip(1e-12)


def crear(nombre, contexto="pregunta+opcion", **hp):
    if nombre == "Prior (pregunta+opción)":
        return BaselinePreguntaOpcion()
    if nombre == "TF-IDF + SVM lineal":
        return ModeloTexto(LinearSVC(C=hp.get("C", 0.3), class_weight=hp.get("class_weight"),
                                     max_iter=5000, random_state=RANDOM_STATE), contexto)
    lr = LogisticRegression(C=hp.get("C", 4.0), class_weight=hp.get("class_weight"),
                            max_iter=3000, random_state=RANDOM_STATE)
    if nombre == "TF-IDF + Regresión logística":
        return ModeloTexto(lr, contexto)
    if nombre == "TF-IDF + Regresión logística jerárquica":
        return LogisticaJerarquica(lr, contexto)
    raise ValueError(nombre)


def alinear(scores, clases_modelo, clases_globales):
    """Reordena una matriz de puntajes al orden de clases_globales (columnas faltantes = 0)."""
    pos = {c: i for i, c in enumerate(clases_modelo)}
    out = np.zeros((scores.shape[0], len(clases_globales)))
    for j, c in enumerate(clases_globales):
        if c in pos:
            out[:, j] = scores[:, pos[c]]
    return out


def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def top_k(scores, classes, k=3):
    return np.asarray(classes)[np.argsort(-scores, axis=1)[:, :k]]


def map_at_3_filas(y_true, top3):
    pesos = np.array([1.0, 1 / 2, 1 / 3])
    return ((np.asarray(top3) == np.asarray(y_true)[:, None]) * pesos).sum(axis=1)


def map_at_3(y_true, top3):
    return float(map_at_3_filas(y_true, top3).mean())


def metricas(y_true, top3):
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = np.asarray(top3)[:, 0]
    return {
        "MAP@3": map_at_3(y_true, top3),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Top-3 accuracy": float((np.asarray(top3) == np.asarray(y_true)[:, None]).any(axis=1).mean()),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def ece(probs, y_idx, bins=15):
    """Error de calibración esperado (Expected Calibration Error) de la primera predicción."""
    conf = probs.max(axis=1)
    acierto = probs.argmax(axis=1) == y_idx
    bordes = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)

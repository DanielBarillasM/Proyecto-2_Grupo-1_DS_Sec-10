"""Aplicación Streamlit del proyecto MAP — Charting Student Math Misunderstandings.

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import map_modelos as mm  # noqa: E402

RES = ROOT / "results"
MODELS = ROOT / "models"
DATOS_APP = ROOT / "app" / "datos"

# Paleta: azul institucional como color principal, acentos análogos y un
# complementario cálido (ámbar) para resaltar; grises neutros para referencia.
AZUL, CIAN, VERDE, AMBAR, GRIS, ROJO = "#2457A7", "#28A9C7", "#2F9B77", "#F2B84B", "#9AA5B1", "#C8553D"
PALETA = {
    "Prior (pregunta+opción)": GRIS,
    "TF-IDF + SVM lineal": CIAN,
    "TF-IDF + Regresión logística jerárquica": AZUL,
    "DeBERTa-v3": AMBAR,
    "MathBERT": ROJO,
    "Ensamble calibrado": VERDE,
}
COLOR_CAT = {"True_Correct": "#2F9B77", "True_Neither": "#8CC7A1", "True_Misconception": "#F2B84B",
             "False_Correct": "#7FB3D5", "False_Neither": "#9AA5B1", "False_Misconception": "#C8553D"}
DESCRIPCION_CAT = {
    "True_Correct": "Opción correcta y razonamiento correcto.",
    "True_Neither": "Opción correcta, pero la explicación no permite evaluar el razonamiento.",
    "True_Misconception": "Opción correcta con un razonamiento erróneo (acertó por casualidad).",
    "False_Correct": "Opción incorrecta, aunque el razonamiento es correcto (descuido).",
    "False_Neither": "Opción incorrecta y explicación no evaluable.",
    "False_Misconception": "Opción incorrecta debida a una concepción errónea.",
}

st.set_page_config(page_title="MAP · Concepciones erróneas", page_icon="📐", layout="wide")
st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {AZUL}; }}
  .tarjeta {{ border-left: 6px solid var(--c); background: rgba(36,87,167,0.06);
             padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.4rem; }}
  .etq {{ font-weight: 700; }}
  .sub {{ font-size: 0.85rem; opacity: 0.8; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de datos y modelos (en caché)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    """Conjunto limpio. Usa data/train.csv si existe; si no, la tabla derivada sin textos."""
    try:
        df = mm.cargar_train()
        df["n_palabras"] = df["Explanation_clean"].str.split().str.len()
        df["n_caracteres"] = df["Explanation_clean"].str.len()
        df["n_digitos"] = df["Explanation_clean"].str.count(r"\d")
        df["n_simbolos"] = df["Explanation_clean"].str.count(r"[+\-*/=<>%]")
        df["diversidad_lexica"] = (df["Explanation_clean"].str.split().apply(lambda w: len(set(w)))
                                   / df["n_palabras"].clip(lower=1))
        return df, True
    except FileNotFoundError:
        return pd.read_csv(DATOS_APP / "variables_limpias.csv.gz"), False


@st.cache_data
def cargar_preguntas():
    return pd.read_csv(DATOS_APP / "preguntas.csv")


@st.cache_resource
def cargar_modelos():
    import joblib
    modelos = {}
    for nombre, archivo in (("Prior (pregunta+opción)", "prior.joblib"),
                            ("TF-IDF + SVM lineal", "svm.joblib"),
                            ("TF-IDF + Regresión logística jerárquica", "logistica.joblib")):
        if (MODELS / archivo).exists():
            modelos[nombre] = joblib.load(MODELS / archivo)
    return modelos


TRANSFORMERS = {"deberta": "DeBERTa-v3", "mathbert": "MathBERT"}


@st.cache_resource
def cargar_transformer(pref):
    """Transformer ajustado: local (models/<pref>_map) o desde Hugging Face Hub (secreto <PREF>_REPO)."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    origen = MODELS / f"{pref}_map"
    if not origen.exists():
        try:
            origen = st.secrets.get(f"{pref.upper()}_REPO", None)
        except Exception:
            origen = None
        if not origen:
            return None
    try:
        tok = AutoTokenizer.from_pretrained(str(origen))
        modelo = AutoModelForSequenceClassification.from_pretrained(str(origen)).float()
        modelo.eval()
        return tok, modelo
    except Exception:
        return None


@st.cache_data
def cargar_resultados():
    r = {"metricas": pd.read_csv(RES / "metricas_prueba.csv"),
         "lineales": json.loads((RES / "resumen_lineales.json").read_text(encoding="utf-8")),
         "final": json.loads((RES / "resumen_final.json").read_text(encoding="utf-8")),
         "transferencia": pd.read_csv(RES / "transferencia.csv"),
         "abstencion": pd.read_csv(RES / "curva_abstencion.csv"),
         "busqueda": pd.read_csv(RES / "busqueda_hiperparametros.csv")}
    ruta = RES / "deberta_resumen.json"
    r["deberta"] = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    ruta = RES / "confusion_category.csv"
    r["confusion"] = pd.read_csv(ruta, index_col=0) if ruta.exists() else None
    return r


CLASES = np.load(RES / "clases.npy", allow_pickle=True)
ENS = json.loads((MODELS / "ensamble.json").read_text()) if (MODELS / "ensamble.json").exists() else None
NOMBRE_T = (ENS or {}).get("nombre_transformer") or "DeBERTa-v3"


def predecir(entrada: pd.DataFrame, nombres):
    """Devuelve {modelo: vector de puntajes alineado a CLASES} para una sola fila."""
    modelos = cargar_modelos()
    d = mm.preprocesar(entrada)  # mismo preprocesamiento que en el entrenamiento
    salida = {}
    for n in nombres:
        if n in modelos:
            m = modelos[n]
            s = mm.alinear(m.scores(d), m.classes_, CLASES)[0]
            if n == "TF-IDF + SVM lineal":
                s = mm.softmax(s[None, :] * 5)[0]  # puntaje relativo (la SVM no estima probabilidades)
            salida[n] = s
    temps = cargar_resultados()["final"].get("transformers", {})
    pref_ens = (ENS or {}).get("transformer")
    for pref, nombre_t in TRANSFORMERS.items():
        usa_ens = "Ensamble calibrado" in nombres and pref == pref_ens
        if nombre_t not in nombres and not usa_ens:
            continue
        cargado = cargar_transformer(pref)
        if cargado is None:
            continue
        import torch
        tok, modelo = cargado
        with torch.no_grad():
            z = modelo(**tok(mm.texto_transformer(d), truncation=True, max_length=160,
                             return_tensors="pt")).logits.float().numpy()
        p_t = mm.softmax(z, temps.get(nombre_t, {}).get("temperatura", 1.0))[0]
        if nombre_t in nombres:
            salida[nombre_t] = p_t
        if usa_ens and "TF-IDF + Regresión logística jerárquica" in modelos:
            lr = modelos["TF-IDF + Regresión logística jerárquica"]
            p_lr = mm.alinear(lr.scores(d), lr.classes_, CLASES)[0]
            mask = mm.alinear(lr.mascara(d).astype(float), lr.classes_, CLASES)[0] > 0
            p = (ENS["alfa"] * p_t + (1 - ENS["alfa"]) * p_lr) * mask
            salida["Ensamble calibrado"] = p / max(p.sum(), 1e-12)
    return salida


def tarjeta(etiqueta, valor, rango, prob=True):
    cat, mis = etiqueta.split(":")
    color = COLOR_CAT.get(cat, GRIS)
    texto_mis = "" if mis == "NA" else f" · concepción errónea: <b>{mis.replace('_', ' ')}</b>"
    medida = f"{valor:.1%}" if prob else f"{valor:.2f}"
    st.markdown(f"<div class='tarjeta' style='--c:{color}'><span class='etq'>{rango}. {cat}</span>"
                f"{texto_mis} <span class='sub'>({medida})</span><br>"
                f"<span class='sub'>{DESCRIPCION_CAT.get(cat, '')}</span></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
st.sidebar.title("📐 MAP")
st.sidebar.caption("Charting Student Math Misunderstandings · Grupo 1 · CC3084")
pagina = st.sidebar.radio("Secciones", ["Clasificar una explicación", "Explorar los datos",
                                        "Rendimiento de los modelos", "Acerca del proyecto"])
st.sidebar.divider()
st.sidebar.caption("Las predicciones orientan la revisión docente; no son un diagnóstico definitivo.")

df, hay_textos = cargar_datos()
preguntas = cargar_preguntas()
disponibles = list(cargar_modelos())
for _pref, _nombre in TRANSFORMERS.items():
    if cargar_transformer(_pref) is not None:
        disponibles.append(_nombre)
if ENS and ENS.get("hay_deberta") and TRANSFORMERS.get(ENS.get("transformer")) in disponibles:
    disponibles.append("Ensamble calibrado")

# ===========================================================================
# 1. CLASIFICAR
# ===========================================================================
if pagina == "Clasificar una explicación":
    st.title("Clasificar la explicación de un estudiante")
    st.write("Elija la pregunta y la opción que marcó el estudiante, escriba su explicación **en inglés** "
             "(idioma del conjunto de entrenamiento) y el sistema propondrá las tres etiquetas más probables.")

    col1, col2 = st.columns([3, 2])
    with col1:
        qid = st.selectbox("Pregunta", preguntas["QuestionId"],
                           format_func=lambda q: f"{q} · {preguntas.set_index('QuestionId').loc[q, 'Pregunta']}")
        fila_q = preguntas.set_index("QuestionId").loc[qid]
        opciones = json.loads(fila_q["Opciones"])
        if st.session_state.get("opcion") not in opciones:
            st.session_state["opcion"] = opciones[0]
        opcion = st.radio("Opción elegida", opciones, horizontal=True, key="opcion",
                          format_func=lambda o: f"{o}  ✓" if o == fila_q["Correcta"] else o)
    with col2:
        st.markdown("**Ejemplos reales**")
        if hay_textos:
            ejemplos = df[df["QuestionId"] == qid]
            cat_ej = st.selectbox("Cargar un ejemplo de la categoría", sorted(ejemplos["Category"].unique()))
            def cargar_ejemplo():
                ej = ejemplos[ejemplos["Category"] == cat_ej].sample(1).iloc[0]
                st.session_state["explicacion"] = ej["StudentExplanation"]
                st.session_state["opcion"] = mm.limpiar_latex(ej["MC_Answer"])
                st.session_state["real"] = ej["target"]

            st.button("🎲 Cargar ejemplo", on_click=cargar_ejemplo)
        else:
            st.caption("Los textos originales no se distribuyen en la versión web; escriba una explicación.")

    explicacion = st.text_area("Explicación del estudiante", key="explicacion", height=110,
                               placeholder="e.g. I think it is 1/3 because 3 of the 9 triangles are not shaded")

    elegir = st.multiselect("Modelos a utilizar", disponibles, default=disponibles,
                            help="Seleccione uno para verlo por separado o varios para compararlos.")
    if "Ensamble calibrado" not in disponibles:
        st.caption("Los Transformers y el ensamble no están disponibles en este entorno "
                   "(modelos no descargados); se muestran los modelos TF-IDF.")

    if st.button("Clasificar", type="primary", disabled=not explicacion.strip() or not elegir):
        entrada = pd.DataFrame([{"QuestionId": qid, "QuestionText": fila_q["Pregunta"],
                                 "MC_Answer": opcion, "StudentExplanation": explicacion}])
        puntajes = predecir(entrada, elegir)
        if hay_textos and st.session_state.get("real") and (df["StudentExplanation"] == explicacion).any():
            st.success(f"Etiqueta real de este ejemplo: **{st.session_state['real']}**")

        cols = st.columns(min(len(puntajes), 3))
        for i, (nombre, s) in enumerate(puntajes.items()):
            with cols[i % len(cols)]:
                st.subheader(nombre)
                orden = np.argsort(-s)[:3]
                for r, j in enumerate(orden, 1):
                    tarjeta(CLASES[j], s[j], r, prob=nombre != "TF-IDF + SVM lineal")
                if ENS and nombre != "TF-IDF + SVM lineal" and nombre != "Prior (pregunta+opción)":
                    if s[orden[0]] < ENS["umbral"]:
                        st.warning(f"Confianza {s[orden[0]]:.0%} < {ENS['umbral']:.0%}: "
                                   "se recomienda **derivar al docente**.")
                    else:
                        st.caption(f"Confianza {s[orden[0]]:.0%} ≥ umbral {ENS['umbral']:.0%}.")
                if nombre == "TF-IDF + SVM lineal":
                    st.caption("Puntaje relativo de la SVM (no es una probabilidad calibrada).")

        st.markdown("#### Comparación entre modelos")
        filas = []
        for nombre, s in puntajes.items():
            for j in np.argsort(-s)[:3]:
                filas.append({"Modelo": nombre, "Etiqueta": CLASES[j], "Puntaje": float(s[j])})
        comp = pd.DataFrame(filas)
        fig = px.bar(comp, x="Puntaje", y="Etiqueta", color="Modelo", orientation="h", barmode="group",
                     color_discrete_map=PALETA, height=120 + 45 * comp["Etiqueta"].nunique())
        fig.update_layout(yaxis_title="", legend_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        acuerdo = comp.groupby("Modelo").first()["Etiqueta"]
        if acuerdo.nunique() == 1:
            st.success(f"Todos los modelos coinciden en la primera etiqueta: **{acuerdo.iloc[0]}**.")
        else:
            st.warning("Los modelos no coinciden en la primera etiqueta: conviene revisar el caso.")

# ===========================================================================
# 2. EXPLORAR
# ===========================================================================
elif pagina == "Explorar los datos":
    st.title("Explorar el conjunto de datos limpio")
    st.write("Variables después de la limpieza: nulos de `Misconception` codificados como `NA`, opciones "
             "convertidas de LaTeX a texto y variables numéricas derivadas del texto, con las que se "
             "construyeron los modelos.")
    f1, f2 = st.columns(2)
    sel_q = f1.multiselect("Preguntas", sorted(df["QuestionId"].unique()), placeholder="Todas")
    sel_c = f2.multiselect("Categorías", sorted(df["Category"].unique()), placeholder="Todas")
    d = df.copy()
    if sel_q:
        d = d[d["QuestionId"].isin(sel_q)]
    if sel_c:
        d = d[d["Category"].isin(sel_c)]

    k = st.columns(5)
    k[0].metric("Observaciones", f"{len(d):,}".replace(",", " "))
    k[1].metric("Preguntas", d["QuestionId"].nunique())
    k[2].metric("Etiquetas compuestas", d["target"].nunique())
    k[3].metric("Mediana de palabras", f"{d['n_palabras'].median():.0f}")
    k[4].metric("% con concepción errónea", f"{(d['Misconception'] != 'NA').mean():.1%}")

    t1, t2, t3, t4 = st.tabs(["Etiquetas", "Longitud y texto", "Por pregunta", "Tabla"])
    with t1:
        c1, c2 = st.columns(2)
        cnt = d["Category"].value_counts().reset_index()
        cnt.columns = ["Category", "n"]
        fig = px.bar(cnt, x="n", y="Category", orientation="h", color="Category",
                     color_discrete_map=COLOR_CAT, title="Distribución de Category")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Observaciones")
        c1.plotly_chart(fig, width="stretch")
        mis = d.loc[d["Misconception"] != "NA", "Misconception"].value_counts().head(20).reset_index()
        mis.columns = ["Misconception", "n"]
        fig = px.bar(mis, x="n", y="Misconception", orientation="h", title="Concepciones erróneas más frecuentes",
                     color_discrete_sequence=[ROJO])
        fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="Observaciones")
        c2.plotly_chart(fig, width="stretch")
        cola = df["target"].value_counts().reset_index(drop=True)
        fig = px.line(x=np.arange(1, len(cola) + 1), y=cola.values, log_y=True, markers=True,
                      title="Cola larga: observaciones por etiqueta compuesta (escala log)",
                      labels={"x": "Rango de la etiqueta", "y": "Observaciones"}, color_discrete_sequence=[AZUL])
        fig.add_hline(y=50, line_dash="dash", line_color=AMBAR, annotation_text="50 ejemplos")
        st.plotly_chart(fig, width="stretch")
        st.caption("Hallazgo: True_Correct concentra el 40 % de los datos y 32 de las 65 etiquetas tienen "
                   "menos de 50 ejemplos; por eso se evalúa con MAP@3 y macro-F1, no solo con exactitud.")
    with t2:
        var = st.selectbox("Variable derivada", ["n_palabras", "n_caracteres", "n_digitos", "n_simbolos",
                                                 "diversidad_lexica"])
        fig = px.box(d, x=var, y="Category", color="Category", color_discrete_map=COLOR_CAT, points=False,
                     title=f"{var} por categoría")
        fig.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Los razonamientos correctos son más largos (mediana 16 palabras) que los erróneos (13) "
                   "y los no evaluables (11), pero las distribuciones se solapan: la longitud solo es una "
                   "señal auxiliar.")
    with t3:
        comp = (pd.crosstab(d["QuestionId"].astype(str), d["Category"], normalize="index") * 100).reset_index()
        comp = comp.melt(id_vars="QuestionId", var_name="Category", value_name="%")
        fig = px.bar(comp, x="%", y="QuestionId", color="Category", orientation="h",
                     color_discrete_map=COLOR_CAT, title="Composición de Category por pregunta")
        fig.update_layout(yaxis_title="Pregunta", legend_title="")
        st.plotly_chart(fig, width="stretch")
        excl = df[df["Misconception"] != "NA"].groupby("Misconception")["QuestionId"].nunique()
        st.info(f"**{(excl == 1).sum()} de {len(excl)}** concepciones erróneas aparecen en una sola pregunta: "
                "el contexto de la pregunta es decisivo y limita la generalización a preguntas nuevas.")
    with t4:
        columnas = [c for c in ["QuestionId", "MC_Answer_clean", "StudentExplanation", "Category",
                                "Misconception", "n_palabras", "n_digitos", "n_simbolos"] if c in d]
        st.dataframe(d[columnas].head(500), width="stretch", hide_index=True)
        st.caption("Se muestran hasta 500 filas del filtro actual.")

# ===========================================================================
# 3. RENDIMIENTO
# ===========================================================================
elif pagina == "Rendimiento de los modelos":
    st.title("Rendimiento de los modelos")
    r = cargar_resultados()
    met = r["metricas"]
    mostrar = st.toggle("Mostrar gráficas de rendimiento", value=True)
    if not mostrar:
        st.info("Las gráficas de rendimiento están ocultas. Active el interruptor para verlas.")
    else:
        sel = st.multiselect("Modelos", met["Modelo"].tolist(), default=met["Modelo"].tolist())
        m = met[met["Modelo"].isin(sel)]
        metrica = st.radio("Métrica", ["MAP@3", "Accuracy", "Top-3 accuracy", "Macro-F1", "Weighted-F1"],
                           horizontal=True)
        fig = px.bar(m.sort_values(metrica), x=metrica, y="Modelo", orientation="h", color="Modelo",
                     color_discrete_map=PALETA, text_auto=".3f",
                     title=f"{metrica} en el conjunto de prueba ({r['final']['n_test']} explicaciones)")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        fig = px.scatter(m, x="Entrenamiento (s)", y="MAP@3", color="Modelo", color_discrete_map=PALETA,
                         log_x=True, size=[18] * len(m), hover_data=["Predicción (s)"],
                         title="Calidad frente a tiempo de entrenamiento")
        fig.update_layout(showlegend=False)
        a.plotly_chart(fig, width="stretch")
        tb = r["transferencia"]
        fig = px.bar(tb, x="MAP@3 preguntas no vistas", y="Modelo", orientation="h", text_auto=".3f",
                     color_discrete_sequence=[AMBAR], title="Validación B: preguntas no vistas")
        fig.update_layout(yaxis_title="", xaxis_range=[0, 1])
        b.plotly_chart(fig, width="stretch")

        with st.expander("Matriz de confusión de Category (ensamble)", expanded=False):
            if r["confusion"] is not None:
                fig = px.imshow(r["confusion"], text_auto=".2f", color_continuous_scale="Blues",
                                labels=dict(x="Predicción", y="Valor real", color="Proporción"))
                st.plotly_chart(fig, width="stretch")
        with st.expander("Ablación del contexto de entrada", expanded=False):
            ab = pd.DataFrame(r["lineales"]["ablacion"])
            fig = px.bar(ab, x="Entrada", y="MAP@3", text_auto=".3f", color_discrete_sequence=[AZUL])
            fig.update_layout(yaxis_range=[max(0, ab["MAP@3"].min() - 0.1), 1])
            st.plotly_chart(fig, width="stretch")
        with st.expander("Abstención y calibración", expanded=False):
            ca = r["abstencion"]
            fig = go.Figure()
            fig.add_scatter(x=ca["umbral"], y=ca["exactitud aceptados"], name="Exactitud de los aceptados",
                            line_color=AZUL)
            fig.add_scatter(x=ca["umbral"], y=ca["cobertura"], name="Cobertura", line_color=AMBAR)
            fig.add_vline(x=r["final"]["abstencion"]["umbral"], line_dash="dash", line_color=VERDE)
            fig.update_layout(xaxis_title="Umbral de confianza", yaxis_title="Proporción")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(r["final"]["ece"].items(), columns=["Modelo", "ECE"]),
                         hide_index=True)
        with st.expander("Ajuste de hiperparámetros", expanded=False):
            bq = r["busqueda"].copy()
            bq["Hiperparámetros"] = bq["Hiperparámetros"].str.replace('"', "")
            fig = px.bar(bq, x="MAP@3 CV (media)", y="Hiperparámetros", color="Modelo", orientation="h",
                         error_x="MAP@3 CV (desv.)", color_discrete_map={"TF-IDF + SVM lineal": CIAN,
                                                                        "TF-IDF + Regresión logística": AZUL})
            fig.update_layout(xaxis_range=[0.8, 0.95], yaxis_title="")
            st.plotly_chart(fig, width="stretch")
            if r["deberta"]:
                h = pd.DataFrame(r["deberta"]["historia"])
                st.plotly_chart(px.line(h, x="época", y="MAP@3 val", markers=True,
                                        title="DeBERTa-v3: MAP@3 de validación por época",
                                        color_discrete_sequence=[AMBAR]), width="stretch")
        st.dataframe(m.round(4), width="stretch", hide_index=True)

# ===========================================================================
# 4. ACERCA
# ===========================================================================
else:
    st.title("Acerca del proyecto")
    st.markdown("""
**Problema.** Dada una pregunta de opción múltiple, la opción elegida y la explicación escrita de un
estudiante, predecir hasta tres etiquetas `Category:Misconception` ordenadas por confianza (métrica MAP@3).

**Modelos.**
- *Prior (pregunta+opción)*: referencia sin texto.
- *TF-IDF + SVM lineal* y *TF-IDF + Regresión logística jerárquica*: líneas base interpretables.
- *DeBERTa-v3* y *MathBERT*: Transformers ajustados con la entrada pregunta + opción + explicación
  (MathBERT está preentrenado con textos de educación matemática).
- *Ensamble calibrado*: combinación ponderada del Transformer con mejor validación (escalado de
  temperatura) y la regresión logística, con restricción jerárquica y umbral de abstención.

**Uso responsable.** El sistema prioriza casos para la revisión docente. Cuando la confianza es baja,
recomienda derivar el caso. No sustituye el juicio pedagógico.

**Datos.** Kaggle — *MAP: Charting Student Math Misunderstandings* (2025).

**Grupo 1 · CC3084 Data Science · Universidad del Valle de Guatemala · 2026.**
Jorge Gabriel Palacios Sales · Pablo Daniel Barillas Moreno · Roberto Emiliano Otoniel Camposeco Torres ·
Abby Sofia Donis Agreda
""")

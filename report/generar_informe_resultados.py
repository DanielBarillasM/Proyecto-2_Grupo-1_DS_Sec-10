"""Genera el informe final de Resultados (Proyecto 2) en formato Word.

Lee las métricas producidas por notebooks/02_modelado.py (carpeta results/) y
las figuras de report/assets/, y construye report/informe_resultados_MAP.docx.

Uso, desde la raíz del repositorio:
    python notebooks/02_modelado.py        # entrena y exporta métricas
    python report/generar_informe_resultados.py
"""

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"
RES = ROOT / "results"
SALIDA = ROOT / "report" / "informe_resultados_MAP.docx"

resumen = json.loads((RES / "resumen_modelado.json").read_text(encoding="utf-8"))
metricas = pd.read_csv(RES / "metricas_prueba.csv")
busqueda = pd.read_csv(RES / "busqueda_hiperparametros.csv")

AZUL = RGBColor(0x24, 0x57, 0xA7)
GRIS = RGBColor(0x52, 0x60, 0x6D)
INTEGRANTES = [
    "Jorge Gabriel Palacios Sales",
    "Pablo Daniel Barillas Moreno",
    "Roberto Emiliano Otoniel Camposeco Torres",
    "Abby Sofia Donis Agreda",
]

doc = Document()

# ---------------------------------------------------------------------------
# Estilos base
# ---------------------------------------------------------------------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)  # carta
for lado in ("left_margin", "right_margin"):
    setattr(sec, lado, Cm(2.5))
sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for nivel, tam in ((1, 16), (2, 13), (3, 11.5)):
    st = doc.styles[f"Heading {nivel}"]
    st.font.name = "Calibri"
    st.font.size = Pt(tam)
    st.font.bold = True
    st.font.color.rgb = AZUL
    st.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    st.paragraph_format.space_before = Pt(14 if nivel == 1 else 10)
    st.paragraph_format.space_after = Pt(6)

cap = doc.styles["Caption"]
cap.font.size = Pt(9)
cap.font.italic = True
cap.font.color.rgb = GRIS


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def p(texto="", negrita_inicial=None, alinear=WD_ALIGN_PARAGRAPH.JUSTIFY, estilo=None):
    par = doc.add_paragraph(style=estilo)
    par.alignment = alinear
    if negrita_inicial:
        par.add_run(negrita_inicial).bold = True
    _runs(par, texto)
    return par


def _runs(par, texto, negrita=False):
    """Interpreta **negrita**, *cursiva* (también anidada) y `código` dentro del texto."""
    import re
    for trozo in re.split(r"(\*\*.+?\*\*(?!\*)|`[^`]+`|\*[^*\s][^*]*\*)", texto):
        if trozo.startswith("**") and trozo.endswith("**") and len(trozo) > 4:
            _runs(par, trozo[2:-2], negrita=True)
        elif trozo.startswith("`"):
            r = par.add_run(trozo[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
            r.bold = negrita
        elif trozo.startswith("*") and trozo.endswith("*") and len(trozo) > 2:
            r = par.add_run(trozo[1:-1])
            r.italic, r.bold = True, negrita
        elif trozo:
            par.add_run(trozo).bold = negrita


def viñetas(items, numerada=False):
    for i, it in enumerate(items, 1):
        if numerada:
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.9)
            par.paragraph_format.first_line_indent = Cm(-0.6)
            par.add_run(f"{i}.  ")
        else:
            par = doc.add_paragraph(style="List Bullet")
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _runs(par, it)


def h(texto, nivel=1):
    return doc.add_heading(texto, level=nivel)


N_FIG = [0]
N_TAB = [0]


def figura(nombre, pie, ancho=15.5):
    ruta = ASSETS / nombre
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(ruta), width=Cm(ancho))
    N_FIG[0] += 1
    c = doc.add_paragraph(f"Figura {N_FIG[0]}. {pie}", style="Caption")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return N_FIG[0]


def _sombrear(celda, color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def tabla(encabezados, filas, titulo, anchos=None, tam=9):
    N_TAB[0] += 1
    c = doc.add_paragraph(f"Tabla {N_TAB[0]}. {titulo}", style="Caption")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(encabezados))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, e in enumerate(encabezados):
        cel = t.rows[0].cells[i]
        cel.text = ""
        run = cel.paragraphs[0].add_run(str(e))
        run.bold = True
        run.font.size = Pt(tam)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _sombrear(cel, "2457A7")
    for k, fila in enumerate(filas):
        celdas = t.add_row().cells
        for i, v in enumerate(fila):
            celdas[i].text = ""
            _runs(celdas[i].paragraphs[0], str(v))
            for r in celdas[i].paragraphs[0].runs:
                r.font.size = Pt(tam)
            if k % 2:
                _sombrear(celdas[i], "EEF3FA")
    if anchos:
        for fila in t.rows:
            for i, w in enumerate(anchos):
                fila.cells[i].width = Cm(w)
    doc.add_paragraph()
    return N_TAB[0]


def salto():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def campo(par, instruccion, relleno=""):
    """Inserta un campo de Word (TOC, número de página)."""
    r = par.add_run()
    for tipo, texto in (("begin", None), (None, instruccion), ("separate", None),
                        (None, relleno),
                        ("end", None)):
        if tipo:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tipo)
            r._r.append(el)
        elif texto == instruccion:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = instruccion
            r._r.append(el)
        else:
            if texto:
                r2 = OxmlElement("w:t")
                r2.text = texto
                r._r.append(r2)


def miles(n):
    return f"{n:,}".replace(",", " ")


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{x * 100:.1f} %"


# Atajos sobre los resultados
M = metricas.set_index("Modelo")
MEJOR = resumen["mejor_modelo"]
TR = {t["Modelo"]: t for t in resumen["transferencia"]}
DIM = resumen["por_dimension"]
HP = resumen["mejores_hiperparametros"]
BASE = "Baseline (pregunta+opción)"

# ===========================================================================
# PORTADA
# ===========================================================================
logo = doc.add_paragraph()
logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
logo.add_run().add_picture(str(ASSETS / "logo_uvg.png"), width=Cm(4.5))
for linea, tam, neg in (
    ("Universidad del Valle de Guatemala", 14, True),
    ("Facultad de Ingeniería · Departamento de Ciencias de la Computación", 11, False),
    ("CC3084 – Data Science · Sección 10 · Semestre II 2026", 11, False),
):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = par.add_run(linea)
    r.font.size, r.bold = Pt(tam), neg
doc.add_paragraph()
for linea, tam, color in (
    ("Proyecto 2 · Resultados", 13, GRIS),
    ("MAP — Charting Student Math Misunderstandings", 22, AZUL),
    ("Clasificación de concepciones erróneas matemáticas en explicaciones "
     "estudiantiles mediante Procesamiento del Lenguaje Natural", 13, GRIS),
):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = par.add_run(linea)
    r.font.size, r.bold, r.font.color.rgb = Pt(tam), True, color
doc.add_paragraph()
par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
par.add_run("Grupo 1\n").bold = True
for nombre in INTEGRANTES:
    par.add_run(nombre + "\n")
par = doc.add_paragraph()
par.alignment = WD_ALIGN_PARAGRAPH.CENTER
par.add_run("Guatemala, 12 de octubre de 2026").italic = True
salto()

# Índice
h("Índice", 1)
campo(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u',
      "Clic derecho → Actualizar campo para generar el índice.")
salto()

# ===========================================================================
# 1. INTRODUCCIÓN
# ===========================================================================
h("1. Introducción", 1)
p("Cuando un estudiante resuelve mal un problema de matemáticas, la respuesta incorrecta es "
  "solo el síntoma. Lo que realmente orienta al docente es el porqué: el razonamiento "
  "defectuoso —la **concepción errónea** o *misconception*— que produjo esa respuesta. Del mismo "
  "modo, una respuesta correcta no garantiza un razonamiento correcto: un estudiante puede acertar "
  "por casualidad o mediante una regla equivocada que funciona en ese caso particular "
  "(Rushton, 2018).")
p("El reto de Kaggle **MAP — Charting Student Math Misunderstandings** (Kaggle, 2025; "
  "Rittle-Johnson et al., 2025) propone automatizar parte de ese diagnóstico. Cada observación "
  "contiene una pregunta de opción múltiple, la opción elegida por el estudiante y una breve "
  "explicación escrita de su razonamiento. A partir de ello se debe predecir una etiqueta "
  "compuesta `Category:Misconception` que indica (i) si la opción es correcta, (ii) si el "
  "razonamiento es correcto, contiene una concepción errónea o no es evaluable, y (iii) cuál es "
  "la concepción errónea específica, cuando la hay.")
p("En la fase anterior del proyecto se realizó un análisis exploratorio completo y una revisión "
  "bibliográfica de algoritmos. Este informe reúne ambos trabajos y presenta la fase de "
  "**resultados**: la selección de algoritmos, la construcción y ajuste de varios modelos, su "
  "evaluación con métricas adecuadas para clasificación multiclase desbalanceada y la discusión "
  "comparativa que permite escoger el modelo final.")
p("**Problema a resolver.** Revisar manualmente miles de explicaciones abiertas exige tiempo y "
  "conocimiento pedagógico especializado. Se busca un clasificador que, dada la pregunta, la "
  "opción y la explicación, proponga las tres etiquetas más probables ordenadas por confianza, "
  "de manera que el docente pueda priorizar la revisión. El sistema se concibe como apoyo a la "
  "decisión y no como diagnóstico automático definitivo del estudiante.")
p("**Pregunta científica.** ¿Qué combinación de representación textual, contexto de la pregunta "
  "y algoritmo de clasificación permite ordenar con mayor precisión las categorías y las "
  "concepciones erróneas presentes en explicaciones matemáticas breves, y cuánto de ese desempeño "
  "se conserva ante preguntas no vistas?")

# ===========================================================================
# 2. OBJETIVOS
# ===========================================================================
h("2. Objetivos", 1)
h("2.1 Objetivo general", 2)
p("Construir, ajustar y comparar modelos de aprendizaje automático supervisado que clasifiquen "
  "explicaciones matemáticas de estudiantes en la etiqueta compuesta `Category:Misconception` "
  "del reto MAP, fundamentando las decisiones de preprocesamiento, validación y selección del "
  "modelo en el análisis exploratorio y en la literatura revisada.")
h("2.2 Objetivos específicos", 2)
viñetas([
    "Caracterizar la estructura, calidad y desbalance del conjunto de datos y derivar de ello "
    "las decisiones de limpieza y preprocesamiento.",
    "Revisar la literatura sobre calificación automática de respuestas cortas y detección de "
    "concepciones erróneas para seleccionar algoritmos candidatos.",
    "Entrenar al menos cuatro modelos (una línea base sin texto y tres clasificadores sobre "
    "TF-IDF) y ajustar sus hiperparámetros con validación cruzada agrupada.",
    "Evaluar los modelos con MAP@3, exactitud, macro-F1 y F1 ponderado, además de tiempos de "
    "entrenamiento y predicción.",
    "Diseñar y evaluar un modelo jerárquico que aproveche la estructura del problema "
    "descubierta en el análisis exploratorio.",
    "Medir la capacidad de generalización a preguntas no vistas mediante validación agrupada "
    "por pregunta.",
], numerada=True)

# ===========================================================================
# 3. MARCO TEÓRICO
# ===========================================================================
h("3. Marco teórico", 1)
h("3.1 Procesamiento del Lenguaje Natural", 2)
p("El **Procesamiento del Lenguaje Natural** (PLN o NLP, por sus siglas en inglés) es el área de "
  "la inteligencia artificial y la lingüística computacional que desarrolla métodos para que una "
  "computadora represente, analice, interprete y genere lenguaje humano. Sus tareas incluyen la "
  "clasificación de textos, el análisis de sentimiento, la extracción de información, la "
  "traducción automática y la respuesta a preguntas. El presente problema es una tarea de "
  "**clasificación de texto corto multiclase**: cada documento (una explicación de unas 13 "
  "palabras) debe asignarse a una de 65 clases.")
p("En el ámbito educativo, el PLN se ha aplicado desde hace más de una década a la "
  "**calificación automática de respuestas cortas** (*Automatic Short Answer Grading*, ASAG). "
  "Burrows et al. (2015) describen la evolución del campo desde sistemas de reglas y "
  "coincidencia con respuestas modelo hacia métodos estadísticos y de aprendizaje automático, y "
  "más recientemente hacia modelos de lenguaje preentrenados (Morley et al., 2026).")

h("3.2 Actividades de preprocesamiento", 2)
p("El preprocesamiento transforma el texto crudo en una representación adecuada para el modelo. "
  "Las actividades habituales, y la forma en que se aplicaron aquí, son:")
viñetas([
    "**Normalización:** conversión a minúsculas y unificación de espacios. Se aplicó a todas las "
    "explicaciones.",
    "**Limpieza de marcado:** eliminación o conversión de notación especial. Las opciones de "
    "respuesta venían en LaTeX (96 %), por ejemplo `\\( \\frac{1}{3} \\)`, y se convirtieron a "
    "texto plano (`1/3`).",
    "**Tokenización:** división del texto en unidades (palabras, símbolos o n-gramas de "
    "caracteres). Se definió un patrón que conserva números y operadores matemáticos "
    "(`+ - * / = < > %`) como tokens.",
    "**Eliminación de palabras vacías (*stopwords*):** se usa con cautela. En la fase de modelado "
    "no se eliminaron, porque palabras como *not* o *than* cambian el sentido matemático.",
    "**Tratamiento de números:** la práctica estándar de eliminarlos se descartó a propósito: el "
    "78.3 % de las explicaciones contiene dígitos y son la principal señal matemática.",
    "**Corrección ortográfica, lematización y *stemming*:** no se aplicaron; en su lugar se usaron "
    "n-gramas de caracteres, que toleran errores ortográficos (\"tree nineth\" por \"three "
    "ninths\") sin alterar la notación.",
    "**Vectorización:** conversión del texto a vectores numéricos mediante TF-IDF.",
])

h("3.3 Representación TF-IDF y n-gramas", 2)
p("La representación **TF-IDF** (*Term Frequency – Inverse Document Frequency*) asigna a cada "
  "término t de un documento d un peso proporcional a su frecuencia en el documento y "
  "inversamente proporcional al número de documentos que lo contienen (Salton & Buckley, 1988):")
p("tfidf(t, d) = tf(t, d) · log((1 + N) / (1 + df(t))) + 1", alinear=WD_ALIGN_PARAGRAPH.CENTER)
p("donde N es el número de documentos y df(t) el número de documentos que contienen t. Así, "
  "términos frecuentes en todo el corpus (\"the\", \"is\") pesan poco, y términos "
  "característicos de un tipo de razonamiento (\"divided by\", \"not shaded\") pesan más. Se usó "
  "la variante sublineal, que reemplaza tf por 1 + log(tf).")
p("Un **n-grama** es una secuencia de n unidades consecutivas. Los n-gramas de palabras (n = 1, 2) "
  "capturan expresiones como \"two negatives\", y los n-gramas de caracteres (n = 2 a 5) capturan "
  "fragmentos como \"3/9\" o \"nine\", lo que da robustez frente a errores ortográficos. El "
  "resultado es una matriz dispersa de muy alta dimensión, adecuada para clasificadores lineales.")

h("3.4 Algoritmos de clasificación para PLN", 2)
p("**Línea base por frecuencia condicional.** Asigna a cada observación las etiquetas más "
  "frecuentes observadas en entrenamiento para el mismo par (pregunta, opción elegida). No lee "
  "el texto; sirve para medir cuánto aporta realmente el lenguaje.")
p("**Naive Bayes Complementario (ComplementNB).** Clasificador probabilístico que aplica el "
  "teorema de Bayes con la suposición de independencia condicional entre términos. La variante "
  "complementaria estima los parámetros de cada clase a partir de los documentos de *todas las "
  "demás* clases, lo que corrige el sesgo del Naive Bayes multinomial hacia las clases "
  "mayoritarias en datos desbalanceados (Rennie et al., 2003). Su hiperparámetro α controla el "
  "suavizado de Laplace.")
p("**Regresión logística multinomial.** Modelo lineal que estima la probabilidad de cada clase k "
  "mediante la función softmax: P(y = k | x) = exp(w_k·x) / Σ_j exp(w_j·x). Se entrena "
  "minimizando la entropía cruzada con regularización L2, cuya intensidad es 1/C. Produce "
  "probabilidades interpretables, lo que la hace idónea para generar un ranking de tres "
  "etiquetas.")
p("**Máquina de vectores de soporte lineal (SVM).** Busca, para cada clase, el hiperplano que "
  "separa con el mayor margen posible los ejemplos de esa clase del resto (esquema uno contra "
  "todos) (Cortes & Vapnik, 1995). Es muy eficaz en espacios dispersos de alta dimensión como "
  "TF-IDF; el hiperparámetro C regula el compromiso entre margen y errores de entrenamiento. "
  "No produce probabilidades, por lo que el ranking se construye con la distancia al hiperplano.")
p("**Modelo jerárquico con restricciones.** Descompone la predicción en etapas que reflejan la "
  "estructura de la etiqueta: primero se determina la veracidad (si la opción es correcta) y luego "
  "el tipo de razonamiento y la concepción errónea, restringiendo los candidatos a las etiquetas "
  "compatibles con la pregunta. Esta idea proviene de la clasificación jerárquica y de la "
  "recomendación de la literatura de incorporar el contexto de la pregunta (Zhang et al., 2020).")
p("**Modelos de lenguaje basados en *Transformers*.** BERT (Devlin et al., 2019) y DeBERTa "
  "(He et al., 2021) aprenden representaciones contextuales mediante preentrenamiento sobre "
  "grandes corpus y luego se ajustan a la tarea. Variantes como MathBERT (Shen et al., 2021) se "
  "preentrenan con textos matemáticos. La literatura reporta mejoras sobre los métodos clásicos "
  "en ASAG (Zhu et al., 2022; Bonthu et al., 2024), aunque no siempre en preguntas no vistas "
  "(Padó et al., 2024). Se discuten como trabajo futuro en la sección 6.")

h("3.5 Métricas de evaluación", 2)
viñetas([
    "**MAP@3 (*Mean Average Precision at 3*):** métrica oficial del reto. El modelo entrega tres "
    "etiquetas ordenadas; un acierto en la primera posición vale 1, en la segunda 1/2 y en la "
    "tercera 1/3; se promedia sobre todas las observaciones.",
    "**Exactitud (*accuracy*):** proporción de observaciones cuya primera predicción es correcta. "
    "Es engañosa con clases desbalanceadas: predecir siempre la clase mayoritaria ya da 40 %.",
    "**Exactitud top-3:** proporción de observaciones cuya etiqueta real está entre las tres "
    "propuestas.",
    "**F1, macro-F1 y F1 ponderado:** el F1 es la media armónica de precisión y exhaustividad "
    "(*recall*) de una clase. El macro-F1 promedia el F1 de todas las clases con igual peso, por lo "
    "que es sensible al desempeño en clases raras; el F1 ponderado pondera por la frecuencia.",
    "**Matriz de confusión:** tabla que cruza clases reales y predichas para identificar qué "
    "categorías se confunden entre sí.",
])

h("3.6 Validación y fuga de información", 2)
p("La **validación cruzada** divide los datos en k pliegues y entrena k veces, usando cada vez un "
  "pliegue distinto como validación. La variante **estratificada** conserva la proporción de "
  "clases en cada pliegue; la **agrupada** (*GroupKFold*) garantiza que todas las observaciones "
  "de un mismo grupo queden en el mismo pliegue. Se habla de **fuga de información** (*data "
  "leakage*) cuando el modelo ve durante el entrenamiento información del conjunto de evaluación "
  "—por ejemplo, textos idénticos repetidos—, lo que infla artificialmente las métricas.")

# ===========================================================================
# 4. METODOLOGÍA
# ===========================================================================
h("4. Metodología", 1)
h("4.1 Pasos seguidos", 2)
viñetas([
    "**Revisión bibliográfica** de algoritmos de ASAG y detección de concepciones erróneas en "
    "revistas indexadas (documento `revision_bibliografica_algoritmos_MAP.pdf`).",
    "**Análisis exploratorio** del conjunto de entrenamiento (`01_analisis_exploratorio.ipynb`): "
    "esquema, faltantes, duplicados, calidad del texto, distribuciones, valores atípicos, "
    "correlaciones, cruces por pregunta y contenido léxico.",
    "**Preprocesamiento** derivado de los hallazgos (sección 5.2).",
    "**Partición** en entrenamiento (80 %) y prueba (20 %) sin fuga de duplicados.",
    "**Construcción de cinco modelos** y **ajuste de hiperparámetros** por búsqueda en malla con "
    "validación cruzada de 3 pliegues sobre el conjunto de entrenamiento.",
    "**Evaluación final** de cada modelo, con sus mejores hiperparámetros, sobre el conjunto de "
    "prueba retenido.",
    "**Prueba de transferencia** a preguntas no vistas con GroupKFold por `QuestionId`.",
    "**Comparación y selección** del modelo final según MAP@3, macro-F1 y costo computacional.",
], numerada=True)
p("Todo el proceso de modelado está en el script reproducible `notebooks/02_modelado.py`, que "
  "exporta las métricas a `results/`, los modelos entrenados a `models/` y las figuras a "
  "`report/assets/modelos/`.")

h("4.2 Selección de los conjuntos de entrenamiento y prueba", 2)
p("El archivo `test.csv` publicado por Kaggle contiene solo 3 filas: el conjunto de prueba real "
  "(≈ 17 000 filas) solo era accesible dentro de los *notebooks* de la competencia, que cerró el "
  "15 de octubre de 2025. Por ello, **toda la evaluación se construyó particionando "
  "`train.csv`** (36 696 filas).")
p(f"Se usó `StratifiedGroupKFold` con 5 pliegues y semilla 42, tomando un pliegue como prueba. "
  f"Esto produce **{miles(resumen['n_train'])} filas de entrenamiento y {miles(resumen['n_test'])} "
  f"de prueba** (80/20). La partición es:")
viñetas([
    "**Estratificada por la etiqueta compuesta**, para que las 65 clases —muchas con menos de "
    "50 ejemplos— conserven su proporción en ambos conjuntos.",
    "**Agrupada por (pregunta, explicación normalizada)**: el análisis exploratorio encontró 970 "
    "explicaciones repetidas (2.6 %). Si una copia quedara en entrenamiento y otra en prueba, el "
    "modelo solo tendría que memorizarla. Al agruparlas, todas las copias quedan del mismo lado.",
])
p("El ajuste de hiperparámetros se hizo **exclusivamente dentro del conjunto de entrenamiento** "
  "(3 pliegues con el mismo esquema), de modo que el conjunto de prueba solo se usó una vez por "
  "modelo, para la evaluación final.")
p("Además, como el hallazgo más importante del análisis exploratorio es que 32 de las 35 "
  "concepciones erróneas aparecen en una sola pregunta, se añadió una **segunda evaluación con "
  "`GroupKFold` por `QuestionId`** (5 pliegues sobre todo el conjunto): cada pliegue de "
  "validación contiene preguntas completas que el modelo nunca vio. Esta prueba mide la "
  "transferencia a preguntas nuevas, no el desempeño operativo.")

h("4.3 Selección de los algoritmos", 2)
p("La selección se basó en la revisión bibliográfica y en las características del conjunto:")
tabla(
    ["Modelo", "Razón de la selección"],
    [
        ["Línea base (pregunta + opción)",
         "El EDA mostró que el par (pregunta, opción) determina la veracidad en 58 de 60 casos. "
         "Mide cuánto se logra sin leer el texto y sirve de piso de comparación."],
        ["Complement Naive Bayes",
         "Línea base clásica y muy rápida en clasificación de texto; la variante complementaria "
         "está diseñada para clases desbalanceadas (Rennie et al., 2003)."],
        ["SVM lineal",
         "Algoritmo de referencia en ASAG sobre bolsa de palabras y TF-IDF (Burrows et al., 2015; "
         "del Gobbo et al., 2023); eficaz en espacios dispersos de alta dimensión."],
        ["Regresión logística",
         "Modelo lineal interpretable que entrega probabilidades calibradas, necesarias para "
         "ordenar tres candidatos (MAP@3) y para una futura revisión docente por incertidumbre."],
        ["Regresión logística jerárquica",
         "Explota dos hallazgos del EDA: la veracidad casi determinista y las concepciones "
         "ligadas a la pregunta. Sigue la recomendación de incorporar el contexto de la pregunta "
         "(Zhang et al., 2020)."],
    ],
    "Algoritmos seleccionados y justificación.", anchos=[4.5, 12],
)
p("Los modelos tipo *Transformer* (BERT, DeBERTa, MathBERT) fueron identificados en la revisión "
  "como el siguiente nivel, pero **no se entrenaron en esta entrega**: requieren GPU para un "
  "ajuste fino con validación cruzada en un tiempo razonable, y la literatura recomienda "
  "establecer primero una línea base lineal sólida y una validación sin fuga (Padó et al., 2024).")

h("4.4 Representación de entrada", 2)
p("Los cuatro modelos de texto comparten la misma matriz de características, construida al "
  "concatenar horizontalmente:")
viñetas([
    "TF-IDF de **palabras** (unigramas y bigramas, frecuencia mínima 2, tf sublineal), con un "
    "patrón de tokenización que conserva números y operadores.",
    "TF-IDF de **caracteres** (n-gramas de 2 a 5 dentro de los límites de palabra, frecuencia "
    "mínima 3, máximo 120 000 rasgos).",
    "Codificación *one-hot* de `QuestionId` y del par (pregunta, opción limpia), que aporta el "
    "contexto de la pregunta y de la respuesta elegida.",
])
p("Los vectorizadores se ajustan solo con los datos de entrenamiento de cada pliegue, para evitar "
  "fuga del vocabulario de prueba.")

h("4.5 Herramientas utilizadas", 2)
p("**Recursos de cómputo.** Todo el entrenamiento se realizó localmente en una computadora "
  "portátil Apple MacBook con procesador **Apple M3 (8 núcleos) y 8 GB de memoria RAM**, sin "
  "GPU dedicada. Los modelos lineales sobre TF-IDF son adecuados para este hardware; los "
  "*Transformers* requerirían Kaggle o Google Colab con GPU.")
tabla(
    ["Herramienta", "Versión", "Uso"],
    [
        ["Python", "3.14", "Lenguaje de programación de todo el proyecto"],
        ["pandas / NumPy", "2.2.3 / 2.2.2", "Carga, limpieza y manipulación de datos"],
        ["scikit-learn", "1.8.0", "TF-IDF, modelos, validación cruzada y métricas "
                                  "(Pedregosa et al., 2011)"],
        ["SciPy", "—", "Matrices dispersas"],
        ["Matplotlib / Seaborn", "3.10.1 / 0.13.2", "Visualizaciones estáticas"],
        ["joblib", "—", "Paralelización de pliegues y persistencia de modelos"],
        ["JupyterLab", "4.6.1", "Análisis exploratorio interactivo"],
        ["Kaggle CLI", "2.2.4", "Descarga de los datos de la competencia"],
        ["Git / GitHub", "—", "Control de versiones y contribuciones individuales"],
        ["python-docx", "1.2.0", "Generación de este informe"],
    ],
    "Lenguajes, bibliotecas y paquetes utilizados.", anchos=[4, 3.5, 9],
)
p("Se eligió Python con scikit-learn porque ofrece en una sola biblioteca coherente todos los "
  "algoritmos clásicos de clasificación de texto, herramientas de validación agrupada y "
  "métricas, con resultados reproducibles mediante una semilla fija (`RANDOM_STATE = 42`).")

# ===========================================================================
# 5. RESULTADOS Y ANÁLISIS DE RESULTADOS
# ===========================================================================
h("5. Resultados y análisis de resultados", 1)

h("5.1 Características del conjunto de datos original", 2)
p("Los datos provienen de la competencia de Kaggle y se descargaron con la API oficial. El "
  "archivo de entrenamiento `train.csv` contiene **36 696 observaciones y 7 variables**; no hay "
  "variables cuantitativas nativas, por lo que las variables numéricas del análisis se derivaron "
  "del texto.")
tabla(
    ["Variable", "Tipo", "Valores únicos", "Nulos", "Rol"],
    [
        ["row_id", "Identificador", "36 696", "0", "Índice, sin valor predictivo"],
        ["QuestionId", "Categórica nominal", "15", "0", "Contexto / agrupamiento"],
        ["QuestionText", "Texto", "15", "0", "Enunciado de la pregunta"],
        ["MC_Answer", "Categórica (LaTeX)", "49", "0", "Opción elegida"],
        ["StudentExplanation", "Texto libre", "35 726", "0", "Predictora principal"],
        ["Category", "Categórica nominal", "6", "0", "Objetivo (parte 1)"],
        ["Misconception", "Categórica nominal", "35", "26 836 (73.1 %)", "Objetivo (parte 2)"],
    ],
    "Diccionario de variables de train.csv.", anchos=[3.4, 3.3, 2.4, 2.7, 4.7],
)
p("La variable `Category` tiene la forma `Veracidad_Tipo`: la **veracidad** (`True`/`False`) "
  "indica si la opción marcada es correcta y el **tipo** (`Correct`, `Misconception`, `Neither`) "
  "describe la calidad del razonamiento escrito. Al combinarla con `Misconception` se obtienen "
  "**65 etiquetas compuestas**, que constituyen el espacio de salida del modelo.")

h("5.2 Limpieza y preprocesamiento", 2)
tabla(
    ["Situación detectada", "Decisión", "Justificación"],
    [
        ["73.1 % de nulos en Misconception",
         "Reemplazar por la categoría explícita \"NA\"",
         "Son estructurales: el valor existe si y solo si Tipo = Misconception. Imputar con la "
         "moda corrompería el 73 % de los datos."],
        ["0 filas idénticas; 736 iguales sin row_id; 970 textos repetidos (2.6 %)",
         "Conservar, pero agrupar en la partición",
         "Pueden ser respuestas legítimas; agruparlas evita fuga entre entrenamiento y prueba."],
        ["LaTeX en 96 % de MC_Answer",
         "Convertir a texto plano (\\frac{1}{3} → 1/3)",
         "Homogeneiza el formato numérico de las opciones."],
        ["62.8 % del texto empieza en minúscula; ortografía variable",
         "Minúsculas, espacios unificados y n-gramas de caracteres",
         "Reduce variación superficial y tolera errores ortográficos."],
        ["78.3 % del texto contiene dígitos; 27.2 % el símbolo /",
         "No eliminar números ni operadores",
         "Son la principal señal matemática."],
        ["≈ 4 % de explicaciones atípicas por longitud",
         "Conservar",
         "Son datos válidos: explicaciones paso a paso o respuestas mínimas (señal de Neither)."],
        ["Etiqueta en dos columnas",
         "Crear target = Category:Misconception",
         "Formato exigido por la métrica MAP@3 de la competencia."],
    ],
    "Decisiones de limpieza y preprocesamiento.", anchos=[4.5, 4.5, 7.5],
)
p("En la aplicación, estas mismas transformaciones (conversión de LaTeX, minúsculas, unificación "
  "de espacios y vectorización TF-IDF ajustada en entrenamiento) se encapsulan dentro del objeto "
  "del modelo guardado, de modo que una explicación nueva se preprocesa de forma idéntica e "
  "inadvertida para el usuario.")

h("5.3 Análisis exploratorio", 2)
p("El análisis exploratorio completo está en `notebooks/01_analisis_exploratorio.ipynb` y en el "
  "informe de la fase anterior. A continuación se resumen los hallazgos que condicionaron el "
  "modelado.")
f = figura("distribucion_category.png",
           "Distribución de Category y su descomposición en veracidad × tipo de razonamiento.")
p(f"**Desbalance de clases (Figura {f}).** `True_Correct` concentra el 40.3 % de las "
  "observaciones, mientras que `True_Misconception` (1.1 %) y `False_Correct` (0.6 %) son muy "
  "raras. Estas dos categorías son pedagógicamente las más interesantes: estudiantes que "
  "aciertan con un razonamiento erróneo, o que razonan bien pero marcan mal.")
f = figura("distribucion_misconceptions.png",
           "Concepciones erróneas más frecuentes y distribución de cola larga de la etiqueta compuesta.")
p(f"**Cola larga (Figura {f}).** De las 65 clases compuestas, 32 tienen menos de 50 "
  "observaciones y 24 menos de 10. La exactitud global es por ello engañosa y se requieren "
  "MAP@3 y macro-F1.")
f = figura("longitud_por_categoria.png",
           "Longitud de la explicación por categoría y por tipo de razonamiento.")
p(f"**Longitud del texto (Figura {f}).** Las explicaciones son muy breves (mediana de 13 "
  "palabras). Los razonamientos correctos son más largos (mediana 16) que los erróneos (13) y "
  "los no evaluables (11), pero las distribuciones se solapan: la longitud discrimina solo "
  "débilmente.")
f = figura("cruces_por_pregunta.png",
           "Composición de Category y tasa de acierto por pregunta.")
p(f"**Dependencia de la pregunta (Figura {f}).** La tasa de acierto varía entre 35.7 % y 79.6 % "
  "según la pregunta. Además, **32 de las 35 concepciones erróneas aparecen en una sola "
  "pregunta** y el par (pregunta, opción) determina la veracidad en 58 de 60 casos. Estos dos "
  "hallazgos motivaron la línea base, el modelo jerárquico y la prueba con preguntas no vistas.")
f = figura("marcadores_lexicos.png",
           "Composición del tipo de razonamiento según palabras marcadoras en la explicación.", 13)
p(f"**Contenido léxico (Figura {f}).** Verbos de procedimiento (*divide*, *multiply*, *half*) se "
  "asocian con razonamiento correcto, y marcadores de incertidumbre (*guess*, *think*) con "
  "`Neither`. El vocabulario contiene señal interpretable, lo que justifica clasificadores "
  "lineales sobre TF-IDF como primera familia de modelos.")

# --- Ajuste de hiperparámetros ---------------------------------------------
h("5.4 Ajuste de hiperparámetros", 2)
p("Cada modelo de texto se ajustó mediante **búsqueda en malla** (*grid search*) con validación "
  "cruzada de 3 pliegues (`StratifiedGroupKFold`) dentro del conjunto de entrenamiento, usando "
  "MAP@3 como criterio de selección. La tabla siguiente resume las combinaciones probadas.")
filas = []
for _, r in busqueda.iterrows():
    hp = json.loads(r["Hiperparámetros"])
    txt = ", ".join(f"{k} = {v}" for k, v in hp.items()).replace("None", "ninguno")
    elegido = "✔" if hp == HP[r["Modelo"]] else ""
    filas.append([r["Modelo"], txt, f"{r['MAP@3 CV (media)']:.4f} ± {r['MAP@3 CV (desv.)']:.4f}",
                  elegido])
tabla(["Modelo", "Hiperparámetros", "MAP@3 CV (media ± desv.)", "Elegido"], filas,
      "Resultados de la búsqueda de hiperparámetros (validación cruzada, 3 pliegues).",
      anchos=[4.3, 5.5, 4.5, 1.8])
f = figura("modelos/busqueda_hiperparametros.png",
           "MAP@3 en validación cruzada según los hiperparámetros de cada modelo "
           "(barras de error: desviación estándar entre pliegues).")
p(f"La Figura {f} muestra tres comportamientos:")
viñetas([
    f"**Regresión logística:** el mejor valor fue **C = {HP['Regresión logística']['C']:g}**. Con "
    "C = 1 el modelo está algo sobre-regularizado y con C = 16 empieza a sobreajustar; las "
    "diferencias son pequeñas (≈ 0.002), lo que indica un modelo estable.",
    f"**SVM lineal:** el óptimo fue **C = {HP['SVM lineal']['C']:g}**, también con una meseta "
    "amplia entre 0.1 y 1.",
    "**Ponderación de clases (`class_weight = balanced`):** empeoró el MAP@3 en ambos modelos "
    "(hasta 6 puntos en SVM con C = 0.1). Al obligar al modelo a favorecer clases raras, se "
    "degrada el orden de los candidatos frecuentes, que dominan la métrica. Se descartó.",
    f"**Complement Naive Bayes:** mejoró al aumentar el suavizado; el mejor valor fue "
    f"**α = {HP['Complement Naive Bayes']['alpha']:g}**, el extremo de la malla. Un α mayor "
    "podría mejorarlo ligeramente, pero su brecha frente a los modelos discriminativos "
    "(≈ 14 puntos) no se cerraría con este ajuste.",
])
p("El modelo jerárquico reutiliza los hiperparámetros de la regresión logística, porque su "
  "restricción no introduce parámetros adicionales.")

# --- Comparación ------------------------------------------------------------
h("5.5 Comparación de los algoritmos", 2)
p(f"Cada modelo se reentrenó con sus mejores hiperparámetros sobre todo el conjunto de "
  f"entrenamiento ({miles(resumen['n_train'])} filas) y se evaluó una sola vez sobre el conjunto "
  f"de prueba retenido ({miles(resumen['n_test'])} filas, {resumen['clases_prueba']} clases "
  f"presentes).")
orden_tabla = metricas.sort_values("MAP@3", ascending=False)
tabla(
    ["Modelo", "MAP@3", "Exactitud", "Top-3", "Macro-F1", "F1 pond.", "Entren. (s)", "Pred. (s)"],
    [[r["Modelo"], f3(r["MAP@3"]), f3(r["Accuracy"]), f3(r["Top-3 accuracy"]), f3(r["Macro-F1"]),
      f3(r["Weighted-F1"]), f"{r['Entrenamiento (s)']:.1f}", f"{r['Predicción (s)']:.2f}"]
     for _, r in orden_tabla.iterrows()],
    "Desempeño y tiempos de los modelos en el conjunto de prueba.",
    anchos=[4.4, 1.5, 1.7, 1.4, 1.7, 1.6, 1.8, 1.6], tam=8.5,
)
f = figura("modelos/comparacion_metricas.png",
           "Comparación de MAP@3, exactitud y macro-F1 en el conjunto de prueba.")
lr, svm, nb, bl = (M.loc["Regresión logística jerárquica"], M.loc["SVM lineal"],
                   M.loc["Complement Naive Bayes"], M.loc[BASE])
p(f"**Efectividad (Figura {f}).** Los tres modelos discriminativos sobre TF-IDF superan con "
  f"claridad a la línea base: la regresión logística jerárquica alcanza **MAP@3 = "
  f"{f3(lr['MAP@3'])}**, frente a {f3(bl['MAP@3'])} de la línea base que solo usa la pregunta y "
  f"la opción. Leer la explicación aporta, por tanto, unos {(lr['MAP@3'] - bl['MAP@3']) * 100:.0f} "
  f"puntos de MAP@3, {(lr['Accuracy'] - bl['Accuracy']) * 100:.0f} puntos de exactitud y casi "
  f"duplica el macro-F1 ({f3(bl['Macro-F1'])} → {f3(lr['Macro-F1'])}).")
p(f"La regresión logística (simple o jerárquica) y la SVM lineal quedan prácticamente empatadas: "
  f"la diferencia de MAP@3 es de {(lr['MAP@3'] - svm['MAP@3']) * 100:.1f} puntos, comparable a la "
  "variación entre pliegues observada en la validación cruzada. La SVM obtiene una exactitud "
  f"ligeramente mayor ({f3(svm['Accuracy'])}), mientras que la regresión logística tiene mejor "
  f"exactitud top-3 ({f3(lr['Top-3 accuracy'])} frente a {f3(svm['Top-3 accuracy'])}): sus "
  "probabilidades ordenan mejor el segundo y tercer candidato, que es justo lo que premia MAP@3.")
p(f"**Complement Naive Bayes** fue el modelo más débil (MAP@3 = {f3(nb['MAP@3'])}), incluso por "
  "debajo de la línea base. Su supuesto de independencia entre términos no se sostiene con "
  "n-gramas de caracteres solapados, que cuentan la misma evidencia muchas veces, y sus "
  "probabilidades resultan mal calibradas para el ranking.")
p("**Modelo jerárquico.** La restricción jerárquica mejora el MAP@3 solo de forma marginal "
  f"({M.loc['Regresión logística', 'MAP@3']:.4f} → {lr['MAP@3']:.4f}) y deja la primera "
  "predicción sin cambios. La razón es que la matriz de entrada ya incluye la codificación del "
  "par (pregunta, opción), por lo que la regresión logística aprendió por sí sola qué etiquetas "
  "son compatibles con cada contexto. La restricción sigue siendo útil como garantía: impide "
  "proponer una concepción errónea que nunca se ha observado en esa pregunta.")
f = figura("modelos/desempeno_vs_tiempo.png",
           "Relación entre MAP@3 y tiempo de entrenamiento (escala logarítmica).", 13.5)
p(f"**Tiempos de procesamiento (Figura {f}).** La SVM lineal entrena en unos "
  f"{svm['Entrenamiento (s)']:.0f} s, entre 5 y 7 veces más rápido que la regresión logística "
  f"(≈ {lr['Entrenamiento (s)']:.0f}–{M.loc['Regresión logística', 'Entrenamiento (s)']:.0f} s), "
  "con una calidad casi idéntica. La predicción es rápida en todos los casos (alrededor de 1 s "
  "para 7 339 explicaciones, es decir, ≈ 0.15 ms por explicación), por lo que cualquiera de "
  "ellos es viable en una aplicación interactiva. Las dos variantes de regresión logística "
  "entrenan el mismo modelo; la diferencia de tiempo entre ellas refleja variación de la "
  "máquina y no un costo adicional.")

h("5.6 Análisis de errores", 2)
p(f"Para el modelo seleccionado, el desempeño por dimensión de la etiqueta es:")
tabla(
    ["Dimensión evaluada", "Exactitud"],
    [[k, pct(v)] for k, v in DIM.items()],
    "Exactitud del modelo seleccionado por nivel de la jerarquía.", anchos=[10, 4],
)
p(f"La **veracidad** se predice casi perfectamente ({pct(DIM['Veracidad (True/False)'])}), "
  "lo que confirma el hallazgo del análisis exploratorio. La dificultad está en el **tipo de "
  "razonamiento**. En las filas que realmente contienen una concepción errónea, el modelo "
  f"propone como primera opción la concepción correcta en el "
  f"{pct(DIM['Misconception (solo filas con concepción)'])} de los casos.")
f = figura("modelos/matriz_confusion_category.png",
           "Matriz de confusión de Category normalizada por fila (modelo seleccionado).", 12.5)
p(f"La matriz de confusión (Figura {f}) muestra dos bloques separados (`True_*` y `False_*`), "
  "sin confusiones entre ellos. Dentro de cada bloque:")
viñetas([
    "`True_Correct` (0.94) y `False_Misconception` (0.87) se reconocen bien.",
    "`True_Neither` se confunde con `True_Correct` en un 33 % de los casos: distinguir una "
    "explicación correcta pero vaga de una suficientemente argumentada es subjetivo incluso para "
    "un evaluador humano.",
    "`True_Misconception` —acertar con un razonamiento erróneo— solo se detecta en un 38 % de los "
    "casos. Con 403 ejemplos en todo el conjunto, es la categoría más difícil y la más valiosa "
    "pedagógicamente.",
    "`False_Correct` (0.50) tiene apenas 227 ejemplos y se confunde con `False_Neither` y "
    "`False_Misconception`.",
])
f = figura("modelos/f1_vs_soporte.png",
           "F1 de cada etiqueta compuesta en prueba frente a su número de ejemplos de entrenamiento.",
           13.5)
p(f"La Figura {f} explica el macro-F1 moderado ({f3(lr['Macro-F1'])}): **todas las clases con "
  "menos de 15 ejemplos de entrenamiento obtienen F1 = 0**, mientras que las clases con más de 100 "
  "ejemplos obtienen F1 entre 0.63 y 0.98, la mayoría por encima de 0.8. El cuello de botella del modelo es la cola larga, no la "
  "capacidad del algoritmo. Esto concuerda con la búsqueda de hiperparámetros: ponderar las "
  "clases no fue suficiente para compensar la falta de ejemplos.")
peor = min(resumen["map3_por_pregunta"].items(), key=lambda kv: kv[1])
mejor = max(resumen["map3_por_pregunta"].items(), key=lambda kv: kv[1])
p(f"Por pregunta, el MAP@3 oscila entre {f3(peor[1])} (pregunta {peor[0]}) y {f3(mejor[1])} "
  f"(pregunta {mejor[0]}); el desempeño es homogéneo entre las 15 preguntas conocidas.")

h("5.7 Generalización a preguntas no vistas", 2)
f = figura("modelos/generalizacion_preguntas.png",
           "MAP@3 con preguntas conocidas (partición estratificada) y con preguntas no vistas "
           "(GroupKFold por QuestionId).", 13.5)
tl, tb = TR["Regresión logística"], TR[BASE]
p(f"Cuando se evalúa con preguntas completas que el modelo nunca vio (Figura {f}), el MAP@3 de "
  f"la regresión logística cae de {f3(M.loc['Regresión logística', 'MAP@3'])} a "
  f"**{f3(tl['MAP@3 preguntas no vistas'])}**, y queda **por debajo** de la línea base "
  f"({f3(tb['MAP@3 preguntas no vistas'])}), que en ese escenario se reduce a proponer las "
  "etiquetas más frecuentes del conjunto.")
p("Este resultado es esperable y coincide con lo reportado por Padó et al. (2024): el 91 % de "
  "las concepciones erróneas de una pregunta nueva no existen en entrenamiento, y el vocabulario "
  "aprendido (números, figuras, operaciones) es específico de cada pregunta. El modelo transfiere "
  "esos patrones a contextos donde no aplican y propone concepciones de otras preguntas. **El "
  "sistema construido es, por tanto, adecuado para las 15 preguntas del banco, pero no para "
  "preguntas nuevas sin datos etiquetados.** Para ellas se requerirían representaciones que "
  "incorporen el texto de la pregunta y de la concepción errónea —por ejemplo, un *Transformer* "
  "que compare la explicación con la descripción de cada concepción— y un número mayor de "
  "preguntas de entrenamiento.")

h("5.8 Selección del modelo", 2)
p("Con base en la comparación se seleccionan dos modelos:")
viñetas([
    f"**Modelo principal: regresión logística jerárquica (C = {HP['Regresión logística']['C']:g}).** "
    f"Obtiene el mejor MAP@3 ({f3(lr['MAP@3'])}) y la mejor exactitud top-3 "
    f"({f3(lr['Top-3 accuracy'])}), entrega probabilidades que permiten mostrar al docente la "
    "confianza de cada sugerencia y aplicar un umbral de revisión humana, y garantiza que las "
    "etiquetas propuestas sean coherentes con la pregunta.",
    f"**Alternativa eficiente: SVM lineal (C = {HP['SVM lineal']['C']:g}).** Prácticamente la misma "
    "calidad con un tiempo de entrenamiento entre 5 y 7 veces menor. Es preferible si se requiere "
    "reentrenar con frecuencia o si se valora la exactitud de la primera sugerencia.",
])
p("La línea base y Naive Bayes se conservan en la aplicación como puntos de comparación, para "
  "que el usuario observe cuánto aporta la lectura del texto.")

h("5.9 Aplicación", 2)
p("**Estado: pendiente de implementación.** La guía del proyecto solicita una aplicación que "
  "permita ingresar una explicación nueva y clasificarla con los modelos entrenados, explorar las "
  "variables del conjunto limpio, ver los resultados de cada algoritmo por separado o en conjunto "
  "y mostrar su rendimiento en gráficas interactivas que el usuario pueda ocultar. Al cierre de "
  "este documento la aplicación no se ha construido; esta sección debe completarse con su "
  "descripción, capturas de pantalla y la justificación de las tecnologías utilizadas.")
p("Los componentes necesarios ya están disponibles en el repositorio: los cinco modelos "
  "entrenados en `models/*.joblib` (cada uno incluye su propio preprocesamiento), las métricas de "
  "prueba en `results/metricas_prueba.csv` y `results/resumen_modelado.json`, y las funciones de "
  "limpieza en `notebooks/02_modelado.py`. Se propone construirla con **Streamlit** y **Plotly**, "
  "por su integración directa con Python y scikit-learn, con la misma paleta de colores de este "
  "informe (azul institucional para el modelo principal y acentos diferenciados por modelo).")

# ===========================================================================
# 6. CONCLUSIONES
# ===========================================================================
h("6. Conclusiones", 1)
viñetas([
    "**Caracterización de los datos (objetivo 1).** El análisis exploratorio mostró que los nulos "
    "de `Misconception` son estructurales, que las clases presentan una cola larga extrema y que "
    "las concepciones erróneas están ligadas a la pregunta. Estas evidencias determinaron todas "
    "las decisiones posteriores: no imputar, conservar números y operadores, usar MAP@3 y "
    "macro-F1 y agrupar duplicados al particionar.",
    "**Revisión bibliográfica (objetivo 2).** La literatura de ASAG orientó la selección de "
    "TF-IDF con clasificadores lineales como primera familia de modelos, la incorporación del "
    "contexto de la pregunta y la necesidad de evaluar con preguntas no vistas.",
    f"**Construcción y ajuste de modelos (objetivo 3).** Se entrenaron cinco modelos y se "
    "ajustaron sus hiperparámetros con validación cruzada agrupada. El ajuste mostró modelos "
    "estables ante cambios en C y que la ponderación de clases perjudica el ranking.",
    f"**Evaluación (objetivo 4).** La regresión logística jerárquica alcanzó **MAP@3 = "
    f"{f3(lr['MAP@3'])}** y exactitud de {pct(lr['Accuracy'])} sobre 65 clases, frente a "
    f"{f3(bl['MAP@3'])} de una línea base sin texto. La SVM lineal logró un desempeño equivalente "
    "con un costo de entrenamiento mucho menor. Naive Bayes resultó inadecuado para este problema.",
    "**Modelo jerárquico (objetivo 5).** La veracidad se predice con 99.9 % de exactitud a partir "
    "del contexto. La restricción jerárquica aportó una mejora mínima porque el contexto ya "
    "estaba en la representación, pero asegura predicciones coherentes con la pregunta.",
    f"**Generalización (objetivo 6).** Con preguntas no vistas el MAP@3 cae a "
    f"{f3(tl['MAP@3 preguntas no vistas'])}. Los modelos son eficaces para el banco de preguntas "
    "conocido, pero no transfieren a preguntas nuevas; este es el principal límite del enfoque.",
])
p("**Efectividad de los algoritmos.** Para el problema planteado —ordenar las tres etiquetas más "
  "probables en preguntas conocidas— los clasificadores lineales sobre TF-IDF de palabras y "
  "caracteres son eficaces, rápidos e interpretables. Sus debilidades son las clases con muy "
  "pocos ejemplos (F1 = 0 por debajo de 15 ejemplos), las categorías pedagógicamente sutiles "
  "(`True_Misconception`, `True_Neither`) y las preguntas nuevas. Por ello las predicciones deben "
  "usarse para **priorizar la revisión docente**, no como diagnóstico definitivo.")
h("Trabajo futuro", 2)
viñetas([
    "Construir la aplicación interactiva descrita en la sección 5.9.",
    "Ajustar DeBERTa o MathBERT con la entrada [pregunta] [opción] [explicación] en GPU y "
    "compararlo con la regresión logística bajo las mismas particiones.",
    "Ensamblar las probabilidades del modelo lineal y del *Transformer*.",
    "Aumentar de forma conservadora los datos de las clases raras, con revisión docente.",
    "Definir un umbral de confianza por debajo del cual la predicción se envía a revisión humana.",
])

# ===========================================================================
# REFERENCIAS
# ===========================================================================
h("Referencias", 1)
REFS = [
    "Bonthu, S., Sree, S. R., & Krishna Prasad, M. H. M. (2024). Framework for automation of short "
    "answer grading based on domain-specific pre-training. *Engineering Applications of Artificial "
    "Intelligence, 137*, 109163. https://doi.org/10.1016/j.engappai.2024.109163",
    "Burrows, S., Gurevych, I., & Stein, B. (2015). The eras and trends of automatic short answer "
    "grading. *International Journal of Artificial Intelligence in Education, 25*(1), 60–117. "
    "https://doi.org/10.1007/s40593-014-0026-8",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. "
    "https://doi.org/10.1007/BF00994018",
    "del Gobbo, E., Guarino, A., Cafarelli, B., & Grilli, L. (2023). GradeAid: A framework for "
    "automatic short answers grading in educational contexts—Design, implementation and "
    "evaluation. *Knowledge and Information Systems, 65*, 4295–4334. "
    "https://doi.org/10.1007/s10115-023-01892-9",
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep "
    "bidirectional transformers for language understanding. En *Proceedings of the 2019 "
    "Conference of the North American Chapter of the Association for Computational Linguistics: "
    "Human Language Technologies* (Vol. 1, pp. 4171–4186). Association for Computational "
    "Linguistics. https://doi.org/10.18653/v1/N19-1423",
    "He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with "
    "disentangled attention. En *International Conference on Learning Representations*. "
    "https://openreview.net/forum?id=XPZIaotutsD",
    "Kaggle. (2025). *MAP—Charting student math misunderstandings* [Competencia de ciencia de "
    "datos]. https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings",
    "Morley, F., Walland, E., & Vidal Rodeiro, C. L. (2026). Auto-marking short answer questions "
    "in science: The foundational years of transformer-based models from BERT to GPT-4. "
    "*International Journal of Artificial Intelligence in Education, 36*(1–2), 100005. "
    "https://doi.org/10.1016/j.ijaied.2026.100005",
    "Padó, U., Eryilmaz, Y., & Kirschner, L. (2024). Short-answer grading for German: Addressing "
    "the challenges. *International Journal of Artificial Intelligence in Education, 34*(4), "
    "1321–1352. https://doi.org/10.1007/s40593-023-00383-w",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
    "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., "
    "Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. "
    "*Journal of Machine Learning Research, 12*, 2825–2830.",
    "Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions "
    "of naive Bayes text classifiers. En *Proceedings of the Twentieth International Conference "
    "on Machine Learning* (pp. 616–623). AAAI Press.",
    "Rittle-Johnson, B., Adler, R., Durkin, K., Burleigh, L., King, J., & Crossley, S. (2025). "
    "Detecting math misconceptions: An AI benchmark dataset. En *Proceedings of the 1st Workshop "
    "on Advancing Artificial Intelligence through Theory of Mind* (pp. 20–24). Association for "
    "Computational Linguistics. https://aclanthology.org/2025.aimecon-wip.3/",
    "Rushton, S. J. (2018). Teaching and learning mathematics through error analysis. *Fields "
    "Mathematics Education Journal, 3*, Artículo 4. https://doi.org/10.1186/s40928-018-0009-y",
    "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. "
    "*Information Processing & Management, 24*(5), 513–523. "
    "https://doi.org/10.1016/0306-4573(88)90021-0",
    "Shen, J. T., Yamashita, M., Prihar, E., Heffernan, N. T., Wu, X., Lee, D., & Wang, D. (2021). "
    "MathBERT: A pre-trained language model for general NLP tasks in mathematics education. "
    "*arXiv*. https://doi.org/10.48550/arXiv.2106.07340",
    "Zhang, Y., Lin, C., & Chi, M. (2020). Going deeper: Automatic short-answer grading by "
    "combining student and question models. *User Modeling and User-Adapted Interaction, 30*(1), "
    "51–80. https://doi.org/10.1007/s11257-019-09251-6",
    "Zhu, X., Wu, H., & Zhang, L. (2022). Automatic short-answer grading via BERT-based deep "
    "neural networks. *IEEE Transactions on Learning Technologies, 15*(3), 364–375. "
    "https://doi.org/10.1109/TLT.2022.3175537",
]
import re as _re
for ref in REFS:
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(1.27)
    par.paragraph_format.first_line_indent = Cm(-1.27)
    for trozo in _re.split(r"(\*[^*]+\*)", ref):
        if trozo.startswith("*"):
            par.add_run(trozo[1:-1]).italic = True
        elif trozo:
            par.add_run(trozo)

# ===========================================================================
# ANEXO: GLOSARIO
# ===========================================================================
h("Anexo A. Glosario", 1)
tabla(
    ["Término", "Definición"],
    [
        ["Concepción errónea (misconception)", "Idea matemática equivocada y persistente que "
         "produce errores sistemáticos de razonamiento."],
        ["Etiqueta compuesta", "Combinación Category:Misconception; 65 clases distintas."],
        ["Nulo estructural", "Valor vacío porque el dato no aplica, no porque se haya perdido."],
        ["Cola larga", "Distribución con muchas clases de muy pocos ejemplos."],
        ["TF-IDF", "Peso de un término proporcional a su frecuencia en el documento e inverso a su "
         "frecuencia en el corpus."],
        ["N-grama", "Secuencia de n palabras o caracteres consecutivos."],
        ["Matriz dispersa", "Matriz con casi todos sus valores en cero."],
        ["Hiperparámetro", "Parámetro fijado antes del entrenamiento (C, α) que controla la "
         "complejidad del modelo."],
        ["Regularización (C)", "Penalización de pesos grandes para evitar sobreajuste; C es su "
         "inverso."],
        ["Validación cruzada", "Evaluación repetida rotando el pliegue de validación."],
        ["StratifiedGroupKFold", "Partición que conserva la proporción de clases y mantiene juntos "
         "los miembros de un grupo."],
        ["GroupKFold", "Partición que deja grupos completos (aquí, preguntas) fuera del "
         "entrenamiento."],
        ["Fuga de información", "Filtración de información de evaluación al entrenamiento que "
         "infla las métricas."],
        ["MAP@3", "Precisión media de las tres predicciones ordenadas: 1, 1/2 o 1/3 según la "
         "posición del acierto."],
        ["Macro-F1", "Promedio del F1 de todas las clases con igual peso."],
        ["Matriz de confusión", "Tabla que cruza clases reales y predichas."],
        ["Línea base (baseline)", "Modelo simple de referencia para medir la ganancia de modelos "
         "más complejos."],
        ["Transformer", "Arquitectura neuronal basada en mecanismos de atención; base de BERT y "
         "DeBERTa."],
    ],
    "Glosario de términos.", anchos=[4.5, 12],
)

# Numeración de páginas en el pie
for s in doc.sections:
    par = s.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    campo(par, "PAGE")

doc.save(SALIDA)
print(f"Informe generado: {SALIDA.relative_to(ROOT)}")

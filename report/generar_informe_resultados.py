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

def leer_json(nombre):
    ruta = RES / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


metricas = pd.read_csv(RES / "metricas_prueba.csv")
busqueda = pd.read_csv(RES / "busqueda_hiperparametros.csv")
LIN = leer_json("resumen_lineales.json")
FIN = leer_json("resumen_final.json")
TRANSF = pd.read_csv(RES / "transferencia.csv")
DEB = leer_json("deberta_resumen.json")
MATH = leer_json("mathbert_resumen.json")
DEB_G = leer_json("deberta_grupos.json")
MATH_G = leer_json("mathbert_grupos.json")

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
PRIOR = "Prior (pregunta+opción)"
SVM = "TF-IDF + SVM lineal"
RL = "TF-IDF + Regresión logística jerárquica"
ENS = "Ensamble calibrado"
HP = LIN["mejores_hiperparametros"]
TB = TRANSF.set_index("Modelo")
ABL = pd.DataFrame(LIN["ablacion"]).set_index("Entrada")
DIM = FIN["por_dimension"]
NT = FIN.get("transformer_ensamble") or "DeBERTa-v3"
HAY_T = bool(FIN.get("hay_deberta"))
mejor = metricas.sort_values("MAP@3").iloc[-1]["Modelo"]

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

h("Índice", 1)
campo(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u',
      "Clic derecho → Actualizar campo para generar el índice.")
salto()

# ===========================================================================
# 1. INTRODUCCIÓN
# ===========================================================================
h("1. Introducción", 1)
p("Cuando un estudiante resuelve mal un problema de matemáticas, la respuesta incorrecta es solo "
  "el síntoma. Lo que orienta al docente es el porqué: el razonamiento defectuoso —la **concepción "
  "errónea** o *misconception*— que produjo esa respuesta. Los errores suelen reflejar reglas "
  "aplicadas fuera de contexto o generalizaciones excesivas, y la explicación escrita permite "
  "distinguir un descuido aislado de un error conceptual persistente (Rushton, 2018). Del mismo "
  "modo, una respuesta correcta no garantiza un razonamiento correcto.")
p("El reto de Kaggle **MAP — Charting Student Math Misunderstandings** (Kaggle, 2025; "
  "Rittle-Johnson et al., 2025) propone automatizar parte de ese diagnóstico. Cada observación "
  "contiene una pregunta de opción múltiple, la opción elegida y una breve explicación escrita. El "
  "sistema debe producir una lista ordenada de hasta tres pares `Category:Misconception`, evaluada "
  "con MAP@3. El problema combina, por tanto, **clasificación multiclase, predicción jerárquica, "
  "ranking y procesamiento de lenguaje matemático**.")
p("Este informe reúne el análisis exploratorio y la revisión bibliográfica realizados previamente "
  "y presenta la fase de resultados: la construcción y el ajuste de modelos de cuatro niveles de "
  "complejidad —líneas base TF-IDF, un *Transformer* contextual, un modelo adaptado al dominio "
  "matemático y un ensamble calibrado—, su evaluación bajo dos protocolos de validación, la "
  "comparación que permite seleccionar el modelo final y la aplicación interactiva construida "
  "para usarlos.")
p("**Problema a resolver.** Revisar manualmente miles de explicaciones abiertas exige tiempo y "
  "conocimiento pedagógico especializado. Un clasificador puede **priorizar casos**, pero no "
  "reemplaza el juicio del docente: cuando la incertidumbre es alta, el sistema debe abstenerse y "
  "derivar el caso a revisión humana. El sistema debe además (1) entender lenguaje informal y "
  "notación matemática, (2) condicionarse a la pregunta y a la opción elegida, (3) aprender con "
  "clases muy desbalanceadas, (4) producir un ranking y (5) generalizar sin memorizar.")
p("**Pregunta científica.** ¿Qué patrones del texto, de su estructura y del contexto permiten "
  "distinguir si la respuesta es correcta, qué tan bueno es el razonamiento y qué concepción "
  "errónea hay detrás, y qué familia de modelos los aprovecha mejor sin perder la capacidad de "
  "generalizar a preguntas nuevas?")

# ===========================================================================
# 2. OBJETIVOS
# ===========================================================================
h("2. Objetivos", 1)
h("2.1 Objetivo general", 2)
p("Construir, ajustar y comparar modelos de aprendizaje automático para clasificar explicaciones "
  "matemáticas de estudiantes en la etiqueta `Category:Misconception` del reto MAP, y seleccionar "
  "una arquitectura que equilibre desempeño, generalización, calibración y costo computacional, "
  "fundamentada en el análisis exploratorio y en la literatura revisada.")
h("2.2 Objetivos específicos", 2)
viñetas([
    "Identificar familias de modelos, representaciones y esquemas de evaluación usados en la "
    "literatura de calificación automática de respuestas cortas y detección de errores conceptuales.",
    "Contrastar sus ventajas y limitaciones con el tamaño, la estructura, el desbalance y la "
    "dependencia contextual del corpus MAP.",
    "Entrenar líneas base TF-IDF, un modelo principal tipo *Transformer*, una alternativa de "
    "adaptación al dominio matemático y un ensamble calibrado con ranking top-3.",
    "Evaluar los modelos con un protocolo que evite fuga entre preguntas y mida el desempeño "
    "global, por clase y por pregunta (MAP@3, macro-F1, exactitud y tiempos).",
    "Establecer controles de confiabilidad y supervisión humana: calibración de probabilidades y "
    "un umbral de abstención.",
    "Construir una aplicación que permita clasificar explicaciones nuevas, explorar los datos y "
    "comparar el rendimiento de los modelos.",
], numerada=True)

# ===========================================================================
# 3. MARCO TEÓRICO
# ===========================================================================
h("3. Marco teórico", 1)
h("3.1 Procesamiento del Lenguaje Natural", 2)
p("El **Procesamiento del Lenguaje Natural** (PLN o NLP) es el área de la inteligencia artificial "
  "y la lingüística computacional que desarrolla métodos para que una computadora represente, "
  "analice, interprete y genere lenguaje humano. Incluye tareas como la clasificación de textos, "
  "la extracción de información, la traducción automática y la respuesta a preguntas. Este "
  "proyecto aborda una tarea de **clasificación de texto corto multiclase con ranking**: cada "
  "explicación (unas 13 palabras de mediana) se asigna a una de 65 clases, y se devuelven las tres "
  "más probables.")
p("En educación, el PLN se ha aplicado a la **calificación automática de respuestas cortas** "
  "(*Automatic Short Answer Grading*, ASAG). La evolución del campo pasa por cuatro etapas: reglas "
  "y similitud con respuestas modelo, interpretables pero frágiles ante paráfrasis; representaciones "
  "léxicas como TF-IDF con clasificadores lineales, competitivas y baratas para texto corto; "
  "modelos de lenguaje preentrenados como BERT y DeBERTa, que capturan contexto bidireccional; y "
  "adaptación al dominio, que incorpora vocabulario y patrones específicos (Burrows et al., 2015; "
  "Morley et al., 2026). La literatura converge en cuatro decisiones: usar el contexto completo "
  "(pregunta y respuesta además de la explicación), aprovechar la transferencia cuando hay clases "
  "con pocos ejemplos, considerar la adaptación al dominio y tratar la calibración, la robustez y la "
  "supervisión humana como requisitos (Zhang et al., 2020; Bonthu et al., 2024; Schneider et al., "
  "2023; Filighera et al., 2024).")

h("3.2 Actividades de preprocesamiento", 2)
viñetas([
    "**Normalización:** conversión a minúsculas y unificación de espacios.",
    "**Limpieza de marcado:** las opciones venían en LaTeX (96 %), por ejemplo "
    "`\\( \\frac{1}{3} \\)`, y se convirtieron a texto plano (`1/3`).",
    "**Tokenización:** división del texto en unidades. Para TF-IDF se definió un patrón que conserva "
    "números y operadores (`+ - * / = < > %`); los *Transformers* usan su propio tokenizador de "
    "subpalabras.",
    "**Palabras vacías (*stopwords*):** no se eliminaron, porque palabras como *not* o *than* "
    "cambian el sentido matemático.",
    "**Números, operadores, signos y negaciones:** se conservaron. El 78.3 % de las explicaciones "
    "contiene dígitos; son la principal señal matemática.",
    "**Corrección ortográfica y lematización:** no se aplicaron; los n-gramas de caracteres y los "
    "tokenizadores de subpalabras toleran errores como \"tree nineth\".",
    "**Vectorización:** TF-IDF para los modelos lineales y *embeddings* contextuales para los "
    "*Transformers*.",
])

h("3.3 Representación TF-IDF y n-gramas", 2)
p("**TF-IDF** (*Term Frequency – Inverse Document Frequency*) asigna a cada término t de un "
  "documento d un peso proporcional a su frecuencia en el documento e inversamente proporcional al "
  "número de documentos que lo contienen (Salton & Buckley, 1988):")
p("tfidf(t, d) = tf(t, d) · [log((1 + N) / (1 + df(t))) + 1]", alinear=WD_ALIGN_PARAGRAPH.CENTER)
p("donde N es el número de documentos y df(t) el número que contiene t; se usó la variante "
  "sublineal 1 + log(tf). Un **n-grama** es una secuencia de n unidades consecutivas: los de "
  "palabras (n = 1, 2) capturan expresiones como \"divided by\", y los de caracteres (n = 2 a 5) "
  "capturan fragmentos como \"3/9\" y dan robustez ortográfica.")

h("3.4 Algoritmos de clasificación", 2)
p("**Prior por pregunta y opción.** Ordena las etiquetas por su frecuencia en entrenamiento dentro "
  "del mismo par (pregunta, opción). No lee el texto: mide cuánto se logra solo con el contexto.")
p("**Regresión logística multinomial.** Estima P(y = k | x) = exp(w_k·x) / Σ_j exp(w_j·x) "
  "minimizando la entropía cruzada con regularización L2 de intensidad 1/C. Entrega probabilidades, "
  "necesarias para ordenar candidatos y aplicar un umbral de abstención.")
p("**Máquina de vectores de soporte (SVM) lineal.** Busca para cada clase el hiperplano de máximo "
  "margen (Cortes & Vapnik, 1995); es muy eficaz en espacios dispersos de alta dimensión. No "
  "produce probabilidades: el ranking se construye con la distancia al hiperplano.")
p("**Modelo jerárquico.** Descompone la predicción según la estructura de la etiqueta: la veracidad "
  "(si la opción es correcta) se determina con la pregunta y la opción, y el tipo de razonamiento "
  "y la concepción errónea se restringen a las etiquetas compatibles con esa veracidad y "
  "observadas en la misma pregunta.")
p("**Transformers.** Arquitectura basada en mecanismos de atención que representa cada token en "
  "función de todos los demás. BERT (Devlin et al., 2019) se preentrena con texto no etiquetado y "
  "se ajusta a la tarea supervisada (*fine-tuning*). **DeBERTa** (He et al., 2021) mejora a BERT "
  "con atención desacoplada —contenido y posición se codifican por separado— y un decodificador "
  "de máscara mejorado; su versión v3 es una de las más competitivas en clasificación. La entrada "
  "contextual concatena pregunta, opción y explicación, siguiendo la evidencia de que el contexto "
  "de la pregunta mejora la representación (Zhang et al., 2020; Zhu et al., 2022).")
p("**Adaptación al dominio.** **MathBERT** (Shen et al., 2021) es un BERT preentrenado con textos "
  "de educación matemática; el preentrenamiento específico puede capturar vocabulario y "
  "estructuras del dominio (Bonthu et al., 2024).")
p("**Ensamble calibrado.** Combina las probabilidades de modelos complementarios, "
  "p_E = α·p_Transformer + (1 − α)·p_lineal, con α elegido en validación (Sahu & Bhowmick, 2020; "
  "del Gobbo et al., 2023).")

h("3.5 Métricas, calibración y abstención", 2)
viñetas([
    "**MAP@3:** métrica oficial. De tres etiquetas ordenadas, un acierto en la primera posición "
    "vale 1, en la segunda 1/2 y en la tercera 1/3.",
    "**Exactitud y exactitud top-3:** proporción de casos con la etiqueta real en la primera "
    "posición o entre las tres primeras. La exactitud es engañosa con clases desbalanceadas.",
    "**Macro-F1 y F1 ponderado:** el F1 combina precisión y exhaustividad; el macro-F1 promedia "
    "todas las clases con igual peso, por lo que es sensible a las clases raras.",
    "**Calibración y ECE:** un modelo está calibrado si, entre los casos en que dice tener 80 % de "
    "confianza, acierta el 80 %. El *Expected Calibration Error* (ECE) promedia la diferencia entre "
    "confianza y exactitud por intervalos. El **escalado de temperatura** divide los logits de una "
    "red entre un escalar T ajustado en validación y es un método simple y eficaz para calibrar "
    "redes neuronales (Guo et al., 2017).",
    "**Abstención:** el sistema solo acepta la predicción si su confianza supera un umbral τ; en "
    "caso contrario deriva el caso al docente. La curva cobertura–exactitud muestra el compromiso.",
])
h("3.6 Validación y fuga de información", 2)
p("La **validación cruzada** rota el conjunto de validación entre k pliegues; la variante "
  "**estratificada** conserva la proporción de clases y la **agrupada** (*GroupKFold*) mantiene "
  "cada grupo en un solo pliegue. Hay **fuga de información** cuando el modelo ve en entrenamiento "
  "información del conjunto de evaluación —por ejemplo, textos duplicados o la misma pregunta—, "
  "lo que infla las métricas. Padó et al. (2024) muestran que los *Transformers* pueden ser "
  "robustos en preguntas vistas y no superar a otros modelos en preguntas no vistas, por lo que "
  "ambos escenarios deben medirse por separado.")

# ===========================================================================
# 4. METODOLOGÍA
# ===========================================================================
h("4. Metodología", 1)
h("4.1 Pasos seguidos", 2)
viñetas([
    "Revisión bibliográfica de algoritmos de ASAG y detección de concepciones erróneas.",
    "Análisis exploratorio (`notebooks/01_analisis_exploratorio.ipynb`).",
    "Preprocesamiento derivado de los hallazgos y particiones sin fuga (`map_modelos.py`).",
    "Líneas base TF-IDF con búsqueda de hiperparámetros, ablación del contexto de entrada y "
    "validación por preguntas no vistas (`notebooks/02_modelado.py`).",
    "Ajuste fino de DeBERTa-v3 y MathBERT en GPU (`notebooks/03_transformer.py`, ejecutado en "
    "Kaggle mediante `kaggle/construir_kernels.py`).",
    "Calibración, ensamble, abstención y figuras (`notebooks/04_ensamble.py`).",
    "Comparación y selección del modelo; construcción de la aplicación (`app/app.py`).",
], numerada=True)
p("La complejidad se adoptó de forma incremental —línea base, *Transformer*, adaptación al "
  "dominio y ensamble—, con el criterio de **incorporar un modelo más costoso solo si mejora el "
  "MAP@3 y el macro-F1 bajo el mismo protocolo de validación**.")

h("4.2 Selección de los conjuntos de entrenamiento, validación y prueba", 2)
p("El `test.csv` publicado tiene solo 3 filas (el conjunto real solo era accesible dentro de la "
  "competencia, ya cerrada), por lo que toda la evaluación se construyó particionando "
  f"`train.csv`. Se definieron dos protocolos complementarios:")
viñetas([
    f"**Validación A — preguntas conocidas.** `StratifiedGroupKFold` (5 pliegues, semilla 42): un "
    f"pliegue es la **prueba** ({miles(LIN['n_test'])} filas, 20 %) y el resto el **entrenamiento** "
    f"({miles(LIN['n_train'])} filas). Es estratificada por la etiqueta compuesta y agrupa las "
    "explicaciones idénticas de una misma pregunta (970 repetidas), para que ninguna copia quede a "
    "ambos lados. Dentro del entrenamiento se reservó un 10 % de **validación** "
    f"({miles(LIN['n_val'])} filas) para detener el entrenamiento de los *Transformers*, calibrar "
    "las probabilidades y fijar el peso del ensamble. Responde a: ¿funciona con respuestas nuevas "
    "a preguntas conocidas?",
    "**Validación B — preguntas no vistas.** `GroupKFold` por `QuestionId` (3 pliegues): cada "
    "pliegue deja fuera 5 preguntas completas. Responde a: ¿transfiere a preguntas nuevas sin "
    "memorizar el identificador? Como 32 de las 35 concepciones erróneas existen en una sola "
    "pregunta, esta prueba es deliberadamente severa.",
])
p("El conjunto de prueba se usó una sola vez por modelo. Los hiperparámetros de los modelos "
  "lineales se eligieron con validación cruzada interna de 3 pliegues; los de los *Transformers*, "
  "con la partición de validación.")
p("**Ablación del contexto de entrada.** Para medir cuánto aporta cada fuente de información se "
  "comparó la regresión logística con tres entradas: texto solo, texto + pregunta y "
  "texto + pregunta + opción.")

h("4.3 Selección de los algoritmos", 2)
tabla(
    ["Nivel", "Modelo", "Razón de la selección"],
    [
        ["Referencia", "Prior (pregunta + opción)",
         "El par (pregunta, opción) determina la veracidad en 58 de 60 casos; mide cuánto se logra "
         "sin leer el texto."],
        ["Línea base", "TF-IDF + SVM lineal",
         "Referencia clásica de ASAG, barata y competitiva en texto corto (Burrows et al., 2015)."],
        ["Línea base", "TF-IDF + regresión logística jerárquica",
         "Interpretable y con probabilidades; explota la estructura jerárquica de la etiqueta."],
        ["Principal", "DeBERTa-v3-base",
         "Contexto bidireccional con atención desacoplada; entrada pregunta + opción + explicación "
         "(He et al., 2021; Zhu et al., 2022)."],
        ["Dominio", "MathBERT",
         "Vocabulario y patrones de educación matemática; se adopta solo si mejora a DeBERTa, en "
         "especial en clases raras y preguntas nuevas (Shen et al., 2021)."],
        ["Salida", "Ensamble calibrado con ranking top-3",
         "Combina evidencia léxica y contextual; optimiza MAP@3 sin ocultar la incertidumbre "
         "(Sahu & Bhowmick, 2020; Guo et al., 2017)."],
    ],
    "Algoritmos seleccionados por nivel y justificación.", anchos=[2.4, 4.5, 9.6],
)
p("**Uso de `QuestionId`.** El análisis exploratorio advirtió que el identificador de la pregunta "
  "puede convertirse en un atajo: el modelo memoriza qué concepciones corresponden a cada pregunta "
  "en lugar de leer el razonamiento. Aun así se incluyó en la línea base, porque la aplicación se "
  "usa sobre el banco conocido de 15 preguntas; su costo se cuantificó explícitamente con la "
  "ablación y con la validación B (secciones 5.6 y 5.7). Los *Transformers* no reciben el "
  "identificador, sino el **texto** de la pregunta, que sí es transferible.")
p("Se descartaron como solución principal los modelos generativos de gran tamaño (LLM) consultados "
  "por API: introducen variabilidad de versión, costo, latencia y riesgos de privacidad, y "
  "dificultan repetir exactamente los experimentos (Grévisse, 2024).")

h("4.4 Representación de entrada", 2)
viñetas([
    "**Modelos lineales:** TF-IDF de palabras (1-2 gramas, frecuencia mínima 2) y de caracteres "
    "(2-5 gramas, máximo 120 000 rasgos) sobre la explicación normalizada, más codificación "
    "*one-hot* de `QuestionId` y del par (pregunta, opción).",
    "**Transformers:** la cadena `Question: <enunciado> Answer: <opción> Explanation: "
    "<explicación>`, truncada a 160 tokens (el percentil 95 es de 80).",
])

h("4.5 Herramientas utilizadas", 2)
p("**Recursos de cómputo.** Los modelos lineales, la calibración, el ensamble y la aplicación se "
  "ejecutaron en una computadora portátil **Apple M3 (8 núcleos) con 8 GB de RAM**. Un primer "
  "intento de ajustar DeBERTa-v3-small en la GPU integrada (MPS) agotó la memoria unificada: una "
  "época tomaba horas por el intercambio con disco. Por ello los *Transformers* se entrenaron en "
  "**Kaggle con GPU NVIDIA** (sesiones gratuitas de hasta 30 h semanales), con precisión mixta "
  "(fp16) y los mismos scripts y particiones; las predicciones se descargaron para el ensamble.")
tabla(
    ["Herramienta", "Versión", "Uso"],
    [
        ["Python", "3.14 (local) · 3.11 (Kaggle)", "Lenguaje de todo el proyecto"],
        ["pandas / NumPy / SciPy", "3.0.3 / 2.5.1 / 1.18.0", "Manipulación de datos y matrices dispersas"],
        ["scikit-learn", "1.9.0", "TF-IDF, modelos lineales, validación y métricas (Pedregosa et al., 2011)"],
        ["PyTorch", "2.x", "Entrenamiento de los Transformers"],
        ["Hugging Face Transformers", "4.x–5.x", "Modelos preentrenados DeBERTa-v3 y MathBERT"],
        ["Matplotlib / Seaborn", "3.11.0 / 0.13.2", "Figuras estáticas del informe"],
        ["Streamlit / Plotly", "1.65.0 / 7.1.0", "Aplicación web y gráficas interactivas"],
        ["joblib", "1.5.3", "Persistencia de modelos y paralelización"],
        ["Kaggle CLI", "2.2.4", "Descarga de datos y ejecución en GPU"],
        ["Git / GitHub", "—", "Control de versiones y contribuciones individuales"],
    ],
    "Lenguajes, bibliotecas y paquetes utilizados.", anchos=[4.2, 4.3, 8],
)

# ===========================================================================
# 5. RESULTADOS Y ANÁLISIS
# ===========================================================================
h("5. Resultados y análisis de resultados", 1)
h("5.1 Características del conjunto de datos original", 2)
p("El archivo `train.csv` contiene **36 696 observaciones y 7 variables**, con 15 preguntas, 49 "
  "opciones distintas, 35 concepciones erróneas y 65 etiquetas compuestas. No hay variables "
  "cuantitativas nativas: las variables numéricas del análisis se derivaron del texto.")
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
p("`Category` tiene la forma `Veracidad_Tipo`: la veracidad (`True`/`False`) indica si la opción "
  "es correcta y el tipo (`Correct`, `Misconception`, `Neither`) describe la calidad del "
  "razonamiento. `Misconception` solo aplica cuando la categoría contiene una concepción errónea. "
  "El conjunto incluye además dos etiquetas casi homónimas, `Wrong_fraction` y `Wrong_Fraction`, "
  "que se conservaron tal como vienen porque corresponden a preguntas distintas.")

h("5.2 Limpieza y preprocesamiento", 2)
tabla(
    ["Situación detectada", "Decisión", "Justificación"],
    [
        ["26 836 nulos en Misconception (73.1 %)", "Codificar como la categoría \"NA\"",
         "Son estructurales: el valor existe si y solo si Tipo = Misconception; imputar inventaría "
         "etiquetas inexistentes."],
        ["736 filas iguales sin row_id; 970 textos repetidos", "Conservar y agrupar en la partición",
         "Pueden ser respuestas legítimas; agruparlas evita fuga entre entrenamiento y prueba."],
        ["LaTeX en el 96 % de MC_Answer y en los enunciados", "Convertir a texto plano",
         "Homogeneiza la notación numérica."],
        ["Mayúsculas, espacios y ortografía irregulares", "Minúsculas y espacios unificados",
         "Reduce variación superficial; los n-gramas de caracteres toleran errores."],
        ["78.3 % del texto con dígitos; operadores y negaciones", "No eliminar",
         "Pueden cambiar por completo el razonamiento matemático."],
        ["≈ 4 % de explicaciones atípicas por longitud", "Auditar y conservar",
         "Son válidas: explicaciones paso a paso o respuestas mínimas (señal de Neither)."],
    ],
    "Decisiones de limpieza y preprocesamiento.", anchos=[4.6, 4.2, 7.7],
)
p("Estas transformaciones están en `map_modelos.py`, que comparten el entrenamiento y la "
  "aplicación; así, una explicación nueva se preprocesa de forma idéntica e inadvertida para el "
  "usuario.")

h("5.3 Análisis exploratorio", 2)
f = figura("distribucion_category.png", "Distribución de Category y su descomposición en veracidad × tipo.")
p(f"**Una clase dominante (Figura {f}).** `True_Correct` concentra el 40.3 % de las observaciones, "
  "mientras que `True_Misconception` (1.1 %) y `False_Correct` (0.6 %) son muy raras, aunque son "
  "pedagógicamente las más interesantes.")
f = figura("distribucion_misconceptions.png", "Concepciones erróneas más frecuentes y cola larga de la etiqueta compuesta.")
p(f"**Cola larga (Figura {f}).** 32 de las 65 etiquetas tienen menos de 50 casos y 24 menos de 10. "
  "Por eso se usan MAP@3 y macro-F1 además de la exactitud.")
f = figura("longitud_por_categoria.png", "Longitud de la explicación por categoría y por tipo de razonamiento.")
p(f"**La longitud no explica por sí sola la etiqueta (Figura {f}).** Mediana de 13 palabras y "
  "máximo de 177. Los razonamientos correctos son más largos (mediana 16) que los erróneos (13) y "
  "los no evaluables (11), pero las distribuciones se solapan.")
f = figura("cruces_por_pregunta.png", "Composición de Category y tasa de acierto por pregunta.")
p(f"**La pregunta estructura el espacio de errores (Figura {f}).** La tasa de acierto varía entre "
  "35.7 % y 79.6 %, 32 de las 35 concepciones aparecen en una sola pregunta y el par (pregunta, "
  "opción) determina la veracidad en 58 de 60 casos. Este hallazgo motivó el modelo jerárquico, la "
  "ablación del contexto y la validación por preguntas no vistas.")
f = figura("marcadores_lexicos.png", "Tipo de razonamiento según palabras marcadoras.", 13)
p(f"**El vocabulario discrimina (Figura {f}).** Verbos de procedimiento (*divide*, *multiply*) se "
  "asocian con razonamiento correcto y marcadores de incertidumbre (*guess*, *think*) con "
  "`Neither`, lo que justifica una línea base léxica.")

# Textos de discusión que dependen de los resultados de los Transformers.
TEXTOS = {
    "SELECCION": (
        f"**Selección provisional.** Mientras se completan los Transformers, el modelo de referencia es "
        f"la **regresión logística jerárquica (C = {HP['TF-IDF + Regresión logística']['C']:g})**: mejor "
        "MAP@3 y exactitud top-3, probabilidades bien calibradas para la abstención y predicciones "
        "coherentes con la pregunta. La SVM lineal es la alternativa eficiente."),
    "CONCLUSION_TRANSFORMERS": ("Los resultados de DeBERTa-v3, MathBERT y el ensamble calibrado se "
                                "incorporan al completarse su entrenamiento en GPU."),
}
_p_original = p


def p(texto="", *a, **k):  # noqa: F811 — sustituye marcadores [[CLAVE]] por su texto
    if texto.startswith("[[") and texto.endswith("]]"):
        texto = TEXTOS.get(texto[2:-2], "")
        if not texto:
            return None
    return _p_original(texto, *a, **k)

# --- 5.4 Hiperparámetros ----------------------------------------------------
h("5.4 Ajuste de hiperparámetros", 2)
p("**Modelos lineales.** Búsqueda en malla con validación cruzada de 3 pliegues "
  "(`StratifiedGroupKFold`) dentro del entrenamiento, con MAP@3 como criterio.")
filas = []
for _, r in busqueda.iterrows():
    hp = json.loads(r["Hiperparámetros"])
    txt = ", ".join(f"{k} = {v}" for k, v in hp.items()).replace("None", "ninguno")
    elegido = "✔" if hp == HP.get(r["Modelo"]) else ""
    filas.append([r["Modelo"], txt, f"{r['MAP@3 CV (media)']:.4f} ± {r['MAP@3 CV (desv.)']:.4f}", elegido])
tabla(["Modelo", "Hiperparámetros", "MAP@3 CV (media ± desv.)", "Elegido"], filas,
      "Búsqueda de hiperparámetros de los modelos lineales (3 pliegues).", anchos=[4.5, 5.3, 4.5, 1.8])
f = figura("modelos/busqueda_hiperparametros.png",
           "MAP@3 en validación cruzada según la regularización C y la ponderación de clases.", 15)
p(f"La Figura {f} muestra mesetas amplias: la regresión logística alcanza su máximo con "
  f"**C = {HP['TF-IDF + Regresión logística']['C']:g}** y la SVM con **C = {HP[SVM]['C']:g}**, con "
  "diferencias menores a 0.004 entre valores vecinos. La **ponderación balanceada de clases empeoró "
  "el MAP@3** en ambos modelos (hasta 6 puntos): favorecer las clases raras degrada el orden de los "
  "candidatos frecuentes, que dominan la métrica. Se descartó.")
tr_info = [(n, d) for n, d in (("DeBERTa-v3", DEB), ("MathBERT", MATH)) if d]
if tr_info:
    p("**Transformers.** Por su costo, no se hizo búsqueda en malla. Se fijaron valores habituales "
      "para ajuste fino —tasa de aprendizaje con calentamiento lineal del 10 %, lotes de 16, "
      "decaimiento de pesos 0.01, recorte de gradiente en 1 y longitud máxima de 160 tokens— y se "
      "eligió la **época** con mejor MAP@3 en la partición de validación (parada temprana).")
    tabla(["Modelo", "Base preentrenada", "Tasa de aprendizaje", "MAP@3 val. por época"],
          [[n, d["modelo_base"], f"{d['hiperparametros']['lr']:g}",
            " → ".join(f"{x['MAP@3 val']:.3f}" for x in d["historia"])] for n, d in tr_info],
          "Ajuste de los Transformers.", anchos=[3, 4.8, 3, 5.7])
    f = figura("modelos/curva_deberta.png", "Curvas de aprendizaje de los Transformers.", 15)
else:
    p("**Transformers.** *[En ejecución en Kaggle: la tabla y la curva de aprendizaje se agregan al "
      "regenerar este documento.]*")

# --- 5.5 Comparación ----------------------------------------------------------
h("5.5 Comparación de los algoritmos", 2)
orden_tabla = metricas.sort_values("MAP@3", ascending=False)
tabla(["Modelo", "MAP@3", "Exactitud", "Top-3", "Macro-F1", "F1 pond.", "Entren. (s)", "Pred. (s)"],
      [[r["Modelo"], f3(r["MAP@3"]), f3(r["Accuracy"]), f3(r["Top-3 accuracy"]), f3(r["Macro-F1"]),
        f3(r["Weighted-F1"]), f"{r['Entrenamiento (s)']:.0f}", f"{r['Predicción (s)']:.2f}"]
       for _, r in orden_tabla.iterrows()],
      f"Desempeño y tiempos en el conjunto de prueba ({miles(LIN['n_test'])} explicaciones).",
      anchos=[4.6, 1.5, 1.7, 1.4, 1.7, 1.6, 1.8, 1.6], tam=8.5)
f = figura("modelos/comparacion_metricas.png", "MAP@3, exactitud y macro-F1 en el conjunto de prueba.")
pr, sv, rl = M.loc[PRIOR], M.loc[SVM], M.loc[RL]
p(f"**Líneas base (Figura {f}).** Leer la explicación aporta: la regresión logística jerárquica "
  f"obtiene MAP@3 = **{f3(rl['MAP@3'])}** frente a {f3(pr['MAP@3'])} del prior, con "
  f"{(rl['Accuracy'] - pr['Accuracy']) * 100:.0f} puntos más de exactitud y casi el doble de "
  f"macro-F1 ({f3(pr['Macro-F1'])} → {f3(rl['Macro-F1'])}). La SVM queda prácticamente empatada "
  f"(MAP@3 = {f3(sv['MAP@3'])}) con un entrenamiento unas {rl['Entrenamiento (s)'] / sv['Entrenamiento (s)']:.0f} "
  "veces más rápido; la regresión logística ordena mejor el segundo y el tercer candidato "
  f"(exactitud top-3 {f3(rl['Top-3 accuracy'])} frente a {f3(sv['Top-3 accuracy'])}).")
if HAY_T:
    lineas = []
    for n in ("DeBERTa-v3", "MathBERT"):
        if n in M.index:
            r = M.loc[n]
            lineas.append(f"**{n}** alcanza MAP@3 = {f3(r['MAP@3'])}, exactitud = {f3(r['Accuracy'])} y "
                          f"macro-F1 = {f3(r['Macro-F1'])}, con {r['Entrenamiento (s)'] / 60:.0f} min de "
                          "entrenamiento en GPU")
    p("**Transformers.** " + "; ".join(lineas) + ".")
    if ENS in M.index:
        e = M.loc[ENS]
        p(f"**Ensamble calibrado.** Con α = {FIN['alfa']:.2f} (peso de {NT}) elegido en validación y "
          f"la restricción jerárquica, el ensamble obtiene MAP@3 = **{f3(e['MAP@3'])}**, exactitud = "
          f"{f3(e['Accuracy'])} y macro-F1 = {f3(e['Macro-F1'])}.")
    p("[[DISCUSION_TRANSFORMERS]]")
else:
    p("*[DeBERTa-v3, MathBERT y el ensamble calibrado se están entrenando en Kaggle; sus filas se "
      "agregan a la tabla al regenerar este documento.]*")
f = figura("modelos/desempeno_vs_tiempo.png", "Calidad (MAP@3) frente a tiempo de entrenamiento.", 13.5)
p(f"**Tiempos (Figura {f}).** La predicción es rápida para todos los modelos lineales (menos de "
  "1 s para más de 7 000 explicaciones), lo que los hace viables en una aplicación interactiva.")

# --- 5.6 Ablación ---------------------------------------------------------------
h("5.6 Ablación del contexto de entrada", 2)
f = figura("modelos/ablacion_contexto.png", "MAP@3 de la regresión logística según el contexto de entrada.", 12)
a1, a2, a3 = (ABL.loc[k, "MAP@3"] for k in ("Texto solo", "Texto + pregunta", "Texto + pregunta + opción"))
p(f"Con preguntas conocidas, el contexto suma (Figura {f}): texto solo {f3(a1)}, + pregunta "
  f"{f3(a2)} y + pregunta y opción **{f3(a3)}**. La opción elegida es la fuente más valiosa porque "
  "fija la veracidad, que el texto por sí solo no siempre permite inferir.")

# --- 5.7 Generalización -------------------------------------------------------
h("5.7 Generalización a preguntas no vistas (validación B)", 2)
f = figura("modelos/generalizacion_preguntas.png", "MAP@3 con preguntas no vistas (GroupKFold por QuestionId).", 14)
tb_txt = TB["MAP@3 preguntas no vistas"]
p(f"Con preguntas completas fuera del entrenamiento (Figura {f}) todos los modelos caen con "
  f"fuerza. El prior se reduce a proponer las etiquetas globales más frecuentes "
  f"({f3(tb_txt[PRIOR])}). La regresión logística con texto solo obtiene "
  f"{f3(tb_txt['TF-IDF + RL (texto solo)'])}, **mejor que la misma regresión con pregunta y opción** "
  f"({f3(tb_txt['TF-IDF + RL (texto + pregunta + opción)'])}).")
p("Este resultado confirma el riesgo de usar `QuestionId` como atajo: el identificador mejora el "
  "desempeño con preguntas conocidas, pero empeora la transferencia, porque el modelo aprende "
  "asociaciones pregunta → concepción que no existen en una pregunta nueva. Coincide con Padó et "
  "al. (2024). Como el 91 % de las concepciones de una pregunta nueva no aparecen en "
  "entrenamiento, ningún modelo puede acertarlas: la validación B mide sobre todo la capacidad de "
  "reconocer la categoría general.")
if HAY_T:
    p("[[DISCUSION_GENERALIZACION]]")

# --- 5.8 Calibración y abstención ---------------------------------------------
h("5.8 Calibración y abstención", 2)
f = figura("modelos/calibracion.png", "Diagrama de confiabilidad de los modelos probabilísticos.", 11.5)
ece_txt = "; ".join(f"{k}: {v:.3f}" for k, v in FIN["ece"].items())
p(f"La Figura {f} compara la confianza declarada con la exactitud observada (ECE: {ece_txt}). "
  + (f"El escalado de temperatura (T = {FIN['temperatura']:.2f}) corrige la sobreconfianza "
     "típica de las redes neuronales (Guo et al., 2017)." if HAY_T else
     "La regresión logística está razonablemente calibrada, lo que permite usar su confianza para "
     "decidir cuándo derivar un caso."))
ab = FIN["abstencion"]
f = figura("modelos/abstencion.png", "Exactitud de los casos aceptados y cobertura según el umbral de confianza.", 13)
p(f"**Abstención (Figura {f}).** Se eligió en validación el menor umbral cuya exactitud sobre los "
  f"casos aceptados es de al menos 95 %: **τ = {ab['umbral']:.2f}**. En prueba, el sistema decide "
  f"automáticamente el {pct(ab['cobertura_prueba'])} de los casos con una exactitud de "
  f"{pct(ab['exactitud_aceptados'])}, y deriva al docente el resto, donde la exactitud sería solo de "
  f"{pct(ab['exactitud_derivados'])}. Así, la revisión humana se concentra donde el modelo es menos "
  "confiable.")

# --- 5.9 Errores y selección --------------------------------------------------
h("5.9 Análisis de errores y selección del modelo", 2)
nombre_sel = ENS if HAY_T and ENS in M.index else RL
tabla(["Dimensión evaluada", "Exactitud"], [[k, pct(v)] for k, v in DIM.items()],
      f"Exactitud por nivel de la jerarquía ({nombre_sel}).", anchos=[10, 4])
f = figura("modelos/matriz_confusion_category.png", f"Matriz de confusión de Category ({nombre_sel}).", 12)
p(f"La veracidad se predice casi perfectamente ({pct(DIM['Veracidad (True/False)'])}); la "
  f"dificultad está en el tipo de razonamiento. La matriz (Figura {f}) muestra dos bloques sin "
  "confusiones entre sí (`True_*` y `False_*`). Los errores se concentran en `True_Neither`, "
  "confundido con `True_Correct` (distinguir una explicación correcta pero vaga es subjetivo), en "
  "`True_Misconception` —acertar con un razonamiento erróneo, solo 403 casos— y en `False_Correct`.")
f = figura("modelos/f1_vs_soporte.png", "F1 por etiqueta frente a su número de ejemplos de entrenamiento.", 13)
p(f"La Figura {f} explica el macro-F1 moderado: las etiquetas con menos de unos 15 ejemplos de "
  "entrenamiento obtienen F1 cercano a cero, mientras que la mayoría de las que superan 100 ejemplos "
  "rebasan 0.8. El cuello de botella es la cola larga, más que la capacidad del algoritmo.")
p("[[SELECCION]]")

h("5.10 Aplicación", 2)
p("Se construyó una aplicación web con **Streamlit** (`app/app.py`) que permite clasificar "
  "explicaciones nuevas con los modelos entrenados, explorar el conjunto de datos limpio y "
  "comparar el rendimiento de los algoritmos. Se ejecuta localmente con `streamlit run app/app.py` "
  "y se publica en Streamlit Community Cloud directamente desde el repositorio de GitHub.")
p("**Clasificar una explicación.** El usuario elige la pregunta y la opción marcada (la correcta "
  "aparece señalada), escribe la explicación o carga un ejemplo real de una categoría, y elige uno "
  "o varios modelos. Para cada modelo se muestran las tres etiquetas más probables con su "
  "probabilidad, una descripción en lenguaje natural de la categoría y, cuando la confianza es "
  "menor que el umbral τ, la recomendación de **derivar el caso al docente**. Una gráfica compara "
  "las respuestas de todos los modelos e indica si coinciden. El preprocesamiento (conversión de "
  "LaTeX, normalización y vectorización) ocurre de forma invisible.")
figura("app/app_clasificar.png", "Aplicación: clasificación de una explicación con varios modelos.", 15)
p("**Explorar los datos.** Filtros por pregunta y categoría, indicadores clave y gráficas "
  "interactivas de la distribución de etiquetas, la cola larga, las variables derivadas del texto "
  "(longitud, dígitos, símbolos, diversidad léxica) y la composición por pregunta, acompañadas de "
  "los hallazgos que explican el comportamiento de los modelos.")
figura("app/app_explorar.png", "Aplicación: exploración interactiva del conjunto limpio.", 15)
p("**Rendimiento de los modelos.** Gráficas interactivas de cada métrica en el conjunto de prueba, "
  "calidad frente a tiempo, validación por preguntas no vistas, matriz de confusión, ablación del "
  "contexto, curva de abstención, calibración y búsqueda de hiperparámetros. Un interruptor "
  "permite **ocultar o mostrar** esta información y un selector elige qué modelos comparar.")
figura("app/app_rendimiento.png", "Aplicación: comparación interactiva del rendimiento.", 15)
p("**Selección de tecnologías.** Streamlit permite construir la interfaz en Python, el mismo "
  "lenguaje de los modelos, y cargar directamente los objetos de scikit-learn y PyTorch sin una "
  "capa de servicio adicional; además se despliega gratis desde GitHub. Plotly aporta gráficas "
  "interactivas (zoom, filtrado y valores al pasar el cursor). La paleta usa el **azul "
  "institucional** como color principal, acentos análogos (cian y verde) para los modelos y un "
  "**complementario cálido** (ámbar) para resaltar; los grises se reservan para la referencia. Los "
  "colores de las categorías siguen una convención semántica (verde para razonamiento correcto, "
  "rojo para concepción errónea) y el tema claro garantiza contraste suficiente del texto.")
p("En la versión publicada no se distribuyen los textos originales de la competencia, conforme a "
  "sus reglas: la exploración usa las variables derivadas y el usuario escribe sus propias "
  "explicaciones. Los ejemplos reales solo están disponibles al ejecutar la aplicación con los "
  "datos descargados de Kaggle.")

# ===========================================================================
# 6. CONCLUSIONES
# ===========================================================================
h("6. Conclusiones", 1)
viñetas([
    "**Revisión y selección (objetivos 1 y 2).** La literatura de calificación automática orientó un "
    "esquema incremental: líneas base léxicas, un Transformer contextual, adaptación al dominio y un "
    "ensamble calibrado, adoptando complejidad solo si mejora el MAP@3 y el macro-F1 bajo el mismo "
    "protocolo. El análisis exploratorio mostró que el problema es contextual, jerárquico y muy "
    "desbalanceado, y que los nulos de `Misconception` son estructurales.",
    f"**Modelos y evaluación (objetivos 3 y 4).** Las líneas base TF-IDF alcanzan MAP@3 = "
    f"{f3(M.loc[RL, 'MAP@3'])} (regresión logística jerárquica) y {f3(M.loc[SVM, 'MAP@3'])} (SVM), "
    f"muy por encima del prior sin texto ({f3(M.loc[PRIOR, 'MAP@3'])}). {TEXTOS.get('CONCLUSION_TRANSFORMERS', '')}",
    f"**Contexto y generalización.** La pregunta y la opción elevan el MAP@3 de {f3(a1)} a {f3(a3)} "
    "con preguntas conocidas, pero con preguntas no vistas el desempeño cae a "
    f"{f3(tb_txt['TF-IDF + RL (texto + pregunta + opción)'])} y el modelo de texto solo generaliza "
    "mejor: el identificador de la pregunta funciona como atajo. El sistema es adecuado para el banco "
    "de preguntas conocido, no para preguntas nuevas sin datos etiquetados.",
    f"**Confiabilidad (objetivo 5).** Con un umbral τ = {ab['umbral']:.2f}, el sistema decide solo el "
    f"{pct(ab['cobertura_prueba'])} de los casos con {pct(ab['exactitud_aceptados'])} de exactitud y "
    "deriva el resto al docente: un clasificador que prioriza casos y no reemplaza el juicio docente.",
    "**Aplicación (objetivo 6).** La aplicación Streamlit clasifica explicaciones nuevas con uno o "
    "todos los modelos, explica las categorías, recomienda la abstención y permite explorar los datos "
    "y el rendimiento con gráficas interactivas que pueden ocultarse.",
])
p("**Efectividad de los algoritmos.** Para ordenar las tres etiquetas más probables en preguntas "
  "conocidas, los modelos son eficaces; sus debilidades son las etiquetas con muy pocos ejemplos, las "
  "categorías pedagógicamente sutiles (`True_Misconception`, `True_Neither`) y las preguntas nuevas. "
  "Por ello las predicciones deben usarse para **priorizar la revisión docente**, no como "
  "diagnóstico definitivo.")
h("Trabajo futuro", 2)
viñetas([
    "Incorporar descripciones textuales de cada concepción errónea para comparar la explicación con "
    "ellas y mejorar la transferencia a preguntas nuevas.",
    "Aumento conservador de datos para las clases raras, revisado por docentes.",
    "Pruebas de robustez: perturbaciones inocuas (palabras irrelevantes, ortografía) y cambios "
    "matemáticamente decisivos (números, signos, negaciones) (Filighera et al., 2024).",
    "Ampliar el banco de preguntas para medir con más precisión la generalización.",
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
    "bidirectional transformers for language understanding. En *Proceedings of the 2019 Conference "
    "of the North American Chapter of the Association for Computational Linguistics: Human Language "
    "Technologies* (Vol. 1, pp. 4171–4186). Association for Computational Linguistics. "
    "https://doi.org/10.18653/v1/N19-1423",
    "Filighera, A., Ochs, S., Steuer, T., & Tregel, T. (2024). Cheating automatic short answer "
    "grading with the adversarial usage of adjectives and adverbs. *International Journal of "
    "Artificial Intelligence in Education, 34*, 616–646. https://doi.org/10.1007/s40593-023-00361-2",
    "Grévisse, C. (2024). LLM-based automatic short answer grading in undergraduate medical "
    "education. *BMC Medical Education, 24*, 1060. https://doi.org/10.1186/s12909-024-06026-5",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural "
    "networks. En *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, "
    "pp. 1321–1330). PMLR.",
    "He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled "
    "attention. En *International Conference on Learning Representations*. "
    "https://openreview.net/forum?id=XPZIaotutsD",
    "Kaggle. (2025). *MAP—Charting student math misunderstandings* [Competencia de ciencia de "
    "datos]. https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings",
    "Morley, F., Walland, E., & Vidal Rodeiro, C. L. (2026). Auto-marking short answer questions in "
    "science: The foundational years of transformer-based models from BERT to GPT-4. "
    "*International Journal of Artificial Intelligence in Education, 36*(1–2), 100005. "
    "https://doi.org/10.1016/j.ijaied.2026.100005",
    "Otero, N., Druga, S., & Lan, A. (2025). A benchmark for math misconceptions: Bridging gaps in "
    "middle school algebra with AI-supported instruction. *Discover Education, 4*, 277. "
    "https://doi.org/10.1007/s44217-025-00742-w",
    "Padó, U., Eryilmaz, Y., & Kirschner, L. (2024). Short-answer grading for German: Addressing the "
    "challenges. *International Journal of Artificial Intelligence in Education, 34*(4), 1321–1352. "
    "https://doi.org/10.1007/s40593-023-00383-w",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
    "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, "
    "M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of "
    "Machine Learning Research, 12*, 2825–2830.",
    "Rittle-Johnson, B., Adler, R., Durkin, K., Burleigh, L., King, J., & Crossley, S. (2025). "
    "Detecting math misconceptions: An AI benchmark dataset. En *Proceedings of the 1st Workshop on "
    "Advancing Artificial Intelligence through Theory of Mind* (pp. 20–24). Association for "
    "Computational Linguistics. https://aclanthology.org/2025.aimecon-wip.3/",
    "Rushton, S. J. (2018). Teaching and learning mathematics through error analysis. *Fields "
    "Mathematics Education Journal, 3*, Artículo 4. https://doi.org/10.1186/s40928-018-0009-y",
    "Sahu, A., & Bhowmick, P. K. (2020). Feature engineering and ensemble-based approach for "
    "improving automatic short-answer grading performance. *IEEE Transactions on Learning "
    "Technologies, 13*(1), 77–90. https://doi.org/10.1109/TLT.2019.2897997",
    "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. "
    "*Information Processing & Management, 24*(5), 513–523. "
    "https://doi.org/10.1016/0306-4573(88)90021-0",
    "Schneider, J., Richner, R., & Riser, M. (2023). Towards trustworthy AutoGrading of short, "
    "multi-lingual, multi-type answers. *International Journal of Artificial Intelligence in "
    "Education, 33*(1), 88–118. https://doi.org/10.1007/s40593-022-00289-z",
    "Shen, J. T., Yamashita, M., Prihar, E., Heffernan, N. T., Wu, X., Lee, D., & Wang, D. (2021). "
    "MathBERT: A pre-trained language model for general NLP tasks in mathematics education. "
    "*arXiv*. https://doi.org/10.48550/arXiv.2106.07340",
    "Zhang, Y., Lin, C., & Chi, M. (2020). Going deeper: Automatic short-answer grading by "
    "combining student and question models. *User Modeling and User-Adapted Interaction, 30*(1), "
    "51–80. https://doi.org/10.1007/s11257-019-09251-6",
    "Zhu, X., Wu, H., & Zhang, L. (2022). Automatic short-answer grading via BERT-based deep neural "
    "networks. *IEEE Transactions on Learning Technologies, 15*(3), 364–375. "
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

h("Anexo A. Glosario", 1)
tabla(
    ["Término", "Definición"],
    [
        ["Concepción errónea (misconception)", "Idea matemática equivocada y persistente que produce errores sistemáticos."],
        ["Etiqueta compuesta", "Combinación Category:Misconception; 65 clases distintas."],
        ["Nulo estructural", "Valor vacío porque el dato no aplica, no porque se haya perdido."],
        ["Cola larga", "Distribución con muchas clases de muy pocos ejemplos."],
        ["TF-IDF", "Peso de un término proporcional a su frecuencia en el documento e inverso a su frecuencia en el corpus."],
        ["N-grama", "Secuencia de n palabras o caracteres consecutivos."],
        ["Transformer", "Red neuronal basada en atención; base de BERT, DeBERTa y MathBERT."],
        ["Ajuste fino (fine-tuning)", "Entrenamiento adicional de un modelo preentrenado sobre la tarea específica."],
        ["Adaptación al dominio", "Uso de un modelo preentrenado con textos del área (aquí, matemáticas escolares)."],
        ["Época", "Una pasada completa del modelo por todos los datos de entrenamiento."],
        ["Hiperparámetro", "Parámetro fijado antes del entrenamiento (C, tasa de aprendizaje, épocas)."],
        ["Ensamble", "Combinación de las predicciones de varios modelos."],
        ["Calibración / ECE", "Coincidencia entre la confianza declarada y la exactitud real; el ECE mide la diferencia."],
        ["Escalado de temperatura", "División de los logits entre T para calibrar las probabilidades de una red."],
        ["Abstención", "Decisión de no clasificar automáticamente y derivar el caso a revisión humana."],
        ["Ablación", "Experimento que quita una parte de la entrada o del modelo para medir su aporte."],
        ["StratifiedGroupKFold", "Partición que conserva la proporción de clases y mantiene juntos los miembros de un grupo."],
        ["GroupKFold", "Partición que deja grupos completos (preguntas) fuera del entrenamiento."],
        ["Fuga de información", "Filtración de información de evaluación al entrenamiento que infla las métricas."],
        ["MAP@3", "Precisión media de tres predicciones ordenadas: 1, 1/2 o 1/3 según la posición del acierto."],
        ["Macro-F1", "Promedio del F1 de todas las clases con igual peso."],
    ],
    "Glosario de términos.", anchos=[4.5, 12],
)

for s in doc.sections:
    par = s.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    campo(par, "PAGE")

doc.save(SALIDA)
print(f"Informe generado: {SALIDA.relative_to(ROOT)}")

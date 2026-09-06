"""Extrae las figuras embebidas del notebook hacia report/assets."""

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_analisis_exploratorio.ipynb"
ASSETS = ROOT / "report" / "assets"

FIGURAS = {
    "sns.countplot": "distribucion_category.png",
    "Top 20 concepciones": "distribucion_misconceptions.png",
    "vars_plot =": "distribuciones_variables_texto.png",
    "sns.boxplot": "longitud_por_categoria.png",
    "corr = df[NUM].corr": "correlaciones_variables_texto.png",
    "Composición de Category por pregunta": "cruces_por_pregunta.png",
    'palabras = ["divide"': "marcadores_lexicos.png",
}

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
exportadas = []
for marcador, nombre in FIGURAS.items():
    celda = next(
        c for c in notebook["cells"]
        if c["cell_type"] == "code" and marcador in "".join(c.get("source", []))
    )
    imagen = next(
        o["data"]["image/png"] for o in celda.get("outputs", [])
        if "image/png" in o.get("data", {})
    )
    (ASSETS / nombre).write_bytes(base64.b64decode(imagen))
    exportadas.append(nombre)

print(f"Figuras exportadas: {len(exportadas)}")
for nombre in exportadas:
    print(f"- {nombre}")

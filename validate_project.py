"""Validación rápida de integridad de los entregables del proyecto MAP."""

import csv
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "01_analisis_exploratorio.ipynb"
HTML = ROOT / "presentation" / "presentacion_MAP.html"
PPTX = ROOT / "presentation" / "presentacion_MAP.pptx"
DOCX = ROOT / "file" / "Presentacion_Repositorio_MAP.docx"
MIEMBRO = "Abby Sofia Donis Agreda"

requeridos = [
    NOTEBOOK,
    HTML,
    PPTX,
    ROOT / "presentation" / "presentacion_MAP.pdf",
    DOCX,
    ROOT / "file" / "Presentacion_Repositorio_MAP.pdf",
    ROOT / "report" / "informe_analisis_exploratorio_MAP.tex",
    ROOT / "report" / "informe_analisis_exploratorio_MAP.pdf",
    ROOT / "report" / "revision_bibliografica_algoritmos_MAP.tex",
    ROOT / "report" / "revision_bibliografica_algoritmos_MAP.pdf",
    ROOT / "requirements.txt",
]
faltantes = [str(p.relative_to(ROOT)) for p in requeridos if not p.exists()]
assert not faltantes, f"Faltan archivos: {faltantes}"

with (ROOT / "data" / "train.csv").open(encoding="utf-8", newline="") as archivo:
    filas = list(csv.DictReader(archivo))
assert len(filas) == 36_696, "Cantidad inesperada de filas en train.csv."
assert len({f["QuestionId"] for f in filas}) == 15, "Cantidad inesperada de preguntas."
assert len({f["MC_Answer"] for f in filas}) == 49, "Cantidad inesperada de opciones."
assert len({f["Misconception"] for f in filas if f["Misconception"] != "NA"}) == 35, (
    "Cantidad inesperada de concepciones erróneas."
)
assert sum(f["Misconception"] == "NA" for f in filas) == 26_836, (
    "Cantidad inesperada de nulos estructurales."
)
assert sum(f["Category"] == "True_Correct" for f in filas) == 14_802, (
    "Cantidad inesperada de True_Correct."
)
targets = {
    f'{f["Category"]}:{f["Misconception"]}' for f in filas
}
assert len(targets) == 65, "Cantidad inesperada de etiquetas compuestas."

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert MIEMBRO in NOTEBOOK.read_text(encoding="utf-8"), "Abby no aparece en el notebook."
codigo = [c for c in notebook["cells"] if c["cell_type"] == "code"]
errores = [
    o for c in codigo for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
conteos = [c.get("execution_count") for c in codigo]
assert not errores, "El notebook contiene salidas de error."
assert conteos == list(range(1, len(codigo) + 1)), "Ejecución no secuencial."

html = HTML.read_text(encoding="utf-8")
assert MIEMBRO in html, "Abby no aparece en la presentación HTML."
assert html.count('<section class="slide') == 18, "El HTML no tiene 18 diapositivas."
for ruta in re.findall(r'<img[^>]+src="([^"]+)', html):
    assert (HTML.parent / ruta).resolve().exists(), f"Imagen HTML ausente: {ruta}"

with zipfile.ZipFile(PPTX) as archivo:
    diapositivas = [
        n for n in archivo.namelist()
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)
    ]
    texto_pptx = " ".join(
        archivo.read(n).decode("utf-8", errors="ignore") for n in diapositivas
    )
assert len(diapositivas) == 18, "El PowerPoint no tiene 18 diapositivas."
assert MIEMBRO in texto_pptx, "Abby no aparece en el PowerPoint."

with zipfile.ZipFile(DOCX) as archivo:
    texto_docx = archivo.read("word/document.xml").decode("utf-8", errors="ignore")
assert MIEMBRO in texto_docx, "Abby no aparece en el documento del repositorio."
assert ">RESUMEN<" in texto_docx, "El encabezado del documento no es Resumen."

for ruta in [
    ROOT / "README.md",
    ROOT / "report" / "informe_analisis_exploratorio_MAP.tex",
    ROOT / "report" / "revision_bibliografica_algoritmos_MAP.tex",
]:
    assert MIEMBRO in ruta.read_text(encoding="utf-8"), f"Abby no aparece en {ruta.name}."

print("OK · train.csv coincide con el perfil documentado")
print(f"OK · {len(codigo)} celdas de código sin errores")
print("OK · 18 diapositivas HTML y 18 diapositivas PowerPoint")
print("OK · todas las imágenes y entregables requeridos están presentes")
print("OK · integrantes y encabezado del DOCX verificados")

"""Genera los notebooks de Kaggle (GPU) que entrenan los Transformers.

Cada kernel empaqueta map_modelos.py y notebooks/03_transformer.py, copia los
datos de la competencia y ejecuta la validación A y la validación B.

Uso:
    python kaggle/construir_kernels.py
    kaggle kernels push -p kaggle/map-deberta
    kaggle kernels push -p kaggle/map-mathbert
    # al terminar:
    kaggle kernels output <usuario>/map-grupo1-deberta -p kaggle/salida-deberta
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USUARIO = "jorgepalaciossales"
COMPETENCIA = "map-charting-student-math-misunderstandings"

KERNELS = {
    "deberta": {"modelo": "microsoft/deberta-v3-base", "lr": 2e-5, "titulo": "MAP Grupo1 DeBERTa"},
    "mathbert": {"modelo": "tbs17/MathBERT", "lr": 3e-5, "titulo": "MAP Grupo1 MathBERT"},
}

PLANTILLA = '''"""Kernel generado por kaggle/construir_kernels.py — no editar a mano."""
import glob, os, shutil, subprocess, sys
W = "/kaggle/working"
open(os.path.join(W, "map_modelos.py"), "w").write({modulo!r})
os.makedirs(os.path.join(W, "notebooks"), exist_ok=True)
open(os.path.join(W, "notebooks", "03_transformer.py"), "w").write({script!r})
os.makedirs(os.path.join(W, "data"), exist_ok=True)
shutil.copy(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0], os.path.join(W, "data", "train.csv"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"], check=False)
base = [sys.executable, "-u", "notebooks/03_transformer.py", "--modelo", {modelo!r}, "--lr", "{lr}",
        "--prefijo", {prefijo!r}, "--batch", "16", "--epocas", "3"]
subprocess.run(base, cwd=W, check=True)
subprocess.run(base + ["--grupos", "--epocas_grupos", "2"], cwd=W, check=True)
shutil.rmtree(os.path.join(W, "data"))  # no se redistribuyen los datos de la competencia
os.remove(os.path.join(W, "map_modelos.py")); shutil.rmtree(os.path.join(W, "notebooks"))
'''

modulo = (ROOT / "map_modelos.py").read_text(encoding="utf-8")
script = (ROOT / "notebooks" / "03_transformer.py").read_text(encoding="utf-8")
for prefijo, cfg in KERNELS.items():
    carpeta = ROOT / "kaggle" / f"map-{prefijo}"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "kernel.py").write_text(PLANTILLA.format(
        modulo=modulo, script=script, modelo=cfg["modelo"], lr=cfg["lr"], prefijo=prefijo), encoding="utf-8")
    (carpeta / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USUARIO}/map-grupo1-{prefijo}",
        "title": cfg["titulo"],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": [COMPETENCIA],
        "dataset_sources": [], "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    print("Kernel listo:", carpeta.relative_to(ROOT))

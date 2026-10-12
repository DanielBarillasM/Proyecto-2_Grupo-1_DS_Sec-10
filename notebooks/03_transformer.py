"""Ajuste fino de DeBERTa-v3 para el reto MAP.

Entrada contextual: "Question: <pregunta> Answer: <opción> Explanation: <explicación>".
Salida: una de las 65 etiquetas Category:Misconception.

Modos:
    python notebooks/03_transformer.py            # validación A (partición estratificada)
    python notebooks/03_transformer.py --grupos   # validación B (GroupKFold por QuestionId)

Usa la GPU de Apple (MPS) o CUDA si están disponibles.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import map_modelos as mm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default="microsoft/deberta-v3-small")
ap.add_argument("--epocas", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_len", type=int, default=160)
ap.add_argument("--grupos", action="store_true")
ap.add_argument("--epocas_grupos", type=int, default=1)
ap.add_argument("--muestra", type=int, default=0, help="solo para pruebas: limita las filas")
ap.add_argument("--prefijo", default="deberta", help="prefijo de los archivos de salida")
args = ap.parse_args()

OUT = mm.ROOT / "results"
DIR_MODELO = mm.ROOT / "models" / f"{args.prefijo}_map"
OUT.mkdir(exist_ok=True)
torch.manual_seed(mm.RANDOM_STATE)
np.random.seed(mm.RANDOM_STATE)
DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")

df = mm.cargar_train()
CLASES = np.array(sorted(df["target"].unique()))
IDX = {c: i for i, c in enumerate(CLASES)}
y_all = df["target"].map(IDX).values
tok = AutoTokenizer.from_pretrained(args.modelo)
textos = mm.texto_transformer(df)
enc = tok(textos, truncation=True, max_length=args.max_len)
largos = np.array([len(x) for x in enc["input_ids"]])
print(f"Dispositivo: {DEV} · tokens p95 = {np.percentile(largos, 95):.0f}", flush=True)


def lotes(indices, barajar):
    """Agrupa por longitud similar para minimizar el relleno."""
    indices = np.asarray(indices)
    if barajar:
        indices = np.random.permutation(indices)
        trozos = [indices[i:i + args.batch * 50] for i in range(0, len(indices), args.batch * 50)]
        orden = np.concatenate([t[np.argsort(largos[t])] for t in trozos])
        bs = [orden[i:i + args.batch] for i in range(0, len(orden), args.batch)]
        return [bs[i] for i in np.random.permutation(len(bs))]
    orden = indices[np.argsort(largos[indices])]
    return [orden[i:i + args.batch * 4] for i in range(0, len(orden), args.batch * 4)]


def tensores(idx):
    b = tok.pad({"input_ids": [enc["input_ids"][i] for i in idx],
                 "attention_mask": [enc["attention_mask"][i] for i in idx]}, return_tensors="pt")
    return {k: v.to(DEV) for k, v in b.items()}


@torch.no_grad()
def logits_de(modelo, indices):
    modelo.eval()
    salida = np.zeros((len(indices), len(CLASES)), np.float32)
    pos = {j: i for i, j in enumerate(indices)}
    for idx in lotes(indices, barajar=False):
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            z = modelo(**tensores(idx)).logits.float().cpu().numpy()
        salida[[pos[j] for j in idx]] = z
    return salida


def entrenar(idx_tr, idx_val=None, epocas=3):
    modelo = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=len(CLASES)).float().to(DEV)
    if DEV.type == "mps":
        # En equipos con memoria unificada limitada se congela la matriz de embeddings
        # del vocabulario, lo que reduce la memoria de gradientes y del optimizador.
        modelo.get_input_embeddings().weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    pasos = epocas * math.ceil(len(idx_tr) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    y_t = torch.tensor(y_all)
    amp = DEV.type == "cuda"  # precisión mixta (fp16) en GPU NVIDIA
    escala = torch.amp.GradScaler("cuda", enabled=amp)
    historia, mejor, mejor_estado = [], -1, None
    for ep in range(1, epocas + 1):
        modelo.train()
        t0, perdida = time.time(), 0.0
        bs = lotes(idx_tr, barajar=True)
        for k, idx in enumerate(bs, 1):
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=amp):
                out = modelo(**tensores(idx), labels=y_t[idx].to(DEV))
            escala.scale(out.loss).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(opt); escala.update(); sched.step(); opt.zero_grad()
            perdida += out.loss.item()
            if k % 200 == 0:
                print(f"  época {ep} · lote {k}/{len(bs)} · pérdida {perdida / k:.4f} · "
                      f"{(time.time() - t0) / k * (len(bs) - k) / 60:.1f} min restantes", flush=True)
        reg = {"época": ep, "pérdida": perdida / len(bs), "minutos": (time.time() - t0) / 60}
        if idx_val is not None:
            z = logits_de(modelo, idx_val)
            reg["MAP@3 val"] = mm.map_at_3(CLASES[y_all[idx_val]], mm.top_k(z, CLASES))
            if reg["MAP@3 val"] > mejor:
                mejor = reg["MAP@3 val"]
                mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        historia.append(reg)
        print("  ", reg, flush=True)
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, historia


if not args.grupos:
    # ----------------------- Validación A ---------------------------------
    idx_full, idx_tr, idx_val, idx_te = mm.particiones(df)
    if args.muestra:
        idx_tr, idx_val, idx_te = idx_tr[:args.muestra], idx_val[:args.muestra], idx_te[:args.muestra]
    t0 = time.time()
    modelo, historia = entrenar(idx_tr, idx_val, args.epocas)
    t_fit = time.time() - t0
    t0 = time.time()
    z_te = logits_de(modelo, idx_te)
    t_pred = time.time() - t0
    z_val = logits_de(modelo, idx_val)
    np.save(OUT / f"{args.prefijo}_logits_val.npy", z_val)
    np.save(OUT / f"{args.prefijo}_logits_test.npy", z_te)
    DIR_MODELO.mkdir(parents=True, exist_ok=True)
    modelo.half().save_pretrained(DIR_MODELO)
    tok.save_pretrained(DIR_MODELO)
    (DIR_MODELO / "clases.json").write_text(json.dumps(CLASES.tolist(), ensure_ascii=False))
    m = mm.metricas(CLASES[y_all[idx_te]], mm.top_k(z_te, CLASES))
    resumen = {"modelo_base": args.modelo, "hiperparametros": vars(args), "historia": historia,
               "entrenamiento_s": t_fit, "prediccion_s": t_pred, "dispositivo": str(DEV),
               "prueba": m}
    (OUT / f"{args.prefijo}_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("[TEST]", m, flush=True)
else:
    # ----------------------- Validación B ---------------------------------
    res = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=3).split(df, groups=df["QuestionId"]), 1):
        if args.muestra:
            a, b = a[:args.muestra], b[:args.muestra]
        print(f"Pliegue {k}: preguntas no vistas {sorted(df['QuestionId'].iloc[b].unique())}", flush=True)
        modelo, _ = entrenar(a, None, args.epocas_grupos)
        z = logits_de(modelo, b)
        top3 = mm.top_k(z, CLASES)
        res.append({"pliegue": k, "MAP@3": mm.map_at_3(CLASES[y_all[b]], top3),
                    "Accuracy Category": float(np.mean(
                        [p.split(":")[0] == t.split(":")[0] for p, t in zip(top3[:, 0], CLASES[y_all[b]])]))})
        print("  ", res[-1], flush=True)
        del modelo
        if DEV.type == "mps":
            torch.mps.empty_cache()
    (OUT / f"{args.prefijo}_grupos.json").write_text(json.dumps(
        {"epocas": args.epocas_grupos, "pliegues": res,
         "MAP@3 media": float(np.mean([r["MAP@3"] for r in res]))}, indent=2, ensure_ascii=False))

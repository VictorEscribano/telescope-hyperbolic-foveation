"""
report.py
=========
Self-contained HTML training report for Telescope drone runs.

Generates ``<run_dir>/report.html`` with:
  • training curves (loss, mAP, recall, LR) — INTERACTIVE (Chart.js): hover a
    point to read its exact value
  • foveation diagnostics over epochs (R, |o-gt| vs centre, o spread), interactive,
    each metric with a hover tooltip explaining what it means
  • the training hyperparameters used, each with a hover tooltip explaining what it
    does and how it affects training
  • detection panels for VAL and TEST: for N sampled images, the image with GT
    (green) + predicted (red) boxes, side-by-side with the foveated/warped image
    the backbone actually saw, annotated with the lens centre o, radius R and α/p

Runs automatically at the end of train.py (rank 0), or standalone:

    python report.py --run_dir runs/drones_et68 \
        --dataset drones --data_dir /path/to/drones_v5 \
        --backbone efficienttam --backbone_ckpt ./checkpoints/efficienttam_s.pt \
        --image_size 512 512 --fov_spatial
"""

import argparse
import base64
import csv
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

# ── Foveation metric glossary (hover tooltips) ────────────────────────────────
FOV_GLOSSARY = {
    "R_mean":      "Radio medio de la lupa. ~0 = no amplía (lente apagada); más alto = más magnificación de la zona de interés.",
    "R_std":       "Variación del radio entre imágenes. >0 indica que la lente ajusta el zoom según la imagen.",
    "o_x_std":     "Dispersión del centro de la lupa en X. ≈0 = la lente apunta siempre al mismo sitio (colapsada); alto = se mueve por imagen.",
    "o_y_std":     "Dispersión del centro de la lupa en Y. ≈0 = colapsada a un punto fijo; alto = sigue al objetivo por imagen.",
    "dist_to_gt":  "Distancia media entre el centro de la lupa (o) y el centroide de los drones reales. MÁS BAJO = mejor (la lente apunta a los drones).",
    "dist_cen_gt": "Línea base: distancia del centro fijo de la imagen (0,0) a los drones. Si dist_to_gt < esto, la lente bate al centro fijo.",
    "mAP50-95":    "Precisión media de detección (IoU 0.50–0.95). Métrica principal de calidad del detector.",
    "mAP50":       "Precisión media a IoU 0.50 (criterio de solape más permisivo).",
    "recall":      "Fracción de drones reales que el modelo detecta (AR@100).",
}

# ── Training-parameter glossary (hover tooltips) ──────────────────────────────
PARAM_GLOSSARY = {
    "lr":               "Learning rate base (AdamW). Alto = aprende rápido pero inestable; bajo = estable pero lento. Decae en coseno hasta lr·lr_min_ratio.",
    "lr_min_ratio":     "Suelo del decay de LR como fracción de lr. El LR baja de lr hasta lr·lr_min_ratio en la última época (el refinamiento tardío gana mucho mAP).",
    "epochs":           "Nº de épocas. Define TODA la curva de decay del LR: con muchas épocas el LR baja muy lento (et67 con 300 se quedó plano y no despegó).",
    "batch_size":       "Imágenes por paso y por GPU. Más grande = gradiente más estable pero menos pasos/época y más VRAM.",
    "image_size":       "Resolución de entrada. Más alta = drones pequeños más visibles, pero más cómputo/memoria. Debe coincidir en train y eval.",
    "fp16":             "Precisión mixta (float16). Acelera y ahorra VRAM; la geometría crítica corre en fp32 para no dar NaN.",
    "weight_decay":     "Regularización L2 de AdamW. Penaliza pesos grandes para reducir sobreajuste.",
    "grad_clip":        "Recorte de norma del gradiente. Evita pasos enormes que desestabilizan el entrenamiento.",
    "eos_coef":         "Peso de la clase 'sin objeto' en la pérdida de clasificación. Bajo = no premia el fondo vacío → favorece recall (clave con ~50% imágenes sin drones).",
    "backbone":         "Encoder de imagen congelado. efficienttam (~22M, edge) o sam3 (453M, máxima precisión).",
    "two_stage":        "Cabeza DINO de 2 etapas: las queries salen de propuestas del encoder en vez de ser fijas. Mejor localización.",
    "denoising":        "Pérdida auxiliar DINO: refina cajas GT con ruido. Acelera y estabiliza la convergencia del detector.",
    "dn_weight":        "Peso de la pérdida de denoising en el total.",
    "enc_weight":       "Peso de la pérdida auxiliar del encoder (supervisa las propuestas de la 1ª etapa).",
    "num_queries":      "Nº de cajas candidatas que predice el DETR por imagen (300 estándar).",
    # ── foveation (the lens) ──
    "fov_spatial":      "Predice el centro o de la lupa con un soft-argmax sobre un heatmap (por imagen) en vez de un valor fijo. Permite que la lente SIGA al dron.",
    "fov_weight":       "Peso de la supervisión de foveación en el total. Alto = la lente manda sobre la detección (en et67 a 2.0 la lupa 'ganó' y la detección se estancó); bajar a ~0.75 deja que la detección respire.",
    "fov_lr_mult":      "Multiplicador de LR SOLO para el scout (la red de la lupa). Alto = la lente se mueve agresiva desde el inicio y desestabiliza la detección; bajarlo (3) suaviza el arranque.",
    "fov_warmup_epochs":"Épocas iniciales con R fijo grande, para que la cabeza aprenda a usar la imagen ampliada antes de que R sea aprendible. Rompe el círculo 'la cabeza no sabe usar el zoom → el zoom parece inútil'.",
    "fov_warmup_R":     "Valor de R fijo durante el warm-up de foveación.",
    "fov_empty_weight": "Peso del término de imágenes vacías. 0 = NO supervisa la lupa en frames sin drones (evita enseñarle a encogerse en la ~mitad vacía del dataset).",
    "fov_w_r":          "Peso del término que empuja R hacia un objetivo según el TAMAÑO del dron (drones pequeños piden lupa más grande). Es lo que le da a R un motivo para CRECER.",
    "fov_w_floor":      "Peso de la bisagra anti-colapso relu(r_floor−R), que impide que R caiga a cero.",
    "fov_r_lo":         "Objetivo de R para drones grandes (límite inferior del mapeo tamaño→R).",
    "fov_r_hi":         "Objetivo de R para drones diminutos (límite superior del mapeo tamaño→R).",
    "fov_r_floor":      "Suelo de R: por debajo de este valor se penaliza (anti-colapso de la lente).",
    "bg_keep_frac":     "Fracción de imágenes de fondo (sin drones) que se conservan en train. 0.5 = elimina la mitad para reequilibrar (~50% del dataset es fondo).",
    "patience":         "Early-stop: para si el mAP no mejora en N épocas. 0 = DESACTIVADO. Ojo: el default es 15 (mató et67 en la meseta de LR plano).",
}

# parameters worth showing (in this order); others from args.json are appended.
_PARAM_ORDER = [
    "backbone", "image_size", "batch_size", "epochs", "lr", "lr_min_ratio",
    "fp16", "eos_coef", "patience",
    "fov_spatial", "fov_weight", "fov_lr_mult", "fov_warmup_epochs",
    "fov_warmup_R", "fov_empty_weight", "fov_w_r", "fov_w_floor",
    "fov_r_lo", "fov_r_hi", "fov_r_floor", "bg_keep_frac",
    "two_stage", "denoising", "dn_weight", "enc_weight", "num_queries",
    "weight_decay", "grad_clip", "init_from", "resume",
]


# ── parsing helpers ─────────────────────────────────────────────────────────--

def _read_results_csv(path: Path):
    if not path.is_file():
        return {}
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return cols


def _read_args_json(path: Path):
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:                                     # noqa: BLE001
            return {}
    return {}


_FOV_RE = re.compile(
    r"o=\(([-+0-9.]+),([-+0-9.]+)\)\s+std=\(([0-9.]+),([0-9.]+)\)\s+"
    r"R=([0-9.]+)±([0-9.]+)(?:\s+\|o-gt\|=([0-9.]+)\s+\(centre=([0-9.]+)\))?"
)


def _parse_fov_log(path: Path):
    keys = ["o_x", "o_y", "o_x_std", "o_y_std", "R_mean", "R_std",
            "dist_to_gt", "dist_cen_gt"]
    out = {k: [] for k in keys}
    if not path.is_file():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        m = _FOV_RE.search(line)
        if not m:
            continue
        for k, v in zip(keys, m.groups()):
            out[k].append(float(v) if v is not None else None)
    return out


# ── interactive Chart.js line charts ──────────────────────────────────────────

_chart_id = [0]


def _jsnum(x):
    """JSON-safe number (NaN/inf → null)."""
    if x is None or (isinstance(x, float) and (x != x or x in (float("inf"), float("-inf")))):
        return "null"
    return repr(float(x))


def _line_chart(labels, series, title, subtitle=""):
    """Return (html, js) for one interactive Chart.js line chart.

    series: list of dicts {label, data (list), color}.
    """
    _chart_id[0] += 1
    cid = f"ch{_chart_id[0]}"
    labels_js = "[" + ",".join(_jsnum(v) for v in labels) + "]"
    ds_js = []
    for s in series:
        data_js = "[" + ",".join(_jsnum(v) for v in s["data"]) + "]"
        ds_js.append(
            f"{{label:{json.dumps(s['label'])},data:{data_js},"
            f"borderColor:'{s['color']}',backgroundColor:'{s['color']}',"
            f"borderWidth:2,pointRadius:2,pointHoverRadius:5,tension:0.25,spanGaps:true}}"
        )
    datasets_js = "[" + ",".join(ds_js) + "]"
    sub = f"<div class='chart-sub'>{subtitle}</div>" if subtitle else ""
    html = (f"<div class='chart-box'><div class='chart-title'>{title}</div>{sub}"
            f"<div class='canvas-wrap'><canvas id='{cid}'></canvas></div></div>")
    js = (f"mkChart('{cid}',{labels_js},{datasets_js});")
    return html, js


# ── detection + foveation image panels (static, matplotlib) ───────────────────

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a"})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_hwc(img_t):
    return img_t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _draw_box(ax, box, H, W, color, label=None):
    cx, cy, w, h = [float(v) for v in box]
    px, py = (cx + 1) / 2 * W, (cy + 1) / 2 * H
    pw, ph = w / 2 * W, h / 2 * H
    ax.add_patch(patches.Rectangle((px - pw / 2, py - ph / 2), pw, ph,
                 linewidth=1.8, edgecolor=color, facecolor="none"))
    if label:
        ax.text(px - pw / 2, py - ph / 2 - 3, label, color=color,
                fontsize=7, weight="bold")


@torch.no_grad()
def _detection_panels_b64(model, dataset, device, indices, alpha, p,
                          score_threshold, fp16):
    from torch.cuda.amp import autocast
    model.eval()
    panels = []
    for idx in indices:
        image, target = dataset[idx]
        img_b = image.unsqueeze(0).to(device)
        with autocast(enabled=fp16 and device.type == "cuda"):
            boxes_eu, logits, o, R = model(img_b)
        warped = model.warp_layer(img_b, o, R)[0]
        probs = logits.softmax(-1)[0, :, :-1]
        scores, labels = probs.max(-1)
        keep = scores > score_threshold

        H, W = image.shape[-2:]
        ox, oy, R_val = float(o[0, 0]), float(o[0, 1]), float(R[0])
        opx, opy = (ox + 1) / 2 * W, (oy + 1) / 2 * H

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 4.3))
        axL.imshow(_to_hwc(image))
        for gb in target["boxes"]:
            _draw_box(axL, gb, H, W, "#37d67a", "GT")
        kept = keep.nonzero(as_tuple=True)[0]
        order = scores[kept].argsort(descending=True)[:10]
        for j in kept[order]:
            _draw_box(axL, boxes_eu[0, j], H, W, "#ff4d4f", f"{float(scores[j]):.2f}")
        axL.set_title(f"{target.get('file_name','')}  ·  GT(verde) vs pred(rojo)", fontsize=8)
        axL.axis("off")
        axR.imshow(_to_hwc(warped))
        axR.plot(opx, opy, "+", color="#ffd23f", ms=14, mew=2.2)
        axR.add_patch(patches.Circle((opx, opy), R_val * W / 2, fill=False,
                      edgecolor="#ffd23f", lw=1.4, ls="--"))
        axR.set_title(f"foveada · o=({ox:+.2f},{oy:+.2f}) R={R_val:.2f} "
                      f"α={alpha:.0f} p={p:.0f}", fontsize=8)
        axR.axis("off")
        fig.tight_layout()
        panels.append(_fig_to_b64(fig))
    return panels


def _sample_indices(dataset, n):
    with_gt = [i for i in range(len(dataset))
               if dataset._label_path(dataset.images[i]).is_file()]
    pool = with_gt if len(with_gt) >= n else list(range(len(dataset)))
    if not pool:
        return []
    step = max(1, len(pool) // n)
    return pool[::step][:n]


# ── HTML assembly ─────────────────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telescope — {title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>{chartjs}</script>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Inter',system-ui,sans-serif;color:#fff;background:#000;
  max-width:1120px;margin:2rem auto;padding:0 1.5rem 4rem;line-height:1.5}}
h1{{font-size:1.5rem;font-weight:600;margin:0 0 .2rem}}
.sub{{color:#888;font-size:.85rem;margin-bottom:2rem}}
.label{{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:#666;margin:2.4rem 0 .8rem}}
.card{{background:#0a0a0a;border:.5px solid #1a1a1a;border-radius:16px;
  padding:1.3rem;margin-bottom:1.2rem}}
.card img{{width:100%;border-radius:8px;display:block}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}
.chart-box{{background:#0d0d0d;border:.5px solid #1a1a1a;border-radius:12px;padding:.9rem}}
.chart-title{{font-size:.8rem;color:#cfcfcf;font-weight:500;margin-bottom:.15rem}}
.chart-sub{{font-size:.68rem;color:#777;margin-bottom:.4rem}}
.canvas-wrap{{position:relative;height:230px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{padding:7px 12px;border-bottom:1px solid #1a1a1a;text-align:left}}
th{{color:#888;font-weight:500}}
.metric{{position:relative;cursor:help;border-bottom:1px dotted #555}}
.metric .tip{{visibility:hidden;opacity:0;transition:.15s;position:absolute;
  z-index:10;bottom:140%;left:0;width:320px;background:#1a1a1a;color:#ddd;
  border:1px solid #333;border-radius:8px;padding:9px 11px;font-size:.78rem;
  font-weight:400;line-height:1.45;letter-spacing:0}}
.metric:hover .tip{{visibility:visible;opacity:1}}
.good{{color:#37d67a}} .bad{{color:#ff4d4f}} .val{{color:#fff;font-weight:600}}
.pgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.3rem 1.4rem}}
.prow{{display:flex;justify-content:space-between;border-bottom:1px solid #141414;padding:5px 2px;font-size:.83rem}}
.pname{{color:#9ab;position:relative;cursor:help;border-bottom:1px dotted #445}}
.pname .tip{{visibility:hidden;opacity:0;transition:.15s;position:absolute;z-index:10;
  bottom:150%;left:0;width:320px;background:#1a1a1a;color:#ddd;border:1px solid #333;
  border-radius:8px;padding:9px 11px;font-size:.78rem;font-weight:400;line-height:1.45}}
.pname:hover .tip{{visibility:visible;opacity:1}}
.pval{{color:#fff;font-weight:600;font-variant-numeric:tabular-nums}}
.grid2{{display:grid;grid-template-columns:1fr;gap:1rem}}
</style></head><body>
<h1>Telescope · {title}</h1><div class="sub">{subtitle}</div>
"""

_CHART_JS = """
<script>
Chart.defaults.color='#888';Chart.defaults.font.family='Inter';
function mkChart(id,labels,datasets){
  new Chart(document.getElementById(id),{type:'line',
    data:{labels:labels,datasets:datasets},
    options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'nearest',intersect:false},
      plugins:{legend:{labels:{color:'#aaa',boxWidth:12,font:{size:10}}},
        tooltip:{enabled:true,callbacks:{title:(it)=>'epoch '+it[0].label,
          label:(c)=>c.dataset.label+': '+(c.parsed.y==null?'—':c.parsed.y.toFixed(4))}}},
      scales:{x:{ticks:{color:'#777',maxTicksLimit:12},grid:{color:'#161616'}},
              y:{ticks:{color:'#777'},grid:{color:'#161616'}}}}});
}
__INITS__
</script>
"""


def _metric_row(name, value, fmt="{:.4f}"):
    tip = FOV_GLOSSARY.get(name, "")
    val = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    return (f'<tr><td><span class="metric">{name}<span class="tip">{tip}</span>'
            f'</span></td><td class="val">{val}</td></tr>')


def _params_section(args):
    if not args:
        return ""
    keys = [k for k in _PARAM_ORDER if k in args]
    keys += [k for k in args if k not in keys and k not in
             ("data_dir", "val_dir", "output_dir", "backbone_ckpt", "et_config",
              "resume", "report", "num_workers", "save_every", "keep_last")]
    rows = []
    for k in keys:
        v = args[k]
        if isinstance(v, list):
            v = "×".join(str(x) for x in v)
        tip = PARAM_GLOSSARY.get(k, "")
        name = (f'<span class="pname">{k}<span class="tip">{tip}</span></span>'
                if tip else f'<span style="color:#9ab">{k}</span>')
        rows.append(f'<div class="prow">{name}<span class="pval">{v}</span></div>')
    return ('<div class="label">Parámetros del entrenamiento '
            '(pasa el ratón por el nombre para ver qué hace)</div>'
            f'<div class="card"><div class="pgrid">{"".join(rows)}</div></div>')


def _chartjs_source() -> str:
    """Return the Chart.js library source to inline (kept next to this file in
    _doc_assets/), so report.html is 100% self-contained — no CDN, works offline
    (needed on the air-gapped server).  Falls back to a CDN <script> tag only if
    the vendored copy is missing."""
    local = Path(__file__).parent / "_doc_assets" / "chart.umd.min.js"
    if local.is_file():
        return local.read_text(errors="ignore")
    return ('</script><script src="https://cdn.jsdelivr.net/npm/'
            'chart.js@4.4.1/dist/chart.umd.min.js">')


def build_html(run_dir, cols, fov, args, val_panels, test_panels, title):
    parts = [_HTML_HEAD.format(title=title, chartjs=_chartjs_source(),
             subtitle=f"run: {run_dir.name} · generado por report.py")]
    inits = []

    # summary
    parts.append('<div class="label">Resumen</div><div class="card"><table>')
    if cols.get("metrics/mAP50-95"):
        m = cols["metrics/mAP50-95"]
        bi = int(np.argmax(m))
        parts.append(f"<tr><th>métrica</th><th>mejor (epoch {int(cols['epoch'][bi])})</th></tr>")
        parts.append(_metric_row("mAP50-95", m[bi]))
        parts.append(_metric_row("mAP50", cols.get("metrics/mAP50", [0])[bi]))
        parts.append(_metric_row("recall", cols.get("metrics/recall", [0])[bi]))
    parts.append("</table></div>")

    # parameters
    parts.append(_params_section(args))

    # training curves (interactive)
    if cols.get("epoch"):
        ep = [int(e) for e in cols["epoch"]]
        charts = [
            ("Loss", "", [("train", cols.get("train/loss", []), "#4da3ff"),
                          ("val", cols.get("val/loss", []), "#ff8c42")]),
            ("mAP@50-95", "precisión media (IoU .50–.95)", [("mAP50-95", cols.get("metrics/mAP50-95", []), "#46c46e")]),
            ("mAP@50", "precisión a IoU .50", [("mAP50", cols.get("metrics/mAP50", []), "#46c46e")]),
            ("Recall (AR@100)", "drones detectados", [("recall", cols.get("metrics/recall", []), "#c46edb")]),
            ("Learning rate", "¿está decayendo?", [("lr", cols.get("lr", []), "#aaaaaa")]),
        ]
        parts.append('<div class="label">Curvas de entrenamiento '
                     '(pasa el ratón por un punto para ver su valor)</div>'
                     '<div class="chart-grid">')
        for title_c, sub_c, series in charts:
            ser = [{"label": l, "data": d, "color": c} for l, d, c in series if d]
            if ser:
                h, j = _line_chart(ep, ser, title_c, sub_c)
                parts.append(h); inits.append(j)
        parts.append("</div>")

    # foveation diagnostics (interactive)
    if fov["R_mean"]:
        ep = list(range(len(fov["R_mean"])))
        fcharts = [
            ("R — radio de la lupa", "↑ = amplía  ·  ~0 = lente apagada",
             [{"label": "R_mean", "data": fov["R_mean"], "color": "#ff8c42"}]),
            ("¿la lente apunta al dron?", "↓ mejor  ·  por debajo del centro fijo = la lupa acierta",
             [{"label": "|o−gt|", "data": fov["dist_to_gt"], "color": "#46c46e"},
              {"label": "centro fijo", "data": fov["dist_cen_gt"], "color": "#777"}]),
            ("¿la lente se mueve por imagen?", "↑ mejor  ·  ≈0 = colapsada a un punto",
             [{"label": "o_x_std", "data": fov["o_x_std"], "color": "#4da3ff"},
              {"label": "o_y_std", "data": fov["o_y_std"], "color": "#c46edb"}]),
        ]
        parts.append('<div class="label">Diagnóstico de la foveación '
                     '(pasa el ratón por un punto o por la métrica)</div>'
                     '<div class="chart-grid">')
        for title_c, sub_c, series in fcharts:
            if any(series[0]["data"]):
                h, j = _line_chart(ep, series, title_c, sub_c)
                parts.append(h); inits.append(j)
        parts.append("</div>")

    # detection panels
    for split, panels in (("VAL", val_panels), ("TEST", test_panels)):
        if not panels:
            continue
        parts.append(f'<div class="label">Detección — {split} '
                     "(izq: GT verde / pred rojo · der: imagen foveada con la lupa)</div>")
        parts.append('<div class="card"><div class="grid2">')
        for b64 in panels:
            parts.append(f'<img src="data:image/png;base64,{b64}">')
        parts.append("</div></div>")

    parts.append(_CHART_JS.replace("__INITS__", "\n".join(inits)))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── public entry point ─────────────────────────────────────────────────────────

def generate_report(run_dir, model, device, dataset_cls, dataset_root,
                    image_size, alpha=2.0, p=2.0, n_images=8,
                    score_threshold=0.3, fp16=True, title="drone run"):
    run_dir = Path(run_dir)
    cols = _read_results_csv(run_dir / "results.csv")
    fov = _parse_fov_log(run_dir / "train.log")
    args = _read_args_json(run_dir / "args.json")

    panels = {"val": [], "test": []}
    if model is not None:
        for split in ("val", "test"):
            try:
                ds = dataset_cls(dataset_root, split=split, image_size=tuple(image_size))
                idx = _sample_indices(ds, n_images)
                panels[split] = _detection_panels_b64(
                    model, ds, device, idx, alpha, p, score_threshold, fp16)
            except Exception as exc:                          # noqa: BLE001
                print(f"[report] skipped {split} panels: {exc}")

    html = build_html(run_dir, cols, fov, args, panels["val"], panels["test"], title)
    out = run_dir / "report.html"
    out.write_text(html)
    print(f"[report] wrote {out}  (val={len(panels['val'])}, test={len(panels['test'])} images)")
    return out


# ── standalone CLI ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Telescope HTML training report")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dataset", default="drones", choices=["drones", "argoverse2"])
    ap.add_argument("--data_dir", required=True, help="dataset root (has val/ test/)")
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to <run_dir>/checkpoint_best.pt")
    ap.add_argument("--backbone", default="efficienttam")
    ap.add_argument("--backbone_ckpt", default=None)
    ap.add_argument("--et_config", default="configs/efficienttam/efficienttam_s.yaml")
    ap.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--fov_spatial", action="store_true", default=False)
    ap.add_argument("--two_stage", action="store_true", default=True)
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--score_threshold", type=float, default=0.3)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "drones":
        from telescope.data_drones import (DronesYoloDataset as DatasetCls,
                                            DRONE_NUM_CLASSES as NUM_CLASSES)
    else:
        from telescope.data import Argoverse2Dataset as DatasetCls, NUM_CLASSES

    from telescope.pipeline import TelescopeModel
    model = TelescopeModel(num_classes=NUM_CLASSES, two_stage=args.two_stage,
                           fov_spatial=args.fov_spatial).to(device)
    if args.backbone_ckpt:
        import train as T
        if args.backbone == "efficienttam":
            T._load_efficienttam_backbone(model, args.backbone_ckpt, device, args.et_config)
        else:
            T._load_sam3_backbone(model, args.backbone_ckpt, device)
    ckpt_path = args.checkpoint or str(Path(args.run_dir) / "checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    generate_report(args.run_dir, model, device, DatasetCls, args.data_dir,
                    args.image_size, alpha=model.alpha, p=model.p,
                    n_images=args.n_images, score_threshold=args.score_threshold,
                    fp16=args.fp16, title=Path(args.run_dir).name)


if __name__ == "__main__":
    main()

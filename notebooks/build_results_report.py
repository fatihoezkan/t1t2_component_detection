"""Build the executed-results notebook source.

Run from the repository root:
    python3 notebooks/build_results_report.py

The generated notebook reads the saved cluster artifacts and checkpoint. Keeping the
notebook construction here makes a large JSON artifact maintainable and reproducible.
"""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_report.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# T1–T2 DETR — Results report

**64 measurements · 1–3 compartments · synthetic signed IR–MSE signals**

This report compares the first **99,999-voxel baseline** with the larger
**750,000-voxel run**. It answers four questions:

1. Did optimization converge stably?
2. Does more training data improve recovery?
3. Where does the detector fail—counting or parameter estimation?
4. Is performance robust across SNR, and do predictions remain physically plausible?

> All headline regression errors are medians over Hungarian-matched compartments. Count
> metrics are per voxel. SNR 20 is an extrapolation test because training used SNR 30–150."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json, os, sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if not (ROOT / "src" / "t1t2").exists():
    raise RuntimeError("Run this notebook from t1t2_component_detection/ or notebooks/.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import HTML, Markdown, display

COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "cyan": "#4BB3C3",
    "orange": "#F28E2B",
    "green": "#2A9D6F",
    "red": "#D9534F",
    "ink": "#203040",
    "muted": "#66788A",
    "grid": "#D9E2EA",
    "paper": "#F6F8FB",
}
mpl.rcParams.update({
    "figure.figsize": (10, 5.2),
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelcolor": COLORS["ink"],
    "axes.edgecolor": COLORS["grid"],
    "text.color": COLORS["ink"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.65,
    "font.size": 10,
})

display(HTML(\"\"\"
<style>
.jp-Notebook { max-width: 1180px; margin: auto; }
.report-banner {
  padding: 18px 22px; border-radius: 12px; color: #17324D;
  background: linear-gradient(105deg, #E8F3FA 0%, #F2F8F5 100%);
  border-left: 5px solid #2878B5; margin: 10px 0 20px 0;
}
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(145px,1fr)); gap:12px; margin:12px 0 22px; }
.metric-card { padding:16px 18px; border:1px solid #D9E2EA; border-radius:12px; background:#FFFFFF; }
.metric-card .value { font-size:27px; font-weight:750; color:#17324D; line-height:1.1; }
.metric-card .label { color:#66788A; font-size:12px; margin-top:6px; }
.metric-card .delta { color:#2A9D6F; font-size:12px; font-weight:650; margin-top:7px; }
.callout { padding:14px 18px; border-radius:10px; background:#FFF6EA; border-left:4px solid #F28E2B; margin:12px 0; }
.good { background:#ECF7F2; border-left-color:#2A9D6F; }
.small-note { color:#66788A; font-size:12px; }
@media (max-width: 850px) { .metric-grid { grid-template-columns:repeat(2,1fr); } }
</style>
\"\"\"))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Load the two completed runs once. Later sections first report each run independently,
# then compare them on the same axes and on the same held-out voxel examples.
RUNS = {
    "100k baseline": {
        "dir": ROOT / "results" / "cluster_baseline",
        "train_voxels": 99_999,
        "color": COLORS["cyan"],
    },
    "750k large": {
        "dir": ROOT / "results" / "cluster_large",
        "train_voxels": 750_000,
        "color": COLORS["blue"],
    },
}
for label, run in RUNS.items():
    run["metrics"] = json.loads((run["dir"] / "metrics_detr.json").read_text())
    run["snr"] = json.loads((run["dir"] / "metrics_snr_ladder.json").read_text())
    run["history"] = json.loads((run["dir"] / "history.json").read_text())
    run["summary"] = json.loads((run["dir"] / "summary.json").read_text())

def metric_cards(label):
    m = RUNS[label]["metrics"]
    values = [
        (f"{m['count_accuracy']:.1%}", "count accuracy"),
        (f"{m['t1_rel_median']:.1%}", "median T1 relative error"),
        (f"{m['t2_rel_median']:.1%}", "median T2 relative error"),
        (f"{m['w_mae']:.1%}", "median weight error"),
    ]
    html = '<div class="metric-grid">'
    for value, text in values:
        html += f'<div class="metric-card"><div class="value">{value}</div><div class="label">{text}</div></div>'
    html += '</div>'
    display(HTML(html))

def single_run_dashboard(label):
    run, m = RUNS[label], RUNS[label]["metrics"]
    h = run["history"]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.0))

    ep = np.array([z["epoch"] + 1 for z in h])
    ax[0,0].plot(ep, [z["train"]["loss"] for z in h], lw=2, ls="--",
                 color=run["color"], alpha=.7, label="train")
    ax[0,0].plot(ep, [z["val"]["loss"] for z in h], lw=2.5,
                 color=run["color"], label="validation")
    best = run["summary"]["best_epoch"] + 1
    ax[0,0].scatter(best, run["summary"]["best_val"], s=75, color=run["color"],
                    edgecolor="white", zorder=4, label=f"best epoch {best}")
    ax[0,0].set(title="Optimization", xlabel="epoch", ylabel="Hungarian loss")
    ax[0,0].grid(True, axis="y"); ax[0,0].legend(frameon=False)

    ns = np.arange(1,4)
    vals = [100*m[f"count_accuracy_n{n}"] for n in ns]
    bars = ax[0,1].bar(ns, vals, color=run["color"], width=.62)
    ax[0,1].bar_label(bars, fmt="%.1f%%", padding=3)
    ax[0,1].set(title="Count accuracy by true complexity", xlabel="true compartment count",
                ylabel="accuracy (%)", xticks=ns, ylim=(0,105))
    ax[0,1].grid(True, axis="y")

    x = np.arange(3); width=.34
    ax[1,0].bar(x-width/2, [100*m[f"n{n}_t1_rel_median"] for n in ns],
                width, color=COLORS["blue"], label="T1")
    ax[1,0].bar(x+width/2, [100*m[f"n{n}_t2_rel_median"] for n in ns],
                width, color=COLORS["orange"], label="T2")
    ax[1,0].set(title="Parameter recovery by complexity", xlabel="true compartment count",
                ylabel="median relative error (%)", xticks=x, xticklabels=ns)
    ax[1,0].grid(True, axis="y"); ax[1,0].legend(frameon=False)

    ladder = sorted(run["snr"].values(), key=lambda z:z["snr"])
    for n, color in zip(ns, [COLORS["green"], COLORS["orange"], COLORS["red"]]):
        ax[1,1].plot([z["snr"] for z in ladder],
                     [100*z[f"count_accuracy_n{n}"] for z in ladder],
                     marker="o", lw=2, color=color, label=f"n={n}")
    ax[1,1].axvspan(15,30,color="#E7E9ED",alpha=.7)
    ax[1,1].set(title="Count robustness by complexity", xlabel="SNR",
                ylabel="count accuracy (%)", xlim=(15,155), ylim=(0,102))
    ax[1,1].grid(True); ax[1,1].legend(frameon=False, ncol=3)

    fig.suptitle(label, fontsize=16, fontweight="bold", color=COLORS["navy"])
    fig.tight_layout(rect=(0,0,1,.965))
    plt.show()

def show_saved_scatter(label):
    p = RUNS[label]["dir"] / "figures" / "scatter_detr.png"
    fig, ax = plt.subplots(figsize=(11.5,5.8))
    ax.imshow(plt.imread(p)); ax.axis("off")
    ax.set_title(f"{label} · all Hungarian-matched test compartments", fontsize=14)
    fig.tight_layout(); plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Shared held-out examples: both checkpoints see the same first 2,000 test voxels per n.
# This is separate from the official stored metrics and is used only for visual/diagnostic parity.
import torch
from scipy.optimize import linear_sum_assignment
from t1t2.config import load_config
from t1t2.data import TargetNormalizer, VoxelDataset
from t1t2.model import build_model
from t1t2.eval import true_compartments, _match

SHARED_LIMIT = 2_000
large_cfg = load_config(RUNS["750k large"]["dir"] / "config.yaml")
normalizer = TargetNormalizer.from_config(large_cfg.data)
SHARED_DS = {
    n: VoxelDataset(ROOT / f"data/full_1to4/n{n}/test.parquet",
                    large_cfg.data, normalizer, limit=SHARED_LIMIT)
    for n in (1,2,3)
}

def decode_rows(out, threshold=.5):
    prob = 1 / (1 + np.exp(-out[:,:,3]))
    decoded = []
    for row, keep in zip(out, prob > threshold):
        t1 = normalizer.denormalize_t1(row[keep,0])
        t2 = normalizer.denormalize_t2(row[keep,1])
        decoded.append([(float(a),float(b),float(c))
                        for a,b,c in zip(t1,t2,row[keep,2])])
    return decoded, prob

EVAL_CACHE = {}
for label, run in RUNS.items():
    cfg = load_config(run["dir"] / "config.yaml")
    model = build_model(cfg.model)
    state = torch.load(run["dir"] / "checkpoints" / "best.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"]); model.eval()
    EVAL_CACHE[label] = {"model": model, "by_n": {}}
    with torch.no_grad():
        for n, ds in SHARED_DS.items():
            chunks = []
            for i in range(0,len(ds),512):
                out = model(ds.X[i:i+512])
                out = out["pred"] if isinstance(out,dict) else out
                chunks.append(out.numpy())
            raw = np.concatenate(chunks)
            pred, prob = decode_rows(raw)
            EVAL_CACHE[label]["by_n"][n] = {
                "raw": raw, "prob": prob, "pred": pred,
                "true": true_compartments(ds),
            }

def match_score(pred, true):
    pairs = _match(pred,true)
    if not pairs:
        return np.inf
    return float(np.mean([
        abs(np.log(p[0]/t[0])) + abs(np.log(p[1]/t[1]))
        for p,t in pairs
    ]))

def choose_example(label, n, predicted_count, quantile=.5):
    d = EVAL_CACHE[label]["by_n"][n]
    idx = [i for i,p in enumerate(d["pred"])
           if len(p)==predicted_count and np.isfinite(match_score(p,d["true"][i]))]
    if not idx:
        idx = [i for i,p in enumerate(d["pred"]) if np.isfinite(match_score(p,d["true"][i]))]
    idx.sort(key=lambda i:match_score(d["pred"][i],d["true"][i]))
    return idx[min(int(quantile*(len(idx)-1)),len(idx)-1)]

def plot_prediction_atlas(label):
    # Typical examples across successes and the two characteristic n=2 failure directions.
    cases = [
        (1,1,.50,"n=1 · correct"),
        (2,1,.50,"n=2 · merged to 1"),
        (2,2,.50,"n=2 · correct"),
        (2,3,.50,"n=2 · split to 3"),
        (3,3,.35,"n=3 · correct"),
        (3,2,.50,"n=3 · missed pool"),
    ]
    fig, axes = plt.subplots(2,3,figsize=(13.5,8.0),sharex=True,sharey=True)
    for ax,(n,pc,q,title) in zip(axes.ravel(),cases):
        i = choose_example(label,n,pc,q)
        d = EVAL_CACHE[label]["by_n"][n]
        pred,true = d["pred"][i],d["true"][i]
        for j,(t1,t2,w) in enumerate(true):
            ax.scatter(t1,t2,s=45+250*w,color=COLORS["blue"],alpha=.78,
                       edgecolor="white",lw=1.1,label="true" if j==0 else None,zorder=3)
        for j,(t1,t2,w) in enumerate(pred):
            ax.scatter(t1,t2,s=45+250*w,marker="X",color=COLORS["orange"],alpha=.92,
                       edgecolor="white",lw=.7,label="predicted" if j==0 else None,zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(45,4500); ax.set_ylim(4.5,3200)
        ax.grid(True,which="both",ls=":",lw=.5)
        ax.set_title(f"{title}\\ntrue {len(true)} · predicted {len(pred)}",fontsize=10)
        ax.set_xlabel("T1 (ms)")
    axes[0,0].set_ylabel("T2 (ms)"); axes[1,0].set_ylabel("T2 (ms)")
    axes[0,0].legend(frameon=False,fontsize=8,loc="lower right")
    fig.suptitle(f"{label} · representative held-out predictions",
                 fontsize=15,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.96)); plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Coverage data. The generator keys every voxel by (seed, n, split, voxel_id), so the
# first 33,333 rows per n in full_1to4 are exactly the nested 99,999-voxel training subset.
import pyarrow.parquet as pq

BASE_PER_N = 33_333
TARGET_COLS = [f"{name}_{i}" for name in ("T1","T2","w") for i in range(1,5)]
SIGNAL_COLS = [f"S_{i}" for i in range(1,65)]

PARAM_POINTS = {"100k baseline": {}, "750k large": {}}
for n in (1,2,3):
    df = pd.read_parquet(ROOT/f"data/full_1to4/n{n}/train.parquet",columns=TARGET_COLS)
    def flatten_params(frame,n):
        t1 = frame[[f"T1_{i}" for i in range(1,n+1)]].to_numpy(float).ravel()
        t2 = frame[[f"T2_{i}" for i in range(1,n+1)]].to_numpy(float).ravel()
        w = frame[[f"w_{i}" for i in range(1,n+1)]].to_numpy(float).ravel()
        return np.c_[t1,t2,w]
    PARAM_POINTS["100k baseline"][n] = flatten_params(df.iloc[:BASE_PER_N],n)
    PARAM_POINTS["750k large"][n] = flatten_params(df,n)
    del df

def _read_first_signals(path,n_rows):
    batches=[]; got=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=min(8192,n_rows),
                                                    columns=SIGNAL_COLS):
        a = batch.to_pandas().to_numpy(np.float32)
        take=min(len(a),n_rows-got); batches.append(a[:take]); got+=take
        if got>=n_rows: break
    return np.concatenate(batches)

def _signal_norm(X):
    scale=np.max(np.abs(X),axis=1,keepdims=True); scale[scale==0]=1
    return X/scale

# A shared descriptive PCA coordinate system, fitted on a balanced sample from n=1..3.
fit = np.concatenate([
    _signal_norm(_read_first_signals(ROOT/f"data/full_1to4/n{n}/train.parquet",4_000))
    for n in (1,2,3)
])
PCA_MEAN = fit.mean(0)
_, singular, vt = np.linalg.svd(fit-PCA_MEAN,full_matrices=False)
PCA_COMPONENTS = vt[:8]
PCA_EXPLAINED = singular**2/np.sum(singular**2)
del fit

def project_family(limit_per_n=None):
    out={}
    for n in (1,2,3):
        path=ROOT/f"data/full_1to4/n{n}/train.parquet"
        if limit_per_n is None:
            X=pd.read_parquet(path,columns=SIGNAL_COLS).to_numpy(np.float32)
        else:
            X=_read_first_signals(path,limit_per_n)
        X=_signal_norm(X)
        out[n]=(X-PCA_MEAN)@PCA_COMPONENTS[:3].T
        del X
    return out

SIGNAL_PCA = {
    "100k baseline": project_family(BASE_PER_N),
    "750k large": project_family(None),
}
all_large=np.concatenate(list(SIGNAL_PCA["750k large"].values()))
PCA_BOUNDS=[
    np.quantile(all_large[:,0],[.002,.998]),
    np.quantile(all_large[:,1],[.002,.998]),
    np.quantile(all_large[:,2],[.002,.998]),
]

def plot_parameter_coverage(label):
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharex=True,sharey=True)
    for ax,n in zip(axes,(1,2,3)):
        p=PARAM_POINTS[label][n]
        ax.hexbin(p[:,0],p[:,1],xscale="log",yscale="log",gridsize=52,
                  bins="log",mincnt=1,cmap="viridis")
        ax.plot([50,3000],[50,3000],ls="--",lw=1,color="white",alpha=.8)
        ax.set(title=f"n={n} · {len(p):,} compartments",xlabel="T1 (ms)",
               xlim=(45,4500),ylim=(4.5,3200))
        ax.grid(False)
    axes[0].set_ylabel("T2 (ms)")
    fig.suptitle(f"{label} · T1–T2 relaxation-spectrum coverage",
                 fontsize=15,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.94)); plt.show()

def plot_signal_coverage(label):
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),sharex=True,sharey=True)
    for ax,n in zip(axes,(1,2,3)):
        z=SIGNAL_PCA[label][n]
        ax.hexbin(z[:,0],z[:,1],gridsize=55,bins="log",mincnt=1,cmap="magma",
                  extent=(PCA_BOUNDS[0][0],PCA_BOUNDS[0][1],
                          PCA_BOUNDS[1][0],PCA_BOUNDS[1][1]))
        ax.set(title=f"n={n} · {len(z):,} signals",xlabel="signal PC1",
               xlim=PCA_BOUNDS[0],ylim=PCA_BOUNDS[1])
        ax.grid(False)
    axes[0].set_ylabel("signal PC2")
    fig.suptitle(f"{label} · coverage of the measured 64D signal manifold",
                 fontsize=15,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.94)); plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """# Part I — 100k baseline

The first report is deliberately self-contained: training behaviour, counting, parameter
recovery, SNR robustness, physical plausibility, matched scatter, representative predictions,
and data coverage are shown before introducing the larger run."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """metric_cards("100k baseline")
single_run_dashboard("100k baseline")
show_saved_scatter("100k baseline")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """### 100k prediction atlas

Blue circles are true compartments; orange crosses are retained predictions at the official
existence threshold of 0.5. Marker area encodes weight."""
    )
)

cells.append(nbf.v4.new_code_cell("""plot_prediction_atlas("100k baseline")"""))

cells.append(
    nbf.v4.new_markdown_cell(
        """### 100k coverage

“Frequency space” is not literally a Fourier frequency axis here: the 64 inputs are irregular
TI–TE acquisition pairs, not an equally sampled time series. The scientifically relevant
coverage views are therefore:

- **T1–T2 relaxation-spectrum space:** where the true compartments lie.
- **Measured-signal space:** where the normalized 64-number signals lie, visualized in shared
  PCA coordinates. PCA is descriptive only; it is not an identifiability proof."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """plot_parameter_coverage("100k baseline")
plot_signal_coverage("100k baseline")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """# Part II — 750k baseline

The architecture and task are unchanged. Only the number of training voxels grows from
99,999 to 750,000."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """metric_cards("750k large")
single_run_dashboard("750k large")
show_saved_scatter("750k large")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """### 750k prediction atlas

The same six outcome types are shown, now using the larger checkpoint."""
    )
)

cells.append(nbf.v4.new_code_cell("""plot_prediction_atlas("750k large")"""))

cells.append(
    nbf.v4.new_markdown_cell(
        """### 750k coverage

The theoretical support does not expand—the sampler ranges are identical. More data should
instead fill the same admissible region more densely, especially at fine resolution."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """plot_parameter_coverage("750k large")
plot_signal_coverage("750k large")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """# Part III — Direct comparison

## Results at a glance

The larger run changes only training-set size; architecture, 64-point protocol, loss,
normalization, and optimizer remain fixed. This makes the comparison a clean data-scaling
result."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """RUNS = {
    "100k baseline": {
        "dir": ROOT / "results" / "cluster_baseline",
        "train_voxels": 99_999,
        "color": COLORS["cyan"],
    },
    "750k large": {
        "dir": ROOT / "results" / "cluster_large",
        "train_voxels": 750_000,
        "color": COLORS["blue"],
    },
}

for label, run in RUNS.items():
    run["metrics"] = json.loads((run["dir"] / "metrics_detr.json").read_text())
    run["snr"] = json.loads((run["dir"] / "metrics_snr_ladder.json").read_text())
    run["history"] = json.loads((run["dir"] / "history.json").read_text())
    run["summary"] = json.loads((run["dir"] / "summary.json").read_text())

b = RUNS["100k baseline"]["metrics"]
l = RUNS["750k large"]["metrics"]

def reduction(old, new):
    return 100 * (old - new) / old

cards = [
    ("69.3%", "compartment-count accuracy", f"+{100*(l['count_accuracy']-b['count_accuracy']):.1f} percentage points"),
    ("7.8%", "median T1 relative error", f"{reduction(b['t1_rel_median'], l['t1_rel_median']):.1f}% lower than baseline"),
    ("11.7%", "median T2 relative error", f"{reduction(b['t2_rel_median'], l['t2_rel_median']):.1f}% lower than baseline"),
    ("3.8 pp", "median weight absolute error", f"{reduction(b['w_mae'], l['w_mae']):.1f}% lower than baseline"),
]
html = '<div class="report-banner"><b>Headline:</b> more synthetic training data produces a clear, broad improvement—but count recovery for two-compartment voxels is still the dominant failure mode.</div>'
html += '<div class="metric-grid">'
for value, label, delta in cards:
    html += f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div><div class="delta">↗ {delta}</div></div>'
html += '</div>'
display(HTML(html))

summary_rows = []
for label, run in RUNS.items():
    s, m = run["summary"], run["metrics"]
    summary_rows.append({
        "run": label,
        "training voxels": f"{run['train_voxels']:,}",
        "test voxels": f"{m['n_voxels']:,}",
        "epochs": s["epochs_run"],
        "best epoch": s["best_epoch"] + 1,
        "best val loss": s["best_val"],
        "wall time": f"{s['wall_seconds']/60:.1f} min",
    })
pd.DataFrame(summary_rows).set_index("run").style.format({"best val loss": "{:.4f}"})"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# A compact, exportable one-page summary.
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
fig.patch.set_facecolor("white")

# Training
ax = axes[0, 0]
for label, run in RUNS.items():
    h = run["history"]
    ep = np.array([x["epoch"] + 1 for x in h])
    ax.plot(ep, [x["val"]["loss"] for x in h], lw=2.2, color=run["color"], label=label)
    best = run["summary"]["best_epoch"] + 1
    ax.scatter(best, run["summary"]["best_val"], s=60, color=run["color"], zorder=4)
ax.set(title="Validation loss", xlabel="epoch", ylabel="Hungarian loss")
ax.grid(True, axis="y"); ax.legend(frameon=False)

# Overall comparison
ax = axes[0, 1]
labels = ["count\\naccuracy", "T1 rel.\\nerror", "T2 rel.\\nerror", "weight\\nerror"]
base_vals = [100*b["count_accuracy"], 100*b["t1_rel_median"], 100*b["t2_rel_median"], 100*b["w_mae"]]
large_vals = [100*l["count_accuracy"], 100*l["t1_rel_median"], 100*l["t2_rel_median"], 100*l["w_mae"]]
x = np.arange(4); width = 0.35
ax.bar(x-width/2, base_vals, width, color=RUNS["100k baseline"]["color"], label="100k")
ax.bar(x+width/2, large_vals, width, color=RUNS["750k large"]["color"], label="750k")
ax.set(title="Headline metrics", ylabel="%  (weight shown as percentage points)", xticks=x, xticklabels=labels)
ax.grid(True, axis="y"); ax.legend(frameon=False)

# Count by complexity
ax = axes[1, 0]
x = np.arange(3)
for i, (label, run) in enumerate(RUNS.items()):
    vals = [100*run["metrics"][f"count_accuracy_n{n}"] for n in (1, 2, 3)]
    ax.bar(x + (i-.5)*width, vals, width, color=run["color"], label=label)
ax.set(title="Count accuracy exposes the bottleneck", xlabel="true compartment count", ylabel="accuracy (%)",
       xticks=x, xticklabels=["1", "2", "3"], ylim=(0, 100))
ax.grid(True, axis="y"); ax.legend(frameon=False)

# SNR
ax = axes[1, 1]
for label, run in RUNS.items():
    vals = sorted(run["snr"].values(), key=lambda z: z["snr"])
    ax.plot([z["snr"] for z in vals], [100*z["count_accuracy"] for z in vals],
            marker="o", lw=2.2, color=run["color"], label=label)
ax.axvspan(0, 30, color="#E7E9ED", alpha=.65, label="outside train range")
ax.set(title="Robustness across SNR", xlabel="SNR", ylabel="count accuracy (%)", xlim=(15, 155))
ax.grid(True); ax.legend(frameon=False)

fig.suptitle("T1–T2 DETR · 64-input synthetic benchmark", fontsize=17, fontweight="bold", color=COLORS["navy"], y=.99)
fig.tight_layout(rect=(0, 0, 1, .965))
report_dir = ROOT / "results" / "results_report"
report_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(report_dir / "results_at_a_glance.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Optimization and data scaling

Both runs early-stop cleanly. The best checkpoint—not the last epoch—is used for every
reported test metric. The larger run reaches a lower validation loss despite seeing far more
unique signals, which is consistent with useful data scaling rather than memorization."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for label, run in RUNS.items():
    h = run["history"]
    ep = np.array([x["epoch"] + 1 for x in h])
    axes[0].plot(ep, [x["train"]["loss"] for x in h], color=run["color"], lw=2, ls="--", alpha=.8)
    axes[0].plot(ep, [x["val"]["loss"] for x in h], color=run["color"], lw=2.4, label=f"{label} · validation")
    be = run["summary"]["best_epoch"] + 1
    axes[0].scatter(be, run["summary"]["best_val"], s=70, color=run["color"], edgecolor="white", zorder=5)

parts = ["t1", "t2", "wt", "ex"]
part_names = ["T1", "T2", "weight", "existence"]
for part, name, color in zip(parts, part_names, [COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["green"]]):
    h = RUNS["750k large"]["history"]
    axes[1].plot([x["epoch"]+1 for x in h], [x["val"][part] for x in h], label=name, color=color, lw=2)

axes[0].set(title="Train and validation loss", xlabel="epoch", ylabel="loss")
axes[0].grid(True, axis="y"); axes[0].legend(frameon=False, fontsize=9)
axes[1].set(title="Large-run validation loss components", xlabel="epoch", ylabel="loss contribution")
axes[1].grid(True, axis="y"); axes[1].legend(frameon=False, ncol=2)
fig.tight_layout()
plt.show()

gain_rows = [
    ("count accuracy", 100*b["count_accuracy"], 100*l["count_accuracy"], "higher"),
    ("T1 relative error", 100*b["t1_rel_median"], 100*l["t1_rel_median"], "lower"),
    ("T2 relative error", 100*b["t2_rel_median"], 100*l["t2_rel_median"], "lower"),
    ("weight absolute error", 100*b["w_mae"], 100*l["w_mae"], "lower"),
    ("CSF T2 relative error", 100*b["t2_rel_median_csf"], 100*l["t2_rel_median_csf"], "lower"),
]
gain_df = pd.DataFrame(gain_rows, columns=["metric", "100k", "750k", "direction"])
gain_df["relative change"] = np.where(
    gain_df.direction.eq("higher"),
    100*(gain_df["750k"]-gain_df["100k"])/gain_df["100k"],
    100*(gain_df["100k"]-gain_df["750k"])/gain_df["100k"],
)
gain_df.drop(columns="direction").set_index("metric").style.format({
    "100k": "{:.1f}%", "750k": "{:.1f}%", "relative change": "+{:.1f}%"
}).background_gradient(subset=["relative change"], cmap="Greens")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Counting is the main bottleneck

Aggregate accuracy hides a strongly non-uniform result. One- and three-compartment voxels
are usually counted correctly, while two-compartment voxels are frequently predicted as
three. More data nearly doubles two-compartment accuracy, but it remains only **34.1%**."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2]})
x = np.arange(3); width = .34
for i, (label, run) in enumerate(RUNS.items()):
    vals = [100*run["metrics"][f"count_accuracy_n{n}"] for n in (1,2,3)]
    bars = axes[0].bar(x + (i-.5)*width, vals, width, label=label, color=run["color"])
    axes[0].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
axes[0].set(title="Correct count by true complexity", xlabel="true count", ylabel="accuracy (%)",
            xticks=x, xticklabels=[1,2,3], ylim=(0,105))
axes[0].grid(True, axis="y"); axes[0].legend(frameon=False, fontsize=8)

for ax, (label, run) in zip(axes[1:], RUNS.items()):
    conf = run["metrics"]["confusion"]["matrix"]
    # Predicted counts above 4 are empty for these runs, so 1–4 is the informative support.
    mat = np.array([conf[str(n)][1:5] for n in (1,2,3)], dtype=float)
    pct = 100 * mat / mat.sum(axis=1, keepdims=True)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for r in range(3):
        for c in range(4):
            ax.text(c, r, f"{pct[r,c]:.0f}", ha="center", va="center",
                    color="white" if pct[r,c] > 52 else COLORS["ink"], fontsize=9)
    ax.set(title=label, xlabel="predicted count", ylabel="true count",
           xticks=range(4), xticklabels=[1,2,3,4], yticks=range(3), yticklabels=[1,2,3])
fig.suptitle("Compartment-count accuracy and confusion · row-normalized (%)", fontsize=14, fontweight="bold")
fig.subplots_adjust(left=.06, right=.98, bottom=.15, top=.82, wspace=.42)
plt.show()

n2_to_3 = 100 * np.array(l["confusion"]["matrix"]["2"])[3] / sum(l["confusion"]["matrix"]["2"])
display(HTML(f'<div class="callout"><b>Specific failure:</b> in the 750k run, {n2_to_3:.1f}% of true two-compartment voxels are assigned three compartments. This points to existence/count calibration or unresolved signal ambiguity—not a general inability to regress T1/T2.</div>'))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Parameter recovery degrades with mixture complexity

For a single compartment, the large model is highly accurate. Errors rise smoothly as more
compartments contribute to the same 64 measurements. The trend is expected for an ill-posed
mixture problem, but the gap from one to three compartments is scientifically important and
must accompany any aggregate metric."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
metric_specs = [
    ("t1_rel_median", "T1 median relative error", "%"),
    ("t2_rel_median", "T2 median relative error", "%"),
    ("w_mae", "Weight median absolute error", "percentage points"),
]
x = np.arange(3); width=.34
for ax, (metric, title, unit) in zip(axes, metric_specs):
    for i, (label, run) in enumerate(RUNS.items()):
        vals = [100*run["metrics"][f"n{n}_{metric}"] for n in (1,2,3)]
        ax.bar(x + (i-.5)*width, vals, width, color=run["color"], label=label)
    ax.set(title=title, xlabel="true compartment count", ylabel=unit,
           xticks=x, xticklabels=[1,2,3])
    ax.grid(True, axis="y")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

large_by_n = pd.DataFrame({
    "true count": [1,2,3],
    "count accuracy": [l[f"count_accuracy_n{n}"] for n in (1,2,3)],
    "T1 median rel. error": [l[f"n{n}_t1_rel_median"] for n in (1,2,3)],
    "T2 median rel. error": [l[f"n{n}_t2_rel_median"] for n in (1,2,3)],
    "weight median abs. error": [l[f"n{n}_w_mae"] for n in (1,2,3)],
}).set_index("true count")
large_by_n.style.format("{:.1%}").background_gradient(cmap="Blues", axis=0)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Robustness across SNR

The fixed-SNR sets contain paired underlying compartments, so movement along each curve is
primarily the effect of noise. Performance improves monotonically across the training range
(30–150). SNR 20 is shown as a deliberately out-of-distribution stress test."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
specs = [
    ("count_accuracy", "Count accuracy", "%", lambda z: 100*z),
    ("t1_rel_median", "T1 median relative error", "%", lambda z: 100*z),
    ("t2_rel_median", "T2 median relative error", "%", lambda z: 100*z),
]
for ax, (key, title, unit, transform) in zip(axes, specs):
    for label, run in RUNS.items():
        vals = sorted(run["snr"].values(), key=lambda z: z["snr"])
        xs = [z["snr"] for z in vals]
        ys = [transform(z[key]) for z in vals]
        ax.plot(xs, ys, marker="o", ms=6, lw=2.2, color=run["color"], label=label)
    ax.axvspan(15, 30, color="#E7E9ED", alpha=.7)
    ax.axvline(30, color=COLORS["muted"], lw=1, ls=":")
    ax.set(title=title, xlabel="SNR", ylabel=unit, xlim=(15,155))
    ax.grid(True)
axes[0].legend(frameon=False, fontsize=8)
axes[0].text(21.5, axes[0].get_ylim()[0] + .04*np.ptp(axes[0].get_ylim()), "extrapolation",
             rotation=90, color=COLORS["muted"], ha="center", va="bottom", fontsize=8)
fig.tight_layout()
plt.show()

snr_rows = []
for label, run in RUNS.items():
    for d in sorted(run["snr"].values(), key=lambda z:z["snr"]):
        snr_rows.append({
            "run": label, "SNR": int(d["snr"]),
            "count accuracy": d["count_accuracy"],
            "T1 rel. error": d["t1_rel_median"],
            "T2 rel. error": d["t2_rel_median"],
            "weight error": d["w_mae"],
            "regime": "extrapolation" if d["extrapolation"] else "train range",
        })
pd.DataFrame(snr_rows).set_index(["run","SNR"]).style.format({
    "count accuracy":"{:.1%}", "T1 rel. error":"{:.1%}",
    "T2 rel. error":"{:.1%}", "weight error":"{:.1%}",
})"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Physical plausibility and the long-T2 limitation

The model is not explicitly constrained to enforce $T_2<T_1$ or weights summing to one.
Measuring violations therefore checks whether it learned those structural properties from
the simulated physics. CSF-like pools ($T_2>1000$ ms) are reported separately because the
protocol's longest TE is only about 150 ms."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """physical = []
for label, run in RUNS.items():
    m = run["metrics"]
    physical.append({
        "run": label,
        "T2 ≥ T1 predictions": m["t2_ge_t1_rate"],
        "median |Σw − 1|": m["weight_sum_dev_median"],
        "non-CSF T2 rel. error": m["t2_rel_median_noncsf"],
        "CSF-like T2 rel. error": m["t2_rel_median_csf"],
    })
display(pd.DataFrame(physical).set_index("run").style.format("{:.1%}"))

display(HTML(f'''
<div class="callout good"><b>Physical outputs:</b> only {l["t2_ge_t1_rate"]:.2%} of retained large-model predictions violate T2 &lt; T1, and the median weight-sum deviation is {l["weight_sum_dev_median"]:.1%}.</div>
<div class="callout"><b>Protocol-limited regime:</b> median T2 error is {l["t2_rel_median_noncsf"]:.1%} outside CSF-like pools but {l["t2_rel_median_csf"]:.1%} for T2 &gt; 1000 ms. This is consistent with weak long-T2 sensitivity at TE<sub>max</sub> ≈ 150 ms.</div>
'''))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 7. Predicted versus true relaxation times

Every point is a Hungarian-matched compartment. The diagonal is perfect recovery. The
larger run visibly tightens both T1 and T2 estimates, while long-T2 predictions retain the
largest spread."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(2, 1, figsize=(12, 10))
for ax, (label, run) in zip(axes, RUNS.items()):
    image_path = run["dir"] / "figures" / "scatter_detr.png"
    ax.imshow(plt.imread(image_path))
    ax.set_title(label, fontsize=13, pad=8)
    ax.axis("off")
fig.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 8. Representative predictions from the large checkpoint

These examples are selected reproducibly from a small prefix of each held-out test stratum.
They show median-quality correct predictions for 1, 2, and 3 compartments, plus a typical
two-compartment overcount. Marker area represents compartment weight."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """import torch
from t1t2.config import load_config
from t1t2.data import TargetNormalizer, VoxelDataset
from t1t2.model import build_model
from t1t2.eval import detr_predictions, true_compartments, _match

cfg = load_config(RUNS["750k large"]["dir"] / "config.yaml")
normalizer = TargetNormalizer.from_config(cfg.data)
model = build_model(cfg.model)
checkpoint = torch.load(
    RUNS["750k large"]["dir"] / "checkpoints" / "best.pt",
    map_location="cpu", weights_only=True,
)
model.load_state_dict(checkpoint["model"])
model.eval()

small = {}
for n in (1,2,3):
    ds = VoxelDataset(ROOT / f"data/full_1to4/n{n}/test.parquet", cfg.data, normalizer, limit=400)
    small[n] = (detr_predictions(model, ds, torch.device("cpu"), normalizer, batch_size=400),
                true_compartments(ds))

def match_score(pred, true):
    pairs = _match(pred, true)
    if not pairs:
        return np.inf
    return float(np.mean([
        abs(np.log(p[0]/t[0])) + abs(np.log(p[1]/t[1]))
        for p, t in pairs
    ]))

def median_example(n, correct=True):
    preds, trues = small[n]
    idx = [i for i,(p,t) in enumerate(zip(preds,trues))
           if (len(p)==len(t)) == correct and np.isfinite(match_score(p,t))]
    idx.sort(key=lambda i: match_score(preds[i],trues[i]))
    return idx[len(idx)//2]

chosen = [
    (1, median_example(1, True), "1 compartment · correct count"),
    (2, median_example(2, True), "2 compartments · correct count"),
    (2, median_example(2, False), "2 compartments · typical overcount"),
    (3, median_example(3, True), "3 compartments · correct count"),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), sharex=True, sharey=True)
for ax, (n, idx, title) in zip(axes, chosen):
    pred, true = small[n][0][idx], small[n][1][idx]
    for j,(t1,t2,w) in enumerate(true):
        ax.scatter(t1,t2,s=45+260*w, color=COLORS["blue"], alpha=.78,
                   edgecolor="white", linewidth=1.2, label="true" if j==0 else None, zorder=3)
    for j,(t1,t2,w) in enumerate(pred):
        ax.scatter(t1,t2,s=45+260*w, marker="X", color=COLORS["orange"], alpha=.9,
                   edgecolor="white", linewidth=.7, label="predicted" if j==0 else None, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(45,4500); ax.set_ylim(4.5,3200)
    ax.grid(True, which="both", ls=":", lw=.5)
    ax.set_title(f"{title}\\ntrue {len(true)} · predicted {len(pred)}", fontsize=10)
    ax.set_xlabel("T1 (ms)")
axes[0].set_ylabel("T2 (ms)")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.suptitle("Held-out examples · marker area encodes weight", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Coverage comparison: how much of the available space is filled?

Coverage is measured at several resolutions. A coarse grid is easy to fill; a fine grid
tests whether rare corners and narrow structures are represented. Because the 100k set is a
deterministic prefix of the 750k family, the ratio below has a direct interpretation:
**what fraction of cells observed in the 750k set were already present in the 100k set?**"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def occupied_ids(points, grid):
    x=(np.log(points[:,0])-np.log(50))/(np.log(4000)-np.log(50))
    y=(np.log(points[:,1])-np.log(5))/(np.log(3000)-np.log(5))
    ix=np.clip((x*grid).astype(int),0,grid-1)
    iy=np.clip((y*grid).astype(int),0,grid-1)
    return np.unique(ix*grid+iy)

param_rows=[]
for n in (1,2,3):
    for grid in (16,32,64,128,256):
        small=occupied_ids(PARAM_POINTS["100k baseline"][n],grid)
        large=occupied_ids(PARAM_POINTS["750k large"][n],grid)
        param_rows.append({
            "n":n,"bins per axis":grid,
            "100k occupied":len(small),"750k occupied":len(large),
            "100k / 750k coverage":len(np.intersect1d(small,large))/len(large),
        })
PARAM_COVERAGE=pd.DataFrame(param_rows)

def pca_occupied(z,grid,dims):
    ids=np.zeros(len(z),dtype=np.int64)
    for d in range(dims):
        lo,hi=PCA_BOUNDS[d]
        q=np.clip(((z[:,d]-lo)/(hi-lo)*grid).astype(int),0,grid-1)
        ids=ids*grid+q
    return np.unique(ids)

signal_rows=[]
for dims in (2,3):
    small=np.concatenate(list(SIGNAL_PCA["100k baseline"].values()))
    large=np.concatenate(list(SIGNAL_PCA["750k large"].values()))
    for grid in (8,12,16,24,32):
        a=pca_occupied(small,grid,dims); b_=pca_occupied(large,grid,dims)
        signal_rows.append({
            "dimensions":dims,"bins per dimension":grid,
            "100k occupied":len(a),"750k occupied":len(b_),
            "100k / 750k coverage":len(np.intersect1d(a,b_))/len(b_),
        })
SIGNAL_COVERAGE=pd.DataFrame(signal_rows)

fig,ax=plt.subplots(1,3,figsize=(14.5,4.4))
for n,color in zip((1,2,3),[COLORS["green"],COLORS["orange"],COLORS["red"]]):
    d=PARAM_COVERAGE.query("n==@n")
    ax[0].plot(d["bins per axis"],100*d["100k / 750k coverage"],
               marker="o",lw=2,color=color,label=f"n={n}")
ax[0].set_xscale("log",base=2)
ax[0].set(title="T1–T2 spectrum coverage",xlabel="bins per axis",
          ylabel="750k occupied cells already covered (%)",ylim=(0,105))
ax[0].grid(True);ax[0].legend(frameon=False)

for dims,color in zip((2,3),[COLORS["blue"],COLORS["orange"]]):
    d=SIGNAL_COVERAGE.query("dimensions==@dims")
    ax[1].plot(d["bins per dimension"],100*d["100k / 750k coverage"],
               marker="o",lw=2,color=color,label=f"{dims} PCA dimensions")
ax[1].set(title="Measured-signal manifold coverage",xlabel="bins per PCA dimension",
          ylabel="750k occupied cells already covered (%)",ylim=(0,105))
ax[1].grid(True);ax[1].legend(frameon=False)

cum=100*np.cumsum(PCA_EXPLAINED[:12])
ax[2].bar(np.arange(1,len(cum)+1),100*PCA_EXPLAINED[:12],color=COLORS["cyan"],label="individual")
ax2=ax[2].twinx()
ax2.plot(np.arange(1,len(cum)+1),cum,color=COLORS["navy"],marker="o",label="cumulative")
ax[2].set(title="Approximate signal-space dimension",xlabel="principal component",
          ylabel="individual variance (%)",xticks=np.arange(1,len(cum)+1))
ax2.set_ylabel("cumulative variance (%)");ax2.set_ylim(0,102)
ax[2].grid(True,axis="y")
fig.suptitle("Coverage gained by scaling 100k → 750k",fontsize=15,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.94));plt.show()

display(PARAM_COVERAGE.pivot(index="bins per axis",columns="n",
                             values="100k / 750k coverage").style.format("{:.1%}"))
display(SIGNAL_COVERAGE.pivot(index="bins per dimension",columns="dimensions",
                              values="100k / 750k coverage").style.format("{:.1%}"))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## n=1, n=2, and n=3 comparison

The comparison must be read by true compartment count. High n=3 count accuracy does **not**
mean n=3 parameter recovery is easier: its T1/T2 regression errors are the largest."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Same held-out voxels, two checkpoints: rows are models, columns are outcome types.
shared_cases=[
    (1,choose_example("750k large",1,1,.5),"n=1 typical"),
    (2,choose_example("750k large",2,1,.5),"n=2 merged"),
    (2,choose_example("750k large",2,2,.5),"n=2 correct"),
    (2,choose_example("750k large",2,3,.5),"n=2 split"),
    (3,choose_example("750k large",3,3,.5),"n=3 typical"),
]
fig,axes=plt.subplots(2,5,figsize=(16,7.0),sharex=True,sharey=True)
for r,label in enumerate(("100k baseline","750k large")):
    for c,(n,i,title) in enumerate(shared_cases):
        ax=axes[r,c];d=EVAL_CACHE[label]["by_n"][n]
        pred,true=d["pred"][i],d["true"][i]
        for t1,t2,w in true:
            ax.scatter(t1,t2,s=35+210*w,color=COLORS["blue"],alpha=.75,
                       edgecolor="white",lw=1,zorder=3)
        for t1,t2,w in pred:
            ax.scatter(t1,t2,s=35+210*w,marker="X",color=COLORS["orange"],alpha=.92,
                       edgecolor="white",lw=.6,zorder=4)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(45,4500);ax.set_ylim(4.5,3200)
        ax.grid(True,which="both",ls=":",lw=.45)
        ax.set_title(f"{title}\\ntrue {len(true)} · pred {len(pred)}",fontsize=9)
        if c==0:ax.set_ylabel(f"{label}\\nT2 (ms)")
        if r==1:ax.set_xlabel("T1 (ms)")
fig.suptitle("Direct visual comparison on identical held-out voxels",
             fontsize=15,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.95));plt.show()

# SNR behaviour separated by n: count accuracy on top, T2 recovery on bottom.
fig,axes=plt.subplots(2,3,figsize=(14,7.5),sharex=True)
for col,n in enumerate((1,2,3)):
    for label,run in RUNS.items():
        ladder=sorted(run["snr"].values(),key=lambda z:z["snr"])
        xs=[z["snr"] for z in ladder]
        axes[0,col].plot(xs,[100*z[f"count_accuracy_n{n}"] for z in ladder],
                         marker="o",lw=2,color=run["color"],label=label)
        axes[1,col].plot(xs,[100*z[f"n{n}_t2_rel_median"] for z in ladder],
                         marker="o",lw=2,color=run["color"],label=label)
    axes[0,col].axvspan(15,30,color="#E7E9ED",alpha=.7)
    axes[1,col].axvspan(15,30,color="#E7E9ED",alpha=.7)
    axes[0,col].set_title(f"n={n}")
    axes[0,col].set_ylabel("count accuracy (%)" if col==0 else "")
    axes[1,col].set_ylabel("T2 median rel. error (%)" if col==0 else "")
    axes[1,col].set_xlabel("SNR")
    axes[0,col].grid(True);axes[1,col].grid(True)
axes[0,0].legend(frameon=False)
fig.suptitle("Per-n robustness comparison",fontsize=15,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.95));plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """# Why is n=2 count accuracy lower?

There are two distinct questions:

1. **Acquisition ambiguity:** do close or weak true pools merge into one observable decay?
2. **Decision calibration:** does the existence head retain too many queries at the fixed
   threshold of 0.5?

The diagnostics below use the shared held-out subset. Threshold selection is done separately
on validation data and is shown only as a diagnostic; the official stored test metrics remain
at the pre-specified threshold of 0.5."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# True n=2 difficulty features.
from t1t2.physics import load_protocol
proto=load_protocol()
n2df=pd.read_parquet(ROOT/"data/full_1to4/n2/test.parquet",
                     columns=["sigma","T1_1","T1_2","T2_1","T2_2","w_1","w_2"]).iloc[:SHARED_LIMIT]
t1=n2df[["T1_1","T1_2"]].to_numpy(float)
t2=n2df[["T2_1","T2_2"]].to_numpy(float)
w=n2df[["w_1","w_2"]].to_numpy(float)
separation=np.sqrt(np.diff(np.log(t1),axis=1).ravel()**2+
                   np.diff(np.log(t2),axis=1).ravel()**2)
min_weight=w.min(1)

# Signal change after dropping the smaller pool, relative to that voxel's actual noise sigma.
ti=proto.ti[None,:,None];te=proto.te[None,:,None]
inv=1-2*np.exp(-ti/t1[:,None,:])+np.exp(-proto.tr/t1[:,None,:])
dec=np.exp(-te/t2[:,None,:])
basis=inv*dec
full=(basis*w[:,None,:]).sum(2)
small_idx=np.argmin(w,axis=1);keep_idx=1-small_idx
reduced=basis[np.arange(len(basis)),:,keep_idx]
detectability=np.median(np.abs(full-reduced),axis=1)/n2df["sigma"].to_numpy(float)

fig,axes=plt.subplots(2,2,figsize=(12.5,8.0))
for r,(label,run) in enumerate(RUNS.items()):
    prob=EVAL_CACHE[label]["by_n"][2]["prob"]
    count=(prob>.5).sum(1)
    groups=[1,2,3,4]
    composition=[100*np.mean(count==g) for g in groups]
    axes[0,0].bar(np.arange(4)+(r-.5)*.35,composition,.35,
                  color=run["color"],label=label)

axes[0,0].set(title="True n=2: predicted-count composition",xlabel="predicted count",
              ylabel="voxels (%)",xticks=range(4),xticklabels=[1,2,3,4])
axes[0,0].grid(True,axis="y");axes[0,0].legend(frameon=False)

large_count=(EVAL_CACHE["750k large"]["by_n"][2]["prob"]>.5).sum(1)
use=[large_count==g for g in (1,2,3)]
labels=["pred 1\\nmerge","pred 2\\ncorrect","pred 3\\nsplit"]
for ax,values,title,ylabel in [
    (axes[0,1],separation,"True-pool separation","log-space distance"),
    (axes[1,0],min_weight,"Smaller true-pool weight","minimum weight"),
    (axes[1,1],np.log10(np.clip(detectability,1e-4,None)),
     "Small-pool signal visibility","log10(signal change / noise σ)"),
]:
    bp=ax.boxplot([values[u] for u in use],labels=labels,patch_artist=True,
                  showfliers=False,medianprops={"color":COLORS["navy"],"lw":2})
    for patch,color in zip(bp["boxes"],[COLORS["red"],COLORS["green"],COLORS["orange"]]):
        patch.set_facecolor(color);patch.set_alpha(.55)
    ax.set(title=title,ylabel=ylabel);ax.grid(True,axis="y")
axes[1,1].axhline(0,color=COLORS["red"],ls="--",lw=1,label="signal change = noise σ")
axes[1,1].legend(frameon=False,fontsize=8)
fig.suptitle("What distinguishes n=2 merges, correct counts, and splits? · 750k model",
             fontsize=14,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.95));plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Count-manifold overlap in the same shared PCA coordinates.
from scipy.ndimage import gaussian_filter

def mass_level(H,mass=.70):
    vals=np.sort(H.ravel())[::-1]
    cum=np.cumsum(vals)
    if cum[-1]==0:return 0
    return vals[np.searchsorted(cum,mass*cum[-1])]

fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharex=True,sharey=True)
for ax,label in zip(axes,("100k baseline","750k large")):
    for n,color in zip((1,2,3),[COLORS["green"],COLORS["orange"],COLORS["red"]]):
        z=SIGNAL_PCA[label][n]
        H,xe,ye=np.histogram2d(z[:,0],z[:,1],bins=90,
                              range=[PCA_BOUNDS[0],PCA_BOUNDS[1]])
        H=gaussian_filter(H,1.2)
        level=mass_level(H,.70)
        xc=(xe[:-1]+xe[1:])/2;yc=(ye[:-1]+ye[1:])/2
        ax.contour(xc,yc,H.T,levels=[level],colors=[color],linewidths=2)
        ax.plot([],[],color=color,lw=2,label=f"n={n} · 70% mass contour")
    ax.set(title=label,xlabel="signal PC1",xlim=PCA_BOUNDS[0],ylim=PCA_BOUNDS[1])
    ax.grid(True,ls=":",lw=.5);ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel("signal PC2")
fig.suptitle("The n=2 signal manifold is the middle, overlapping class",
             fontsize=14,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.94));plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Validation-only threshold sweep, then apply the selected threshold to the shared test subset.
THRESHOLDS=np.arange(.30,.901,.025)
VAL_CACHE={}
for label in RUNS:
    model=EVAL_CACHE[label]["model"]
    probs=[];truth=[]
    with torch.no_grad():
        for n in (1,2,3):
            ds=VoxelDataset(ROOT/f"data/full_1to4/n{n}/val.parquet",
                            large_cfg.data,normalizer,limit=1_500)
            chunks=[]
            for i in range(0,len(ds),512):
                out=model(ds.X[i:i+512])
                out=out["pred"] if isinstance(out,dict) else out
                chunks.append(torch.sigmoid(out[:,:,3]).numpy())
            probs.append(np.concatenate(chunks));truth.extend([n]*len(ds))
    VAL_CACHE[label]={"prob":np.concatenate(probs),"truth":np.array(truth)}

fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),sharey=True)
calibration_rows=[]
for ax,label in zip(axes,RUNS):
    p,y=VAL_CACHE[label]["prob"],VAL_CACHE[label]["truth"]
    overall=[];per_n={n:[] for n in (1,2,3)}
    for th in THRESHOLDS:
        c=(p>th).sum(1);overall.append(np.mean(c==y))
        for n in (1,2,3):per_n[n].append(np.mean(c[y==n]==n))
    best_i=int(np.argmax(overall));best_th=float(THRESHOLDS[best_i])
    ax.plot(THRESHOLDS,100*np.array(overall),color=COLORS["navy"],lw=3,label="overall")
    for n,color in zip((1,2,3),[COLORS["green"],COLORS["orange"],COLORS["red"]]):
        ax.plot(THRESHOLDS,100*np.array(per_n[n]),color=color,lw=1.8,label=f"n={n}")
    ax.axvline(.5,color=COLORS["muted"],ls="--",lw=1.3,label="reported 0.5")
    ax.axvline(best_th,color=COLORS["blue"],ls=":",lw=2,label=f"validation best {best_th:.3f}")
    ax.set(title=label,xlabel="existence threshold",ylabel="count accuracy (%)")
    ax.grid(True);ax.legend(frameon=False,fontsize=8,ncol=2)

    # Freeze validation-selected threshold and evaluate only on the shared diagnostic test subset.
    test_p=np.concatenate([EVAL_CACHE[label]["by_n"][n]["prob"] for n in (1,2,3)])
    test_y=np.repeat([1,2,3],SHARED_LIMIT)
    for name,th in [("reported",.5),("validation-selected",best_th)]:
        c=(test_p>th).sum(1)
        calibration_rows.append({
            "run":label,"threshold":name,"value":th,
            "overall":np.mean(c==test_y),
            "n=1":np.mean(c[test_y==1]==1),
            "n=2":np.mean(c[test_y==2]==2),
            "n=3":np.mean(c[test_y==3]==3),
        })
fig.suptitle("Existence-threshold calibration changes the apparent n=2 failure",
             fontsize=14,fontweight="bold")
fig.tight_layout(rect=(0,0,1,.94));plt.show()

cal=pd.DataFrame(calibration_rows).set_index(["run","threshold"])
display(cal.style.format({"value":"{:.3f}","overall":"{:.1%}","n=1":"{:.1%}",
                          "n=2":"{:.1%}","n=3":"{:.1%}"}))

display(HTML('''
<div class="callout"><b>Interpretation:</b> close/weak n=2 pools do merge into one, so physical
ambiguity is real. But the dominant official-threshold error is the opposite: the model retains
three queries. The high n=3 accuracy at threshold 0.5 is therefore partly a count=3 bias, while
n=3 parameter errors remain worst. A validation-selected threshold improves n=2 and overall
count accuracy but trades away some n=3 accuracy. This points first to calibration and the
existence loss—not immediately to a new count head.</div>
'''))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Thesis-ready interpretation

<div class="callout good">
<b>Core result — the port works.</b> The 64-input T1–T2 DETR trains stably, recovers
relaxation parameters with useful accuracy on held-out synthetic data, and almost always
respects the simulated physical relation T2 &lt; T1 without a hard constraint.
</div>

<div class="callout">
<b>Main limitation — count recovery is not solved.</b> Overall count accuracy is 69.3%, but
the per-count result is essential: 85.1% for one compartment, 34.1% for two, and 88.8% for
three. The dominant error is splitting a true two-compartment voxel into three predictions.
</div>

**What the scaling comparison establishes**

- Increasing training data from 99,999 to 750,000 voxels improves every headline metric.
- Median relative error falls from 10.1% to 7.8% for T1 and from 16.3% to 11.7% for T2.
- The largest relative gain is for CSF-like T2, but it remains much worse than non-CSF T2
  because the acquisition has little sensitivity to very long decay constants.
- SNR trends are monotonic; the model degrades gracefully at the out-of-range SNR 20 probe.

**What these results do not yet establish**

- They do not demonstrate sim-to-real generalization.
- They do not show that fewer than 64 inputs are sufficient.
- They do not prove a separate count head is needed. The validation-only threshold diagnostic
  shows that a meaningful part of the n=2 deficit is calibration, while close/weak pools explain
  the smaller undercounting branch.

**Recommended next steps:** freeze a validation-selected existence threshold, rerun the full
held-out evaluation once at that frozen threshold, and then use the 750k/64-input checkpoint as
the reference for the planned 48 → 32 → 16 input-count ablation."""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """# Optional post-processing: merge highly similar query predictions

Johannes Schlund's predecessor pipeline did not stop after the existence threshold: it also
grouped highly similar query predictions into one peak. The analysis below adds that missing
post-processing step **without changing or retraining either checkpoint**.

The procedure is deliberately conservative:

1. Discard queries below an existence threshold.
2. Cluster the survivors by complete-linkage distance in `(log T1, log T2)` space.
3. Within each cluster, take an existence-confidence-weighted geometric mean of T1 and T2 and
   an existence-confidence-weighted mean of the predicted compartment weight.
4. Renormalize the merged compartment weights to sum to one.

Complete linkage requires every member of a cluster to remain close to the others, reducing the
risk of a chain of predictions accidentally joining two genuine nearby compartments. Both the
existence threshold and clustering radius are selected on validation data only, frozen, and then
applied once to the full test split."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Full-split inference for post-processing. Tuning uses the complete balanced
# validation split; the frozen settings are evaluated on every test voxel.
from scipy.cluster.hierarchy import linkage, fcluster

CLUSTER_VAL_LIMIT = None  # use the complete balanced validation split
CLUSTER_BATCH = 512

def infer_post_split(label, split, limit_per_n=None):
    model = EVAL_CACHE[label]["model"]
    by_n = {}
    with torch.no_grad():
        for n in (1, 2, 3):
            ds = VoxelDataset(
                ROOT / f"data/full_1to4/n{n}/{split}.parquet",
                large_cfg.data, normalizer, limit=limit_per_n,
            )
            chunks = []
            for i in range(0, len(ds), CLUSTER_BATCH):
                out = model(ds.X[i:i + CLUSTER_BATCH])
                out = out["pred"] if isinstance(out, dict) else out
                chunks.append(out.numpy())
            raw = np.concatenate(chunks)
            by_n[n] = {
                "raw": raw,
                "prob": 1 / (1 + np.exp(-raw[:, :, 3])),
                "t1": normalizer.denormalize_t1(raw[:, :, 0]),
                "t2": normalizer.denormalize_t2(raw[:, :, 1]),
                "true": true_compartments(ds),
            }
    return by_n

POST_CACHE = {}
for label in RUNS:
    POST_CACHE[label] = {
        "val": infer_post_split(label, "val", CLUSTER_VAL_LIMIT),
        "test": infer_post_split(label, "test", None),
    }

print("Post-processing cache:")
for label in RUNS:
    n_val = sum(len(POST_CACHE[label]["val"][n]["raw"]) for n in (1, 2, 3))
    n_test = sum(len(POST_CACHE[label]["test"][n]["raw"]) for n in (1, 2, 3))
    print(f"  {label}: {n_val:,} validation + {n_test:,} test voxels")"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Validation-only grid search. For each thresholded set, complete-linkage merge heights
# tell us the number of clusters at every candidate radius without repeatedly reclustering it.
CLUSTER_THRESHOLDS = np.arange(.35, .901, .025)
CLUSTER_RADII = np.arange(0.0, .501, .025)

def cluster_count_curve(t1_row, t2_row, prob_row, threshold):
    keep = prob_row > threshold
    k = int(keep.sum())
    if k <= 1:
        return np.full(len(CLUSTER_RADII), k, dtype=np.int8)
    points = np.column_stack([np.log(t1_row[keep]), np.log(t2_row[keep])])
    merge_heights = linkage(points, method="complete", metric="euclidean")[:, 2]
    merged = np.searchsorted(merge_heights, CLUSTER_RADII, side="right")
    return (k - merged).astype(np.int8)

def tune_clustering(by_n):
    per_n_accuracy = np.zeros(
        (3, len(CLUSTER_THRESHOLDS), len(CLUSTER_RADII)), dtype=float
    )
    for ni, n in enumerate((1, 2, 3)):
        d = by_n[n]
        for ti, threshold in enumerate(CLUSTER_THRESHOLDS):
            counts = np.empty((len(d["raw"]), len(CLUSTER_RADII)), dtype=np.int8)
            for i in range(len(counts)):
                counts[i] = cluster_count_curve(
                    d["t1"][i], d["t2"][i], d["prob"][i], threshold
                )
            per_n_accuracy[ni, ti] = np.mean(counts == n, axis=0)
    macro = per_n_accuracy.mean(axis=0)

    # First tune the threshold with clustering off. Then, holding that threshold fixed, choose
    # the best genuinely non-zero radius. Also retain the unconstrained global preference: if
    # it chooses radius zero, validation is explicitly telling us not to cluster.
    threshold_i = int(np.argmax(macro[:, 0]))
    nonzero_radius_i = 1 + int(np.argmax(macro[threshold_i, 1:]))
    preferred_i = np.unravel_index(int(np.argmax(macro)), macro.shape)
    return {
        "per_n": per_n_accuracy,
        "macro": macro,
        "threshold_only": {
            "threshold": float(CLUSTER_THRESHOLDS[threshold_i]),
            "radius": 0.0,
            "val_macro": float(macro[threshold_i, 0]),
        },
        "clustered": {
            "threshold": float(CLUSTER_THRESHOLDS[threshold_i]),
            "radius": float(CLUSTER_RADII[nonzero_radius_i]),
            "val_macro": float(macro[threshold_i, nonzero_radius_i]),
        },
        "preferred": {
            "threshold": float(CLUSTER_THRESHOLDS[preferred_i[0]]),
            "radius": float(CLUSTER_RADII[preferred_i[1]]),
            "val_macro": float(macro[preferred_i]),
        },
    }

CLUSTER_TUNING = {
    label: tune_clustering(POST_CACHE[label]["val"]) for label in RUNS
}

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True, sharey=True)
for ax, label in zip(axes, RUNS):
    tune = CLUSTER_TUNING[label]
    image = ax.pcolormesh(
        CLUSTER_RADII, CLUSTER_THRESHOLDS, 100 * tune["macro"],
        shading="nearest", cmap="viridis"
    )
    preferred = tune["preferred"]
    clustered = tune["clustered"]
    ax.scatter(
        preferred["radius"], preferred["threshold"], marker="*", s=190,
        color="white", edgecolor=COLORS["navy"], linewidth=1.2,
        label=f"validation preference: t={preferred['threshold']:.3f}, r={preferred['radius']:.3f}",
    )
    ax.scatter(
        clustered["radius"], clustered["threshold"], marker="o", s=80,
        color=COLORS["orange"], edgecolor="white", linewidth=1.0,
        label=f"best non-zero radius: {clustered['radius']:.3f}",
    )
    ax.set(
        title=label, xlabel="complete-linkage radius in log(T1,T2)",
        ylabel="existence threshold",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.colorbar(image, ax=ax, label="validation macro count accuracy (%)")
fig.suptitle(
    "Validation-only selection of existence threshold and clustering radius",
    fontsize=14, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, .94))
cluster_fig_dir = ROOT / "results" / "results_report" / "figures"
cluster_fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(cluster_fig_dir / "clustering_validation_surface.png", dpi=160, bbox_inches="tight")
plt.show()

tuning_table = pd.DataFrame([
    {
        "run": label,
        "threshold-only threshold": tune["threshold_only"]["threshold"],
        "threshold-only validation macro": tune["threshold_only"]["val_macro"],
        "preferred radius (may be zero)": tune["preferred"]["radius"],
        "preferred validation macro": tune["preferred"]["val_macro"],
        "best non-zero radius": tune["clustered"]["radius"],
        "best non-zero validation macro": tune["clustered"]["val_macro"],
    }
    for label, tune in CLUSTER_TUNING.items()
]).set_index("run")
display(tuning_table.style.format({
    "threshold-only threshold": "{:.3f}",
    "threshold-only validation macro": "{:.1%}",
    "preferred radius (may be zero)": "{:.3f}",
    "preferred validation macro": "{:.1%}",
    "best non-zero radius": "{:.3f}",
    "best non-zero validation macro": "{:.1%}",
}))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Freeze the validation choices and apply them to the full held-out test split.
def threshold_predictions(d, threshold):
    predictions = []
    for i in range(len(d["raw"])):
        keep = d["prob"][i] > threshold
        predictions.append([
            (float(t1), float(t2), float(w))
            for t1, t2, w in zip(
                d["t1"][i, keep], d["t2"][i, keep], d["raw"][i, keep, 2]
            )
        ])
    return predictions

def merge_predictions(d, threshold, radius):
    merged_predictions = []
    for i in range(len(d["raw"])):
        keep = d["prob"][i] > threshold
        indices = np.flatnonzero(keep)
        if len(indices) == 0:
            merged_predictions.append([])
            continue

        t1 = d["t1"][i, indices]
        t2 = d["t2"][i, indices]
        w = d["raw"][i, indices, 2].astype(float)
        confidence = d["prob"][i, indices].astype(float)
        if len(indices) == 1 or radius <= 0:
            labels = np.arange(len(indices))
        else:
            points = np.column_stack([np.log(t1), np.log(t2)])
            tree = linkage(points, method="complete", metric="euclidean")
            labels = fcluster(tree, t=radius, criterion="distance")

        peaks = []
        for cluster_id in np.unique(labels):
            member = labels == cluster_id
            reliability = confidence[member]
            reliability = reliability / reliability.sum()
            peaks.append([
                float(np.exp(np.sum(reliability * np.log(t1[member])))),
                float(np.exp(np.sum(reliability * np.log(t2[member])))),
                float(np.sum(reliability * w[member])),
            ])

        # Duplicate queries are alternative estimates, so average their weights rather than
        # summing them; normalize the final set exactly as the predecessor post-processing did.
        weight_sum = sum(p[2] for p in peaks)
        if weight_sum > 0:
            for peak in peaks:
                peak[2] /= weight_sum
        merged_predictions.append([tuple(p) for p in peaks])
    return merged_predictions

POST_RESULTS = {}
for label in RUNS:
    tune = CLUSTER_TUNING[label]
    POST_RESULTS[label] = {}
    for n in (1, 2, 3):
        d = POST_CACHE[label]["test"][n]
        thresholded = threshold_predictions(d, tune["threshold_only"]["threshold"])
        clustered = merge_predictions(
            d, tune["clustered"]["threshold"], tune["clustered"]["radius"]
        )
        POST_RESULTS[label][n] = {
            "true": d["true"],
            "thresholded": thresholded,
            "clustered": clustered,
            "raw_count": np.full(len(d["raw"]), d["raw"].shape[1], dtype=int),
            "thresholded_count": np.array([len(p) for p in thresholded]),
            "clustered_count": np.array([len(p) for p in clustered]),
        }

def count_stage_row(label, stage):
    per_n = {}
    all_counts, all_truth = [], []
    count_key = f"{stage}_count"
    for n in (1, 2, 3):
        counts = POST_RESULTS[label][n][count_key]
        per_n[n] = np.mean(counts == n)
        all_counts.append(counts)
        all_truth.append(np.full(len(counts), n))
    counts = np.concatenate(all_counts)
    truth = np.concatenate(all_truth)
    if stage == "raw":
        threshold, radius = np.nan, np.nan
    elif stage == "thresholded":
        threshold = CLUSTER_TUNING[label]["threshold_only"]["threshold"]
        radius = 0.0
    else:
        threshold = CLUSTER_TUNING[label]["clustered"]["threshold"]
        radius = CLUSTER_TUNING[label]["clustered"]["radius"]
    return {
        "run": label, "stage": stage,
        "threshold": threshold, "radius": radius,
        "overall": np.mean(counts == truth),
        "macro": np.mean(list(per_n.values())),
        "n=1": per_n[1], "n=2": per_n[2], "n=3": per_n[3],
        "mean predicted count": np.mean(counts),
    }

stage_rows = [
    count_stage_row(label, stage)
    for label in RUNS
    for stage in ("raw", "thresholded", "clustered")
]
STAGE_TABLE = pd.DataFrame(stage_rows).set_index(["run", "stage"])
display(STAGE_TABLE.style.format({
    "threshold": lambda x: "—" if pd.isna(x) else f"{x:.3f}",
    "radius": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
    "overall": "{:.1%}", "macro": "{:.1%}",
    "n=1": "{:.1%}", "n=2": "{:.1%}", "n=3": "{:.1%}",
    "mean predicted count": "{:.2f}",
}))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
stage_colors = {
    "raw": COLORS["muted"],
    "thresholded": COLORS["orange"],
    "clustered": COLORS["green"],
}
xlabels = ["n=1", "n=2", "n=3", "macro"]
for ax, label in zip(axes, RUNS):
    x = np.arange(len(xlabels))
    for si, stage in enumerate(("raw", "thresholded", "clustered")):
        row = STAGE_TABLE.loc[(label, stage)]
        values = 100 * np.array([row["n=1"], row["n=2"], row["n=3"], row["macro"]])
        bars = ax.bar(
            x + (si - 1) * .25, values, .24,
            color=stage_colors[stage], label=stage,
        )
        if stage != "raw":
            ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set(
        title=label, xlabel="true compartment count",
        ylabel="test count accuracy (%)", xticks=x, xticklabels=xlabels, ylim=(0, 105),
    )
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(
    "Full test split: all query slots vs thresholding vs best non-zero clustering arm",
    fontsize=14, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(cluster_fig_dir / "clustering_stage_comparison.png", dpi=160, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Does clustering rescue n=2 without merging genuinely close n=3 compartments?
impact_rows = []
for label in RUNS:
    n2 = POST_RESULTS[label][2]
    n3 = POST_RESULTS[label][3]
    t2c, c2c = n2["thresholded_count"], n2["clustered_count"]
    t3c, c3c = n3["thresholded_count"], n3["clustered_count"]
    impact_rows.append({
        "run": label,
        "n=2 rescued (% all n=2)": np.mean((t2c != 2) & (c2c == 2)),
        "n=2 damaged (% all n=2)": np.mean((t2c == 2) & (c2c != 2)),
        "n=3 rescued (% all n=3)": np.mean((t3c != 3) & (c3c == 3)),
        "n=3 damaged (% all n=3)": np.mean((t3c == 3) & (c3c != 3)),
        "n=2 net accuracy change": np.mean(c2c == 2) - np.mean(t2c == 2),
        "n=3 net accuracy change": np.mean(c3c == 3) - np.mean(t3c == 3),
    })
IMPACT_TABLE = pd.DataFrame(impact_rows).set_index("run")
display(IMPACT_TABLE.style.format("{:+.1%}"))

def minimum_true_separation(trues):
    values = []
    for true in trues:
        points = np.log(np.array([[p[0], p[1]] for p in true], dtype=float))
        distance = np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(2))
        distance[np.eye(len(points), dtype=bool)] = np.inf
        values.append(distance.min())
    return np.array(values)

sep_bins = np.array([0, .20, .35, .50, .75, 1.10, np.inf])
sep_labels = ["<.20", ".20–.35", ".35–.50", ".50–.75", ".75–1.10", ">1.10"]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
for ax, label in zip(axes, RUNS):
    r = POST_RESULTS[label][3]
    separation = minimum_true_separation(r["true"])
    bin_id = np.digitize(separation, sep_bins) - 1
    x = np.arange(len(sep_labels))
    threshold_accuracy, cluster_accuracy, sizes = [], [], []
    for b in range(len(sep_labels)):
        use = bin_id == b
        sizes.append(int(use.sum()))
        threshold_accuracy.append(np.mean(r["thresholded_count"][use] == 3) if use.any() else np.nan)
        cluster_accuracy.append(np.mean(r["clustered_count"][use] == 3) if use.any() else np.nan)
    ax.plot(
        x, 100 * np.array(threshold_accuracy), marker="o", lw=2.2,
        color=COLORS["orange"], label="thresholded",
    )
    ax.plot(
        x, 100 * np.array(cluster_accuracy), marker="D", lw=2.2,
        color=COLORS["green"], label="thresholded + clustered",
    )
    for xi, size in enumerate(sizes):
        ax.text(xi, 3, f"{size:,}", ha="center", va="bottom", fontsize=7, color=COLORS["muted"])
    ax.set(
        title=label, ylabel="n=3 count accuracy (%)" if ax is axes[0] else "",
        xticks=x, xticklabels=sep_labels, ylim=(0, 105),
    )
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(
    "Safety check: does clustering damage n=3 when two true pools are close?",
    fontsize=14, fontweight="bold",
)
fig.supxlabel("minimum true pair distance in log(T1,T2)", y=.02)
fig.tight_layout(rect=(0, .04, 1, .94))
fig.savefig(cluster_fig_dir / "clustering_n3_close_damage.png", dpi=160, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Concrete held-out examples: left is thresholding only; right is the merged result.
def pick_case(label, n, kind):
    r = POST_RESULTS[label][n]
    before, after = r["thresholded_count"], r["clustered_count"]
    if kind == "n2 rescued":
        candidates = np.flatnonzero((before != 2) & (after == 2))
    elif kind == "n3 preserved":
        candidates = np.flatnonzero((before == 3) & (after == 3))
    else:  # n3 damaged
        candidates = np.flatnonzero((before == 3) & (after != 3))
    if len(candidates) == 0:
        candidates = np.arange(len(before))
    if n == 3:
        separation = minimum_true_separation([r["true"][i] for i in candidates])
        return int(candidates[np.argmin(separation)])
    # Prefer a representative example rather than an extreme relaxation-time outlier.
    scores = np.array([
        match_score(r["thresholded"][i], r["true"][i]) for i in candidates
    ])
    finite = np.isfinite(scores)
    if finite.any():
        ordered = candidates[finite][np.argsort(scores[finite])]
        return int(ordered[len(ordered) // 2])
    return int(candidates[0])

example_rows = [
    ("100k baseline", 2, pick_case("100k baseline", 2, "n2 rescued"), "n=2 rescued"),
    ("750k large", 2, pick_case("750k large", 2, "n2 rescued"), "n=2 rescued"),
    ("750k large", 3, pick_case("750k large", 3, "n3 preserved"), "close n=3 preserved"),
]
damaged = np.flatnonzero(
    (POST_RESULTS["750k large"][3]["thresholded_count"] == 3)
    & (POST_RESULTS["750k large"][3]["clustered_count"] != 3)
)
if len(damaged):
    example_rows.append(
        ("750k large", 3, pick_case("750k large", 3, "n3 damaged"), "close n=3 damaged")
    )

fig, axes = plt.subplots(len(example_rows), 2, figsize=(11.5, 4.0 * len(example_rows)), squeeze=False)
for row_i, (label, n, index, description) in enumerate(example_rows):
    r = POST_RESULTS[label][n]
    true = r["true"][index]
    for col, (stage, color, marker) in enumerate([
        ("thresholded", COLORS["orange"], "X"),
        ("clustered", COLORS["green"], "D"),
    ]):
        ax = axes[row_i, col]
        pred = r[stage][index]
        for t1, t2, w in true:
            ax.scatter(
                t1, t2, s=45 + 230 * w, color=COLORS["blue"], alpha=.72,
                edgecolor="white", linewidth=1, zorder=3,
                label="true" if row_i == 0 and col == 0 else None,
            )
        for t1, t2, w in pred:
            ax.scatter(
                t1, t2, s=45 + 230 * w, color=color, marker=marker, alpha=.92,
                edgecolor="white", linewidth=.7, zorder=4,
                label=stage if row_i == 0 and col == 0 else None,
            )
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(45, 4500); ax.set_ylim(4.5, 3200)
        ax.grid(True, which="both", ls=":", lw=.45)
        ax.set(
            title=f"{label} · {description}\\ntrue {len(true)} · {stage} {len(pred)}",
            xlabel="T1 (ms)", ylabel="T2 (ms)",
        )
fig.suptitle(
    "Held-out examples before and after confidence-weighted peak merging",
    fontsize=15, fontweight="bold", y=.998,
)
from matplotlib.lines import Line2D
fig.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"],
               markeredgecolor="white", markersize=9, label="true"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["orange"],
               markeredgecolor="white", markersize=9, label="thresholded prediction"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["green"],
               markeredgecolor="white", markersize=9, label="clustered prediction"),
    ],
    loc="upper center", bbox_to_anchor=(.5, .972), ncol=3, frameon=False,
)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(cluster_fig_dir / "clustering_examples.png", dpi=160, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Data-driven conclusion from the frozen full-test evaluation.
large_thresholded = STAGE_TABLE.loc[("750k large", "thresholded")]
large_clustered = STAGE_TABLE.loc[("750k large", "clustered")]
large_impact = IMPACT_TABLE.loc["750k large"]
large_tune = CLUSTER_TUNING["750k large"]["clustered"]
large_preferred = CLUSTER_TUNING["750k large"]["preferred"]

display(Markdown(f\"\"\"
## Clustering conclusion

For the 750k model, unrestricted validation selection preferred threshold
**{large_preferred['threshold']:.3f}** and radius **{large_preferred['radius']:.3f}**.
To test the clustering hypothesis rather than silently skip it, the best genuinely non-zero
arm held the calibrated threshold at **{large_tune['threshold']:.3f}** and selected radius
**{large_tune['radius']:.3f}** in log(T1,T2) space. With that non-zero arm frozen:

- Macro count accuracy changes from **{large_thresholded['macro']:.1%}** after thresholding
  to **{large_clustered['macro']:.1%}** after clustering.
- True n=2 accuracy changes from **{large_thresholded['n=2']:.1%}** to
  **{large_clustered['n=2']:.1%}**.
- True n=3 accuracy changes from **{large_thresholded['n=3']:.1%}** to
  **{large_clustered['n=3']:.1%}**.
- Clustering newly rescues **{large_impact['n=2 rescued (% all n=2)']:.1%}** of all n=2
  voxels and damages **{large_impact['n=3 damaged (% all n=3)']:.1%}** of all n=3 voxels
  that thresholding had counted correctly.

The close-n=3 separation plot is the safety check: clustering is useful only if the n=2 gain
is not purchased by systematically merging genuine neighboring compartments. This remains
optional post-processing; raw-query results stay visible so it cannot conceal an
existence-score problem.
\"\"\"))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """---
<span class="small-note">Generated from saved artifacts in
<code>results/cluster_baseline</code> and <code>results/cluster_large</code>.
Re-running the notebook refreshes every table and plot from those files.</span>"""
    )
)

nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")

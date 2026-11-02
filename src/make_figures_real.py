"""Generate real-data figures for the ISBI paper.

fig1_method.pdf  : method framework (no synthetic phantom)
fig2_real.pdf    : real COVID-19 CT slice + lung/infection mask overlay
fig3_results.pdf : 4-config MAE / Spearman comparison (from results_real.json)

Run: python3 src/make_figures_real.py
"""
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/Users/a1/.openclaw/workspace/isbi-paper"
FIG = f"{ROOT}/paper/figures"
RAW = f"{ROOT}/data/raw"

# --------------------------------------------------------------------------- #
# fig1: method framework
# --------------------------------------------------------------------------- #
def draw_framework():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc="#1f6feb", tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.05,rounding_size=0.12",
                     fc=fc, ec="#0b3d91", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=12,
                     lw=1.2, color=color))

    box(3.4, 8.7, 3.2, 0.9, "CT volume\n(per axial slice)", fc="#30363d")
    box(1.7, 6.4, 6.6, 1.4,
        "Feature extraction (22-D)\n16-bin lung HU histogram + texture\n(mean/std/skew, gradient) + anatomy\n(lung-area ratio, L/R ratio)",
        fc="#1f6feb")
    arrow(5.0, 8.7, 5.0, 7.8)
    box(0.6, 3.8, 3.4, 1.2, "Ridge regression", fc="#2da44e")
    box(6.0, 3.8, 3.4, 1.2, "2-layer MLP\n(32 tanh, Adam)", fc="#2da44e")
    arrow(3.4, 6.6, 2.3, 5.0)
    arrow(6.6, 6.6, 7.7, 5.0)
    box(0.4, 1.4, 3.8, 1.3, "A: all 22 features", fc="#8957e5")
    box(4.1, 1.4, 2.0, 1.3, "B: drop 2\nanatomy", fc="#bf8700")
    box(6.4, 1.4, 3.2, 1.3, "A: all 22 features", fc="#8957e5")
    arrow(2.3, 3.8, 1.6, 2.7)
    arrow(2.3, 3.8, 4.5, 2.7)
    arrow(7.7, 3.8, 6.7, 2.7)
    arrow(7.7, 3.8, 8.0, 2.7)
    box(3.1, 0.0, 3.8, 1.1, "Slice infection-burden\nscore  y = Ninf / Nlung",
        fc="#cf222e", tc="white")
    arrow(2.3, 1.4, 5.0, 1.1)
    arrow(8.0, 1.4, 5.0, 1.1)
    ax.text(5.0, 9.85, "By-case 4-fold CV; label from segmentation masks",
            ha="center", fontsize=8, color="#666", style="italic")
    plt.tight_layout()
    fig.savefig(f"{FIG}/fig1_method.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_method.pdf")

# --------------------------------------------------------------------------- #
# fig2: real CT slice + mask overlay
# --------------------------------------------------------------------------- #
def draw_ct_overlay():
    ct_dir = f"{RAW}/ct"
    mask_dir = f"{RAW}/masks_tmp"
    cases = sorted([c[:-7] for c in __import__("os").listdir(ct_dir)
                    if c.endswith(".nii.gz")])
    target = None
    for case in cases:
        ct = nib.load(f"{ct_dir}/{case}.nii.gz").get_fdata()
        mk = nib.load(f"{mask_dir}/{case}.nii.gz").get_fdata()
        # anatomical axis last
        if ct.shape[0] in (512,):
            ct = np.transpose(ct, (2, 1, 0))
            mk = np.transpose(mk, (2, 1, 0))
        inf = (mk == 3).sum(axis=(1, 2))
        for s in range(len(inf)):
            if 400 <= inf[s] <= 6000:
                target = (case, s); break
        if target:
            break
    if target is None:
        target = (cases[0], ct.shape[0] // 2)
    case, s = target
    ct = nib.load(f"{ct_dir}/{case}.nii.gz").get_fdata()
    mk = nib.load(f"{mask_dir}/{case}.nii.gz").get_fdata()
    if ct.shape[0] in (512,):
        ct = np.transpose(ct, (2, 1, 0)); mk = np.transpose(mk, (2, 1, 0))
    sl = np.clip(ct[s], -1000, 400)
    lung = np.isin(mk[s], [1, 2])
    inf = (mk[s] == 3)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    axes[0].imshow(sl, cmap="gray", vmin=-1000, vmax=400)
    axes[0].set_title(f"CT (case {case}, slice {s})", fontsize=9)
    axes[1].imshow(sl, cmap="gray", vmin=-1000, vmax=400)
    axes[1].contour(lung.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
    axes[1].set_title("Lung contour", fontsize=9)
    axes[2].imshow(sl, cmap="gray", vmin=-1000, vmax=400)
    axes[2].imshow(np.ma.masked_where(~inf, inf), cmap="autumn", alpha=0.55)
    axes[2].set_title("Infection mask overlay", fontsize=9)
    for a in axes:
        a.axis("off")
    fig.suptitle("Real COVID-19 CT with Lung and Infection Segmentation",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{FIG}/fig2_real.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig2_real.pdf from {case} slice {s}")

# --------------------------------------------------------------------------- #
# fig3: all model families, A (all) vs B (no anatomy), with case-bootstrap CI
# --------------------------------------------------------------------------- #
def draw_results():
    j = json.load(open(f"{ROOT}/results/results_real_v2.json"))
    models = ["median", "ridge", "mlp", "rf", "hgb", "mlpsk"]
    mlbl = ["Median", "Ridge", "MLP\n(2-layer)", "Random\nForest",
            "HistGrad\nBoost", "MLP\nReg"]
    arms = ["A", "B"]
    col = {"A": "#1f6feb", "B": "#bf8700"}

    def vals(metric):
        out = {}
        for a in arms:
            pt = [j["summary"][f"{m}_{a}"][metric + "_point"] for m in models]
            ci = [j["ci"][f"{m}_{a}"][metric] for m in models]
            out[a] = (pt, ci)
        return out

    mae = vals("mae")
    spr = vals("spearman")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(models)); w = 0.36
    for i, a in enumerate(arms):
        pos = x + (i - 0.5) * w
        pt, ci = mae[a]
        lo = [ (pt[k] - ci[k][0]) if ci[k] else 0 for k in range(len(models)) ]
        hi = [ (ci[k][1] - pt[k]) if ci[k] else 0 for k in range(len(models)) ]
        a1.bar(pos, pt, width=w, yerr=[lo, hi], capsize=3, color=col[a],
               label=("A: all features" if a == "A" else "B: no anatomy"))
    a1.set_xticks(x); a1.set_xticklabels(mlbl, fontsize=7.5)
    a1.set_ylabel("MAE  (lower better)"); a1.set_title("(a) MAE  [95% case-bootstrap CI]")
    a1.legend(fontsize=7); a1.grid(axis="y", alpha=0.3)

    xs = np.arange(len(models) - 1)  # Spearman panel skips Median (undefined)
    for i, a in enumerate(arms):
        pos = xs + (i - 0.5) * w
        pt = [ (j["summary"][f"{models[k+1]}_{a}"]["spearman_point"] or 0)
               for k in range(len(models) - 1) ]
        ci = [ j["ci"][f"{models[k+1]}_{a}"]["spearman"] for k in range(len(models) - 1) ]
        lo = [ (pt[k] - ci[k][0]) if ci[k] else 0 for k in range(len(models) - 1) ]
        hi = [ (ci[k][1] - pt[k]) if ci[k] else 0 for k in range(len(models) - 1) ]
        a2.bar(pos, pt, width=w, yerr=[lo, hi], capsize=3, color=col[a])
    a2.set_xticks(xs); a2.set_xticklabels([mlbl[k+1] for k in range(len(models)-1)], fontsize=7.5)
    a2.set_ylabel("Spearman  (higher better)"); a2.set_title("(b) Spearman  [CI] (Median omitted)")
    a2.grid(axis="y", alpha=0.3)
    fig.suptitle("All baselines: arm A (all 22 features) vs B (drop anatomy)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{FIG}/fig3_results.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_results.pdf")


# --------------------------------------------------------------------------- #
# fig4: predicted vs true scatter (from data/real_preds.npz, OOF predictions)
# --------------------------------------------------------------------------- #
def draw_scatter():
    d = np.load(f"{ROOT}/data/real_preds.npz")
    y = d["y"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, (key, title) in zip(axes, [("ridge_A", "Ridge (A)"),
                                       ("mlp_B", "MLP (B)")]):
        p = d[key]
        ax.scatter(y, p, s=6, alpha=0.35, color="#1f6feb", edgecolors="none")
        lim = max(y.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("true burden"); ax.set_ylabel("predicted")
        ax.set_title(title, fontsize=9)
        r = float(np.corrcoef(y, p)[0, 1])
        ax.text(0.05, 0.92, f"Pearson r={r:.2f}", transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Out-of-fold predicted vs. true slice infection burden",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{FIG}/fig4_real_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_real_scatter.pdf")


# --------------------------------------------------------------------------- #
# fig5: A vs B delta Spearman per model (the effect is model-dependent)
# --------------------------------------------------------------------------- #
def draw_deltas():
    j = json.load(open(f"{ROOT}/results/results_real_v2.json"))
    models = ["ridge", "mlp", "rf", "hgb", "mlpsk"]
    lbl = ["Ridge", "MLP\n(2-layer)", "Random\nForest", "HistGrad\nBoost",
           "MLP\nReg"]
    delta = []
    for m in models:
        a = j["summary"][f"{m}_A"]["spearman_mean_fold"]
        b = j["summary"][f"{m}_B"]["spearman_mean_fold"]
        delta.append(a - b)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#2da44e" if v >= 0 else "#cf222e" for v in delta]
    x = np.arange(len(models))
    ax.bar(x, delta, color=colors, width=0.6)
    ax.axhline(0, color="k", lw=1)
    for i, v in enumerate(delta):
        ax.text(i, v + (0.006 if v >= 0 else -0.013), f"{v:+.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel(r"$\Delta\rho = \rho_A - \rho_B$")
    ax.set_title("Anatomy effect is model-dependent (all 95% CIs include 0)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIG}/fig5_delta.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_delta.pdf")


if __name__ == "__main__":
    draw_framework()
    draw_ct_overlay()
    draw_results()
    draw_scatter()
    draw_deltas()

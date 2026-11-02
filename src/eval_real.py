#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot real-data predictions -> paper/figures/fig4_real_scatter.pdf

Reads results/real_preds.npz (produced by src/train_real.py):
  y_true, ridge_A, ridge_B, mlp_A, mlp_B  (one entry per test slice, pooled folds)

Two panels:
  (a) predicted vs true slice-infection fraction (ridge_A vs ridge_B, and mlp_A
      vs mlp_B) -- demonstrates the A/B (with/without anatomy) comparison.
  (b) error distribution (pred - true) for A and B, per model.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS = os.path.join(ROOT, "results", "real_preds.npz")
FIG = os.path.join(ROOT, "paper", "figures", "fig4_real_scatter.pdf")

MODELS = [("ridge", "Ridge"), ("mlp", "MLP")]


def main():
    d = np.load(PREDS, allow_pickle=True)
    y = np.asarray(d["y_true"], dtype=np.float64)
    keys = [k for k in d.files if k != "y_true"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- panel (a): pred vs true ----
    ax = axes[0]
    lo, hi = 0.0, float(np.quantile(y, 0.99)) if y.max() > 0 else 1.0
    for m, label in MODELS:
        for f, fc, mk in (("A", "C0", "o"), ("B", "C1", "s")):
            key = f"{m}_{f}"
            if key not in d:
                continue
            pr = np.asarray(d[key], dtype=np.float64)
            ax.scatter(y, pr, s=9, alpha=0.35, color=fc,
                       marker=mk, label=f"{label} {f}")
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("true infection fraction")
    ax.set_ylabel("predicted infection fraction")
    ax.set_title("(a) predicted vs true")
    ax.legend(fontsize=7, loc="upper left")

    # ---- panel (b): error distribution ----
    ax = axes[1]
    for m, label in MODELS:
        for f, fc in (("A", "C0"), ("B", "C1")):
            key = f"{m}_{f}"
            if key not in d:
                continue
            err = np.asarray(d[key], dtype=np.float64) - y
            ax.hist(err, bins=40, alpha=0.45, color=fc,
                    label=f"{label} {f}", density=True)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("prediction error (pred - true)")
    ax.set_ylabel("density")
    ax.set_title("(b) error distribution")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG)
    print(f"written: {FIG}")
    # also print a quick numeric summary for the log
    for m, _ in MODELS:
        for f in ("A", "B"):
            key = f"{m}_{f}"
            if key in d:
                err = np.asarray(d[key], dtype=np.float64) - y
                print(f"{key:10s} MAE={np.mean(np.abs(err)):.4f} "
                      f"bias={err.mean():+.4f}")


if __name__ == "__main__":
    main()

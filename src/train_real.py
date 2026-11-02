#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-validated real-data experiment -> results/results_real.json + real_preds.npz

Design (no torch):
  * 4-fold cross-validation SPLIT BY CASE (never by slice -> avoids leakage).
  * Two feature sets:
        A  = all features
        B  = A minus anatomy features -> the "decoupled" control
  * Two models, each implemented in plain numpy:
        - ridge regression (closed form, L2 penalty)
        - a small 2-layer MLP (tanh hidden, manual Adam backprop)

Outputs:
  results/results_real.json : per-fold MAE / Spearman + aggregated mean +/- std
  results/real_preds.npz    : y_true and predictions for every model/feature set
                              (consumed by src/eval_real.py)
"""
import os
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real_dataset.npz")
OUT_JSON = os.path.join(ROOT, "results", "results_real.json")
OUT_PREDS = os.path.join(ROOT, "results", "real_preds.npz")

K_FOLDS = 4
RIDGE_ALPHA = 1.0
MLP_HIDDEN = 32
MLP_EPOCHS = 300
MLP_LR = 0.01
SEED = 0

ANATOMY_IDX = [20, 21]  # must match src/data_real.py


# --------------------------------------------------------------------------- #
def standardize(tr, te):
    mu = tr.mean(0)
    sd = tr.std(0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    XtX = X.T @ X
    XtX += alpha * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)
    return w


def ridge_pred(w, X):
    return X @ w


def _adam(params, grads, m, v, t, lr=MLP_LR, b1=0.9, b2=0.999, eps=1e-8):
    out_p, out_m, out_v = [], [], []
    for p, g, mm, vv in zip(params, grads, m, v):
        mm = b1 * mm + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * (g ** 2)
        mh = mm / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        p = p - lr * mh / (np.sqrt(vh) + eps)
        out_p.append(p); out_m.append(mm); out_v.append(vv)
    return out_p, out_m, out_v


def mlp_fit(X, y, hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, lr=MLP_LR, seed=SEED):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, 0.1, (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, (hidden, 1))
    b2 = np.zeros(1)
    ymu, ysd = y.mean(), y.std() + 1e-8
    yt = (y - ymu) / ysd
    m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    n = X.shape[0]
    for t in range(1, epochs + 1):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        # keep err as a column vector: a 1-D err would broadcast gW2 from
        # (hidden,1) to (hidden,hidden) and silently corrupt W2.
        err = z2 - yt.reshape(-1, 1)
        gW2 = (a1.T @ err) / n
        gb2 = err.mean(axis=0)
        da1 = (err @ W2.T) * (1 - a1 ** 2)
        gW1 = (X.T @ da1) / n
        gb1 = da1.sum(0)
        (W1, b1, W2, b2), m, v = _adam(
            [W1, b1, W2, b2], [gW1, gb1, gW2, gb2], m, v, t, lr)
    return (W1, b1, W2, b2, ymu, ysd)


def mlp_pred(params, X):
    W1, b1, W2, b2, ymu, ysd = params
    a1 = np.tanh(X @ W1 + b1)
    return (a1 @ W2 + b2).ravel() * ysd + ymu


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def spear(pred, true):
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return 0.0
    r = spearmanr(pred, true).correlation
    return 0.0 if np.isnan(r) else float(r)


def run_model(name, fit, pred, Xtr, ytr, Xte, yte):
    p = fit(Xtr, ytr)
    pr = pred(p, Xte)
    return {"mae": mae(pr, yte), "spearman": spear(pr, yte)}, pr


def main():
    d = np.load(DATA, allow_pickle=True)
    X, y = d["X"], d["y"]
    case_ids = d["case_id"]
    uniq = np.array(sorted(set(case_ids.tolist())), dtype=object)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(uniq))
    folds_cases = np.array_split(uniq[idx], K_FOLDS)

    Xb = np.delete(X, ANATOMY_IDX, axis=1)

    model_specs = [
        ("ridge", ridge_fit, ridge_pred),
        ("mlp", mlp_fit, mlp_pred),
    ]
    feat_specs = [("A", X), ("B", Xb)]

    fold_records = []
    preds_store = {"y_true": None}
    for k in range(K_FOLDS):
        test_cases = set(folds_cases[k].tolist())
        tr = np.array([c not in test_cases for c in case_ids.tolist()])
        te = ~tr
        Xtr_a, Xte_a = standardize(X[tr], X[te])
        Xtr_b, Xte_b = standardize(Xb[tr], Xb[te])
        ytr, yte = y[tr], y[te]
        rec = {"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum())}
        if preds_store["y_true"] is None:
            preds_store["y_true"] = yte
        else:
            preds_store["y_true"] = np.concatenate([preds_store["y_true"], yte])
        for mname, fit, pred in model_specs:
            for fname, Xtr, Xte in (("A", Xtr_a, Xte_a), ("B", Xtr_b, Xte_b)):
                key = f"{mname}_{fname}"
                metrics, pr = run_model(key, fit, pred, Xtr, ytr, Xte, yte)
                rec[key] = metrics
                if key in preds_store:
                    preds_store[key] = np.concatenate([preds_store[key], pr])
                else:
                    preds_store[key] = pr
        fold_records.append(rec)
        print(f"fold {k}: " + "  ".join(
            f"{mname}_{fname} MAE={rec[mname+'_'+fname]['mae']:.4f}"
            for mname, _, _ in model_specs for fname in ('A', 'B')))

    summary = {}
    for mname, _, _ in model_specs:
        for fname in ('A', 'B'):
            key = f"{mname}_{fname}"
            maes = np.array([f[key]["mae"] for f in fold_records])
            sps = np.array([f[key]["spearman"] for f in fold_records])
            summary[key] = {
                "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
                "spearman_mean": float(sps.mean()),
                "spearman_std": float(sps.std()),
            }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {"k_folds": K_FOLDS, "n_cases": len(uniq),
           "anatomy_idx_dropped": ANATOMY_IDX,
           "folds": fold_records, "summary": summary}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    np.savez(OUT_PREDS, **preds_store)

    print("\n=== summary (mean +/- std across folds) ===")
    for key, s in summary.items():
        print(f"{key:10s}  MAE={s['mae_mean']:.4f}+/-{s['mae_std']:.4f}"
              f"  Spearman={s['spearman_mean']:.4f}+/-{s['spearman_std']:.4f}")
    print(f"written: {OUT_JSON}")
    print(f"written: {OUT_PREDS}")


if __name__ == "__main__":
    main()

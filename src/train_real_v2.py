#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-data benchmark v2 -> results/results_real_v2.json + data/features.csv
+ data/splits.json

Extends src/train_real.py with:
  * Three scikit-learn baselines (RandomForest, HistGradientBoosting,
    MLPRegressor) and a trivial Median floor, run *alongside* the existing
    numpy ridge and 2-layer MLP.
  * The same A (all features) / B (drop anatomy) ablation for every model.
  * 4-fold CV split BY CASE (no leakage); the case partition is fixed by a
    seed and dumped to data/splits.json so the benchmark is reproducible.
  * Case-level bootstrap (1000 resamples of *cases*, not slices) producing
    95% CIs for MAE and Spearman -- the uncertainty that matters, because the
    2240 slices are clustered within 20 cases.

Outputs:
  results/results_real_v2.json : per-fold metrics, point estimates, CI, configs
  data/features.csv            : one row per slice (case_id, slice_idx, 22 feats, y)
  data/splits.json             : fixed 4-fold case partition (seed=0)
"""
import os
import json
import csv

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real_dataset.npz")
OUT_JSON = os.path.join(ROOT, "results", "results_real_v2.json")
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
SPLITS_JSON = os.path.join(ROOT, "data", "splits.json")

K_FOLDS = 4
SEED = 0
N_BOOT = 1000
ANATOMY_IDX = [20, 21]  # dropped for arm B; must match src/data_real.py

RIDGE_ALPHA = 1.0
MLP_HIDDEN = 32
MLP_EPOCHS = 300
MLP_LR = 0.01


# --------------------------------------------------------------------------- #
# numpy ridge / 2-layer MLP (kept from train_real.py for continuity)
# --------------------------------------------------------------------------- #
def standardize(tr, te):
    mu = tr.mean(0)
    sd = tr.std(0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    XtX = X.T @ X
    XtX += alpha * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def ridge_pred(w, X):
    return X @ w


def _adam(params, grads, m, v, t, lr=MLP_LR, b1=0.9, b2=0.999, eps=1e-8):
    new_p, new_m, new_v = [], [], []
    for p, g, mm, vv in zip(params, grads, m, v):
        mm = b1 * mm + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * (g ** 2)
        mh = mm / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        new_p.append(p - lr * mh / (np.sqrt(vh) + eps))
        new_m.append(mm)
        new_v.append(vv)
    return new_p, new_m, new_v


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
        a1 = np.tanh(X @ W1 + b1)
        z2 = a1 @ W2 + b2
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
    return (np.tanh(X @ W1 + b1) @ W2 + b2).ravel() * ysd + ymu


# --------------------------------------------------------------------------- #
# trivial Median floor + sklearn baselines
# --------------------------------------------------------------------------- #
def median_fit(X, y):
    return float(np.median(y))


def median_pred(m, X):
    return np.full(X.shape[0], m)


def _sk_fit(make, X, y):
    est = make()
    est.fit(X, y)
    return est


def _sk_pred(est, X):
    return est.predict(X)


def make_rf():
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)


def make_hgb():
    return HistGradientBoostingRegressor(random_state=SEED)


def make_mlpsk():
    return MLPRegressor(hidden_layer_sizes=(32,), max_iter=500,
                        random_state=SEED)


# model registry: name -> (needs_feature_scaling, fit, pred)
MODELS = {
    "median": (False, median_fit, median_pred),
    "ridge":  (True,  ridge_fit, ridge_pred),
    "mlp":    (True,  mlp_fit, mlp_pred),
    "rf":     (False, lambda X, y: _sk_fit(make_rf, X, y), _sk_pred),
    "hgb":    (False, lambda X, y: _sk_fit(make_hgb, X, y), _sk_pred),
    "mlpsk":  (True,  lambda X, y: _sk_fit(make_mlpsk, X, y), _sk_pred),
}


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def spear(pred, true):
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return 0.0
    r = spearmanr(pred, true).correlation
    return 0.0 if np.isnan(r) else float(r)


def case_bootstrap_ci(y_true, y_pred, case_ids, n_boot=N_BOOT, seed=SEED):
    """Case-level bootstrap: resample 20 cases with replacement, pool their
    slices, recompute MAE / Spearman. Returns 95% percentile CI."""
    rng = np.random.default_rng(seed)
    cases = np.unique(case_ids)
    n = len(cases)
    idx_by_case = {c: np.where(case_ids == c)[0] for c in cases}
    maes = np.empty(n_boot)
    sprs = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(cases, size=n, replace=True)
        ii = np.concatenate([idx_by_case[c] for c in samp])
        yt, yp = y_true[ii], y_pred[ii]
        maes[b] = np.mean(np.abs(yp - yt))
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            r = spearmanr(yp, yt).correlation
            sprs[b] = 0.0 if np.isnan(r) else r
        else:
            sprs[b] = np.nan
    return {
        "mae": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "spearman": [float(np.nanpercentile(sprs, 2.5)),
                     float(np.nanpercentile(sprs, 97.5))],
    }


def main():
    d = np.load(DATA, allow_pickle=True)
    # keep float32 to exactly reproduce the v1 (train_real.py) numbers
    X, y = np.asarray(d["X"], dtype=np.float32), np.asarray(d["y"], dtype=np.float32)
    case_ids = np.asarray(d["case_id"])
    slice_idx = np.asarray(d["slice_idx"])
    n_feat = X.shape[1]
    assert n_feat == 22, n_feat
    uniq = np.array(sorted(set(case_ids.tolist())), dtype=object)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(uniq))
    folds_cases = [c.tolist() for c in np.array_split(uniq[idx], K_FOLDS)]

    Xb = np.delete(X, ANATOMY_IDX, axis=1)

    # all (model, featset) configs
    configs = []
    for mname in MODELS:
        for fs in ("A", "B"):
            configs.append((mname, fs))

    fold_records = []
    oof = {f"{m}_{fs}": np.empty(X.shape[0]) for (m, fs) in configs}
    y_true_oof = np.empty(X.shape[0])
    assigned = np.zeros(X.shape[0], dtype=bool)

    for k in range(K_FOLDS):
        test_cases = set(folds_cases[k])
        tr = np.array([c not in test_cases for c in case_ids.tolist()])
        te = ~tr
        ytr, yte = y[tr], y[te]
        y_true_oof[te] = yte
        assigned[te] = True

        # per-fold standardized views (train stats only -> no leakage)
        Xtr_a, Xte_a = standardize(X[tr], X[te])
        Xtr_b, Xte_b = standardize(Xb[tr], Xb[te])
        raw = {("A", True): (X[tr], X[te]), ("B", True): (Xb[tr], Xb[te]),
               ("A", False): (X[tr], X[te]), ("B", False): (Xb[tr], Xb[te])}

        rec = {"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum())}
        for mname, fs in configs:
            needs_scale, fit, pred = MODELS[mname]
            if needs_scale:
                Xtr, Xte = (Xtr_a, Xte_a) if fs == "A" else (Xtr_b, Xte_b)
            else:
                Xtr, Xte = raw[(fs, False)]
            p = fit(Xtr, ytr)
            pr = pred(p, Xte)
            key = f"{mname}_{fs}"
            rec[key] = {"mae": mae(pr, yte), "spearman": spear(pr, yte)}
            oof[key][te] = pr
        fold_records.append(rec)
        print(f"fold {k}: n_test={int(te.sum())}  " + "  ".join(
            f"{mname}_{fs}={rec[mname+'_'+fs]['mae']:.4f}"
            for mname, fs in configs))

    assert assigned.all(), "some slices not assigned to a test fold"

    # point estimates (over all OOF) + case-level bootstrap CIs
    summary, ci = {}, {}
    for mname, fs in configs:
        key = f"{mname}_{fs}"
        maes = np.array([f[key]["mae"] for f in fold_records])
        sps = np.array([f[key]["spearman"] for f in fold_records])
        yt = y_true_oof
        yp = oof[key]
        summary[key] = {
            "mae_point": mae(yp, yt),
            "mae_mean_fold": float(maes.mean()), "mae_std_fold": float(maes.std()),
            "spearman_point": spear(yp, yt),
            "spearman_mean_fold": float(sps.mean()),
            "spearman_std_fold": float(sps.std()),
        }
        if mname == "median":
            # constant predictor: no ranking ability -> Spearman undefined
            summary[key]["spearman_point"] = None
            ci[key] = {"mae": case_bootstrap_ci(yt, yp, case_ids)["mae"],
                       "spearman": None}
        else:
            ci[key] = case_bootstrap_ci(yt, yp, case_ids)

    # ------------------------------------------------------------------ #
    # exports: features.csv, splits.json
    # ------------------------------------------------------------------ #
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "slice_idx"] + [f"f{i}" for i in range(n_feat)] + ["y"])
        for i in range(X.shape[0]):
            w.writerow([case_ids[i], int(slice_idx[i])] +
                       [f"{v:.6g}" for v in X[i]] + [f"{y[i]:.6g}"])

    splits = {"seed": SEED, "k_folds": K_FOLDS,
              "folds": [{"fold": k, "test_cases": folds_cases[k]}
                        for k in range(K_FOLDS)]}
    with open(SPLITS_JSON, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2, ensure_ascii=False)

    # OOF predictions for the scatter figure (fig4)
    REAL_PREDS = os.path.join(ROOT, "data", "real_preds.npz")
    np.savez(REAL_PREDS, y=y_true_oof, ridge_A=oof["ridge_A"],
             mlp_B=oof["mlp_B"], median=oof["median_A"])

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {"k_folds": K_FOLDS, "n_cases": len(uniq), "n_slices": int(X.shape[0]),
           "anatomy_idx_dropped": ANATOMY_IDX, "n_boot": N_BOOT,
           "configs": [{"model": m, "features": fs} for (m, fs) in configs],
           "folds": fold_records, "summary": summary, "ci": ci}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\n=== point estimate [95% case-bootstrap CI] ===")
    for mname, fs in configs:
        key = f"{mname}_{fs}"
        s, c = summary[key], ci[key]
        sp = (f"{s['spearman_point']:.3f} "
              f"[{c['spearman'][0]:.3f}, {c['spearman'][1]:.3f}]"
              if s['spearman_point'] is not None else "— (constant predictor)")
        print(f"{key:10s} MAE      = {s['mae_point']:.4f} "
              f"[{c['mae'][0]:.4f}, {c['mae'][1]:.4f}]")
        print(f"{'':10s} Spearman = {sp}")
    print(f"written: {OUT_JSON}")
    print(f"written: {FEATURES_CSV}")
    print(f"written: {SPLITS_JSON}")


if __name__ == "__main__":
    main()

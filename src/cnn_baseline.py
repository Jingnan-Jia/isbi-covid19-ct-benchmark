#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-based learned baseline (pure numpy, no torch).

Two implementations behind one switch:
  * CNN  : conv(8x3x3, stride 2) -> ReLU -> maxpool 2 -> flatten -> linear,
           trained with hand-written batch-gradient-descent backprop.
  * RCF  : if the trained CNN fails to learn (collapses to the training mean,
           OOF Spearman<=0 or MAE>=median floor) or exceeds the time budget,
           we downgrade to Random Conv Features + Ridge (fixed random conv,
           closed-form ridge). This downgrade is reported honestly.

Both run the SAME fixed by-case 4-fold partition as train_real_v2.py (read from
data/splits.json) so the OOF predictions are directly comparable, and feed the
same case-level bootstrap CI. Results are merged into
results/results_real_v2.json under the key "cnn" or "rcf".
"""
import os
import sys
import time
import json
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data", "real_dataset.npz")
SPLITS = os.path.join(ROOT, "data", "splits.json")
IMG_CACHE = os.path.join(ROOT, "data", "real_images.npz")
OUT_JSON = os.path.join(ROOT, "results", "results_real_v2.json")

EPOCHS = 250
LR = 1e-3
SEED = 0
FOLD_CAP = 90.0          # seconds per fold: time-based downgrade trigger
N_BOOT = 1000


# --------------------------------------------------------------------------- #
def build_images():
    """32x32 lung-cropped, HU-normalised single-channel slices (pure numpy)."""
    if os.path.exists(IMG_CACHE):
        z = np.load(IMG_CACHE, allow_pickle=True)
        return z["img"], z["cid"], z["sidx"]
    import data_real
    ct_cases = data_real._find_ct_cases(data_real.RAW)
    cases = sorted(ct_cases.keys())
    recs = []
    for c in cases:
        d = data_real.load_case(c)
        ct, lung = d["ct"], d["lung_mask"]
        for z_ in range(ct.shape[-1]):
            lu = lung[..., z_]
            if lu.mean() < data_real.MIN_LUNG_AREA_RATIO:
                continue
            sl = ct[..., z_].astype(np.float32)
            rs, cs = np.where(lu)
            r0, r1 = rs.min(), rs.max() + 1
            c0, c1 = cs.min(), cs.max() + 1
            crop = sl[r0:r1, c0:c1]
            h, w = crop.shape
            im = zoom(crop, (32.0 / h, 32.0 / w), order=1)
            im = np.clip(im, data_real.HU_MIN, data_real.HU_MAX)
            im = (im - data_real.HU_MIN) / (data_real.HU_MAX - data_real.HU_MIN)
            recs.append((c, int(z_), im.astype(np.float32)))
    cid = np.array([r[0] for r in recs], dtype=object)
    sidx = np.array([r[1] for r in recs], dtype=np.int32)
    img = np.stack([r[2] for r in recs])
    np.savez(IMG_CACHE, img=img, cid=cid, sidx=sidx)
    print(f"built images: {img.shape}")
    return img, cid, sidx


def _im2col(x, kh, kw, sh, sw):
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    win = win[:, :, ::sh, ::sw, :, :]
    return win.reshape(x.shape[0], -1, kh * kw)


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def spear(pred, true):
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return 0.0
    r = spearmanr(pred, true).correlation
    return 0.0 if np.isnan(r) else float(r)


# --------------------------------------------------------------------------- #
# CNN (trained conv)
# --------------------------------------------------------------------------- #
def cnn_forward(x, W1, b1, W2, b2):
    pat = _im2col(x, 3, 3, 2, 2)              # (N,225,9)
    conv = pat @ W1 + b1                      # (N,225,8)
    conv = conv.reshape(x.shape[0], 8, 15, 15)
    a = np.maximum(conv, 0.0)
    mp = np.lib.stride_tricks.sliding_window_view(a, (2, 2), axis=(2, 3))
    mp = mp[:, :, ::2, ::2, :, :]             # (N,8,7,7,2,2)
    mp4 = mp.reshape(x.shape[0], 8, 7, 7, 4)  # collapse 2x2 into one axis
    pooled = mp4.max(axis=4)                  # (N,8,7,7)
    amax = mp4.argmax(axis=4)                 # (N,8,7,7)
    flat = pooled.reshape(x.shape[0], 392)
    out = flat @ W2 + b2
    return conv, a, pooled, amax, flat, out


def cnn_train(x, y, epochs=EPOCHS, lr=LR, seed=SEED):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, (9, 8))
    b1 = np.zeros(8)
    W2 = rng.normal(0, 0.1, (392, 1))
    b2 = np.zeros(1)
    n = x.shape[0]
    yc = y.reshape(-1, 1)
    for _ in range(epochs):
        conv, a, pooled, amax, flat, out = cnn_forward(x, W1, b1, W2, b2)
        dout = (out - yc) / n
        dW2 = flat.T @ dout
        db2 = dout.sum(0)
        dpooled = (dout @ W2.T).reshape(x.shape[0], 8, 7, 7)
        dmp = np.zeros((x.shape[0], 8, 15, 15))
        for di in range(2):
            for dj in range(2):
                idx = di * 2 + dj
                sel = (amax == idx)
                r = (np.arange(7)[:, None]) * 2 + di
                c = (np.arange(7)[None, :]) * 2 + dj
                dmp[:, :, r, c][sel] = dpooled[sel]
        da = dmp * (a > 0).astype(np.float32)
        pat = _im2col(x, 3, 3, 2, 2)
        dconv = da.reshape(x.shape[0], 225, 8)
        dW1 = np.einsum("nkj,nkf->jf", pat, dconv)
        db1 = dconv.sum(axis=(0, 1))
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return W1, b1, W2, b2


def cnn_predict(x, params):
    W1, b1, W2, b2 = params
    _, _, _, _, _, out = cnn_forward(x, W1, b1, W2, b2)
    return out.ravel()


# --------------------------------------------------------------------------- #
# RCF (random conv features + ridge) -- downgrade path
# --------------------------------------------------------------------------- #
def rcf_features(x, W1, b1):
    pat = _im2col(x, 3, 3, 2, 2)
    conv = (pat @ W1 + b1).reshape(x.shape[0], 8, 15, 15)
    a = np.maximum(conv, 0.0)
    mp = np.lib.stride_tricks.sliding_window_view(a, (2, 2), axis=(2, 3))
    mp = mp[:, :, ::2, ::2, :, :]
    mp4 = mp.reshape(x.shape[0], 8, 7, 7, 4)
    pooled = mp4.max(axis=4)
    return pooled.reshape(x.shape[0], 392)


def ridge_fit(X, y, alpha=1.0):
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def ridge_pred(w, X):
    return X @ w


def case_bootstrap_ci(y_true, y_pred, case_ids, n_boot=N_BOOT, seed=SEED):
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
    return {"mae": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
            "spearman": [float(np.nanpercentile(sprs, 2.5)),
                         float(np.nanpercentile(sprs, 97.5))]}


def main():
    d = np.load(DATA, allow_pickle=True)
    y = np.asarray(d["y"], dtype=np.float32)
    cid0 = np.asarray(d["case_id"]); sidx0 = np.asarray(d["slice_idx"])
    img, cid, sidx = build_images()
    key2 = {(cid[i], int(sidx[i])): img[i] for i in range(len(cid))}
    img_arr = np.stack([key2[(cid0[i], int(sidx0[i]))] for i in range(len(cid0))])
    img_arr = img_arr[:, None, :, :]           # (N,1,32,32)

    splits = json.load(open(SPLITS))
    folds_cases = [f["test_cases"] for f in splits["folds"]]
    K = len(folds_cases)

    def run_rcf():
        oof = np.empty(len(y)); assigned = np.zeros(len(y), dtype=bool)
        rec = [{} for _ in range(K)]
        rng = np.random.default_rng(SEED)
        W1r = rng.normal(0, 1.0, (9, 8)); b1r = np.zeros(8)

        def _std(tr, te):
            mu = tr.mean(0); sd = tr.std(0); sd[sd == 0] = 1.0
            return (tr - mu) / sd, (te - mu) / sd

        for k in range(K):
            te = np.array([c in set(folds_cases[k]) for c in cid0.tolist()])
            Ftr = rcf_features(img_arr[~te], W1r, b1r)
            Fte = rcf_features(img_arr[te], W1r, b1r)
            Ftr, Fte = _std(Ftr, Fte)
            w = ridge_fit(Ftr, y[~te])
            pr = ridge_pred(w, Fte)
            oof[te] = pr; assigned[te] = True
            rec[k] = {"rcf": {"mae": mae(pr, y[te]), "spearman": spear(pr, y[te])}}
        assert assigned.all()
        return oof, rec

    # ---- attempt trained CNN ----
    mode, key, downgrade = "cnn", "cnn", None
    t_start = time.time()
    oof_cnn = np.empty(len(y)); assigned = np.zeros(len(y), dtype=bool)
    fold_cnn = [{} for _ in range(K)]
    time_blown = False
    for k in range(K):
        te = np.array([c in set(folds_cases[k]) for c in cid0.tolist()])
        params = cnn_train(img_arr[~te], y[~te])
        pr = cnn_predict(img_arr[te], params)
        oof_cnn[te] = pr; assigned[te] = True
        fold_cnn[k] = {"cnn": {"mae": mae(pr, y[te]), "spearman": spear(pr, y[te])}}
        if (time.time() - t_start) > FOLD_CAP * K:
            time_blown = True
            break
    assert assigned.all()

    s_cnn = {"mae_point": mae(oof_cnn, y), "spearman_point": spear(oof_cnn, y)}
    floor_mae = json.load(open(OUT_JSON))["summary"]["median_A"]["mae_point"]
    if time_blown:
        downgrade = "time (>8 min budget)"
    elif s_cnn["spearman_point"] <= 0 or s_cnn["mae_point"] >= floor_mae:
        downgrade = ("CNN underfit (collapsed to the training mean; OOF "
                     "Spearman<=0 / MAE>=median floor) on 20 cases")

    if downgrade:
        mode, key = "rcf", "rcf"
        oof, fold_records_extra = run_rcf()
        summary = {key: {"mae_point": mae(oof, y), "spearman_point": spear(oof, y)}}
        ci = {key: case_bootstrap_ci(y, oof, cid0)}
        print(f"DOWNGRADE to RCF ({downgrade})")
    else:
        oof, fold_records_extra = oof_cnn, fold_cnn
        summary = {"cnn": s_cnn}
        ci = {"cnn": case_bootstrap_ci(y, oof_cnn, cid0)}

    # merge into existing v2 json (clean any prior cnn/rcf entries first)
    j = json.load(open(OUT_JSON))
    for kk in ("cnn", "rcf"):
        j["summary"].pop(kk, None); j["ci"].pop(kk, None)
        for f in j["folds"]:
            f.pop(kk, None)
        j["configs"] = [c for c in j.get("configs", [])
                        if c.get("model") not in ("cnn", "rcf")]
    for k in range(K):
        j["folds"][k][key] = fold_records_extra[k][key]
    j["configs"].append({"model": key, "features": "img"})
    j["summary"][key] = summary[key]
    j["ci"][key] = ci[key]
    j["cnn_mode"] = mode
    j["cnn_downgrade"] = downgrade
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(j, fh, indent=2, ensure_ascii=False)

    s, c = summary[key], ci[key]
    print(f"\nmode={mode}  {key}  MAE={s['mae_point']:.4f} [{c['mae'][0]:.4f}, "
          f"{c['mae'][1]:.4f}]  Spearman={s['spearman_point']:.3f} "
          f"[{c['spearman'][0]:.3f}, {c['spearman'][1]:.3f}]")
    print(f"merged into {OUT_JSON} (key={key})")


if __name__ == "__main__":
    main()

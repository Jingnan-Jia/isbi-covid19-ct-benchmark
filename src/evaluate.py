#!/usr/bin/env python3
"""评测（第 2 轮修订版）

- 跨种子：每个变体在多个随机初始化种子上评测，报 mean ± std（跨 seed）
- 显著性：ours vs b1、ours vs b2 用 **Wilcoxon 符号秩检验**（按测试病例配对，n=160）
- 产物：
    results/results.json        全量结果（论文数字来源）
    results/results_seeds.json  跨种子汇总 + p 值（本次要求的交付物）

红线：所有数字来自真实运行，不写死。
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from model import D_DENSITY, D_LESION, Model, apply_scaler, build_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
VARIANTS = ["ours", "b1", "b1b", "b4", "b2", "leaky"]


def variant_input(base, ds):
    X, Xa = ds["X"], ds["Xa"]
    if base == "b1":
        return np.concatenate([X, Xa], axis=-1)
    if base == "b2":
        return X[:, :, :D_DENSITY]
    return X[:, :, :D_LESION]


def load_models(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    seeds = [int(s) for s in meta.get("seeds", [0])]
    models = {}
    for group in ("variants", "sweep"):
        for name, rec in meta.get(group, {}).items():
            base = rec.get("base", name)
            per_seed = []
            for s in seeds:
                m = Model(variant=base, hidden=int(rec["hidden"]))
                for k in m.p:
                    m.p[k] = z[f"{name}::s{s}::{k}"]
                m.scaler = {"mu": z[f"{name}::mu"], "sd": z[f"{name}::sd"]}
                m.ana_scaler = {"mu": z["ana_mu"], "sd": z["ana_sd"]}
                per_seed.append(m)
            models[name] = (base, per_seed)
    prior = z["lobar_prior"] if "lobar_prior" in z else None
    return models, meta, prior, seeds


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def ridge_probe_r2(H, a, lam=1e-2, folds=5):
    n = H.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    r2s = []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.array([i for i in idx if i not in set(te.tolist())])
        if len(te) < 2 or len(tr) < 2:
            continue
        Xb = np.hstack([H[tr], np.ones((len(tr), 1))])
        w = np.linalg.solve(Xb.T @ Xb + np.eye(Xb.shape[1]) * lam, Xb.T @ a[tr])
        Hte = np.hstack([H[te], np.ones((len(te), 1))])
        r2s.append(_r2(a[te], Hte @ w))
    return float(np.mean(r2s)), float(np.std(r2s))


def metrics_for_model(m, ds, Xin):
    Xs = apply_scaler(Xin, m.scaler)
    Xa_s = apply_scaler(ds["Xa"], m.ana_scaler)
    out = m.forward(Xs, Xa_s, ds["vf"])
    s, gt = out["s"], ds["gt"] * 100.0
    per_case_mae = np.abs(s - gt).mean(1)
    H = out["H"].mean(1)
    r2, _ = ridge_probe_r2(H, ds["a"])
    return {
        "per_lobe_mae": float(per_case_mae.mean()),
        "spearman": float(spearmanr(out["yhat"], ds["y"] * 100.0).statistic),
        "case_mae": float(np.abs(out["yhat"] - ds["y"] * 100.0).mean()),
        "disentangle_r2": r2,
    }, per_case_mae


def agg(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--shape", default="32,64,64")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))

    path = os.path.join(RESULTS, "model.npz")
    if not os.path.exists(path):
        raise SystemExit(f"未找到 {path}，请先运行 src/train.py")
    models, meta, prior, seeds = load_models(path)

    t0 = time.time()
    print("=" * 62)
    print("生成测试集（与训练集不同种子 → 未见过的合成数据）")
    print("=" * 62)
    ds = build_dataset(n_per=args.n_per, shape=shape, seed_base=args.seed)
    print(f"测试集 {ds['n_cases']} 例（{len(ds['grid'])} 组 × {args.n_per}）")
    print(f"评测 {len(models)} 组配置 × {len(seeds)} seeds\n")

    res = {
        "test_set": {"n_cases": int(ds["n_cases"]), "shape": list(shape),
                     "seed_base": args.seed, "n_per": args.n_per,
                     "grid": ds["grid"]},
        "train_set": meta["train"], "n_seeds": len(seeds), "seeds": seeds,
        "note": "std = 跨随机初始化种子的标准差；p_value 为按测试病例配对的 Wilcoxon 符号秩检验",
    }

    ref, ref_s = ridge_probe_r2(ds["Xa"].mean(1), ds["a"])
    res["reference"] = {"anatomy_feature_to_anatomy_strength_r2":
                        {"mean": ref, "std": ref_s}}
    print(f"[参考] 解剖特征→解剖强度 探针 R² = {ref:.4f} ± {ref_s:.4f}\n")

    per_case = {}
    for name, (base, ms) in models.items():
        Xin = variant_input(base, ds)
        acc = {k: [] for k in ("per_lobe_mae", "spearman", "case_mae",
                               "disentangle_r2")}
        pcs = []
        for m in ms:
            met, pc = metrics_for_model(m, ds, Xin)
            for k in acc:
                acc[k].append(met[k])
            pcs.append(pc)
        res[name] = {k: agg(v) for k, v in acc.items()}
        per_case[name] = np.mean(np.stack(pcs), axis=0)   # 跨 seed 平均的逐病例 MAE
        d = res[name]
        print(f"[{name:12s}] MAE {d['per_lobe_mae']['mean']:.3f}"
              f"±{d['per_lobe_mae']['std']:.3f}  "
              f"Spearman {d['spearman']['mean']:.3f}"
              f"±{d['spearman']['std']:.3f}  "
              f"解耦R² {d['disentangle_r2']['mean']:+.3f}"
              f"±{d['disentangle_r2']['std']:.3f}")

    # ---------------- 显著性检验（配对 Wilcoxon，按测试病例配对）----------------
    tests = {}
    for other in ("b1", "b1b", "b2"):
        if "ours" in per_case and other in per_case:
            w = wilcoxon(per_case["ours"], per_case[other])
            tests[f"ours_vs_{other}"] = {
                "metric": "per_lobe_mae (paired per test case)",
                "n_pairs": int(len(per_case["ours"])),
                "statistic": float(w.statistic),
                "p_value": float(w.pvalue),
                "ours_mean": float(per_case["ours"].mean()),
                "other_mean": float(per_case[other].mean()),
            }
    res["tests"] = tests
    # 同时把 p 值挂到 b1/b2 各自块里，方便查表
    for other in ("b1", "b2"):
        key = f"ours_vs_{other}"
        if key in tests and other in res:
            res[other]["p_value_vs_ours"] = tests[key]["p_value"]

    print("\n[Wilcoxon 配对检验] 逐肺叶 MAE（按测试病例配对）:")
    for k, v in tests.items():
        print(f"  {k}: p = {v['p_value']:.4g}  "
              f"(ours {v['ours_mean']:.3f} vs other {v['other_mean']:.3f}, "
              f"n={v['n_pairs']})")

    # ---------------- 不定位基线 ----------------
    gt100 = ds["gt"] * 100.0
    b3 = np.abs(gt100 - (ds["y"] * 100.0)[:, None]).mean(1)
    b3b = (np.abs(gt100 - (prior * 100.0)[None, :]).mean(1)
           if prior is not None else None)
    res["baselines"] = {"B3_constant_case_severity": agg([float(b3.mean())])}
    if b3b is not None:
        res["baselines"]["B3b_lobar_prior"] = agg([float(b3b.mean())])
    print(f"\n[B3  常数=病例级真值] MAE = {b3.mean():.3f}")
    if b3b is not None:
        print(f"[B3b 常数=肺叶先验]   MAE = {b3b.mean():.3f}")

    # ---------------- λ 敏感性 ----------------
    sweep = {n: res[n] for n in models if n.startswith("ours_lam")}
    if sweep:
        res["lambda_sweep"] = {
            "ours_lam1.0": {k: res["ours"][k] for k in
                            ("per_lobe_mae", "disentangle_r2")},
            **{n: {k: sweep[n][k] for k in ("per_lobe_mae", "disentangle_r2")}
               for n in sweep},
        }
        print("\n[λ 敏感性] ours:")
        print(f"  λ=1.0  MAE {res['ours']['per_lobe_mae']['mean']:.3f}  "
              f"R² {res['ours']['disentangle_r2']['mean']:+.3f}")
        for n in sorted(sweep):
            print(f"  {n:<12s} MAE {sweep[n]['per_lobe_mae']['mean']:.3f}  "
                  f"R² {sweep[n]['disentangle_r2']['mean']:+.3f}")

    # ---------------- 分层 + 标签噪声 ----------------
    strat = {}
    for name in ("ours", "b1b", "b1"):
        if name not in per_case:
            continue
        strat[name] = [{"anatomy_var": float(l),
                        "per_lobe_mae": float(per_case[name][ds["a"] == l].mean()),
                        "n": int((ds["a"] == l).sum())}
                       for l in sorted(set(ds["a"].tolist()))]
    res["stratified_by_anatomy"] = strat

    rng = np.random.default_rng(0)
    y_true = ds["y"]
    noisy = {}
    for sig in (0.005, 0.01, 0.02):
        y_n = np.clip(y_true + rng.normal(0, sig, y_true.shape), 0, None)
        vals = []
        for m in models["ours"][1]:
            Xin = variant_input("ours", ds)
            Xs = apply_scaler(Xin, m.scaler)
            Xa_s = apply_scaler(ds["Xa"], m.ana_scaler)
            out = m.forward(Xs, Xa_s, ds["vf"])
            vals.append(float(spearmanr(out["yhat"], y_n * 100.0).statistic))
        noisy[f"sigma_{sig}"] = agg(vals)
    res["label_noise_ours_spearman"] = noisy

    with open(os.path.join(RESULTS, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, "results_seeds.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 results/results.json 与 results/results_seeds.json"
          f"  （耗时 {time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""后验分析（第 3 轮）

目的：回答第 2 轮评审的 **统计类** 意见（D1 / D2 / D6 / D4 的可计算部分）。
本脚本**不训练任何新模型**：它只加载 results/model.npz 里已保存的权重，
在既有测试集上重新计算指标与检验。因此不含任何编造的数字。

产出：
  results/results_posthoc.json
    primary_tests     3 组主比较的配对 Wilcoxon + Holm 校正（m=3）
    seed_level_tests  保守的 seed 级检验（n=5）
    lambda_gap        每个 λ 下 Ours 相对匹配臂 B1b 的差值 + 配对检验
    stratified_paired 按解剖变异强度分层后的 Ours vs B1b 配对检验
    stratified_lesion_ratio  按病灶比例分层 + 真值均值 + 归一化误差

红线：所有数字来自真实重算，不写死。
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon

from model import apply_scaler, build_dataset
from evaluate import load_models, variant_input, metrics_for_model, agg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

PRIMARY = [("b1", "B1"), ("b1b", "B1b"), ("b2", "B2")]


def holm(pvals):
    """Holm–Bonferroni step-down：返回与输入同序的校正后 p（下界 1.0）。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def paired_per_case(a, b):
    """逐病例配对检验（输入已是跨 seed 平均后的逐病例 MAE）。"""
    w = wilcoxon(a, b)
    return {"n_pairs": int(len(a)), "statistic": float(w.statistic),
            "p_value": float(w.pvalue), "mean_diff": float(a.mean() - b.mean())}


def pooled_p(per_seed_a, per_seed_b, pairs=5):
    """保守的 seed 级检验：把每个 seed 的整体均值当作一个观测（n=5）。"""
    diffs = np.asarray(per_seed_a, float) - np.asarray(per_seed_b, float)
    if np.allclose(diffs, diffs[0]):
        return {"p_value": float("nan"), "note": "all seed diffs identical"}
    w = wilcoxon(diffs)
    return {"n_pairs": pairs, "statistic": float(w.statistic),
            "p_value": float(w.pvalue),
            "mean_diff": float(diffs.mean()),
            "note": "seed-level (n=5); minimum attainable p = 0.0625"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--shape", default="32,64,64")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))

    path = os.path.join(RESULTS, "model.npz")
    models, meta, _prior, seeds = load_models(path)

    t0 = time.time()
    print("=" * 66)
    print("后验分析：加载已训练权重 + 重建测试集（不训练新模型）")
    print("=" * 66)
    ds = build_dataset(n_per=args.n_per, shape=shape, seed_base=args.seed)
    print(f"测试集 {ds['n_cases']} 例；{len(models)} 组配置 × {len(seeds)} seeds\n")

    per_case, per_seed = {}, {}
    for name, (base, ms) in models.items():
        Xin = variant_input(base, ds)
        pcs, scalars = [], []
        for m in ms:
            met, pc = metrics_for_model(m, ds, Xin)
            pcs.append(pc)
            scalars.append(met["per_lobe_mae"])
        per_case[name] = np.mean(np.stack(pcs), axis=0)
        per_seed[name] = scalars
        print(f"  [{name:12s}] MAE {np.mean(scalars):.3f}"
              f" ± {np.std(scalars, ddof=1):.3f}")

    out = {"meta": {"source": "re-evaluation of existing checkpoints "
                              "(results/model.npz); no new training",
                    "n_cases": int(ds["n_cases"]), "seeds": seeds,
                    "test_seed_base": args.seed,
                    "pairing": "per-case values are first averaged across "
                               "seeds, then paired by test case"}}

    # ---------------- 1. 主比较 + Holm 校正（m=3）----------------
    prim = {}
    for other, nice in PRIMARY:
        prim[f"ours_vs_{other}"] = paired_per_case(per_case["ours"],
                                                   per_case[other])
    keys = list(prim.keys())
    adj = holm([prim[k]["p_value"] for k in keys])
    for k, a in zip(keys, adj):
        prim[k]["p_value_holm"] = float(a)
    out["primary_tests"] = prim
    out["seed_level_tests"] = {
        f"ours_vs_{other}": pooled_p(per_seed["ours"], per_seed[other])
        for other, _ in PRIMARY}

    print("\n[主比较 · 逐病例配对 + Holm(m=3)]")
    for k in keys:
        v = prim[k]
        print(f"  {k:14s} ΔMAE {v['mean_diff']:+.3f}  "
              f"p = {v['p_value']:.3g}  Holm p = {v['p_value_holm']:.3g}")
    print("[保守 seed 级检验 n=5]")
    for k, v in out["seed_level_tests"].items():
        print(f"  {k:14s} ΔMAE {v['mean_diff']:+.3f}  p = {v['p_value']:.3g}")

    # ---------------- 2. 每个 λ 的 Ours-vs-B1b gap ----------------
    lam_names = {"ours": 1.0}
    for n in models:
        if n.startswith("ours_lam"):
            lam_names[n] = float(n.split("lam")[1])
    gap = {}
    print("\n[λ 敏感性 · Ours(λ) 相对 B1b 的 gap]")
    for n, lam in sorted(lam_names.items(), key=lambda kv: kv[1]):
        pc = per_case[n]
        rec = {"lambda_adv": lam,
               "mae": agg(per_seed[n]),
               "gap_vs_b1b": float(pc.mean() - per_case["b1b"].mean()),
               "paired_test_vs_b1b": paired_per_case(pc, per_case["b1b"]),
               "seed_level_test_vs_b1b": pooled_p(per_seed[n],
                                                  per_seed["b1b"])}
        gap[f"lam_{lam:.1f}"] = rec
        m = rec["mae"]
        print(f"  λ={lam:<4.1f} MAE {m['mean']:.3f}±{m['std']:.3f}  "
              f"gap vs B1b {rec['gap_vs_b1b']:+.3f}  "
              f"p = {rec['paired_test_vs_b1b']['p_value']:.3g}  "
              f"(seed-level p = {rec['seed_level_test_vs_b1b']['p_value']:.3g})")
    out["lambda_gap_vs_b1b"] = gap

    # ---------------- 3. 按解剖变异强度分层：Ours vs B1b 配对检验 ----------------
    avs = sorted(set(ds["a"].tolist()))
    strat = {}
    print("\n[按解剖变异强度分层 · Ours vs B1b 逐层配对]")
    for lev in avs:
        idx = ds["a"] == lev
        rec = {
            "anatomy_var": float(lev), "n": int(idx.sum()),
            "ours": float(per_case["ours"][idx].mean()),
            "b1b": float(per_case["b1b"][idx].mean()),
            "b1": float(per_case["b1"][idx].mean()),
            "paired_test": paired_per_case(per_case["ours"][idx],
                                          per_case["b1b"][idx]),
        }
        strat[f"av_{lev:.2f}"] = rec
        print(f"  av={lev:<4.2f}  Ours {rec['ours']:.3f}  B1b {rec['b1b']:.3f}"
              f"  Δ {rec['ours'] - rec['b1b']:+.3f}"
              f"  p = {rec['paired_test']['p_value']:.3g}")
    lo, hi = strat[f"av_{avs[0]:.2f}"], strat[f"av_{avs[-1]:.2f}"]
    dla = {"ours_relative_rise_percent":
           100 * (hi["ours"] - lo["ours"]) / lo["ours"],
           "b1b_relative_rise_percent":
           100 * (hi["b1b"] - lo["b1b"]) / lo["b1b"],
           "b1_relative_rise_percent":
           100 * (hi["b1"] - lo["b1"]) / lo["b1"],
           "note": "anatomy-variation levels are between-case factors "
                   "(not paired), so only within-level paired tests are "
                   "reported; the low-to-high rise is descriptive"}
    out["stratified_by_anatomy_paired"] = strat
    out["anatomy_degradation"] = dla
    print(f"  低→高变异相对上升：Ours {dla['ours_relative_rise_percent']:.1f}%"
          f"  B1b {dla['b1b_relative_rise_percent']:.1f}%"
          f"  B1 {dla['b1_relative_rise_percent']:.1f}%")

    # ---------------- 4. 按病灶比例分层 + 归一化误差 ----------------
    grid, n_per = ds["grid"], args.n_per
    lrs = sorted(set(g[0] for g in grid))
    lr_idx = {lr: np.array([i for i in range(ds["n_cases"])
                            if grid[i // n_per][0] == lr]) for lr in lrs}
    gtp = ds["gt"] * 100.0
    ssee = {}
    print("\n[按病灶比例分层 · MAE 与归一化误差]")
    for lr in lrs:
        idx = lr_idx[lr]
        rec = {"lesion_ratio": float(lr), "n": int(idx.sum()),
               "gt_mean_pp": float(gtp[idx].mean())}
        for name in ("ours", "b1b", "b1"):
            mae = float(per_case[name][idx].mean())
            rec[name] = {"mae_pp": mae,
                         "nmae_percent": 100 * mae / rec["gt_mean_pp"]}
        ssee[f"lr_{lr:.2f}"] = rec
        print(f"  lr={lr:<5.2f} GT均值 {rec['gt_mean_pp']:.2f}pp  "
              f"Ours {rec['ours']['mae_pp']:.3f} "
              f"({rec['ours']['nmae_percent']:.0f}%)  "
              f"B1b {rec['b1b']['mae_pp']:.3f} "
              f"({rec['b1b']['nmae_percent']:.0f}%)")
    out["stratified_by_lesion_ratio"] = ssee

    with open(os.path.join(RESULTS, "results_posthoc.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 results/results_posthoc.json  （耗时 {time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()

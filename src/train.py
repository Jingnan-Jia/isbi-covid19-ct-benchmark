#!/usr/bin/env python3
"""训练（第 2 轮修订版）

针对第 1 轮评审补的东西：
- C1: 新增 b1b（13-D 无对抗，与 ours 只差对抗这一项）与 b4（正交约束替代对抗）
- C1/C16: 新增 leaky（打分分支被主动要求预测解剖强度 = 阳性对照）
- C4: --seed 跑多个随机初始化种子（默认 0,1,2,3,4），每个种子都保存完整参数
- λ 敏感性：对 ours 额外训练 λ_adv ∈ {0.1, 0.5, 2.0}（λ=1.0 即默认 ours）

变体一览（R1 指出原 b1 与 ours 差了两个因素，故拆开）：
  ours  13-D 病灶特征 + 对抗(λ=+1)
  b1    15-D（病灶+解剖混在一起）+ 无对抗        [原基线]
  b1b   13-D（同 ours）+ 无对抗                 [对抗的唯一消融]
  b4    13-D + 正交约束（替代机制）
  b2    10-D 仅密度 + 对抗
  leaky 13-D + 对抗头但 λ=-1（主动泄漏，阳性对照）

红线：不写死任何结果数字，所有打印来自真实运行。
"""
import argparse
import json
import os
import time

import numpy as np

from model import (Adam, D_DENSITY, D_LESION, Model, apply_scaler,
                   build_dataset, scaler_from)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

VARIANTS = ["ours", "b1", "b1b", "b4", "b2", "leaky"]
# λ 敏感性扫描：（保存名, 基础变体, λ_adv）
SWEEP = [("ours_lam0.1", "ours", 0.1),
         ("ours_lam0.5", "ours", 0.5),
         ("ours_lam2.0", "ours", 2.0)]


def variant_input(name, ds):
    X = ds["X"]
    Xa = ds["Xa"]
    if name == "b1":
        return np.concatenate([X, Xa], axis=-1)
    if name == "b2":
        return X[:, :, :D_DENSITY]
    return X[:, :, :D_LESION]


def lam_for(name):
    """leaky 的对抗权重取负 → 打分分支被推向"更能预测解剖强度"。"""
    return -1.0 if name == "leaky" else 1.0


def train_one(name, ds, seed=0, steps=400, batch=32, lr=0.01,
              hidden=16, lam_ortho=1.0, lam_adv=None, log_every=400):
    base = name.split("_lam")[0] if name.startswith("ours_lam") else name
    Xin = variant_input(base, ds)
    sc = scaler_from(Xin)
    Xs = apply_scaler(Xin, sc)
    Xa_s = apply_scaler(ds["Xa"], scaler_from(ds["Xa"]))
    y, a, vf = ds["y"], ds["a"], ds["vf"]

    m = Model(variant=base, hidden=hidden, seed=seed)
    opt = Adam(m.p, lr=lr)
    rng = np.random.default_rng(seed + 1)
    N = Xs.shape[0]
    lam = lam_for(base) if lam_adv is None else lam_adv

    for step in range(steps + 1):
        idx = rng.choice(N, size=min(batch, N), replace=False)
        out = m.forward(Xs[idx], Xa_s[idx], vf[idx])
        L = m.losses(out, y[idx], a[idx])
        if step % log_every == 0 or step == steps:
            msg = f"[{name} s{seed}] step {step:4d} score={L['score']:.3f}"
            for k in ("adv", "ortho", "ana"):
                if k in L:
                    msg += f"  {k}={L[k]:.4f}"
            print(msg)
        if step == steps:
            break
        g = m.backward(Xs[idx], Xa_s[idx], vf[idx], y[idx], a[idx], out,
                       lam_adv=lam, lam_ortho=lam_ortho)
        tot = np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))
        if tot > 5.0:
            for k in g:
                g[k] *= 5.0 / tot
        opt.step(m.p, g)
    return m, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=20)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="随机初始化种子列表（默认 0 1 2 3 4）")
    ap.add_argument("--shape", default="32,64,64")
    args = ap.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))
    seeds = list(args.seed)

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    print("=" * 62)
    print("生成训练集（synthetic phantom，已知真值）")
    print("=" * 62)
    ds = build_dataset(n_per=args.n_per, shape=shape, seed_base=0, verbose=True)
    print(f"{ds['n_cases']} 例（{len(ds['grid'])} 组 × {args.n_per}），shape={shape}，"
          f"耗时 {time.time()-t0:.1f}s")
    print(f"seeds = {seeds}")

    # C2：训练集肺叶先验，供评测端构造"不定位"基线 B3b
    out = {"lobar_prior": ds["gt"].mean(0),
           "train_mean_severity": np.array(float(ds["y"].mean())),
           "ana_mu": scaler_from(ds["Xa"])["mu"],
           "ana_sd": scaler_from(ds["Xa"])["sd"]}
    meta = {"variants": {}, "sweep": {}, "seeds": seeds, "steps": args.steps,
            "train": {"n_cases": int(ds["n_cases"]), "grid": ds["grid"],
                      "n_per": args.n_per, "shape": list(shape)}}

    jobs = [(n, n, None) for n in VARIANTS] + \
           [(sname, sbase, slam) for (sname, sbase, slam) in SWEEP]

    for name, base, slam in jobs:
        print("-" * 62)
        for s in seeds:
            m, sc = train_one(name, ds, seed=s, steps=args.steps, lam_adv=slam)
            for k, v in m.p.items():
                out[f"{name}::s{s}::{k}"] = v
            if s == seeds[0]:
                out[f"{name}::mu"] = sc["mu"]
                out[f"{name}::sd"] = sc["sd"]
                rec = {"d_in": int(m.d_in), "hidden": int(m.hidden),
                       "cfg": m.cfg, "base": base,
                       "lam_adv": (lam_for(base) if slam is None else slam)}
                (meta["sweep"] if slam is not None else meta["variants"])[name] = rec
        print(f"[{name}] {len(seeds)} 个种子训练完毕")

    out["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    path = os.path.join(RESULTS, "model.npz")
    np.savez_compressed(path, **out)
    print("-" * 62)
    print(f"已保存 → {path}（{len(jobs)} 组配置 × {len(seeds)} seeds）")
    print(f"总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

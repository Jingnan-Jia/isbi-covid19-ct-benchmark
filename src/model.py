#!/usr/bin/env python3
"""解剖解耦 + 弱监督逐肺叶打分模型（纯 numpy 实现，无 torch）。

设计要点
--------
1. 特征提取（每个肺叶 ROI 一组）
   - 密度直方图 (a)：HU 在 [-1000, -100] 分 10 箱，按 ROI 体素数归一化 → 10 维
   - 纹理特征 (b)：局部均值、局部标准差、梯度幅值均值 → 3 维
   - 解剖形状特征 (c)：ROI 体积分数、球形度 → 2 维
     **只用于解耦正则与解剖分支，绝不进入打分分支**

2. 弱监督打分头
   - 病灶特征 → 每个肺叶一个 0-100 的连续分数
   - 病例级聚合：按肺叶体积分数加权求和（体积分数是可观测量，来自分割）
   - 监督信号**只有病例级严重度**（= 真实病灶比例），逐肺叶真值不进训练

3. 解剖解耦（对抗 + 梯度反转）
   - 解剖分支：解剖形状特征 → 预测"解剖变异强度"（合成数据中已知）
   - 对抗分支：从打分分支的隐表示 h 预测解剖变异强度
   - 打分分支反向时对对抗梯度取负（梯度反转），使 h 无法预测解剖强度

红线：本模块只做前向/反向/优化，不含任何写死的实验结果数字。
"""
import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter

from gen_phantom import LOBE_NAMES, generate_phantom

EPS = 1e-8
HU_LO, HU_HI, N_BINS = -1000.0, -100.0, 10
D_DENSITY = N_BINS          # 10
D_TEXTURE = 3               # local mean / local std / grad magnitude
D_LESION = D_DENSITY + D_TEXTURE   # 13 → 打分分支输入
D_ANA = 2                   # volume fraction / sphericity → 解剖分支输入
D_ALL = D_LESION + D_ANA    # 15 → B1（不解耦）输入

LOBE_IDS = sorted(LOBE_NAMES)      # [1,2,3,4,5]


# --------------------------------------------------------------------------- #
# 特征提取
# --------------------------------------------------------------------------- #
def _sphericity(mask):
    """球形度代理：((36πV²)^(1/3)) / A，A 用边界体素数近似。球体≈1，越不规则越大。"""
    v = float(mask.sum())
    if v <= 1:
        return 1.0
    inner = binary_erosion(mask)
    area = float(mask.sum() - inner.sum())
    return float(((36.0 * np.pi * v * v) ** (1.0 / 3.0)) / max(area, 1.0))


def lobe_features(vol, lobes, lesion=None):
    """对单个体数据提取 5 个肺叶的特征。

    返回 (X_lesion, X_ana, vol_frac, lobe_gt)
      X_lesion : (5, 13)  密度直方图 + 纹理
      X_ana    : (5, 2)   体积分数 + 球形度（不进打分分支）
      vol_frac : (5,)     肺叶体积占全肺比例（弱监督聚合权重，可观测量）
      lobe_gt  : (5,)     逐肺叶真实病灶比例（**仅评测用，训练不可见**）
    """
    lung = lobes > 0
    n_lung = max(int(lung.sum()), 1)

    smoothed = uniform_filter(vol, size=3)
    gz, gy, gx = np.gradient(vol)
    grad_mag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)

    Xd, Xt, Xa, vf, gt = [], [], [], [], []
    for lid in LOBE_IDS:
        m = lobes == lid
        n = int(m.sum())
        if n < 32:                      # 退化 ROI：给零特征
            Xd.append(np.zeros(D_DENSITY)); Xt.append(np.zeros(D_TEXTURE))
            Xa.append(np.zeros(D_ANA)); vf.append(0.0); gt.append(0.0)
            continue
        hu = vol[m]
        # (a) 密度直方图：只看肺/病灶的 HU 区间
        hist, _ = np.histogram(hu, bins=N_BINS, range=(HU_LO, HU_HI))
        Xd.append(hist.astype(np.float64) / n)
        # (b) 纹理
        Xt.append([
            float(smoothed[m].mean()),
            float((hu - smoothed[m]).std()),
            float(grad_mag[m].mean()),
        ])
        # (c) 解剖形状
        Xa.append([n / n_lung, _sphericity(m)])
        vf.append(n / n_lung)
        gt.append(float((lesion & m).sum()) / n if lesion is not None else 0.0)

    X_les = np.hstack([np.array(Xd), np.array(Xt)]).astype(np.float64)  # (5,13)
    X_ana = np.array(Xa, dtype=np.float64)                              # (5,2)
    vf = np.array(vf, dtype=np.float64)
    s = vf.sum()
    if s > EPS:
        vf = vf / s
    return X_les, X_ana, vf, np.array(gt, dtype=np.float64)


def case_features(vol, lobes, lesion):
    """单病例 → (X_lesion(5,13), X_ana(5,2), vol_frac(5), lobe_gt(5), severity)"""
    X_les, X_ana, vf, gt = lobe_features(vol, lobes, lesion)
    lung = lobes > 0
    sev = float((lesion & lung).sum()) / max(int(lung.sum()), 1)
    return X_les, X_ana, vf, gt, sev


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
# 4 档病灶比例 × 4 档解剖变异 = 16 组（≥ 要求的 8 组）。
# 解剖变异给 4 档（而非 2 档）是刻意的：只有解剖轴上有足够动态范围，
# "病灶分支能否预测解剖强度"这个解耦指标才测得出东西。
DEFAULT_GRID = [(lr, av) for lr in (0.02, 0.05, 0.08, 0.12)
                for av in (0.05, 0.35, 0.70, 1.00)]   # 4 × 4 = 16 组


def build_dataset(grid=DEFAULT_GRID, n_per=20, shape=(32, 64, 64),
                  noise=25.0, seed_base=0, verbose=False):
    """用 gen_phantom 生成数据集。返回 dict of arrays。

    标签约定：
      y  = 病例级严重度（真实全肺病灶比例）—— **弱监督唯一监督信号**
      gt = 逐肺叶真实病灶比例 —— 仅用于评测
      a  = 解剖变异强度（合成时已知）—— 解剖分支/对抗分支的目标
    """
    cases = []
    k = 0
    for (lr, av) in grid:
        for i in range(n_per):
            seed = seed_base + k
            vol, lobes, lesion = generate_phantom(
                shape=shape, lesion_ratio=lr, anatomy_var=av,
                noise=noise, seed=seed)
            Xl, Xa, vf, gt, sev = case_features(vol, lobes, lesion)
            cases.append((Xl, Xa, vf, gt, sev, av))
            k += 1
            if verbose and k % 20 == 0:
                print(f"    generated {k}/{len(grid)*n_per}")
    n = len(cases)
    return {
        "X": np.stack([c[0] for c in cases]),        # (N,5,13)
        "Xa": np.stack([c[1] for c in cases]),       # (N,5,2)
        "vf": np.stack([c[2] for c in cases]),       # (N,5)
        "gt": np.stack([c[3] for c in cases]),       # (N,5) 逐肺叶真值(评测用)
        "y": np.array([c[4] for c in cases]),        # (N,)  病例级严重度
        "a": np.array([c[5] for c in cases]),        # (N,)  解剖变异强度
        "grid": [list(g) for g in grid],
        "n_per": n_per, "shape": list(shape), "noise": noise,
        "n_cases": n,
    }


def scaler_from(X):
    """按 (N*5, d) 计算标准化统计量。"""
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd = np.where(sd < EPS, 1.0, sd)
    return {"mu": mu, "sd": sd}


def apply_scaler(X, sc):
    return (X - sc["mu"]) / sc["sd"]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def relu(t):
    return np.maximum(t, 0.0)


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


class Model:
    """打分头 + 解剖分支 + 对抗分支。

    variant:
      'ours' → 输入 13 维（密度+纹理），启用对抗解耦 + 解剖分支
      'b1'   → 输入 15 维（病灶+解剖混在一起），**不启用解耦**
      'b2'   → 输入 10 维（仅密度），启用对抗解耦 + 解剖分支
    """

    VARIANTS = {
        # ours : 13-D 病灶特征 + 对抗解耦（梯度反转）+ 解剖分支
        "ours": dict(use_ana_feat=False, use_adv=True, use_ana_branch=True),
        # b1   : 病灶+解剖特征混在一起(15-D)，且无对抗 —— 与 ours 差两个因素
        "b1":   dict(use_ana_feat=True,  use_adv=False, use_ana_branch=False),
        # b1b  : 13-D（同 ours）但**不加对抗** —— 与 ours 唯一差异 = 有无对抗
        "b1b":  dict(use_ana_feat=False, use_adv=False, use_ana_branch=False),
        # b4   : 13-D（同 ours），用**正交约束**替代对抗（替代机制对比）
        "b4":   dict(use_ana_feat=False, use_adv=False, use_ana_branch=True,
                     use_ortho=True),
        # b2   : 仅密度特征(10-D) + 对抗解耦
        "b2":   dict(use_density_only=True, use_ana_feat=False,
                     use_adv=True, use_ana_branch=True),
        # leaky: 13-D，但打分分支被**主动要求预测解剖强度**（阳性对照：
        #        若探针在它身上 R² 明显升高，说明探针本身有辨别力）
        "leaky": dict(use_ana_feat=False, use_adv=True, use_ana_branch=True,
                      leak=True),
    }

    def __init__(self, variant="ours", hidden=16, seed=0):
        cfg = dict(self.VARIANTS.get(variant, self.VARIANTS["ours"]))
        cfg.setdefault("use_density_only", False)
        self.variant = variant
        self.cfg = cfg
        self.hidden = hidden
        self.d_in = D_DENSITY if cfg["use_density_only"] else (
            D_ALL if cfg["use_ana_feat"] else D_LESION)
        rng = np.random.default_rng(seed)
        s = 0.08
        self.p = {
            "W1": rng.normal(0, s, (hidden, self.d_in)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, s, hidden),
            "b2": np.zeros(1),
            "Wa": rng.normal(0, s, (hidden, D_ANA)),
            "ba": np.zeros(hidden),
            "va": rng.normal(0, s, hidden),
            "ca": np.zeros(1),
            "u": rng.normal(0, s, hidden),      # 对抗分支
            "c": np.zeros(1),
        }
        self.scaler = None

    # ---------- 前向 ----------
    def forward(self, X, Xa, vf, a=None):
        p = self.p
        H = relu(X @ p["W1"].T + p["b1"])            # (N,5,H)
        z = H @ p["w2"] + p["b2"]                    # (N,5)
        s = 100.0 * sigmoid(z)                       # 逐肺叶分数 0-100
        yhat = (s * vf).sum(1) / np.maximum(vf.sum(1), EPS)   # 病例级聚合 (N,)

        out = {"H": H, "z": z, "s": s, "yhat": yhat}
        if self.cfg["use_ana_branch"]:
            Ha = relu(Xa @ p["Wa"].T + p["ba"])      # (N,5,H)
            a_hat = (Ha @ p["va"] + p["ca"]).mean(1)  # (N,)
            out.update(Ha=Ha, a_hat=a_hat)
        if self.cfg["use_adv"]:
            out["adv"] = H @ p["u"] + p["c"]         # (N,5)
        if self.cfg.get("use_ortho"):
            # 正交约束：打分隐表示 H 与解剖特征 Xa 的互协方差 Frobenius 范数平方。
            # 压到 0 即 H 与解剖特征线性无关（替代对抗的解耦机制）。
            Hf = H.reshape(-1, self.hidden)
            Xf = Xa.reshape(-1, Xa.shape[-1]).astype(np.float64)
            Hc = Hf - Hf.mean(0)
            Xc = Xf - Xf.mean(0)
            n = max(Hf.shape[0], 1)
            C = (Hc.T @ Xc) / n                      # (H, 2)
            out["ortho"] = float((C ** 2).sum())
            out["_oC"], out["_oHc"], out["_oXc"], out["_on"] = C, Hc, Xc, n
        return out

    # ---------- 损失 ----------
    def losses(self, out, y, a):
        p = self.p
        L_score = float(np.mean((out["yhat"] - 100.0 * y) ** 2))
        res = {"score": L_score}
        if self.cfg["use_ana_branch"]:
            res["ana"] = float(np.mean((out["a_hat"] - a) ** 2))
        if self.cfg["use_adv"]:
            res["adv"] = float(np.mean((out["adv"] - a[:, None]) ** 2))
        if self.cfg.get("use_ortho"):
            res["ortho"] = out["ortho"]
        return res

    # ---------- 反向 ----------
    def backward(self, X, Xa, vf, y, a, out, lam_adv=1.0, lam_ana=1.0,
                 lam_ortho=1.0):
        p = self.p
        N, L, Hh = X.shape[0], X.shape[1], self.hidden
        g = {k: np.zeros_like(v) for k, v in p.items()}

        # --- 打分损失 ---
        dy = 2.0 * (out["yhat"] - 100.0 * y) / N                 # (N,)
        w = vf / np.maximum(vf.sum(1, keepdims=True), EPS)
        ds = dy[:, None] * w                                      # (N,5)
        dz = ds * out["s"] * (100.0 - out["s"]) / 100.0           # d s/d z
        g["w2"] += (dz[:, :, None] * out["H"]).sum((0, 1))
        g["b2"] += dz.sum()
        dH = dz[:, :, None] * p["w2"]                             # (N,5,H)

        # --- 对抗分支（梯度反转）---
        if self.cfg["use_adv"]:
            dadv = 2.0 * (out["adv"] - a[:, None]) / (N * L)      # (N,5)
            g["u"] += (dadv[:, :, None] * out["H"]).sum((0, 1))
            g["c"] += dadv.sum()
            dH -= lam_adv * (dadv[:, :, None] * p["u"])           # 反转：减号

        # --- 正交约束（B4：压低 H 与解剖特征的互协方差）---
        if self.cfg.get("use_ortho"):
            C, Xc, n = out["_oC"], out["_oXc"], out["_on"]
            dHf = (Xc @ (2.0 * C).T) / n
            dHf = dHf - dHf.mean(0)               # 中心化操作的梯度修正
            dH += lam_ortho * dHf.reshape(N, L, Hh)

        # --- 过 ReLU 回传 ---
        dpre = dH * (out["H"] > 0)
        g["W1"] += np.einsum("nlh,nld->hd", dpre, X)
        g["b1"] += dpre.sum((0, 1))

        # --- 解剖分支（独立优化，不受反转影响）---
        if self.cfg["use_ana_branch"]:
            Ha = out["Ha"]
            da_hat = 2.0 * (out["a_hat"] - a) / N                 # (N,)
            dHa = (da_hat[:, None] / L)[:, :, None] * p["va"]     # (N,5,H)
            g["va"] += (dHa * Ha).sum((0, 1))
            g["ca"] += da_hat.sum()
            dpa = dHa * (Ha > 0)
            g["Wa"] += np.einsum("nlh,nld->hd", dpa, Xa)
            g["ba"] += dpa.sum((0, 1))
            g["Wa"] *= lam_ana
            g["ba"] *= lam_ana
            g["va"] *= lam_ana
            g["ca"] *= lam_ana
        return g


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

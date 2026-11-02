#!/usr/bin/env python3
"""合成肺 CT 体数据生成器（SYNTHETIC PHANTOM）。

⚠️ 红线声明
-----------
本脚本产出的是**程序合成的合成体模（synthetic phantom）**，不是真实患者数据，
也不代表任何真实疾病的统计分布。凡是由它产生的数字，在论文/报告中都必须明确
标注为 "synthetic phantom"，不得写成 "we collected N patients"。

解剖设定（近似，非精确解剖学）
------------------------------
- 体模外空气 ≈ -1000 HU；躯干软组织 ≈ +40 HU；肺实质 ≈ -850 HU
- 5 个肺叶：右肺 上/中/下（RUL / RML / RLL），左肺 上/下（LUL / LLL）
  （以图像 x 较小侧为右肺，仅为一致性约定）
- 病灶：以**基底部 + 胸膜下**分布为主，成簇出现，模拟 SSc-ILD 的
  磨玻璃密度影（GGO）与网格/纤维化影

三个可调参数（对应 BRIEF 第一阶段要求）
--------------------------------------
- lesion_ratio  病灶比例：病灶体素 / 肺体素 的目标值
- anatomy_var   解剖变异强度：肺大小/位置/肺裂倾角的抖动强度（0 = 完全对称）
- noise         噪声：加性高斯噪声标准差（HU）

输出
----
  <out>.npy            CT 体数据（float32, HU-like）
  <out>_lobes.npy      肺叶标签（int16, 0=非肺, 1..5）
  <out>_lesion.npy     病灶掩膜（uint8）
  <out>_meta.json      参数 + 统计量（含逐肺叶病灶比例 = 后续弱监督的目标值）

用法
----
  python3 src/gen_phantom.py
  python3 src/gen_phantom.py --lesion-ratio 0.15 --anatomy-var 0.8 --noise 40 --seed 7
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

LOBE_NAMES = {1: "RUL", 2: "RML", 3: "RLL", 4: "LUL", 5: "LLL"}

HU_AIR = -1000.0
HU_SOFT = 40.0
HU_LUNG = -850.0


def _ellipsoid(shape, center, radii):
    """返回布尔掩膜：以 center 为中心、radii 为半轴的椭球（含内部）。"""
    zz, yy, xx = (a.astype(np.float32) for a in np.indices(shape))
    cz, cy, cx = center
    rz, ry, rx = radii
    return (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def generate_phantom(shape=(64, 128, 128), lesion_ratio=0.08, anatomy_var=0.30,
                     noise=25.0, seed=0):
    """生成一个合成肺 CT 体数据。

    返回 (volume, lobe_label, lesion_mask)，分别为
    float32 HU 体数据 / int16 肺叶标签 / uint8 病灶掩膜。
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = (a.astype(np.float32) for a in np.indices(shape))
    ny = (yy - H / 2.0) / (H / 2.0)     # 归一化 -1..1
    nx = (xx - W / 2.0) / (W / 2.0)
    av = float(anatomy_var)

    # ---------- 1) 躯干（软组织） ----------
    body = _ellipsoid(shape, (D / 2.0, H / 2.0, W / 2.0),
                      (D * 0.50, H * 0.46, W * 0.46))
    clean = np.full(shape, HU_AIR, np.float32)
    clean[body] = HU_SOFT + rng.normal(0, 8.0, int(body.sum())).astype(np.float32)

    # ---------- 2) 双肺（解剖变异：中心 / 大小抖动）----------
    cz = D * 0.50 + rng.normal(0, 1) * av * D * 0.03
    cy = H * 0.50 + rng.normal(0, 1) * av * H * 0.03
    sep = W * 0.16 * (1.0 + rng.normal(0, 1) * av * 0.20)
    cxl, cxr = W * 0.50 - sep, W * 0.50 + sep

    def _lung_radii():
        return (D * 0.33 * (1.0 + rng.normal(0, 1) * av * 0.10),
                H * 0.30 * (1.0 + rng.normal(0, 1) * av * 0.10),
                W * 0.15 * (1.0 + rng.normal(0, 1) * av * 0.12))

    lung_r = _ellipsoid(shape, (cz, cy, cxr), _lung_radii())   # 右肺：3 叶
    lung_l = _ellipsoid(shape, (cz, cy, cxl), _lung_radii())   # 左肺：2 叶
    lung = lung_r | lung_l

    # 肺实质：健康肺 ≈ -850 HU（带轻微不均匀）
    clean[lung] = HU_LUNG + rng.normal(0, 25.0, int(lung.sum())).astype(np.float32)

    # ---------- 3) 肺叶（肺裂平面，倾角受 anatomy_var 控制）----------
    def _fissure(z0_frac, tilt_y_scale=0.22, tilt_x_scale=0.12):
        a = rng.normal(0, 1) * av * D * tilt_y_scale
        b = rng.normal(0, 1) * av * D * tilt_x_scale
        return D * z0_frac + a * ny + b * nx

    z_minor = _fissure(0.44)   # 右肺 上/中 叶之间（水平裂）
    z_major = _fissure(0.64)   # 右肺 中/下 叶之间（斜裂）
    z_left = _fissure(0.55)    # 左肺 上/下 叶之间（斜裂）

    lobes = np.zeros(shape, np.int16)
    lobes[lung_r & (zz < z_minor)] = 1                                    # RUL
    lobes[lung_r & (zz >= z_minor) & (zz < z_major)] = 2                  # RML
    lobes[lung_r & (zz >= z_major)] = 3                                   # RLL
    lobes[lung_l & (zz < z_left)] = 4                                     # LUL
    lobes[lung_l & (zz >= z_left)] = 5                                    # LLL

    # ---------- 4) 病灶：基底 + 胸膜下为主，成簇 ----------
    lesion = np.zeros(shape, bool)
    n_lung = int(lung.sum())
    target = int(round(lesion_ratio * n_lung))

    if target > 0 and n_lung > 0:
        # 权重：基底部（z 大）× 胸膜下（距肺边界近）
        w_basal = np.clip(zz / float(D), 0.0, 1.0) ** 2
        dist = distance_transform_edt(lung).astype(np.float32)
        w_sub = np.exp(-dist / max(0.18 * float(dist.max()), 1e-6))
        w = (w_basal * w_sub) * lung
        p = w.ravel().astype(np.float64)
        s = p.sum()
        if s <= 0:                       # 退化情况：退回均匀采样
            p = lung.ravel().astype(np.float64)
            s = p.sum()
        cdf = np.cumsum(p) / s

        lesion_hu = np.full(shape, np.nan, np.float32)
        guard = 0
        while int(lesion.sum()) < target and guard < 4000:
            guard += 1
            flat = int(np.searchsorted(cdf, rng.random()))
            if flat >= cdf.size:
                flat = cdf.size - 1
            sz, sy, sx = np.unravel_index(flat, shape)
            if not lung[sz, sy, sx]:
                continue
            r0 = float(rng.uniform(1.2, 3.5))
            rz0 = r0 * float(rng.uniform(0.6, 1.0))     # 轴位略扁
            blob = _ellipsoid(shape, (float(sz), float(sy), float(sx)),
                              (rz0, r0, r0)) & lung
            # GGO（-600..-380）为主，少量网格/纤维化影（-350..-80）
            hu = (rng.uniform(-600, -380) if rng.random() < 0.7
                  else rng.uniform(-350, -80))
            lesion_hu[blob] = hu
            lesion |= blob

    # ---------- 5) 融合：软边病灶 + CT 平滑 + 噪声 ----------
    if lesion.any():
        alpha = gaussian_filter(lesion.astype(np.float32), sigma=0.9)
        tgt = np.where(np.isfinite(lesion_hu), lesion_hu, clean)
        vol = clean * (1.0 - alpha) + tgt * alpha
    else:
        vol = clean.copy()
    vol = gaussian_filter(vol, sigma=0.7).astype(np.float32)   # CT 模糊
    vol = vol + rng.normal(0.0, float(noise), shape).astype(np.float32)
    vol = np.clip(vol, -1024.0, 3071.0).astype(np.float32)

    return vol, lobes, lesion.astype(np.uint8)


def stats(vol, lobes, lesion, params):
    """计算并打印形状与统计量；返回 dict（同时写入 meta.json）。"""
    lung = lobes > 0
    n_lung = int(lung.sum())
    n_les = int(lesion.sum())
    out = {
        "params": params,
        "shape": list(vol.shape),
        "dtype": str(vol.dtype),
        "hu_mean": float(vol.mean()),
        "hu_std": float(vol.std()),
        "hu_min": float(vol.min()),
        "hu_max": float(vol.max()),
        "lung_voxels": n_lung,
        "lung_fraction_of_volume": float(n_lung / vol.size),
        "lung_hu_mean": float(vol[lung].mean()) if n_lung else 0.0,
        "lung_hu_std": float(vol[lung].std()) if n_lung else 0.0,
        "lesion_voxels": n_les,
        "lesion_fraction_of_lung": float(n_les / n_lung) if n_lung else 0.0,
        "case_level_severity": float(n_les / n_lung) if n_lung else 0.0,
        "per_lobe": [],
    }
    for lid in sorted(LOBE_NAMES):
        m = lobes == lid
        nv = int(m.sum())
        nl = int((lesion & m).sum())
        out["per_lobe"].append({
            "lobe_id": lid, "name": LOBE_NAMES[lid], "voxels": nv,
            "lesion_voxels": nl,
            "lesion_fraction": float(nl / nv) if nv else 0.0,
        })

    p = params
    print("=" * 62)
    print("SYNTHETIC PHANTOM — 合成肺 CT 体数据（非真实患者数据）")
    print("=" * 62)
    print(f"参数: lesion_ratio={p['lesion_ratio']}  anatomy_var={p['anatomy_var']}  "
          f"noise={p['noise']}  seed={p['seed']}")
    print(f"形状: {tuple(out['shape'])}  dtype={out['dtype']}  体素总数={vol.size}")
    print(f"HU:   mean={out['hu_mean']:.1f}  std={out['hu_std']:.1f}  "
          f"min={out['hu_min']:.1f}  max={out['hu_max']:.1f}")
    print(f"肺:   体素={out['lung_voxels']}  占全容积={out['lung_fraction_of_volume']*100:.1f}%  "
          f"HU mean={out['lung_hu_mean']:.1f} std={out['lung_hu_std']:.1f}")
    print(f"病灶: 体素={out['lesion_voxels']}  占肺比例={out['lesion_fraction_of_lung']:.4f} "
          f"(目标 {p['lesion_ratio']})")
    print(f"病例级严重度(连续标签) = {out['case_level_severity']:.4f}")
    print("逐肺叶病灶比例（后续弱监督的逐叶目标值）:")
    for d in out["per_lobe"]:
        print(f"  - {d['name']:<4} (id={d['lobe_id']})  体素={d['voxels']:<7} "
              f"病灶={d['lesion_voxels']:<6} 比例={d['lesion_fraction']:.4f}")
    print("=" * 62)
    return out


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="生成合成肺 CT 体数据（synthetic phantom）")
    ap.add_argument("--shape", default="64,128,128",
                    help="体数据形状 D,H,W（默认 64,128,128）")
    ap.add_argument("--lesion-ratio", type=float, default=0.08,
                    help="病灶比例：病灶体素/肺体素（默认 0.08）")
    ap.add_argument("--anatomy-var", type=float, default=0.30,
                    help="解剖变异强度（默认 0.30，0=完全对称）")
    ap.add_argument("--noise", type=float, default=25.0,
                    help="加性高斯噪声标准差 HU（默认 25）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--out", default=None,
                    help="输出前缀（默认 data/phantom_lr.._av.._nz.._s..）")
    ap.add_argument("--no-save", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    params = {
        "shape": list(shape),
        "lesion_ratio": args.lesion_ratio,
        "anatomy_var": args.anatomy_var,
        "noise": args.noise,
        "seed": args.seed,
    }

    vol, lobes, lesion = generate_phantom(
        shape=shape, lesion_ratio=args.lesion_ratio,
        anatomy_var=args.anatomy_var, noise=args.noise, seed=args.seed)

    meta = stats(vol, lobes, lesion, params)

    if args.no_save:
        return
    if args.out:
        prefix = args.out
    else:
        tag = (f"phantom_lr{args.lesion_ratio:.3f}_av{args.anatomy_var:.2f}"
               f"_nz{args.noise:.0f}_s{args.seed}")
        prefix = os.path.join(ROOT, "data", tag)
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.save(prefix + ".npy", vol)
    np.save(prefix + "_lobes.npy", lobes)
    np.save(prefix + "_lesion.npy", lesion)
    with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("已保存:")
    for suf in (".npy", "_lobes.npy", "_lesion.npy", "_meta.json"):
        print(f"  {prefix}{suf}")


if __name__ == "__main__":
    main()

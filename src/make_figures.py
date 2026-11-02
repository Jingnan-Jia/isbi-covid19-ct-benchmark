#!/usr/bin/env python3
"""生成论文插图（PDF 矢量图）→ paper/figures/

- fig1_framework.pdf  方法框架（解剖解耦 + 弱监督打分，方框箭头）
- fig2_phantom.pdf    合成数据三联图：CT-like 切片 / 病灶 mask / 肺叶分区
- fig3_results.pdf    柱状图对比 ours / b1 / b2 的逐肺叶 MAE（带误差棒）

规约：300 dpi、字体够大、**图内不带标题**（标题写在 LaTeX caption 里）。
fig3 的数值全部从 results/results.json 读取，保证"图上的数 = 结果里的数"。
fig2 的体数据复用 data/phantom_lr0.080_av0.30_nz25_s0*.npy（第一阶段产物）。
论文为英文（ISBI），故图内标注用英文。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(ROOT, "data")
PHANTOM = "phantom_lr0.080_av0.30_nz25_s0"

ACC = "#2b6cb0"; ORANGE = "#c05621"; GREEN = "#2f855a"; RED = "#c53030"
INK = "#1a202c"; GREY = "#4a5568"

LOBE_NAMES = ["RUL", "RML", "RLL", "LUL", "LLL"]
LOBE_COLORS = ["#4c78d8", "#f2b13c", "#59a14f", "#e4575b", "#b07aa1"]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.linewidth": 0.8})


def _box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.02",
                                linewidth=1.1, edgecolor=color,
                                facecolor="#edf2f7"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=5, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color=GREY, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=lw,
                                 color=color, linestyle=ls, zorder=4))


def fig1_framework(path):
    """第 3 轮重写：更少方框（6 个主框）、字号 9.5–10pt、对抗分支不再与框重叠。"""
    fig, ax = plt.subplots(figsize=(3.45, 3.35), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # R1 输入
    _box(ax, 0.03, 0.87, 0.94, 0.11, "CT volume  →  5 lobe ROIs", INK, 10)
    _arrow(ax, 0.25, 0.87, 0.25, 0.79, GREY, lw=1.0)
    _arrow(ax, 0.75, 0.87, 0.75, 0.79, GREY, lw=1.0)

    # R2 特征
    _box(ax, 0.03, 0.63, 0.44, 0.16, "Lesion features\ndensity + texture",
         ACC, 9.5)
    _box(ax, 0.53, 0.63, 0.44, 0.16, "Anatomy features\nvolume, sphericity",
         ORANGE, 9.5)
    _arrow(ax, 0.25, 0.63, 0.25, 0.56)
    _arrow(ax, 0.75, 0.63, 0.75, 0.56)

    # R3 打分头 / 解剖分支
    _box(ax, 0.03, 0.40, 0.44, 0.16, "Shared score head\n h  →  lobe score s",
         ACC, 9.5)
    _box(ax, 0.53, 0.40, 0.44, 0.16, "Anatomy branch\n predicts a", ORANGE, 9.5)

    # 对抗分支（右下方空隙），文字两行放进框内，不再与箭头重叠
    _box(ax, 0.42, 0.205, 0.56, 0.135,
         "Adversary:  uᵀh + c\n(gradient reversal)", RED, 9)
    ax.add_patch(FancyArrowPatch((0.47, 0.45), (0.52, 0.345), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.2, color=RED,
                                 linestyle="--", zorder=4))

    # R4 聚合与监督
    _box(ax, 0.03, 0.04, 0.44, 0.14, "Volume-weighted\naggregation", GREEN, 9.5)
    _box(ax, 0.53, 0.04, 0.44, 0.14, "Case severity y\n(only supervision)",
         GREEN, 9.5)
    _arrow(ax, 0.25, 0.40, 0.25, 0.19)
    _arrow(ax, 0.47, 0.11, 0.525, 0.11)

    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _side_of_three_lobe_lung(lb3):
    """三叶的那一侧肺（= 右肺）在图像的哪一边。返回 True 表示它落在图像左侧。"""
    # 用整个三维标签体计算（单张切片不一定五叶都出现）
    cols_r = [float(np.nonzero(lb3 == i)[2].mean()) for i in (1, 2, 3)
              if (lb3 == i).any()]
    cols_l = [float(np.nonzero(lb3 == i)[2].mean()) for i in (4, 5)
              if (lb3 == i).any()]
    if not cols_r or not cols_l:
        return False
    return np.mean(cols_r) < np.mean(cols_l)


def _best_axial_slice(vol, lobes, lesion):
    """选一张能同时看清最多肺叶、且病灶最多的轴位层。"""
    best, best_score = vol.shape[0] // 2, (-1, -1)
    for z in range(vol.shape[0]):
        n_lobe = len(set(np.unique(lobes[z]).tolist()) - {0})
        score = (n_lobe, int(lesion[z].sum()))
        if score > best_score:
            best, best_score = z, score
    return best


def _add_scale_bar(ax, sl, vox=20, color="#edf2f7"):
    """体素比例尺。仿真是按体素网格定义的、没有物理间距，因此标 voxel 而非 mm。"""
    h = sl.shape[0]
    x0, x1 = 3, 3 + vox
    y = h - 4
    ax.plot([x0, x1], [y, y], color=INK, linewidth=3.4, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.8, solid_capstyle="butt")
    for xx in (x0, x1):
        ax.plot([xx, xx], [y - 1.5, y + 1.5], color=INK, linewidth=3.4)
        ax.plot([xx, xx], [y - 1.5, y + 1.5], color=color, linewidth=1.8)
    ax.text((x0 + x1) / 2, y - 3.0, f"{vox} vox", color=color, fontsize=7,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#1a202c",
                      edgecolor="none", alpha=0.65))


def fig2_phantom(path):
    """第 3 轮补全：HU colorbar、体素比例尺、左右方位、五肺叶配色图例。

    三张图同为一张轴位切片，切片位置自动选为"可见肺叶数最多、且病灶最多"
    的那一层（本病例为 z=29，同时可见 RUL / RML / LUL，可见水平裂线）。
    """
    vol = np.load(os.path.join(DATA, PHANTOM + ".npy"))
    lobes = np.load(os.path.join(DATA, PHANTOM + "_lobes.npy"))
    lesion = np.load(os.path.join(DATA, PHANTOM + "_lesion.npy"))
    VMIN, VMAX = -1000, 200

    z = _best_axial_slice(vol, lobes, lesion)
    sl, lm, lb = vol[z], lesion[z], lobes[z]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), dpi=300)
    im = axes[0].imshow(sl, cmap="gray", vmin=VMIN, vmax=VMAX)
    axes[1].imshow(sl, cmap="gray", vmin=VMIN, vmax=VMAX)
    axes[1].imshow(np.ma.masked_where(lm == 0, lm), cmap="autumn",
                   alpha=0.85, vmin=0, vmax=1)
    cmap = matplotlib.colors.ListedColormap(LOBE_COLORS)
    norm = matplotlib.colors.BoundaryNorm(
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    axes[2].imshow(sl, cmap="gray", vmin=VMIN, vmax=VMAX)
    axes[2].imshow(np.ma.masked_where(lb == 0, lb), cmap=cmap, norm=norm,
                   alpha=0.65)

    right_on_left = _side_of_three_lobe_lung(lobes)
    lab_left, lab_right = ("R", "L") if right_on_left else ("L", "R")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.6); s.set_color("#a0aec0")
        a.text(0.04, 0.5, lab_left, transform=a.transAxes, fontsize=10,
               color="#edf2f7", ha="left", va="center", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.12", facecolor="#1a202c",
                         edgecolor="none", alpha=0.55))
        a.text(0.96, 0.5, lab_right, transform=a.transAxes, fontsize=10,
               color="#edf2f7", ha="right", va="center", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.12", facecolor="#1a202c",
                         edgecolor="none", alpha=0.55))
        _add_scale_bar(a, sl)
    for i, a in enumerate(axes):
        a.text(0.04, 0.96, f"({chr(97 + i)})", transform=a.transAxes,
               fontsize=9, color="#edf2f7", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.12", facecolor="#1a202c",
                         edgecolor="none", alpha=0.55))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.32, wspace=0.05)
    cax = fig.add_axes([0.26, 0.185, 0.44, 0.024])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("greyscale [HU]", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.outline.set_linewidth(0.6)

    handles = [Patch(facecolor=c, edgecolor="none", label=n)
               for c, n in zip(LOBE_COLORS, LOBE_NAMES)]
    handles.append(Patch(facecolor="#f28e2b", edgecolor="none", label="lesion"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.012),
               handlelength=1.2, columnspacing=1.0, handletextpad=0.4)

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig3_results(path, res, post):
    keys = ["ours", "b1b", "leaky", "b1", "b4", "b2"]
    labels = ["Ours", "B1b", "Leaky", "B1", "B4", "B2"]
    vals = [res[k]["per_lobe_mae"]["mean"] for k in keys]
    errs = [res[k]["per_lobe_mae"]["std"] for k in keys]
    b3 = res["baselines"]["B3_constant_case_severity"]["mean"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.24,
                        wspace=0.52)

    # (a) 六臂 MAE + 误差棒 + 不定位参考线
    ax = axes[0]
    ax.bar(range(len(keys)), vals, yerr=errs, capsize=3.5,
           color=[ACC, GREEN, "#d69e2e", ORANGE, "#805ad5", "#718096"],
           edgecolor=INK, linewidth=0.9, width=0.66)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Per-lobe MAE [pp]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.axhline(b3, color=RED, linestyle="--", linewidth=1.0)
    ax.text(len(keys) - 0.5, b3 + 0.10, "non-localizing ref.", fontsize=7,
            color=RED, ha="right", va="bottom")
    ax.set_ylim(0, b3 * 1.24)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("(a)", fontsize=8.5, loc="left")

    # (b) 每个 λ 下 Ours(λ) 相对匹配臂 B1b 的差值
    ax2 = axes[1]
    lam = post["lambda_gap_vs_b1b"]
    lv = sorted(lam, key=lambda k: lam[k]["lambda_adv"])
    xs = [lam[k]["lambda_adv"] for k in lv]
    ys = [lam[k]["gap_vs_b1b"] for k in lv]
    ps = [lam[k]["paired_test_vs_b1b"]["p_value"] for k in lv]
    ax2.plot(xs, ys, marker="o", color=ACC, linewidth=1.4, markersize=5)
    ax2.axhline(0.0, color=GREEN, linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
    ax2.set_xlabel(r"$\lambda_{\mathrm{adv}}$", fontsize=8.5)
    ax2.set_ylabel(r"Ours($\lambda$) $-$ B1b [pp]", fontsize=8.5)
    ax2.set_ylim(-0.003, max(ys) * 1.25)
    ax2.tick_params(labelsize=8)
    ax2.grid(alpha=0.3, linewidth=0.5)
    for x, y, p in zip(xs, ys, ps):
        ax2.annotate("n.s." if p >= 0.05 else "*", (x, y),
                     textcoords="offset points", xytext=(0, 7), ha="center",
                     fontsize=7, color=INK)
    ax2.text(0.03, 0.97, "0 = equal to\nno adversary", transform=ax2.transAxes,
             fontsize=6.8, color=GREEN, ha="left", va="top")
    ax2.set_title("(b)", fontsize=8.5, loc="left")

    # (c) 按解剖变异强度分层：Ours vs B1b（匹配对照），B1 作淡参考
    ax3 = axes[2]
    strat = post["stratified_by_anatomy_paired"]
    sk = sorted(strat, key=lambda k: strat[k]["anatomy_var"])
    xs = [strat[k]["anatomy_var"] for k in sk]
    ax3.plot(xs, [strat[k]["ours"] for k in sk], marker="o", color=ACC,
             linewidth=1.6, markersize=5, label="Ours (+adv)")
    ax3.plot(xs, [strat[k]["b1b"] for k in sk], marker="s", color=GREEN,
             linewidth=1.6, markersize=5, label="B1b (matched, no adv)")
    if "b1" in res and strat[sk[0]].get("b1") is not None:
        ax3.plot(xs, [strat[k]["b1"] for k in sk], marker="^", color=ORANGE,
                 linewidth=1.2, markersize=4, alpha=0.55,
                 label="B1 (confounded)")
    ax3.set_xlabel("anatomical variation strength", fontsize=8.5)
    ax3.set_ylabel("Per-lobe MAE [pp]", fontsize=8.5)
    ax3.tick_params(labelsize=8)
    ax3.grid(alpha=0.3, linewidth=0.5)
    ax3.legend(fontsize=7, frameon=False)
    ax3.set_title("(c)", fontsize=8.5, loc="left")

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rp = os.path.join(RESULTS, "results.json")
    if not os.path.exists(rp):
        raise SystemExit(f"未找到 {rp}，请先运行 src/evaluate.py")
    with open(rp, encoding="utf-8") as f:
        res = json.load(f)
    pp = os.path.join(RESULTS, "results_posthoc.json")
    if not os.path.exists(pp):
        raise SystemExit(f"未找到 {pp}，请先运行 src/analyze_posthoc.py")
    with open(pp, encoding="utf-8") as f:
        post = json.load(f)

    paths = [os.path.join(FIGDIR, "fig1_framework.pdf"),
             os.path.join(FIGDIR, "fig2_phantom.pdf"),
             os.path.join(FIGDIR, "fig3_results.pdf")]
    fig1_framework(paths[0])
    fig2_phantom(paths[1])
    fig3_results(paths[2], res, post)
    for p in paths:
        print(f"  {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()

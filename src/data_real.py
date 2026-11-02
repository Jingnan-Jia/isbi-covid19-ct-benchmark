#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real public CT dataset pipeline -> data/real_dataset.npz

Source (must be cited in the paper):
    COVID-19-CT-Seg dataset (Ma et al.), Zenodo record 3757476,
    licensed CC-BY 4.0. Lung & infection masks are part of the same release.
No patient-identifiable information is used; the data are a public, open-licensed
research dataset.

Per-slice samples, slice-level infection fraction label, and a fixed feature set.
No torch; only numpy + nibabel (+ a guarded pydicom import for completeness).
"""
import os
import glob

import numpy as np
import nibabel as nib
from scipy.stats import skew

# pydicom is listed in the requested stack but this dataset ships as NIfTI,
# so DICOM is not actually needed here. Import guarded so the script still
# runs where pydicom is absent.
try:
    import pydicom  # noqa: F401  (unused for NIfTI inputs)
except Exception:
    pydicom = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "real_dataset.npz")

DATASET_CITATION = (
    "COVID-19-CT-Seg (Ma et al.), Zenodo record 3757476, licensed CC-BY 4.0"
)

HU_MIN, HU_MAX = -1000.0, 400.0
N_HIST = 16
MIN_LUNG_AREA_RATIO = 0.05  # keep slices where lung covers >5% of the frame

# Feature layout (length 22):
#   [0:16]  lung-region HU histogram (16 bins, normalised to sum=1)
#   [16]    mean HU (lung)            [17] std HU (lung)
#   [18]    skewness (lung)           [19] mean gradient magnitude (lung)
#   [20]    lung-area ratio           [21] left/(left+right) lung-area ratio
ANATOMY_IDX = [20, 21]  # dropped for the "decoupled" control arm B


# --------------------------------------------------------------------------- #
def _find_ct_cases(raw_dir):
    """Return {case_id: path_to_case.nii.gz} by recursively searching raw_dir."""
    cases = {}
    for f in glob.glob(os.path.join(raw_dir, "**", "*.nii.gz"), recursive=True):
        base = os.path.basename(f)
        if "_CT_" in base:          # mask files contain '_CT_', skip
            continue
        case_id = base.replace(".nii.gz", "").replace(".nii", "")
        cases.setdefault(case_id, f)
    return cases


def _find_masks(raw_dir, case_id):
    """Return mask paths for a case.

    This release ships a single multi-label volume per case
    (0=background, 1=left lung, 2=right lung, 3=infection), so we look for
    that first and fall back to the three-file layout.
    """
    combined = os.path.join(raw_dir, f"{case_id}.nii.gz")
    if os.path.exists(combined):
        return {"combined": combined}
    out = {}
    for f in glob.glob(os.path.join(raw_dir, "**", f"{case_id}_CT_*.nii.gz"),
                       recursive=True):
        low = os.path.basename(f).lower()
        if "infection" in low:
            out["infection"] = f
        elif "left" in low:
            out["left"] = f
        elif "right" in low:
            out["right"] = f
    return out


def _to_slice_last(vol):
    """Move the smallest spatial axis (axial slices) to the last position."""
    ax = int(np.argmin(vol.shape))   # 512x512xN -> N is smallest
    return np.moveaxis(vol, ax, -1)


def load_case(case_id):
    """Load one case.

    Returns dict(ct, lung_mask, left_mask, right_mask, infection_mask, spacing).
    All volumes share shape (H, W, N) with N = number of axial slices.
    """
    ct_cases = _find_ct_cases(RAW)
    if case_id not in ct_cases:
        raise KeyError(f"case {case_id} not found under {RAW}")
    ct_img = nib.load(ct_cases[case_id])
    ct = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    spacing = np.asarray(ct_img.header.get_zooms(), dtype=np.float64)

    masks = _find_masks(RAW, case_id)
    if "combined" in masks:
        mv = np.asarray(nib.load(masks["combined"]).dataobj).astype(np.uint8)
        mv = _to_slice_last(mv)
        n = min(ct.shape[-1], mv.shape[-1])
        ct = ct[..., :n]
        mv = mv[..., :n]
        left = (mv == 1)
        right = (mv == 2)
        infection = (mv == 3)
        lung = left | right
    else:
        if not ({"left", "right", "infection"} <= set(masks)):
            raise FileNotFoundError(f"incomplete masks for {case_id}: {masks}")

        def _load(p):
            v = np.asarray(nib.load(p).dataobj, dtype=np.float32)
            return _to_slice_last(v)

        left = _load(masks["left"])
        right = _load(masks["right"])
        infection = _load(masks["infection"])
        n = min(ct.shape[-1], left.shape[-1], right.shape[-1], infection.shape[-1])
        ct = ct[..., :n]
        left = (left[..., :n] > 0)
        right = (right[..., :n] > 0)
        infection = (infection[..., :n] > 0)
        lung = left | right
    return {"ct": ct, "lung_mask": lung, "left_mask": left,
            "right_mask": right, "infection_mask": infection, "spacing": spacing}


def _features(slice_img, lung, left, right):
    vals = slice_img[lung]
    if vals.size == 0:
        return None
    hist, _ = np.histogram(vals, bins=N_HIST, range=(HU_MIN, HU_MAX))
    hist = hist / max(hist.sum(), 1)
    mean = float(vals.mean())
    std = float(vals.std())
    sk = float(skew(vals)) if std > 0 else 0.0
    gx, gy = np.gradient(slice_img)
    grad = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    grad_mean = float(grad[lung].mean())
    lung_ratio = float(lung.mean())
    lp = float(left.sum())
    rp = float(right.sum())
    lr_ratio = lp / (lp + rp) if (lp + rp) > 0 else 0.0
    return np.array([*hist, mean, std, sk, grad_mean, lung_ratio, lr_ratio],
                    dtype=np.float32)


def build_dataset():
    ct_cases = _find_ct_cases(RAW)
    case_ids = sorted(ct_cases.keys())
    X, y, cid, sidx = [], [], [], []
    for c in case_ids:
        d = load_case(c)
        ct = d["ct"]
        lung = d["lung_mask"]
        left = d["left_mask"]
        right = d["right_mask"]
        inf = d["infection_mask"]
        for z in range(ct.shape[-1]):
            lu = lung[..., z]
            if lu.mean() < MIN_LUNG_AREA_RATIO:
                continue
            # Labels are mutually exclusive in this release: 1=left lung,
            # 2=right lung, 3=infection. Infection voxels are therefore NOT
            # inside the lung mask, so the denominator must include them.
            inf_slice = inf[..., z]
            n_inf = float(inf_slice.sum())
            lab = n_inf / max(lu.sum() + n_inf, 1.0)
            lab = min(max(lab, 0.0), 1.0)
            f = _features(ct[..., z], lu, left[..., z], right[..., z])
            if f is None:
                continue
            X.append(f)
            y.append(lab)
            cid.append(c)
            sidx.append(z)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    cid = np.array(cid, dtype=object)
    sidx = np.array(sidx, dtype=np.int32)
    np.savez(OUT, X=X, y=y, case_id=cid, slice_idx=sidx)
    return X, y, cid, case_ids


def main():
    X, y, cid, case_ids = build_dataset()
    print(f"cases       : {len(case_ids)} ({', '.join(case_ids)})")
    print(f"slices kept : {X.shape[0]} (lung area > {MIN_LUNG_AREA_RATIO:.0%})")
    print(f"feature dim : {X.shape[1]}")
    print(f"label min/median/max : {y.min():.4f} / {np.median(y):.4f} / {y.max():.4f}")
    print(f"zero-label fraction : {(y == 0).mean():.3f}")
    print(f"anatomy cols dropped for arm B : {ANATOMY_IDX}")
    print(f"source      : {DATASET_CITATION}")
    print(f"written     : {OUT}")


if __name__ == "__main__":
    main()

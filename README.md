# A Reproducible Benchmark for Slice-Level Infection Burden Scoring on COVID-19 CT

Code, figures, and results for the ISBI 2027 submission.

## What this is

A small, reproducible benchmark that fixes two common problems in COVID-19 CT scoring studies:

1. **Slices are not independent samples.** 20 patients yield thousands of slices, but the effective sample size is 20. We therefore use a **fixed 4-fold split by patient**, not by slice.
2. **A pooled error hides ranking quality.** We report MAE and Spearman correlation with **case-level 95% bootstrap confidence intervals**.

The benchmark's first finding is a negative one: adding two anatomical features (lung-area ratio, left/right ratio) helps a ridge model slightly and *hurts* an MLP — so "anatomy helps" cannot be claimed without specifying the model.

## Data

[COVID-19 CT Lung and Infection Segmentation Dataset](https://doi.org/10.5281/zenodo.3757476) — Ma et al., **CC-BY 4.0**.

- 20 CT volumes, 2240 lung slices (lung area > 5%)
- Lung and infection masks annotated by two radiologists, verified by a third

Raw data is **not** included in this repository (1.1 GB). Download it from Zenodo and place it under `data/raw/`.

## Results (4-fold, by patient)

| Model | MAE (A) | MAE (B) | Spearman (A) | Spearman (B) |
|---|---|---|---|---|
| Ridge | 0.0924 | 0.0931 | **0.597** | 0.576 |
| MLP | 0.0658 | **0.0582** | 0.367 | **0.504** |
| Random Forest | 0.0685 | 0.0616 | 0.407 | — |
| Median baseline | 0.0776 | — | — | — |

A = all 22 features; B = without the two anatomical features. 1000 case-level bootstrap resamples.

## Repository layout

```
src/            feature extraction, training, evaluation, figure generation
paper/          LaTeX source, references, figures
results/        JSON outputs (authoritative numbers)
```

## Reproducing

```bash
# 1. Download the dataset from Zenodo (see Data) into data/raw/
# 2. Build features
python3 src/data_real.py
# 3. Train and evaluate (4-fold by patient)
python3 src/train_real.py
# 4. Figures
python3 src/make_figures_real.py
```

Requires: Python 3, numpy, scipy, matplotlib, nibabel.

## Status

Submitted to IEEE ISBI 2027. Code and splits will be released publicly; this repository stays private until the review process completes.

## Citation

```bibtex
@inproceedings{jia2027benchmark,
  title  = {A Reproducible Benchmark for Slice-Level Infection Burden Scoring on COVID-19 CT},
  author = {Jia, Jingnan and Wang, Lisi},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year   = {2027}
}
```

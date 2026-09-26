# uhpc-ml-transferability

Code for **Tabani, A., Biswas, R. — Transferability of Machine-Learning Models for UHPC Strength Prediction: Cross-Study Validation, Diagnosis and Calibration.**

The notebook `uhpc_ml_transferability.ipynb` reproduces the analyses of Sections 4.1–4.10 and 5.1 of the paper (Tables 3–10, Figures 3–14).

## Data

The notebook expects the following files in `data/` (not included in this repository):

| File | Source |
|---|---|
| `uhpc_v2_clean.xlsx` | Cleaned modelling table derived from Malik et al. (2025), Mendeley Data V2, https://doi.org/10.17632/czb7ww5pkz.2 (CC BY 4.0). Available from the corresponding author on reasonable request. |
| `uhpc_compressive_strength_1.csv`, `uhpc_flexural_strength_1.csv` | External database of Bolbolvand et al. (2025), *Construction and Building Materials* 493, 143135. Obtain from its authors. |

## Environment

```
conda env create -f environment.yml
conda activate uhpc-transfer
jupyter lab
```

## Notebook map

| Step | Paper output |
|---|---|
| 1 | Modelling sets (982 flexural / 89 studies; 243 tensile / 38 studies) and the 10 × 5 random and study-grouped fold assignments (Section 3.5) |
| 2 | Table 3 (model families and study-mean diagnostic); exposure statistics (Section 4.2) |
| 3 | Table 6, study-level split conformal (Section 4.7) |
| 4 | Table 7, few-shot calibration (Section 4.8) |
| 5 | Table 3, transparent baselines B1–B4 |
| 6 | Section 4.10, aspect-ratio encoding; Table 8, external transfer and overlap rules |
| 7 | Variance shares (Section 4.3), Spearman ranking (Section 4.8), Table 5 (SHAP) |
| 8 | Table 4 (learning curve), Section 4.5 (nested selection, ablation, compact few-shot) |
| 9 | Figures 3–14 |
| 10 | Table 9, nested tuning sensitivity (Section 4.10); about 25 min on 32 threads, 70 min on 12 |
| 11 | Tables 2A/2B and code relations on the external set (Table 8) |

All outputs are written to `output/`. Seeds are fixed (base seed 2026). Run time is about 60–75 min on a 12-thread CPU. Results were produced with the package versions in `environment.yml`; other versions may change the last decimal places.

## Licence

The code is released under the MIT licence.

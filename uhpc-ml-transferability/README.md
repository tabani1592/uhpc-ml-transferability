# uhpc-ml-transferability

Code for **Tabani, A., and R. Biswas. "Transferability of Machine-Learning Models for UHPC Strength Prediction: Cross-Study Validation, Diagnosis, and Calibration."** Submitted to *Journal of Materials in Civil Engineering* (ASCE).

## Quick start

```
conda env create -f environment.yml
conda activate uhpc-transfer
python run_paper.py      # all analyses, about 60-75 min on a 12-thread CPU
python verify_paper.py   # about 1 min; writes output/VERIFY_REPORT.txt
```

`run_paper.py` contains the same code as `uhpc_ml_transferability.ipynb` (Steps 0-11), with progress bars added. `verify_paper.py` draws Fig. 1 and compares 175 numbers quoted in the paper and supplement with the computed values (PASS/CHECK). With the package versions in `environment.yml`, 175 of 175 values reproduce the paper. Seeds are fixed (base seed 2026).

## Data (folder `data/`, not included)

| File | Source |
|---|---|
| `uhpc_v2_clean.xlsx` | Cleaned modelling table derived from Malik et al. (2025), Mendeley Data V2, https://doi.org/10.17632/czb7ww5pkz.2 (CC BY 4.0). Available from the corresponding author on reasonable request. |
| `uhpc_compressive_strength_1.csv`, `uhpc_flexural_strength_1.csv` | External database of Bolbolvand et al. (2025), *Construction and Building Materials* 493, 143135. Obtain from its authors. |

## Where each paper item comes from

| Paper item | Script step | Output file (in `output/`) |
|---|---|---|
| Table 1 (code relations) | Step 11 | `step11_code_relations.xlsx` (sheet Table2A) |
| Table 2 (protocols, diagnostic, baselines) | Steps 2, 5 | `step2_table3.xlsx`, `step5_baselines_all.csv` |
| Table 3 (few-shot calibration) | Step 4 | `step4_fewshot_per_rep.csv` |
| Table 4 (external transfer) | Steps 6, 11 | `step6b_external_overlap.xlsx`, `step11_code_relations.xlsx` (sheet external_codes) |
| Fig. 1 | verify_paper.py | `Fig1_paper_code_relations.png` |
| Fig. 2 | Step 9 | `Fig3_catboost_random_vs_grouped.png` |
| Fig. 3 | Step 9 | `Fig5_variance_share.png` |
| Fig. 4 | Step 9 | `Fig12_fewshot_per_study.png` |
| Fig. 5 | Step 9 | `Fig14_validation_hierarchy.png` |
| Table S2 (data description) | verify_paper.py | printed in `VERIFY_REPORT.txt` |
| Table S3 (minimum provisions) | Step 11 | `step11_code_relations.xlsx` (sheet Table2B) |
| Table S4 (nested tuning) | Step 10 | `step10_tuning_sensitivity.xlsx` |
| Table S5 (learning curve) | Step 8 | `step9_results.xlsx` (sheet learning_curve) |
| Table S6 (SHAP) | Step 7 | `step8_table5_shap.xlsx` |
| Table S7 (conformal intervals) | Step 3 | `step3_conformal_all.csv` |
| Fig. S2 | Step 9 | `Fig4_families_random_vs_grouped.png` |
| Fig. S3 | Step 9 | `Fig13_external_learning_ranking.png` |
| Fig. S4 | Step 9 | `Fig6_tensile_selection.png` |
| Fig. S5 | Step 9 | `Fig7_tensile_ablation.png` |
| Fig. S6 | Step 9 | `Fig9_shap_dependence_flex.png` |
| Fig. S7 | Step 9 | `Fig8_shap_summary_flex.png` |
| Fig. S8 | Step 9 | `Fig10_shap_summary_dts.png` |
| Fig. S9 | Step 9 | `Fig11_conformal_per_study.png` |
| Supplement, aspect-ratio encoding | Step 6 | `step6a_aspect_ratio.xlsx` |

Output file names keep the numbering of the original notebook; the table above maps them to the submitted paper. Table S1 (literature summary) and Fig. S1 (data-flow diagram) are not computed.

## Citation

See `CITATION.cff`. The code is released under the MIT licence.

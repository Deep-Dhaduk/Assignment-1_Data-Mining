# Assignment-1_Data-Mining

Chatgpt history link(exported as pdf as chat history link sharing option was not showing) :- [link text](https://drive.google.com/file/d/1QBeSL6Zh_kKYtIBl-ho8mmEaXL9tTRBX/view?usp=sharing)

Published Medium Article Link : [link text](https://medium.com/@dhadukdeep123/a-compact-crisp-dm-walkthrough-from-raw-csv-to-decisions-that-matter-9ddcd36f87e0)
# Animal Bites — CRISP-DM

A compact, reproducible pipeline for the Kaggle/Louisville **Animal Bites** dataset.  
Implements **cleaning → feature engineering → regression (with baselines) → clustering → outlier analysis → final report** in a single Python script.

## Quickstart
```bash
# 1) Install deps (Python 3.9+ recommended)
pip install -U pandas numpy scikit-learn scipy joblib

# 2) Run the pipeline (replace the path to your CSV)
python animal_bites_pipeline.py --data Animal_Bites.csv --outdir outputs --run all
```

Artifacts will be written under `outputs/`:
- `animal_bites_clean_all.parquet`, `animal_bites_model_strict.parquet`
- `model_selection_summary.csv`, `model_quarantine_days_best.joblib`
- `permutation_importance_top20.csv`
- `cluster_profile.csv`, `kmeans_labels.parquet`, `kmeans_internal_scores.csv`
- `outlier_strategies_compare.csv`, `outlier_flags_train.parquet`
- `model_card.md` and `final_report/` bundle

## Notes
- Uses **temporal split**, **OHE with min_frequency** (falls back gracefully), and a **GBRT** default.
- The **strict** feature set avoids post-outcome leakage.
- Outlier strategies compare: status quo vs. drop (multi-flag) vs. winsorized target vs. hybrid.
- Keep README short; see comments in the script for details.

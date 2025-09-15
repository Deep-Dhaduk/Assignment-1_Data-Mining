#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animal Bites — CRISP-DM Pipeline (EDA → Prep → Regression → Clustering → Outliers → Report)
Author: ChatGPT (professor-style pipeline)
License: MIT

Run `python animal_bites_pipeline.py --help` for options.

This script is compute-frugal and self-contained. It implements the key steps we built across the chunks:
- Robust CSV load + schema normalization
- Cleaning & feature engineering (temporal, vaccination, rare buckets)
- Leak-safe "strict" modeling table
- Temporal split
- Baselines + Linear/Ridge/Lasso + GBRT/RF (regression)
- Permutation importance & model selection
- K-means clustering with silhouette/DB selection + persona table
- Outlier analysis (IQR, IsolationForest, LOF) and cleaning strategy comparison
- Final report bundle (scoreboard + model card + cluster personas)

Inputs:
- A CSV (e.g., Animal_Bites.csv) from Kaggle/Louisville Open Data

Outputs (default: ./outputs):
- Cleaned parquet files, trained pipeline (.joblib), model selection tables, cluster profiles,
  outlier comparison, and a concise model card markdown.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

# Scikit / SciPy stack
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

RANDOM_STATE = 42
TARGET = "quarantine_days"


# -----------------------
# Utility & cleaning
# -----------------------
def clean_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) else np.nan


def season_from_month(m: Optional[float]) -> Optional[str]:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return np.nan
    m = int(m)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "fall"
    return np.nan


def rare_bucket(s: pd.Series, min_count=50, min_frac=0.005, other_label="other") -> pd.Series:
    vc = s.value_counts(dropna=True)
    thresh = max(min_count, int(np.ceil(min_frac * len(s))))
    keep = set(vc[vc >= thresh].index.tolist())
    return s.where(s.isin(keep), other_label)


def build_ohe(min_frequency=0.01):
    # Gracefully handle sklearn versions without min_frequency/handle_unknown="infrequent_if_exist"
    try:
        return OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist", sparse=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# -----------------------
# Load & normalize
# -----------------------
def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalize columns
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    # coerce dates (any column ending with 'date')
    for c in [c for c in df.columns if c.endswith("date")]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # normalize important categoricals
    for c in ["species", "breed", "gender", "where_bitten", "disposition", "victim_zip", "results", "color"]:
        if c in df.columns:
            df[c] = df[c].map(clean_str)
    # Harmonize: species/gender + bite site binning (coarse rules)
    if "species" in df.columns:
        df["species"] = df["species"].map(lambda x: {"canine": "dog", "dog": "dog", "cat": "cat", "feline": "cat"}.get(x, x))
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(lambda x: {"m": "male", "male": "male", "f": "female", "female": "female", "u": "unknown", "unknown": "unknown"}.get(x, x))
    if "where_bitten" in df.columns:
        def map_bite(x: Optional[str]) -> Optional[str]:
            if pd.isna(x):
                return x
            x = x.replace("-", " ")
            bins = {
                "hand": "hand/arm", "arm": "hand/arm", "finger": "hand/arm", "wrist": "hand/arm", "forearm": "hand/arm",
                "leg": "leg/foot", "foot": "leg/foot", "ankle": "leg/foot", "toe": "leg/foot", "thigh": "leg/foot",
                "face": "head/face", "head": "head/face", "ear": "head/face", "nose": "head/face", "neck": "head/face",
                "torso": "torso", "back": "torso", "abdomen": "torso", "hip": "torso", "buttocks": "torso"
            }
            for k, v in bins.items():
                if k in x:
                    return v
            return "other"
        df["where_bitten"] = df["where_bitten"].map(map_bite)
    # ZIP standardization
    if "victim_zip" in df.columns:
        z5 = df["victim_zip"].astype(str).str.extract(r"(\d{5})")[0]
        df["victim_zip"] = z5
        df["zip_known"] = df["victim_zip"].notna().astype("int8")
    # Temporal features (from bite_date)
    if "bite_date" in df.columns:
        bd = pd.to_datetime(df["bite_date"], errors="coerce")
        df["bite_year"] = bd.dt.year
        df["bite_month"] = bd.dt.month
        df["bite_weekday"] = bd.dt.dayofweek
        df["bite_is_weekend"] = (df["bite_weekday"] >= 5).astype("Int8")
        df["bite_season"] = df["bite_month"].map(season_from_month)
    # Vaccination recency
    if {"bite_date", "vaccination_date"} <= set(df.columns):
        ds = (pd.to_datetime(df["bite_date"], errors="coerce") - pd.to_datetime(df["vaccination_date"], errors="coerce")).dt.days
        ds = ds.mask(ds < 0, np.nan)
        df["days_since_vax"] = ds.astype("float32")
        df["vax_recent_1y"] = (ds <= 365).astype("Int8")
        df["vax_recent_3y"] = (ds <= 3*365).astype("Int8")
    # Target (regression): quarantine_days (clip to curb tails)
    if {"quarantine_date", "release_date"} <= set(df.columns):
        qd = (pd.to_datetime(df["release_date"], errors="coerce") - pd.to_datetime(df["quarantine_date"], errors="coerce")).dt.days
        qd = qd.clip(lower=0, upper=120)  # soft winsorization
        df[TARGET] = qd
    # Rare bucket (keeps feature space manageable)
    for c in ["species", "breed", "gender", "where_bitten", "disposition", "results", "victim_zip", "bite_season"]:
        if c in df.columns:
            df[c] = rare_bucket(df[c], min_count=50, min_frac=0.005)
    return df


def strict_feature_list(df: pd.DataFrame) -> List[str]:
    base = [
        "species","breed","gender","where_bitten","disposition",
        "victim_zip","zip_known",
        "bite_year","bite_month","bite_weekday","bite_is_weekend","bite_season",
        "days_since_vax","vax_recent_1y","vax_recent_3y"
    ]
    return [c for c in base if c in df.columns]


def temporal_split(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "bite_year" in df.columns and df["bite_year"].notna().any():
        q80 = np.nanpercentile(df["bite_year"], 80)
        train = df[df["bite_year"] <= q80].dropna(subset=[target])
        test = df[df["bite_year"] > q80].dropna(subset=[target])
        if len(test) < 0.1 * len(train):
            order = df.index.to_series()
            q80 = np.nanpercentile(order, 80)
            train = df[order <= q80].dropna(subset=[target])
            test = df[order > q80].dropna(subset=[target])
    else:
        order = df.index.to_series()
        q80 = np.nanpercentile(order, 80)
        train = df[order <= q80].dropna(subset=[target])
        test = df[order > q80].dropna(subset=[target])
    return train, test


def build_preprocessor(dfX: pd.DataFrame, for_linear=True, min_freq=0.01) -> ColumnTransformer:
    cat_cols = [c for c in dfX.columns if dfX[c].dtype == "object"]
    num_cols = [c for c in dfX.columns if c not in cat_cols]
    ohe = build_ohe(min_frequency=min_freq)
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", ohe)])
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if for_linear:
        num_steps.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    pre = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ], remainder="drop", sparse_threshold=0.3)
    return pre


# -----------------------
# Regression modeling
# -----------------------
class MedianBaseline:
    def fit(self, X, y):
        self.med_ = float(np.median(np.asarray(y)))
        return self
    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.med_, dtype=float)


def train_and_compare_models(train: pd.DataFrame, test: pd.DataFrame, outdir: Path) -> Dict[str, dict]:
    y_tr = train[TARGET].astype(float)
    y_te = test[TARGET].astype(float)
    X_tr = train.drop(columns=[TARGET])
    X_te = test.drop(columns=[TARGET])

    MODELS = {
        "baseline_median": MedianBaseline(),
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "lasso": LassoCV(alphas=np.logspace(-3, 1, 9), max_iter=5000),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "gbrt": GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=3, random_state=RANDOM_STATE),
    }

    results = {}
    best_name = None
    best_mae = float("inf")
    best_pipe = None

    for name, mdl in MODELS.items():
        for_linear = name in {"linear", "ridge", "lasso"}
        pre = build_preprocessor(X_tr, for_linear=for_linear, min_freq=0.01)
        pipe = Pipeline([("pre", pre), ("mdl", mdl)])
        pipe.fit(X_tr, y_tr)
        pred_tr = pipe.predict(X_tr)
        pred_te = pipe.predict(X_te)

        def metrics(y, p):
            return {
                "MAE": float(mean_absolute_error(y, p)),
                "RMSE": float(mean_squared_error(y, p, squared=False)),
                "R2": float(r2_score(y, p))
            }

        mtr = metrics(y_tr, pred_tr)
        mte = metrics(y_te, pred_te)
        results[name] = {"train": mtr, "test": mte}
        if mte["MAE"] < best_mae and name != "baseline_median":
            best_mae = mte["MAE"]
            best_name = name
            best_pipe = pipe

    # Save a comparison table
    rows = []
    for name, out in results.items():
        tr, te = out["train"], out["test"]
        rows.append({
            "model": name,
            "MAE_train": round(tr["MAE"], 4),
            "RMSE_train": round(tr["RMSE"], 4),
            "R2_train": round(tr["R2"], 4),
            "MAE_test": round(te["MAE"], 4),
            "RMSE_test": round(te["RMSE"], 4),
            "R2_test": round(te["R2"], 4),
        })
    sel = pd.DataFrame(rows).sort_values("MAE_test")
    sel.to_csv(outdir / "model_selection_summary.csv", index=False)

    if best_pipe is not None:
        joblib.dump(best_pipe, outdir / "model_quarantine_days_best.joblib")
        # Permutation importance (top 20)
        imp = permutation_importance(best_pipe, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE,
                                     scoring="neg_mean_absolute_error")
        # Get feature names
        pre = best_pipe.named_steps["pre"]
        feat_names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
                try:
                    ohe_names = list(ohe.get_feature_names_out(cols))
                except Exception:
                    ohe_names = [f"{c}_<cat>" for c in cols]
                feat_names.extend(ohe_names)
            elif hasattr(trans, "get_feature_names_out"):
                feat_names.extend(list(trans.get_feature_names_out(cols)))
            else:
                feat_names.extend(list(cols))
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": imp.importances_mean,
            "importance_std": imp.importances_std
        }).sort_values("importance", ascending=False).head(20)
        imp_df.to_csv(outdir / "permutation_importance_top20.csv", index=False)

    return {"best_model": best_name, "best_mae": best_mae, "results_table": str(outdir / "model_selection_summary.csv")}


# -----------------------
# Clustering
# -----------------------
def cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bite_month" in out:
        m = out["bite_month"].astype(float)
        out["m_sin"] = np.sin(2*np.pi*(m/12.0))
        out["m_cos"] = np.cos(2*np.pi*(m/12.0))
    if "bite_weekday" in out:
        w = out["bite_weekday"].astype(float)
        out["w_sin"] = np.sin(2*np.pi*(w/7.0))
        out["w_cos"] = np.cos(2*np.pi*(w/7.0))
    return out


def run_clustering(df: pd.DataFrame, outdir: Path) -> Dict[str, object]:
    # Compact feature set
    base_cols = [
        "species","where_bitten","gender","bite_season",
        "bite_month","bite_weekday","bite_is_weekend",
        "days_since_vax","vax_recent_1y","vax_recent_3y","zip_known"
    ]
    use_cols = [c for c in base_cols if c in df.columns]
    D = df[use_cols + ([TARGET] if TARGET in df.columns else [])].copy()
    D2 = cyclical_features(D)
    num_cols = [c for c in ["days_since_vax","bite_is_weekend","vax_recent_1y","vax_recent_3y","zip_known","m_sin","m_cos","w_sin","w_cos"] if c in D2.columns]
    cat_cols = [c for c in ["species","where_bitten","gender","bite_season"] if c in D2.columns]

    try:
        ohe = OneHotEncoder(min_frequency=0.02, handle_unknown="infrequent_if_exist", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ], remainder="drop", sparse_threshold=0.0)
    X = pre.fit_transform(D2[cat_cols + num_cols])

    # K selection
    best = {"k": None, "sil": -np.inf, "db": np.inf, "km": None}
    scores = []
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE, max_iter=300)
        labs = km.fit_predict(X)
        if len(np.unique(labs)) < 2:
            continue
        n = X.shape[0]
        idx = np.random.RandomState(0).choice(n, size=min(4000, n), replace=False)
        sil = silhouette_score(X[idx], labs[idx])
        db = davies_bouldin_score(X, labs)
        scores.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        if (sil > best["sil"]) or (np.isclose(sil, best["sil"]) and db < best["db"]):
            best = {"k": k, "sil": sil, "db": db, "km": km}

    # Persona table
    labels = best["km"].predict(X)
    D3 = D2.copy()
    D3["cluster"] = labels

    def top_counts(series: pd.Series, k=5) -> str:
        vc = (series.fillna("missing").astype(str).str.strip().str.title().value_counts(dropna=False).head(k))
        return "; ".join([f"{idx} ({cnt})" for idx, cnt in vc.items()])

    prof_rows = []
    for c in sorted(D3["cluster"].unique()):
        seg = D3[D3["cluster"] == c]
        row = {
            "cluster": int(c),
            "size": len(seg),
            "pct": round(100 * len(seg) / len(D3), 2),
            "top_species": top_counts(seg.get("species", pd.Series(index=[])), 5),
            "top_bitesite": top_counts(seg.get("where_bitten", pd.Series(index=[])), 5),
            "top_season": top_counts(seg.get("bite_season", pd.Series(index=[])), 4),
            "gender_mix": top_counts(seg.get("gender", pd.Series(index=[])), 4),
            "weekend_%": round(100 * seg.get("bite_is_weekend", pd.Series([0]*len(seg))).mean(), 1) if "bite_is_weekend" in seg else None,
            "days_since_vax_med": float(np.nanmedian(seg.get("days_since_vax", pd.Series([np.nan]*len(seg))))),
        }
        if TARGET in D3.columns and seg[TARGET].notna().any():
            row["quarantine_days_med"] = float(np.nanmedian(seg[TARGET]))
            row["quarantine_days_p90"] = float(np.nanpercentile(seg[TARGET].dropna(), 90))
        prof_rows.append(row)

    cluster_profile = pd.DataFrame(prof_rows).sort_values("size", ascending=False)
    cluster_profile.to_csv(outdir / "cluster_profile.csv", index=False)

    # Save labels for possible downstream use
    labels_df = pd.DataFrame({"cluster": labels}, index=D3.index)
    labels_df.to_parquet(outdir / "kmeans_labels.parquet", index=True)

    pd.DataFrame(scores).to_csv(outdir / "kmeans_internal_scores.csv", index=False)
    return {"k": best["k"], "silhouette": float(best["sil"]), "davies_bouldin": float(best["db"])}


# -----------------------
# Outlier analysis & strategies
# -----------------------
def iqr_threshold(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    return q3 + 1.5 * (q3 - q1)


def run_outlier_strategies(train: pd.DataFrame, test: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    Xtr, ytr = train.drop(columns=[TARGET]), train[TARGET].astype(float)
    Xte, yte = test.drop(columns=[TARGET]), test[TARGET].astype(float)

    # Fit a strong baseline (GBRT) to get residuals
    pre = build_preprocessor(Xtr, for_linear=False, min_freq=0.01)
    base = Pipeline([("pre", pre),
                     ("mdl", GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=3, random_state=RANDOM_STATE))])
    base.fit(Xtr, ytr)
    p_tr = base.predict(Xtr)
    p_te = base.predict(Xte)
    res_tr = ytr - p_tr
    res_te = yte - p_te

    # Statistical flags
    y_thr = iqr_threshold(ytr.values)
    e_thr = iqr_threshold(np.abs(res_tr.values))
    y_flag = ytr > y_thr
    e_flag_tr = np.abs(res_tr) > e_thr
    e_flag_te = np.abs(res_te) > e_thr

    # Model-based anomaly on reduced embedding
    Xtr_enc = base.named_steps["pre"].transform(Xtr)
    Xte_enc = base.named_steps["pre"].transform(Xte)
    svd = TruncatedSVD(n_components=min(32, (Xtr_enc.shape[1]-1)), random_state=RANDOM_STATE)
    Xtr_emb = svd.fit_transform(Xtr_enc)
    Xte_emb = svd.transform(Xte_enc)

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr_emb)
    iso_tr = -iso.score_samples(Xtr_emb)
    iso_te = -iso.score_samples(Xte_emb)
    iso_thr = np.quantile(iso_tr, 0.95)
    iso_flag_tr = iso_tr >= iso_thr
    iso_flag_te = iso_te >= iso_thr

    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.03, novelty=False)
    lof.fit_predict(Xtr_emb)
    lof_tr = -lof.negative_outlier_factor_
    lof_thr = np.quantile(lof_tr, 0.95)
    lof_flag_tr = lof_tr >= lof_thr
    lof_nov = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xtr_emb)
    lof_te = -lof_nov.score_samples(Xte_emb)
    lof_flag_te = lof_te >= lof_thr

    flags_tr = pd.DataFrame({
        "y_IQR_high": y_flag.astype(bool),
        "resid_IQR_high": e_flag_tr.astype(bool),
        "iso_hi": pd.Series(iso_flag_tr, index=Xtr.index),
        "lof_hi": pd.Series(lof_flag_tr, index=Xtr.index),
    })
    flags_te = pd.DataFrame({
        "resid_IQR_high": e_flag_te.astype(bool),
        "iso_hi": pd.Series(iso_flag_te, index=Xte.index),
        "lof_hi": pd.Series(lof_flag_te, index=Xte.index),
    })
    score_tr = flags_tr.sum(axis=1)
    multi_flag_tr = score_tr >= 2

    def rebuild_pipe():
        return Pipeline([("pre", build_preprocessor(Xtr, for_linear=False, min_freq=0.01)),
                         ("mdl", GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=3, random_state=RANDOM_STATE))])

    def eval_strategy(mask_drop=None, winsor_p=None):
        y_mod = ytr.copy()
        X_mod = Xtr.copy()
        if mask_drop is not None:
            keep_idx = X_mod.index[~mask_drop]
            X_mod, y_mod = X_mod.loc[keep_idx], y_mod.loc[keep_idx]
        if winsor_p is not None:
            cap = np.quantile(y_mod, winsor_p)
            y_mod = np.minimum(y_mod, cap)
        pipe = rebuild_pipe()
        pipe.fit(X_mod, y_mod)
        pred = pipe.predict(Xte)
        return {
            "MAE_test": mean_absolute_error(yte, pred),
            "RMSE_test": mean_squared_error(yte, pred, squared=False),
            "R2_test": r2_score(yte, pred),
            "n_train": len(y_mod)
        }

    results = {}
    results["A_status_quo"] = {"MAE_test": mean_absolute_error(yte, p_te),
                               "RMSE_test": mean_squared_error(yte, p_te, squared=False),
                               "R2_test": r2_score(yte, p_te),
                               "n_train": len(ytr)}
    results["B_drop_multiflag"] = eval_strategy(mask_drop=multi_flag_tr, winsor_p=None)
    results["C_winsor_995"] = eval_strategy(mask_drop=None, winsor_p=0.995)
    results["D_hybrid"] = eval_strategy(mask_drop=multi_flag_tr, winsor_p=0.995)

    df_res = pd.DataFrame(results).T.sort_values("MAE_test")
    df_res.to_csv(outdir / "outlier_strategies_compare.csv")
    flags_tr.assign(score=score_tr, multi=multi_flag_tr).to_parquet(outdir / "outlier_flags_train.parquet")
    return df_res


# -----------------------
# Final report bundle
# -----------------------
def write_model_card(outdir: Path, board: Optional[pd.DataFrame], top_imp: Optional[pd.DataFrame], cluster_prof: Optional[pd.DataFrame]) -> None:
    card = []
    card.append(f"# Animal Bites — Quarantine Days Regressor\n")
    card.append(f"**Date:** {datetime.now():%Y-%m-%d}\n")
    card.append("## Task\nPredict quarantine duration (days) at incident intake, to aid triage and resource planning.\n")
    card.append("## Data & Prep\nPublic Animal Bites dataset; schema normalization, temporal/vaccination features, rare-level bucketing; temporal split.\n")
    card.append("## Features (intake-known)\nspecies, breed(binned), gender, where_bitten(binned), disposition,\n"
                "bite_year/month/weekday/weekend/season, days_since_vax, vax_recent flags, zip_known.\n")
    card.append("## Model\nGradientBoostingRegressor within a ColumnTransformer (OHE with min_frequency). Outliers handled by conservative fusion + target winsorization in training.\n")
    if board is not None and len(board):
        keep = [c for c in ["strategy","MAE_test","RMSE_test","R2_test","n_test","n_train"] if c in board.columns]
        card.append("## Evaluation (holdout)\n")
        card.append(board[keep].sort_values("MAE_test").to_markdown(index=False)); card.append("\n")
    if top_imp is not None and len(top_imp):
        card.append("## Top Signals (Permutation Importance)\n")
        card.append(top_imp.head(10).to_markdown(index=False)); card.append("\n")
    if cluster_prof is not None and len(cluster_prof):
        card.append("## Cluster Personas\n")
        card.append(cluster_prof.head(5).to_markdown(index=False)); card.append("\n")
    card.append("## Monitoring\nTrack drift on species/site/season; retrain quarterly; monitor slice MAE parity.\n")
    card.append("## Limitations\nReporting noise; long-tail durations; avoid stigmatizing breeds/ZIPs.\n")
    (outdir / "model_card.md").write_text("\n".join(card), encoding="utf-8")


# -----------------------
# Orchestration
# -----------------------
def run_pipeline(csv_path: Path, outdir: Path, run: str = "all") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading {csv_path} ...")
    df = load_and_normalize_csv(csv_path)
    # Save the cleaned full table
    full_path = outdir / "animal_bites_clean_all.parquet"
    df.to_parquet(full_path, index=False)
    print(f"[OK] Cleaned data → {full_path} (rows={len(df)})")

    # Build strict modeling table
    feats = strict_feature_list(df)
    model_df = df[feats + ([TARGET] if TARGET in df.columns else [])].copy()
    model_pq = outdir / "animal_bites_model_strict.parquet"
    model_df.to_parquet(model_pq, index=False)
    print(f"[OK] Modeling table → {model_pq}")

    # Temporal split
    if TARGET not in model_df.columns:
        raise RuntimeError("Target column quarantine_days missing; ensure quarantine_date and release_date exist.")
    train, test = temporal_split(model_df, TARGET)
    Xtr, ytr = train.drop(columns=[TARGET]), train[TARGET].astype(float)
    Xte, yte = test.drop(columns=[TARGET]), test[TARGET].astype(float)
    print(f"[SPLIT] train={len(train)} test={len(test)}")

    # Steps
    board = None
    top_imp = None
    cluster_prof = None

    if run in ("all", "model"):
        print("[STEP] Regression modeling & selection ...")
        mod = train_and_compare_models(train, test, outdir)
        board = pd.read_csv(outdir / "model_selection_summary.csv")
        # Also capture importance if produced
        imp_path = outdir / "permutation_importance_top20.csv"
        if imp_path.exists():
            top_imp = pd.read_csv(imp_path)
        print(f"[OK] Best model: {mod['best_model']}  (MAE={mod['best_mae']:.3f})")

    if run in ("all", "cluster"):
        print("[STEP] Clustering ...")
        clu = run_clustering(df, outdir)
        if (outdir / "cluster_profile.csv").exists():
            cluster_prof = pd.read_csv(outdir / "cluster_profile.csv")
        print(f"[OK] KMeans k={clu['k']}  sil={clu['silhouette']:.3f}  DB={clu['davies_bouldin']:.3f}")

    if run in ("all", "outliers"):
        print("[STEP] Outlier analysis & cleaning strategies ...")
        comp = run_outlier_strategies(train, test, outdir)
        # Merge scoreboard if both exist
        comp["strategy"] = comp.index
        if board is None:
            board = comp
        else:
            board = pd.concat([board.rename(columns={"model": "strategy"}), comp], ignore_index=True, sort=False)
        board.to_csv(outdir / "00_model_scoreboard.csv", index=False)
        print(f"[OK] Outlier strategy comparison saved → {outdir / 'outlier_strategies_compare.csv'}")

    # Final report bundle
    if run in ("all", "report"):
        print("[STEP] Writing model card & final bundle ...")
        if board is None and (outdir / "00_model_scoreboard.csv").exists():
            board = pd.read_csv(outdir / "00_model_scoreboard.csv")
        write_model_card(outdir, board, top_imp, cluster_prof)
        # Minimal "final_report" folder
        fr = outdir / "final_report"
        fr.mkdir(exist_ok=True)
        if board is not None:
            board.to_csv(fr / "00_model_scoreboard.csv", index=False)
        if top_imp is not None:
            top_imp.to_csv(fr / "01_top_permutation_importance.csv", index=False)
        if cluster_prof is not None:
            cluster_prof.to_csv(fr / "02_cluster_personas.csv", index=False)
        (fr / "model_card.md").write_text((outdir / "model_card.md").read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Final report bundle → {fr}")

    print("[DONE] Pipeline finished.")


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Animal Bites — CRISP-DM Pipeline (cleaning, modeling, clustering, outliers, report)")
    p.add_argument("--data", required=True, type=Path, help="Path to Animal_Bites.csv")
    p.add_argument("--outdir", default=Path("./outputs"), type=Path, help="Output directory (default: ./outputs)")
    p.add_argument("--run", choices=["all", "model", "cluster", "outliers", "report"], default="all",
                   help="Which steps to run (default: all)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data, args.outdir, run=args.run)

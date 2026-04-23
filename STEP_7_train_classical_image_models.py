#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

DEFECTS = ["dropout", "banding", "weak_print", "geometry_distortion"]
CORE_FACTORS = ["dpi", "pulse_width", "voltage", "height_mm"]
META_COLS = {
    "experiment", "pattern_num", "pattern_name", "setting_id", "source_folder", "source_csv", "resolved_image_path",
    "image_id", "processed_image_file", "image_file", *CORE_FACTORS, *DEFECTS,
    *[f"{d}_present" for d in DEFECTS], *[f"{d}_severe" for d in DEFECTS],
    "defect_burden", "any_defect", "max_defect_severity"
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name(x) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(x).strip().lower()).replace(".", "p")


def numeric_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    all_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    image_cols = [c for c in all_num if c not in META_COLS and not c.endswith("_present") and not c.endswith("_severe")]
    meta_cols = [c for c in CORE_FACTORS if c in df.columns and df[c].nunique(dropna=True) > 1]

    if mode == "image_only":
        cols = image_cols
    elif mode == "metadata_only":
        cols = meta_cols
    elif mode == "image_plus_metadata":
        cols = image_cols + meta_cols
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    return [c for c in cols if df[c].notna().sum() > 0 and df[c].nunique(dropna=True) > 1]


def make_models(n_samples: int, n_pos: int) -> dict:
    # KNN cannot use more neighbours than the number of likely training samples.
    # 5 is a reasonable cap for this small dataset.
    k = max(1, min(5, n_samples - 1))
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]),
        "decision_tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)),
        ]),
        "knn": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=k)),
        ]),
    }


def get_cv(y: pd.Series, groups: pd.Series):
    n_groups = groups.nunique()
    pos_groups = pd.Series(groups[y == 1].unique()).nunique()
    neg_groups = pd.Series(groups[y == 0].unique()).nunique()

    n_splits = int(min(5, n_groups, pos_groups, neg_groups))
    if n_splits >= 2:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits, "StratifiedGroupKFold"

    n_splits = int(min(5, n_groups))
    if n_splits >= 2:
        return GroupKFold(n_splits=n_splits), n_splits, "GroupKFold_fallback"

    return None, 0, "no_valid_group_cv"


def metric_pack(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validate_model(model, X, y, groups, cv):
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            rows.append({"fold": fold, "skipped": 1, "reason": "single class in train or test fold"})
            continue

        # FIX: joblib has load/dump for files, not loads/dumps for in-memory cloning.
        # sklearn.base.clone safely creates a fresh unfitted estimator for each fold.
        m = clone(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train, y_train)

        pred = m.predict(X_test)
        rows.append({
            "fold": fold,
            "skipped": 0,
            **metric_pack(y_test, pred),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_positive_count": int(y_test.sum()),
        })
    return rows


def save_feature_importance(model, X, y, model_name, base_info):
    rows = []
    model = clone(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    if model_name in {"random_forest", "decision_tree"}:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            for f, imp in zip(X.columns, importances):
                rows.append({**base_info, "model": model_name, "feature": f, "importance": float(imp), "signed_effect": np.nan})

    elif model_name == "logistic_regression":
        coefs = model.named_steps["clf"].coef_[0]
        for f, coef in zip(X.columns, coefs):
            rows.append({**base_info, "model": model_name, "feature": f, "importance": abs(float(coef)), "signed_effect": float(coef)})

    return rows


def save_tree_rules(model, X, y, path: Path):
    try:
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        path.write_text(export_text(m, feature_names=list(X.columns), decimals=4), encoding="utf-8")
    except Exception as exc:
        path.write_text(f"Could not export tree rules: {exc}", encoding="utf-8")


def plot_importance(df: pd.DataFrame, out: Path, top_n: int = 12):
    ensure_dir(out)
    if df.empty:
        return

    for keys, g in df.groupby(["analysis_group", "feature_mode", "target", "model"]):
        if g["importance"].max() <= 0:
            continue
        top = g.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.35 * len(top))))
        plt.barh(top["feature"], top["importance"])
        plt.title(" | ".join(keys))
        plt.xlabel("importance")
        plt.tight_layout()
        plt.savefig(out / f"importance_{safe_name('_'.join(keys))}.png", dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="phase3_outputs/feature_tables/image_feature_table.csv")
    ap.add_argument("--output_dir", default="phase3_outputs")
    ap.add_argument("--feature_modes", default="image_only,image_plus_metadata,metadata_only")
    args = ap.parse_args()

    out = Path(args.output_dir)
    table_dir = out / "tables"
    model_dir = out / "models"
    rule_dir = model_dir / "tree_rules"
    imp_plot_dir = out / "plots" / "feature_importance"
    for p in [table_dir, model_dir, rule_dir, imp_plot_dir]:
        ensure_dir(p)

    df = pd.read_csv(args.features_csv)
    feature_modes = [x.strip() for x in args.feature_modes.split(",") if x.strip()]
    metric_rows, fold_rows, importance_rows, skipped = [], [], [], []

    # Primary analysis: separate experiment and pattern, matching the Phase 2 structure.
    for (exp, pattern), g in df.groupby(["experiment", "pattern_name"]):
        analysis_group = f"experiment_{int(exp)}_{pattern}"
        groups = g["setting_id"].astype(str)

        for feature_mode in feature_modes:
            feats = numeric_feature_columns(g, feature_mode)
            if not feats:
                skipped.append({"analysis_group": analysis_group, "feature_mode": feature_mode, "reason": "no usable features"})
                continue

            X = g[feats].replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True))

            for defect in DEFECTS:
                target = f"{defect}_present"
                y = g[target].astype(int)
                n_pos, n_neg = int(y.sum()), int((y == 0).sum())
                base = {
                    "analysis_group": analysis_group,
                    "experiment": exp,
                    "pattern_name": pattern,
                    "feature_mode": feature_mode,
                    "target": target,
                    "n_images": len(g),
                    "n_setting_combinations": groups.nunique(),
                    "n_features": len(feats),
                    "positive_count": n_pos,
                    "negative_count": n_neg,
                }

                if y.nunique() < 2 or n_pos < 3 or n_neg < 3:
                    skipped.append({**base, "reason": "too few positive or negative samples for defensible supervised modelling"})
                    continue

                cv, n_splits, cv_name = get_cv(y, groups)
                if cv is None:
                    skipped.append({**base, "reason": "not enough setting groups for grouped CV"})
                    continue

                models = make_models(len(g), n_pos)
                for model_name, model in models.items():
                    fold_metrics = cross_validate_model(model, X, y, groups, cv)
                    valid = [r for r in fold_metrics if not r.get("skipped", 0)]

                    for r in fold_metrics:
                        fold_rows.append({**base, "model": model_name, "cv": cv_name, **r})

                    if not valid:
                        skipped.append({**base, "model": model_name, "reason": "all folds skipped"})
                        continue

                    summary = {**base, "model": model_name, "cv": cv_name, "n_splits": n_splits, "valid_folds": len(valid)}
                    for metric in ["balanced_accuracy", "f1", "precision", "recall"]:
                        vals = [r[metric] for r in valid]
                        summary[f"{metric}_mean"] = float(np.mean(vals))
                        summary[f"{metric}_std"] = float(np.std(vals))

                    summary["tp_sum"] = int(sum(r["tp"] for r in valid))
                    summary["fp_sum"] = int(sum(r["fp"] for r in valid))
                    summary["tn_sum"] = int(sum(r["tn"] for r in valid))
                    summary["fn_sum"] = int(sum(r["fn"] for r in valid))
                    metric_rows.append(summary)

                    if model_name in ["logistic_regression", "decision_tree", "random_forest"] and feature_mode != "metadata_only":
                        importance_rows.extend(save_feature_importance(model, X, y, model_name, base))

                    if model_name == "decision_tree":
                        save_tree_rules(model, X, y, rule_dir / f"tree_{safe_name(analysis_group)}_{safe_name(feature_mode)}_{safe_name(target)}.txt")

    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    importance = pd.DataFrame(importance_rows)
    skipped_df = pd.DataFrame(skipped)

    metrics.to_csv(table_dir / "image_ml_model_metrics.csv", index=False)
    folds.to_csv(table_dir / "image_ml_fold_metrics.csv", index=False)
    skipped_df.to_csv(table_dir / "image_ml_skipped_models.csv", index=False)

    if not importance.empty:
        importance = importance.sort_values(
            ["analysis_group", "feature_mode", "target", "model", "importance"],
            ascending=[True, True, True, True, False],
        )
        importance.to_csv(table_dir / "image_ml_feature_importance.csv", index=False)
        importance.groupby(["analysis_group", "feature_mode", "target", "model"]).head(10).to_csv(table_dir / "image_ml_top_features.csv", index=False)
        plot_importance(importance, imp_plot_dir)
    else:
        pd.DataFrame().to_csv(table_dir / "image_ml_feature_importance.csv", index=False)
        pd.DataFrame().to_csv(table_dir / "image_ml_top_features.csv", index=False)

    notes = {
        "fix_applied": "Replaced joblib.loads(joblib.dumps(model)) with sklearn.base.clone(model).",
        "method": [
            "Primary evaluation is grouped by setting_id to prevent repeat leakage.",
            "Models are fitted separately by experiment and pattern.",
            "Targets are binary defect presence labels derived from the manual 0/1/2 severity labels.",
            "Dummy majority baseline is included so other models must beat a trivial classifier.",
            "Use balanced accuracy, F1, precision, and recall rather than raw accuracy.",
            "Feature importance is interpretive support, not causal proof.",
        ],
        "outputs": [
            "tables/image_ml_model_metrics.csv",
            "tables/image_ml_fold_metrics.csv",
            "tables/image_ml_skipped_models.csv",
            "tables/image_ml_feature_importance.csv",
            "models/tree_rules/*.txt",
        ],
    }
    (out / "step7_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print("STEP 7 complete")
    print(f"Metrics: {(table_dir / 'image_ml_model_metrics.csv').resolve()}")
    print(f"Skipped models: {(table_dir / 'image_ml_skipped_models.csv').resolve()}")


if __name__ == "__main__":
    main()

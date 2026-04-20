#!/usr/bin/env python3
"""
Repeatable metadata-first factor analysis for the labelled inkjet defect CSVs.

Run in VS Code terminal:
    python run_metadata_factor_analysis.py --input_dir . --output_dir outputs

The script:
1. loads Experiment_*_Pattern_*.csv files
2. builds an image-level master table
3. aggregates to one row per unique setting combination
4. makes Experiment 1 screening plots
5. makes Experiment 2 sensitivity heatmaps
6. fits light metadata-only models:
   logistic regression, decision tree, random forest

The currently uploaded file Experiment_3_Pattern_3.csv is treated as Experiment 2 Pattern 3.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

DEFECTS = ["dropout", "banding", "weak_print", "geometry_distortion"]
CORE_FACTORS = ["dpi", "pulse_width", "voltage", "height_mm"]
PATTERN_NAMES = {1: "stripes", 2: "qr", 3: "star"}
PATTERN_IDS_TO_NUMBERS = {"PATTERN001": 1, "PATTERN002": 2, "PATTERN003": 3}

EXPERIMENT_1_BASELINE = {"dpi": 600, "pulse_width": 2.2, "voltage": 9.0, "height_mm": 1}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_\.\-]+", "_", text)
    return text.replace(".", "p")


def parse_file_metadata(path: Path) -> tuple[int, int, str]:
    match = re.search(r"Experiment_(\d+)_Pattern_(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse filename: {path.name}")
    return int(match.group(1)), int(match.group(2)), ""


def load_all_csvs(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    csv_paths = sorted(input_dir.glob("Experiment_*_Pattern_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Experiment_*_Pattern_*.csv files found in {input_dir.resolve()}")

    frames = []
    notes = []

    for path in csv_paths:
        experiment, pattern_num_from_file, note = parse_file_metadata(path)
        if note:
            notes.append(f"{path.name}: {note}")

        df = pd.read_csv(path)
        missing = sorted(set(CORE_FACTORS + DEFECTS + ["review_complete"]) - set(df.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")

        df = df.copy()
        df["source_file"] = path.name
        df["experiment"] = experiment

        if "pattern_id" in df.columns:
            pattern_numbers = df["pattern_id"].map(PATTERN_IDS_TO_NUMBERS)
            pattern_num = int(pattern_numbers.dropna().iloc[0]) if pattern_numbers.notna().any() else pattern_num_from_file
        else:
            pattern_num = pattern_num_from_file

        df["pattern_num"] = pattern_num
        df["pattern_name"] = PATTERN_NAMES.get(pattern_num, f"pattern_{pattern_num}")

        for col in CORE_FACTORS + DEFECTS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["review_complete"] == 1].copy()

        for defect in DEFECTS:
            df[f"{defect}_present"] = (df[defect] > 0).astype(int)
            df[f"{defect}_severe"] = (df[defect] == 2).astype(int)

        df["defect_burden"] = df[DEFECTS].sum(axis=1)
        df["any_defect"] = (df["defect_burden"] > 0).astype(int)
        df["max_defect_severity"] = df[DEFECTS].max(axis=1)

        setting_str = (
            "E" + df["experiment"].astype(str)
            + "_P" + df["pattern_num"].astype(str)
            + "_dpi" + df["dpi"].astype(str)
            + "_pw" + df["pulse_width"].astype(str)
            + "_v" + df["voltage"].astype(str)
            + "_h" + df["height_mm"].astype(str)
        )
        df["setting_id"] = setting_str.map(safe_name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True), notes


def aggregate_combinations(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    group_cols = ["experiment", "pattern_num", "pattern_name"] + CORE_FACTORS

    if "QC" in master.columns:
        master["qc_handwriting"] = (master["QC"].astype(str).str.lower() == "handwriting").astype(int)
    else:
        master["qc_handwriting"] = 0

    agg = {
        "setting_id": "size",
        "qc_handwriting": "sum",
        "defect_burden": ["mean", "max"],
        "any_defect": "mean",
        "max_defect_severity": "mean",
    }
    for defect in DEFECTS:
        agg[defect] = ["mean", "max"]
        agg[f"{defect}_present"] = "mean"
        agg[f"{defect}_severe"] = "mean"

    combo = master.groupby(group_cols, dropna=False).agg(agg)
    combo.columns = ["_".join([str(x) for x in c if str(x)]).strip("_") for c in combo.columns]
    combo = combo.reset_index()

    combo = combo.rename(
        columns={
            "setting_id_size": "n_images",
            "qc_handwriting_sum": "n_handwriting",
            "defect_burden_mean": "mean_defect_burden",
            "defect_burden_max": "max_defect_burden",
            "any_defect_mean": "any_defect_rate",
            "max_defect_severity_mean": "mean_max_defect_severity",
        }
    )

    for defect in DEFECTS:
        combo = combo.rename(
            columns={
                f"{defect}_mean": f"{defect}_mean_severity",
                f"{defect}_max": f"{defect}_max_severity",
                f"{defect}_present_mean": f"{defect}_presence_rate",
                f"{defect}_severe_mean": f"{defect}_severe_rate",
            }
        )

    setting_str = (
        "E" + combo["experiment"].astype(str)
        + "_P" + combo["pattern_num"].astype(str)
        + "_dpi" + combo["dpi"].astype(str)
        + "_pw" + combo["pulse_width"].astype(str)
        + "_v" + combo["voltage"].astype(str)
        + "_h" + combo["height_mm"].astype(str)
    )
    combo["setting_id"] = setting_str.map(safe_name)
    return combo.sort_values(["experiment", "pattern_num"] + CORE_FACTORS).reset_index(drop=True)


def write_tables(master: pd.DataFrame, combo: pd.DataFrame, out_tables: Path) -> None:
    ensure_dir(out_tables)
    master.to_csv(out_tables / "master_image_level.csv", index=False)
    combo.to_csv(out_tables / "combination_level_summary.csv", index=False)

    overview = master.groupby(["experiment", "pattern_num", "pattern_name"]).agg(
        n_images=("setting_id", "size"),
        n_unique_setting_combinations=("setting_id", "nunique"),
        any_defect_rate=("any_defect", "mean"),
        mean_defect_burden=("defect_burden", "mean"),
        mean_max_defect_severity=("max_defect_severity", "mean"),
    ).reset_index()

    for defect in DEFECTS:
        extra = master.groupby(["experiment", "pattern_num", "pattern_name"])[f"{defect}_present"].mean().reset_index(name=f"{defect}_presence_rate")
        overview = overview.merge(extra, on=["experiment", "pattern_num", "pattern_name"], how="left")
    overview.to_csv(out_tables / "dataset_overview.csv", index=False)

    repeat_distribution = combo.groupby(["experiment", "pattern_num", "pattern_name"]).agg(
        n_setting_combinations=("setting_id", "size"),
        min_repeats=("n_images", "min"),
        median_repeats=("n_images", "median"),
        mean_repeats=("n_images", "mean"),
        max_repeats=("n_images", "max"),
    ).reset_index()
    repeat_distribution.to_csv(out_tables / "repeat_distribution.csv", index=False)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def nearly_equal(series: pd.Series, value: float, atol: float = 1e-9) -> pd.Series:
    return np.isclose(series.astype(float), float(value), atol=atol)


def plot_overview(master: pd.DataFrame, combo: pd.DataFrame, out_plots: Path) -> None:
    out = out_plots / "00_overview"
    ensure_dir(out)

    table = master.groupby(["experiment", "pattern_name"]).size().unstack("pattern_name").fillna(0).sort_index()
    ax = table.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Image-level sample count after QC")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Number of images")
    ax.legend(title="Pattern")
    savefig(out / "image_counts_by_experiment_and_pattern.png")

    repeats = combo["n_images"].dropna()
    plt.figure(figsize=(7, 5))
    plt.hist(repeats, bins=range(int(repeats.min()), int(repeats.max()) + 2))
    plt.title("Retained repeats per unique setting combination")
    plt.xlabel("Retained images per setting")
    plt.ylabel("Setting combinations")
    savefig(out / "retained_repeats_histogram.png")

    presence_cols = [f"{d}_presence_rate" for d in DEFECTS]
    prev = combo.groupby(["experiment", "pattern_name"])[presence_cols].mean().reset_index()
    for _, row in prev.iterrows():
        label = f"E{int(row['experiment'])}_{row['pattern_name']}"
        plt.figure(figsize=(8, 5))
        plt.bar(DEFECTS, [row[c] for c in presence_cols])
        plt.ylim(0, 1)
        plt.title(f"Combination-level defect prevalence: {label}")
        plt.ylabel("Mean presence rate")
        plt.xticks(rotation=25, ha="right")
        savefig(out / f"defect_prevalence_{safe_name(label)}.png")


def plot_experiment1_screening(combo: pd.DataFrame, out_plots: Path) -> None:
    exp1 = combo[combo["experiment"] == 1].copy()
    out = out_plots / "01_experiment1_screening"
    ensure_dir(out)

    for pattern_name, pattern_df in exp1.groupby("pattern_name"):
        pdir = out / safe_name(pattern_name)
        ensure_dir(pdir)

        for factor in CORE_FACTORS:
            mask = pd.Series(True, index=pattern_df.index)
            for other in CORE_FACTORS:
                if other != factor:
                    mask &= nearly_equal(pattern_df[other], EXPERIMENT_1_BASELINE[other])
            data = pattern_df[mask].sort_values(factor)
            if data.empty or data[factor].nunique() < 2:
                continue

            plt.figure(figsize=(8, 5))
            for defect in DEFECTS:
                plt.plot(data[factor], data[f"{defect}_mean_severity"], marker="o", label=defect)
            plt.title(f"Experiment 1 screening: {pattern_name}, factor = {factor}")
            plt.xlabel(factor)
            plt.ylabel("Mean severity")
            plt.legend()
            savefig(pdir / f"screening_mean_severity_vs_{factor}.png")

            plt.figure(figsize=(8, 5))
            for defect in DEFECTS:
                plt.plot(data[factor], data[f"{defect}_presence_rate"], marker="o", label=defect)
            plt.title(f"Experiment 1 presence rate: {pattern_name}, factor = {factor}")
            plt.xlabel(factor)
            plt.ylabel("Presence rate")
            plt.ylim(-0.05, 1.05)
            plt.legend()
            savefig(pdir / f"screening_presence_rate_vs_{factor}.png")

            plt.figure(figsize=(8, 5))
            plt.plot(data[factor], data["mean_defect_burden"], marker="o")
            plt.title(f"Experiment 1 overall burden: {pattern_name}, factor = {factor}")
            plt.xlabel(factor)
            plt.ylabel("Mean defect burden")
            savefig(pdir / f"screening_defect_burden_vs_{factor}.png")


def plot_heatmap(pivot: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values.astype(float), aspect="auto", origin="lower")
    plt.colorbar(label="Mean value")
    plt.title(title)
    plt.xlabel("pulse_width")
    plt.ylabel("voltage")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    savefig(path)


def plot_experiment2_heatmaps(combo: pd.DataFrame, out_plots: Path) -> None:
    exp2 = combo[combo["experiment"] == 2].copy()
    out = out_plots / "02_experiment2_heatmaps"
    ensure_dir(out)
    targets = ["mean_defect_burden"] + [f"{d}_mean_severity" for d in DEFECTS]

    for pattern_name, pattern_df in exp2.groupby("pattern_name"):
        pdir = out / safe_name(pattern_name)
        ensure_dir(pdir)
        for height in sorted(pattern_df["height_mm"].dropna().unique()):
            sub = pattern_df[nearly_equal(pattern_df["height_mm"], height)]
            if sub["pulse_width"].nunique() < 2 or sub["voltage"].nunique() < 2:
                continue
            for target in targets:
                pivot = sub.pivot_table(index="voltage", columns="pulse_width", values=target, aggfunc="mean").sort_index().sort_index(axis=1)
                title = f"{pattern_name}, Experiment 2, height={height}, {target}"
                plot_heatmap(pivot, title, pdir / f"heatmap_{safe_name(target)}_height_{safe_name(height)}.png")


def setting_weights(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("setting_id")["setting_id"].transform("count").astype(float)
    return 1.0 / counts


def fit_metadata_models(master: pd.DataFrame, out_tables: Path, out_models: Path) -> None:
    ensure_dir(out_tables)
    rule_dir = out_models / "decision_tree_rules"
    ensure_dir(rule_dir)

    ranking_rows = []
    metric_rows = []

    for (experiment, pattern_name), group in master.groupby(["experiment", "pattern_name"]):
        group = group.copy()
        analysis_group = f"experiment_{int(experiment)}_{pattern_name}"
        features = [c for c in CORE_FACTORS if group[c].nunique(dropna=True) > 1]
        if not features:
            continue
        X = group[features].astype(float)
        weights = setting_weights(group)

        for defect in DEFECTS:
            target = f"{defect}_present"
            y = group[target].astype(int)
            class_counts = y.value_counts().to_dict()
            base = {
                "analysis_group": analysis_group,
                "target": target,
                "n_images": len(group),
                "n_setting_combinations": group["setting_id"].nunique(),
                "features_used": ", ".join(features),
                "class_0_count": int(class_counts.get(0, 0)),
                "class_1_count": int(class_counts.get(1, 0)),
            }
            if y.nunique() < 2:
                metric_rows.append({**base, "model": "skipped", "apparent_balanced_accuracy": np.nan, "apparent_f1_macro": np.nan, "note": "Only one class present."})
                continue

            # Logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
            logreg.fit(X_scaled, y, sample_weight=weights)
            pred = logreg.predict(X_scaled)
            metric_rows.append({**base, "model": "logistic_regression", "apparent_balanced_accuracy": balanced_accuracy_score(y, pred), "apparent_f1_macro": f1_score(y, pred, average="macro", zero_division=0), "note": "Apparent fit only. Use for sanity check, not final proof."})
            for feature, coef in zip(features, logreg.coef_[0]):
                ranking_rows.append({**base, "model": "logistic_regression", "feature": feature, "importance": abs(float(coef)), "signed_effect": float(coef), "interpretation": "Positive signed_effect means higher feature value increases predicted defect probability."})

            # Decision tree
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=42)
            tree.fit(X, y, sample_weight=weights)
            pred = tree.predict(X)
            metric_rows.append({**base, "model": "decision_tree", "apparent_balanced_accuracy": balanced_accuracy_score(y, pred), "apparent_f1_macro": f1_score(y, pred, average="macro", zero_division=0), "note": "Apparent fit only. Read tree rules for interpretation."})
            for feature, imp in zip(features, tree.feature_importances_):
                ranking_rows.append({**base, "model": "decision_tree", "feature": feature, "importance": float(imp), "signed_effect": np.nan, "interpretation": "Unsigned. Inspect decision tree rule file for direction."})
            rule_file = rule_dir / f"tree_rules_{safe_name(analysis_group)}_{safe_name(target)}.txt"
            rule_file.write_text(export_text(tree, feature_names=features, decimals=4), encoding="utf-8")

            # Random forest
            forest = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
            forest.fit(X, y, sample_weight=weights)
            pred = forest.predict(X)
            metric_rows.append({**base, "model": "random_forest", "apparent_balanced_accuracy": balanced_accuracy_score(y, pred), "apparent_f1_macro": f1_score(y, pred, average="macro", zero_division=0), "note": "Apparent fit only. Use feature importance as a ranking, not causal proof."})
            for feature, imp in zip(features, forest.feature_importances_):
                ranking_rows.append({**base, "model": "random_forest", "feature": feature, "importance": float(imp), "signed_effect": np.nan, "interpretation": "Unsigned ranking. Do not treat as causal proof."})

    metrics = pd.DataFrame(metric_rows)
    rankings = pd.DataFrame(ranking_rows)
    metrics.to_csv(out_tables / "metadata_model_metrics.csv", index=False)
    rankings.sort_values(["analysis_group", "target", "model", "importance"], ascending=[True, True, True, False]).to_csv(out_tables / "metadata_model_feature_rankings.csv", index=False)
    rankings.sort_values(["analysis_group", "target", "model", "importance"], ascending=[True, True, True, False]).groupby(["analysis_group", "target", "model"]).head(3).to_csv(out_tables / "metadata_model_top3_features.csv", index=False)


def write_notes(notes: list[str], output_dir: Path) -> None:
    payload = {
        "important_method_points": [
            "Use combination_level_summary.csv as the main factor-analysis table.",
            "Use master_image_level.csv for traceability and later image ML.",
            "Models are fitted separately for each experiment and pattern.",
            "Model sample weights are 1 divided by retained repeat count for each setting, so each unique setting contributes equally.",
            "Model metrics are apparent fit metrics only. Use plots and feature rankings for interpretation.",
        ],
        "warnings": notes,
    }
    (output_dir / "run_notes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=".", help="Folder containing Experiment_*_Pattern_*.csv files")
    parser.add_argument("--output_dir", default="outputs", help="Folder to write outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    out_tables = output_dir / "tables"
    out_plots = output_dir / "plots"
    out_models = output_dir / "models"
    ensure_dir(output_dir)

    master, notes = load_all_csvs(input_dir)
    combo = aggregate_combinations(master)
    write_tables(master, combo, out_tables)
    plot_overview(master, combo, out_plots)
    plot_experiment1_screening(combo, out_plots)
    plot_experiment2_heatmaps(combo, out_plots)
    fit_metadata_models(master, out_tables, out_models)
    write_notes(notes, output_dir)

    print("Metadata factor analysis complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Main table: outputs/tables/combination_level_summary.csv")
    print("Model ranking: outputs/tables/metadata_model_feature_rankings.csv")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()

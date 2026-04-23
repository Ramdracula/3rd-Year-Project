#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFECTS = ["dropout", "banding", "weak_print", "geometry_distortion"]
CORE_FACTORS = ["dpi", "pulse_width", "voltage", "height_mm"]
PATTERN_NAMES = {1: "stripes", 2: "qr", 3: "star"}
DATASET_MAP = [
    (1, 1, "dataset_PATTERN001/processed_cropped"),
    (1, 2, "dataset_PATTERN002/processed_cropped"),
    (1, 3, "dataset_PATTERN003/processed_cropped"),
    (2, 1, "PATTERN001_Sensitivity/processed_cropped"),
    (2, 2, "PATTERN002_Sensitivity/processed_cropped"),
    (2, 3, "PATTERN003_Sensitivity/processed_cropped"),
]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_name(x) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(x).strip().lower()).replace(".", "p")

def find_csv(folder: Path) -> Path:
    for name in ["index_processed.csv", "index.csv"]:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No index_processed.csv or index.csv in {folder}")

def resolve_image(folder: Path, row: pd.Series) -> str:
    names = []
    for col in ["processed_image_file", "image_file", "image_id"]:
        if col in row.index and pd.notna(row[col]):
            names.append(str(row[col]).strip())
    for name in dict.fromkeys(names):
        p = folder / name
        candidates = [p]
        if not p.suffix:
            candidates += [folder / f"{name}{ext}" for ext in IMAGE_EXTS]
        for c in candidates:
            if c.exists() and c.is_file():
                return str(c)
    return ""

def setting_id(df: pd.DataFrame) -> pd.Series:
    s = (
        "E" + df["experiment"].astype(str)
        + "_P" + df["pattern_num"].astype(str)
        + "_dpi" + df["dpi"].astype(str)
        + "_pw" + df["pulse_width"].astype(str)
        + "_v" + df["voltage"].astype(str)
        + "_h" + df["height_mm"].astype(str)
    )
    return s.map(safe_name)

def load_rows(root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, warnings = [], []
    for exp, pat, rel in DATASET_MAP:
        folder = root / rel
        if not folder.exists():
            warnings.append(f"Missing folder: {folder}")
            continue
        csv_path = find_csv(folder)
        df = pd.read_csv(csv_path)
        missing = sorted(set(DEFECTS + CORE_FACTORS + ["review_complete"]) - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")
        df = df.copy()
        df["experiment"] = exp
        df["pattern_num"] = pat
        df["pattern_name"] = PATTERN_NAMES[pat]
        df["source_folder"] = str(folder)
        df["source_csv"] = str(csv_path)
        for c in DEFECTS + CORE_FACTORS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["review_complete"] == 1].copy()
        for d in DEFECTS:
            df[f"{d}_present"] = (df[d] > 0).astype(int)
            df[f"{d}_severe"] = (df[d] == 2).astype(int)
        df["defect_burden"] = df[DEFECTS].sum(axis=1)
        df["any_defect"] = (df["defect_burden"] > 0).astype(int)
        df["max_defect_severity"] = df[DEFECTS].max(axis=1)
        df["setting_id"] = setting_id(df)
        df["resolved_image_path"] = [resolve_image(folder, row) for _, row in df.iterrows()]
        missing_images = int((df["resolved_image_path"] == "").sum())
        if missing_images:
            warnings.append(f"{rel}: {missing_images} images could not be resolved")
        df = df[df["resolved_image_path"] != ""].copy()
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No usable labelled rows found.")
    return pd.concat(frames, ignore_index=True), warnings

def projection_features(ink: np.ndarray, gray: np.ndarray) -> dict:
    row_ink = ink.mean(axis=1)
    col_ink = ink.mean(axis=0)
    row_int = gray.mean(axis=1) / 255.0
    col_int = gray.mean(axis=0) / 255.0
    def fft(signal, name):
        signal = signal.astype(float) - float(np.mean(signal))
        spec = np.abs(np.fft.rfft(signal))[1:]
        if len(spec) == 0 or np.allclose(spec.sum(), 0):
            return {f"{name}_fft_dominant_ratio": 0.0, f"{name}_fft_energy": 0.0}
        return {f"{name}_fft_dominant_ratio": float(spec.max()/(spec.sum()+1e-9)),
                f"{name}_fft_energy": float(np.mean(spec**2))}
    out = {
        "row_ink_std": float(row_ink.std()), "col_ink_std": float(col_ink.std()),
        "row_ink_max": float(row_ink.max()), "col_ink_max": float(col_ink.max()),
        "row_intensity_std": float(row_int.std()), "col_intensity_std": float(col_int.std()),
    }
    out.update(fft(row_ink, "row_ink")); out.update(fft(col_ink, "col_ink"))
    out.update(fft(row_int, "row_intensity")); out.update(fft(col_int, "col_intensity"))
    return out

def lbp_features(gray: np.ndarray) -> dict:
    c = gray[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    neigh = [gray[:-2,:-2], gray[:-2,1:-1], gray[:-2,2:], gray[1:-1,2:],
             gray[2:,2:], gray[2:,1:-1], gray[2:,:-2], gray[1:-1,:-2]]
    for i, n in enumerate(neigh):
        code |= ((n >= c).astype(np.uint8) << i)
    hist, _ = np.histogram(code.ravel(), bins=16, range=(0,256), density=True)
    return {f"lbp_bin_{i:02d}": float(v) for i, v in enumerate(hist)}

def extract_features(path: str, resize: int) -> dict:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {path}")
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (resize, resize), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32) / 255.0
    otsu, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (inv > 0).astype(np.uint8)
    sobx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(sobx**2 + soby**2)
    edges = cv2.Canny(gray, 50, 150)
    nlab, labels, stats, cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float) if nlab > 1 else np.array([])
    total = float(resize*resize)
    hist, _ = np.histogram(gray.ravel(), bins=16, range=(0,256), density=True)
    f = {
        "image_width_original": gray0.shape[1], "image_height_original": gray0.shape[0],
        "mean_intensity": float(g.mean()), "std_intensity": float(g.std()),
        "p05_intensity": float(np.percentile(g,5)), "p50_intensity": float(np.percentile(g,50)),
        "p95_intensity": float(np.percentile(g,95)),
        "contrast_p95_p05": float(np.percentile(g,95)-np.percentile(g,5)),
        "otsu_threshold": float(otsu/255.0), "ink_fraction_otsu": float(ink.mean()),
        "dark_fraction_50": float((gray < 50).mean()), "dark_fraction_80": float((gray < 80).mean()),
        "dark_fraction_120": float((gray < 120).mean()), "light_fraction_200": float((gray > 200).mean()),
        "edge_density": float((edges > 0).mean()), "sobel_mean": float(sob.mean()),
        "sobel_std": float(sob.std()), "sobel_p95": float(np.percentile(sob,95)),
        "component_count": float(len(areas)), "small_component_count": float(np.sum(areas < 20)) if len(areas) else 0.0,
        "largest_component_area_frac": float(areas.max()/total) if len(areas) else 0.0,
        "mean_component_area_frac": float(areas.mean()/total) if len(areas) else 0.0,
    }
    f.update({f"intensity_hist_bin_{i:02d}": float(v) for i, v in enumerate(hist)})
    f.update(projection_features(ink, gray))
    f.update(lbp_features(gray))
    return f

def plot_class_balance(df: pd.DataFrame, out: Path) -> None:
    ensure_dir(out)
    for (exp, pattern), g in df.groupby(["experiment", "pattern_name"]):
        vals = [int(g[f"{d}_present"].sum()) for d in DEFECTS]
        plt.figure(figsize=(8,5)); plt.bar(DEFECTS, vals)
        plt.title(f"Positive defect counts: E{exp}, {pattern}")
        plt.ylabel("positive images"); plt.xticks(rotation=25, ha="right")
        plt.tight_layout(); plt.savefig(out / f"positive_counts_e{exp}_{safe_name(pattern)}.png", dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default=".")
    ap.add_argument("--output_dir", default="phase3_outputs")
    ap.add_argument("--resize", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.output_dir); feat_dir = out/"feature_tables"; table_dir = out/"tables"; plot_dir = out/"plots"/"class_balance"
    for p in [feat_dir, table_dir, plot_dir]: ensure_dir(p)
    rows, warnings = load_rows(Path(args.root_dir))
    feature_rows, failed = [], []
    meta_cols = ["experiment","pattern_num","pattern_name","setting_id","source_folder","source_csv","resolved_image_path",
                 "image_id","processed_image_file","image_file", *CORE_FACTORS, *DEFECTS,
                 *[f"{d}_present" for d in DEFECTS], *[f"{d}_severe" for d in DEFECTS],
                 "defect_burden","any_defect","max_defect_severity"]
    for _, r in rows.iterrows():
        try:
            feats = extract_features(r["resolved_image_path"], args.resize)
            meta = {c: r[c] for c in meta_cols if c in r.index}
            feature_rows.append({**meta, **feats})
        except Exception as e:
            failed.append({"path": r["resolved_image_path"], "error": str(e)})
    features = pd.DataFrame(feature_rows)
    features.to_csv(feat_dir/"image_feature_table.csv", index=False)
    overview = []
    for (exp, pattern), g in features.groupby(["experiment","pattern_name"]):
        row = {"experiment": exp, "pattern_name": pattern, "n_images": len(g),
               "n_setting_combinations": g["setting_id"].nunique(),
               "mean_defect_burden": g["defect_burden"].mean()}
        for d in DEFECTS:
            row[f"{d}_positive_count"] = int(g[f"{d}_present"].sum())
            row[f"{d}_positive_rate"] = float(g[f"{d}_present"].mean())
            row[f"{d}_severe_count"] = int(g[f"{d}_severe"].sum())
        overview.append(row)
    pd.DataFrame(overview).to_csv(table_dir/"image_ml_dataset_overview.csv", index=False)
    if failed: pd.DataFrame(failed).to_csv(table_dir/"failed_image_reads.csv", index=False)
    plot_class_balance(features, plot_dir)
    (out/"step6_notes.json").write_text(json.dumps({"warnings": warnings, "failed_image_reads": len(failed)}, indent=2))
    print("STEP 6 complete")
    print(f"Feature table: {(feat_dir/'image_feature_table.csv').resolve()}")
    print(f"Rows written: {len(features)}")
    if warnings:
        print("Warnings:"); [print(" -", w) for w in warnings]
    if failed:
        print(f"Failed image reads: {len(failed)}")
if __name__ == "__main__":
    main()

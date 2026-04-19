from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import shutil
import sys


ALLOWED_QC = {"keep", "handwriting"}


def find_paper_bbox(img):
    """
    Find bounding box of the visible white paper.
    Returns (x, y, w, h).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_h, img_w = gray.shape
    if w * h < 0.1 * img_w * img_h:
        return None

    return (x, y, w, h)


def crop_visible_paper(img):
    bbox = find_paper_bbox(img)
    if bbox is None:
        return None, None

    x, y, w, h = bbox
    cropped = img[y:y+h, x:x+w]
    return cropped, bbox


def copy_matching_json(image_path: Path, output_image_name: str, output_dir: Path):
    """
    Copy the JSON file with the same stem as the original image into output_dir,
    and rename it to match the processed cropped image stem.
    Example:
      original image: PATTERN001_000_xxx.jpg
      original json:  PATTERN001_000_xxx.json
      output image:   PATTERN001_000_xxx_crop.jpg
      output json:    PATTERN001_000_xxx_crop.json
    """
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        output_json_name = Path(output_image_name).with_suffix(".json").name
        shutil.copy2(json_path, output_dir / output_json_name)
        return output_json_name
    return ""


def process_dataset(dataset_dir: Path):
    csv_path = dataset_dir / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No index.csv found in {dataset_dir}")

    df = pd.read_csv(csv_path)

    if "QC" not in df.columns:
        raise ValueError("QC column not found in index.csv")
    if "image_file" not in df.columns:
        raise ValueError("image_file column not found in index.csv")

    df["QC"] = df["QC"].fillna("").astype(str).str.strip().str.lower()
    df_keep = df[df["QC"].isin(ALLOWED_QC)].copy()

    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows selected for processing: {len(df_keep)}")

    output_dir = dataset_dir / "processed_cropped"
    output_dir.mkdir(exist_ok=True)

    processed_rows = []

    for i, (_, row) in enumerate(df_keep.iterrows(), start=1):
        image_name = str(row["image_file"])
        image_path = dataset_dir / image_name

        if not image_path.exists():
            print(f"[{i}] Missing file: {image_name}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[{i}] Unreadable image: {image_name}")
            continue

        cropped, bbox = crop_visible_paper(img)
        if cropped is None:
            print(f"[{i}] Could not detect paper: {image_name}")
            continue

        out_name = Path(image_name).stem + "_crop.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), cropped)

        copied_json_name = copy_matching_json(image_path, out_name, output_dir)

        new_row = row.copy()
        new_row["processed_image_file"] = out_name
        new_row["processed_json_file"] = copied_json_name
        new_row["crop_x"] = bbox[0]
        new_row["crop_y"] = bbox[1]
        new_row["crop_w"] = bbox[2]
        new_row["crop_h"] = bbox[3]
        processed_rows.append(new_row)

        print(f"[{i}] Saved image: {out_name}")
        if copied_json_name:
            print(f"[{i}] Copied json: {copied_json_name}")

    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        processed_csv_path = output_dir / "index_processed.csv"
        processed_df.to_csv(processed_csv_path, index=False)
        print(f"\nProcessed CSV saved to: {processed_csv_path}")
    else:
        print("\nNo images were processed.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(r'python STEP_2_crop_to_paper.py "C:\Users\gauth\Ram_3YP\dataset_PATTERN001"')
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder not found: {dataset_dir}")

    process_dataset(dataset_dir)


if __name__ == "__main__":
    main()
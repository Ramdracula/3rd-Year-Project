from pathlib import Path
import pandas as pd
import cv2
import shutil
import sys


VALID_QC = {"k": "keep", "p": "partial", "h": "handwriting", "b": "blur"}
WINDOW_NAME = "QC Review"


def resolve_image_path(dataset_dir: Path, image_file: str) -> Path:
    """
    Resolve the image path from the CSV entry.
    Assumes image_file is usually just a filename like PATTERN001_001_xxx.jpg
    stored in the same folder as index.csv.
    """
    image_path = dataset_dir / str(image_file)
    return image_path


def backup_csv(csv_path: Path) -> None:
    backup_path = csv_path.with_name(csv_path.stem + "_backup.csv")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "image_file" not in df.columns:
        raise ValueError(f"'image_file' column not found in {csv_path}")

    if "QC" not in df.columns:
        df["QC"] = ""

    # Force string so blanks stay easy to handle
    df["QC"] = df["QC"].fillna("").astype(str)

    return df


def get_unreviewed_indices(df: pd.DataFrame):
    return df.index[df["QC"].str.strip() == ""].tolist()


def make_display_image(img, row, idx, total, dataset_name):
    """
    Resize image for display and overlay useful review info.
    """
    display = img.copy()

    # Resize large images to fit typical laptop screen better
    max_w = 1200
    max_h = 900
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    overlay_lines = [
        f"Dataset: {dataset_name}",
        f"Image {idx + 1} / {total}",
        "Keys: k=keep  p=partial  h=handwriting  b=blur",
        "Other: u=undo  q=quit"
    ]

    y = 30
    for line in overlay_lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        y += 35

    return display


def review_dataset(dataset_dir: Path):
    csv_path = dataset_dir / "index.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No index.csv found in {dataset_dir}")

    backup_csv(csv_path)
    df = load_csv(csv_path)

    total = len(df)
    dataset_name = dataset_dir.name
    history = []

    unreviewed = get_unreviewed_indices(df)
    if not unreviewed:
        print(f"All rows already reviewed in {csv_path}")
        return

    print(f"Starting review for: {dataset_name}")
    print(f"Rows in CSV: {total}")
    print(f"Remaining unreviewed: {len(unreviewed)}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(unreviewed):
        row_idx = unreviewed[i]
        row = df.loc[row_idx]
        image_path = resolve_image_path(dataset_dir, row["image_file"])

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            df.at[row_idx, "QC"] = "missing_file"
            df.to_csv(csv_path, index=False)
            i += 1
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Unreadable image: {image_path}")
            df.at[row_idx, "QC"] = "unreadable_file"
            df.to_csv(csv_path, index=False)
            i += 1
            continue

        display = make_display_image(img, row, row_idx, total, dataset_name)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFF
        key_char = chr(key).lower() if 32 <= key <= 126 else ""

        if key_char in VALID_QC:
            decision = VALID_QC[key_char]
            previous_value = df.at[row_idx, "QC"]
            df.at[row_idx, "QC"] = decision
            df.to_csv(csv_path, index=False)
            history.append((row_idx, previous_value))
            print(f"[{row_idx}] -> {decision}")
            i += 1

        elif key_char == "u":
            if history:
                last_row_idx, last_previous_value = history.pop()
                df.at[last_row_idx, "QC"] = last_previous_value
                df.to_csv(csv_path, index=False)

                # Step back one item in the review list if possible
                i = max(i - 1, 0)
                print(f"Undo applied to row {last_row_idx}")
            else:
                print("Nothing to undo.")

        elif key_char == "q":
            print("Quitting review. Progress saved.")
            break

        else:
            print("Invalid key. Use k, p, h, b, u, or q.")

    cv2.destroyAllWindows()
    print("Review session ended.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(r'python qc_review.py "C:\Users\gauth\Ram_3YP\dataset_PATTERN001"')
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder not found: {dataset_dir}")

    review_dataset(dataset_dir)


if __name__ == "__main__":
    main()

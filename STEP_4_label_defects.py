from pathlib import Path
import pandas as pd
import cv2
import shutil
import sys

WINDOW_NAME = "Defect Labelling"

DEFECT_KEYS = {
    "d": "dropout",
    "b": "banding",
    "w": "weak_print",
    "g": "geometry_distortion",
}

DEFECT_COLUMNS = list(DEFECT_KEYS.values())

# Red in BGR
TEXT_COLOUR = (0, 0, 255)


def find_csv(dataset_dir: Path) -> Path:
    processed_csv = dataset_dir / "index_processed.csv"
    normal_csv = dataset_dir / "index.csv"

    if processed_csv.exists():
        return processed_csv
    if normal_csv.exists():
        return normal_csv

    raise FileNotFoundError(f"No index_processed.csv or index.csv found in {dataset_dir}")


def backup_csv(csv_path: Path):
    backup_path = csv_path.with_name(csv_path.stem + "_labels_backup.csv")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "image_file" not in df.columns and "processed_image_file" not in df.columns:
        raise ValueError(
            f"Neither 'image_file' nor 'processed_image_file' found in {csv_path}"
        )

    for col in DEFECT_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    if "review_complete" not in df.columns:
        df["review_complete"] = 0

    for col in DEFECT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["review_complete"] = pd.to_numeric(
        df["review_complete"], errors="coerce"
    ).fillna(0).astype(int)

    return df


def get_display_filename(row) -> str:
    if "processed_image_file" in row.index:
        val = str(row["processed_image_file"]).strip()
        if val and val.lower() != "nan":
            return val

    if "image_file" in row.index:
        val = str(row["image_file"]).strip()
        if val and val.lower() != "nan":
            return val

    raise ValueError("No usable filename found in row.")


def resolve_image_path(dataset_dir: Path, row) -> Path:
    return dataset_dir / get_display_filename(row)


def get_unlabelled_indices(df: pd.DataFrame):
    return df.index[df["review_complete"] == 0].tolist()


def make_display_image(
    img,
    row,
    row_idx,
    total,
    dataset_name,
    current_labels,
    pending_defect,
    completed_count,
    remaining_count,
):
    display = img.copy()

    max_w = 1400
    max_h = 950
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    shown_filename = get_display_filename(row)

    lines = [
        f"Dataset: {dataset_name}",
        f"CSV row: {row_idx + 1} / {total}",
        f"Completed: {completed_count}",
        f"Remaining: {remaining_count}",
        f"File: {shown_filename}",
        "",
        "Defect keys:",
        "d=dropout, b=banding, w=weak_print, g=geometry_distortion",
        "Then press severity: 0, 1, or 2",
        "Enter=save/next, u=undo, q=quit",
        "",
        f"Pending defect: {pending_defect if pending_defect else 'None'}",
        "",
        "Current labels:",
        f"dropout = {current_labels['dropout']}",
        f"banding = {current_labels['banding']}",
        f"weak_print = {current_labels['weak_print']}",
        f"geometry_distortion = {current_labels['geometry_distortion']}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOUR,
            2,
            cv2.LINE_AA,
        )
        y += 30

    return display


def label_dataset(dataset_dir: Path):
    csv_path = find_csv(dataset_dir)
    backup_csv(csv_path)
    df = load_csv(csv_path)

    total = len(df)
    dataset_name = dataset_dir.name
    history = []

    if len(get_unlabelled_indices(df)) == 0:
        print("All images already reviewed.")
        return

    print(f"Using CSV: {csv_path}")
    print(f"Starting labelling for: {dataset_name}")
    print(f"Total rows: {total}")
    print(f"Remaining to review: {len(get_unlabelled_indices(df))}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        unlabelled = get_unlabelled_indices(df)
        if not unlabelled:
            break

        # Always take the first remaining unreviewed row
        row_idx = unlabelled[0]
        row = df.loc[row_idx]
        image_path = resolve_image_path(dataset_dir, row)

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            df.at[row_idx, "review_complete"] = -1
            df.to_csv(csv_path, index=False)
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Unreadable image: {image_path}")
            df.at[row_idx, "review_complete"] = -1
            df.to_csv(csv_path, index=False)
            continue

        current_labels = {
            "dropout": int(df.at[row_idx, "dropout"]),
            "banding": int(df.at[row_idx, "banding"]),
            "weak_print": int(df.at[row_idx, "weak_print"]),
            "geometry_distortion": int(df.at[row_idx, "geometry_distortion"]),
        }

        pending_defect = None
        row_history = []

        while True:
            completed_count = int((df["review_complete"] == 1).sum())
            remaining_count = int((df["review_complete"] == 0).sum())

            display = make_display_image(
                img=img,
                row=row,
                row_idx=row_idx,
                total=total,
                dataset_name=dataset_name,
                current_labels=current_labels,
                pending_defect=pending_defect,
                completed_count=completed_count,
                remaining_count=remaining_count,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(0) & 0xFF

            if key == 13:  # Enter
                old_values = []
                for col in DEFECT_COLUMNS:
                    old_values.append((col, int(df.at[row_idx, col])))
                    df.at[row_idx, col] = current_labels[col]

                old_review_complete = int(df.at[row_idx, "review_complete"])
                df.at[row_idx, "review_complete"] = 1
                df.to_csv(csv_path, index=False)

                history.append(
                    {
                        "row_idx": row_idx,
                        "old_values": old_values,
                        "old_review_complete": old_review_complete,
                    }
                )

                print(f"[{row_idx}] Saved: {current_labels}")
                break

            elif key == ord("q"):
                print("Quitting. Previous progress remains saved.")
                cv2.destroyAllWindows()
                return

            elif key == ord("u"):
                if pending_defect is not None:
                    pending_defect = None
                    print("Pending defect cleared.")

                elif row_history:
                    last_defect, last_old_value = row_history.pop()
                    current_labels[last_defect] = last_old_value
                    print(f"Undo: restored {last_defect} to {last_old_value}")

                elif history:
                    last_saved = history.pop()
                    prev_row_idx = last_saved["row_idx"]

                    for defect_name, old_value in last_saved["old_values"]:
                        df.at[prev_row_idx, defect_name] = old_value
                    df.at[prev_row_idx, "review_complete"] = last_saved["old_review_complete"]
                    df.to_csv(csv_path, index=False)

                    print(f"Returned previous image to unreviewed: row {prev_row_idx}")
                    break

                else:
                    print("Nothing to undo.")

            elif 32 <= key <= 126:
                ch = chr(key).lower()

                if ch in DEFECT_KEYS:
                    pending_defect = DEFECT_KEYS[ch]
                    print(f"Selected defect: {pending_defect}")

                elif ch in {"0", "1", "2"}:
                    if pending_defect is None:
                        print("Select defect key first: d, b, w, or g")
                    else:
                        sev = int(ch)
                        old_value = current_labels[pending_defect]
                        row_history.append((pending_defect, old_value))
                        current_labels[pending_defect] = sev
                        print(f"Set {pending_defect} = {sev}")
                        pending_defect = None

                else:
                    print("Invalid key.")

    cv2.destroyAllWindows()
    print("Labelling complete.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(r'python label_defects.py "C:\Users\gauth\Ram_3YP\dataset_PATTERN001\processed_cropped"')
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder not found: {dataset_dir}")

    label_dataset(dataset_dir)


if __name__ == "__main__":
    main()
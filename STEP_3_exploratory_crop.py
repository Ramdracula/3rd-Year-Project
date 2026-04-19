from pathlib import Path
import cv2
import numpy as np
import shutil
import sys


WHITE_THRESHOLD = 120
BLACK_THRESHOLD = 100
STEP_BACK = 20
MIN_RUN = 5
MIN_BLACK_IN_ROW = 5

TOP_BOTTOM_OFFSET = 0
ROW_MIDDLE_FRAC = 0.70

SIDE_ROW_OFFSET = 0
SIDE_ROW_FRAC = 0.90

SMOOTH_WINDOW = 9

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_white(v):
    return v > WHITE_THRESHOLD


def is_black(v):
    return v < BLACK_THRESHOLD


def get_middle_x_bounds(w, frac=ROW_MIDDLE_FRAC):
    keep_w = int(w * frac)
    x1 = max(0, (w - keep_w) // 2)
    x2 = min(w, x1 + keep_w)
    return x1, x2


def get_middle_y_bounds(h, frac=SIDE_ROW_FRAC, edge_offset=SIDE_ROW_OFFSET):
    usable_top = min(edge_offset, h - 1)
    usable_bottom = max(usable_top + 1, h - edge_offset)
    usable_h = usable_bottom - usable_top

    keep_h = int(usable_h * frac)
    y1 = usable_top + max(0, (usable_h - keep_h) // 2)
    y2 = min(usable_bottom, y1 + keep_h)
    return y1, y2


def smooth_1d(arr, window=SMOOTH_WINDOW):
    if window <= 1:
        return arr.astype(float)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def black_run_horizontal(gray, y, x, direction=1, min_run=MIN_RUN):
    h, w = gray.shape
    count = 0
    xx = x

    while 0 <= xx < w and count < min_run:
        if is_black(gray[y, xx]):
            count += 1
        else:
            break
        xx += direction

    return count >= min_run


def first_black_from_left_in_row(gray, y):
    h, w = gray.shape
    start = gray[y, 0]

    if is_white(start):
        for x in range(w):
            if is_black(gray[y, x]) and black_run_horizontal(gray, y, x, direction=1):
                return x

    elif is_black(start):
        seen_white = False
        for x in range(w):
            v = gray[y, x]
            if not seen_white:
                if is_white(v):
                    seen_white = True
            else:
                if is_black(v) and black_run_horizontal(gray, y, x, direction=1):
                    return x

    return None


def first_black_from_right_in_row(gray, y):
    h, w = gray.shape
    start = gray[y, w - 1]

    if is_white(start):
        for x in range(w - 1, -1, -1):
            if is_black(gray[y, x]) and black_run_horizontal(gray, y, x, direction=-1):
                return x

    elif is_black(start):
        seen_white = False
        for x in range(w - 1, -1, -1):
            v = gray[y, x]
            if not seen_white:
                if is_white(v):
                    seen_white = True
            else:
                if is_black(v) and black_run_horizontal(gray, y, x, direction=-1):
                    return x

    return None


def find_left_boundary(gray):
    hits = []
    h, w = gray.shape
    y1, y2 = get_middle_y_bounds(h)

    for y in range(y1, y2):
        x_hit = first_black_from_left_in_row(gray, y)
        if x_hit is not None:
            hits.append(x_hit)

    if not hits:
        return 0
    return max(0, min(hits) - STEP_BACK)


def find_right_boundary(gray):
    hits = []
    h, w = gray.shape
    y1, y2 = get_middle_y_bounds(h)

    for y in range(y1, y2):
        x_hit = first_black_from_right_in_row(gray, y)
        if x_hit is not None:
            hits.append(x_hit)

    if not hits:
        return w - 1
    return min(w - 1, max(hits) + STEP_BACK)


def compute_row_black_profile(gray):
    h, w = gray.shape
    x1, x2 = get_middle_x_bounds(w)
    profile = np.zeros(h, dtype=np.int32)

    for y in range(h):
        row = gray[y, x1:x2]
        profile[y] = np.count_nonzero(row < BLACK_THRESHOLD)

    return profile


def find_top_boundary(gray):
    h, w = gray.shape
    profile = compute_row_black_profile(gray)
    smooth_profile = smooth_1d(profile, SMOOTH_WINDOW)

    start_y = min(TOP_BOTTOM_OFFSET, h - 1)

    for y in range(start_y, h):
        if smooth_profile[y] >= MIN_BLACK_IN_ROW:
            return max(0, y - STEP_BACK)

    best_y = int(np.argmax(smooth_profile[start_y:])) + start_y
    return max(0, best_y - STEP_BACK)


def find_bottom_boundary(gray):
    h, w = gray.shape
    profile = compute_row_black_profile(gray)
    smooth_profile = smooth_1d(profile, SMOOTH_WINDOW)

    start_y = max(0, h - 1 - TOP_BOTTOM_OFFSET)

    for y in range(start_y, -1, -1):
        if smooth_profile[y] >= MIN_BLACK_IN_ROW:
            return min(h - 1, y + STEP_BACK)

    best_y = int(np.argmax(smooth_profile[:start_y + 1]))
    return min(h - 1, best_y + STEP_BACK)


def crop_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    left = find_left_boundary(gray)
    right = find_right_boundary(gray)
    top = find_top_boundary(gray)
    bottom = find_bottom_boundary(gray)

    h, w = gray.shape
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w - 1))
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - 1))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop bounds: left={left}, right={right}, top={top}, bottom={bottom}"
        )

    cropped = image_bgr[top:bottom + 1, left:right + 1]
    bounds = {"left": left, "right": right, "top": top, "bottom": bottom}
    return cropped, bounds


def copy_matching_json(image_path: Path, output_image_name: str, output_dir: Path):
    """
    Carry forward the JSON matching the input image stem,
    renamed to match the new output image stem.
    """
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        output_json_name = Path(output_image_name).with_suffix(".json").name
        shutil.copy2(json_path, output_dir / output_json_name)
        return output_json_name
    return ""


def process_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    output_dir = folder / "rowwise_tb_cropped"
    debug_dir = folder / "rowwise_tb_cropped_debug"
    output_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    image_files = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS]
    image_files.sort()

    if not image_files:
        print("No image files found.")
        return

    print(f"Found {len(image_files)} images")

    for i, image_path in enumerate(image_files, start=1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}")

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Could not read {image_path.name}")
            continue

        try:
            cropped, bounds = crop_image(img)

            out_name = f"{image_path.stem}_rowwiseTB{image_path.suffix}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), cropped)

            copied_json_name = copy_matching_json(image_path, out_name, output_dir)

            debug = img.copy()
            cv2.rectangle(
                debug,
                (bounds["left"], bounds["top"]),
                (bounds["right"], bounds["bottom"]),
                (0, 255, 0),
                3
            )
            debug_path = debug_dir / f"{image_path.stem}_debug{image_path.suffix}"
            cv2.imwrite(str(debug_path), debug)

            print(f"  Saved crop: {out_path.name}")
            if copied_json_name:
                print(f"  Copied json: {copied_json_name}")
            print(f"  Bounds: {bounds}")

        except Exception as e:
            print(f"  Failed on {image_path.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(r'python STEP_3_exploratory_crop.py "C:\Users\gauth\Ram_3YP\dataset_PATTERN001\processed_cropped"')
        sys.exit(1)

    process_folder(sys.argv[1])
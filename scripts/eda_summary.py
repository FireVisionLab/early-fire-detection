from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}
SMALL_BOX_AREA_THRESHOLD = 0.01


def read_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} has {len(parts)} values, expected 5")

        class_id_text, x_center_text, y_center_text, width_text, height_text = parts
        class_id = int(class_id_text)
        x_center = float(x_center_text)
        y_center = float(y_center_text)
        width = float(width_text)
        height = float(height_text)

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def validate_yolo_box(
    label_path: Path,
    line_number: int,
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> list[str]:
    errors = []

    if class_id not in CLASS_NAMES:
        errors.append(f"{label_path}:{line_number} unknown class id: {class_id}")

    values = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{label_path}:{line_number} {name}={value} is outside [0, 1]")

    if width <= 0:
        errors.append(f"{label_path}:{line_number} width must be positive")
    if height <= 0:
        errors.append(f"{label_path}:{line_number} height must be positive")

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
        errors.append(
            f"{label_path}:{line_number} box extends outside image bounds "
            f"(x_min={x_min:.4f}, y_min={y_min:.4f}, x_max={x_max:.4f}, y_max={y_max:.4f})"
        )

    return errors


def analyze_split(split: str) -> tuple[list[dict], list[str]]:
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    image_paths = sorted(image_dir.glob("*.jpg"))
    label_paths = sorted(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    errors = []
    for stem in sorted(image_stems - label_stems):
        errors.append(f"{split}: missing label for image {stem}.jpg")
    for stem in sorted(label_stems - image_stems):
        errors.append(f"{split}: missing image for label {stem}.txt")

    rows = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        if label_path.exists():
            boxes = read_yolo_label(label_path)
        else:
            boxes = []

        for line_number, box in enumerate(boxes, start=1):
            class_id, x_center, y_center, width, height = box
            errors.extend(
                validate_yolo_box(
                    label_path=label_path,
                    line_number=line_number,
                    class_id=class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            )

        rows.append(
            {
                "split": split,
                "image_name": image_path.name,
                "label_name": label_path.name,
                "image_width": image_width,
                "image_height": image_height,
                "box_count": len(boxes),
                "classes": [box[0] for box in boxes],
                "box_areas": [box[3] * box[4] for box in boxes],
            }
        )

    return rows, errors


def print_split_summary(df: pd.DataFrame, split: str) -> None:
    split_df = df[df["split"] == split]
    class_counts = Counter(class_id for classes in split_df["classes"] for class_id in classes)
    box_areas = [area for areas in split_df["box_areas"] for area in areas]
    small_box_count = sum(area < SMALL_BOX_AREA_THRESHOLD for area in box_areas)

    print(f"\n## {split}")
    print(f"images: {len(split_df)}")
    print(f"empty label files: {(split_df['box_count'] == 0).sum()}")
    print(f"total boxes: {sum(split_df['box_count'])}")
    print(f"average boxes per image: {split_df['box_count'].mean():.2f}")
    print(f"unique image sizes: {split_df[['image_width', 'image_height']].drop_duplicates().shape[0]}")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name} boxes: {class_counts[class_id]}")

    if box_areas:
        print(f"small boxes (< {SMALL_BOX_AREA_THRESHOLD:.0%} of image): {small_box_count}")
        print(f"small box ratio: {small_box_count / len(box_areas):.2%}")


def main() -> None:
    all_rows = []
    all_errors = []

    for split in SPLITS:
        rows, errors = analyze_split(split)
        all_rows.extend(rows)
        all_errors.extend(errors)

    df = pd.DataFrame(all_rows)

    for split in SPLITS:
        print_split_summary(df, split)

    print("\n## dataset")
    print(f"total images: {len(df)}")
    print(f"total boxes: {sum(df['box_count'])}")
    print(f"validation errors: {len(all_errors)}")

    if all_errors:
        print("\nFirst validation errors:")
        for error in all_errors[:20]:
            print(f"- {error}")


if __name__ == "__main__":
    main()

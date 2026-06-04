import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda_samples"
CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}
CLASS_COLORS = {
    0: "red",
    1: "dodgerblue",
}
SMALL_BOX_AREA_THRESHOLD = 0.01
RANDOM_SEED = 42


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    label_path: Path
    boxes: list[YoloBox]


def read_yolo_label(label_path: Path) -> list[YoloBox]:
    boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        class_id, x_center, y_center, width, height = line.split()
        boxes.append(
            YoloBox(
                class_id=int(class_id),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )

    return boxes


def collect_samples(split: str) -> list[Sample]:
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    samples = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_label(label_path)
        samples.append(
            Sample(
                split=split,
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
            )
        )

    return samples


def box_to_pixel_coordinates(box: YoloBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_min = (box.x_center - box.width / 2) * image_width
    y_min = (box.y_center - box.height / 2) * image_height
    box_width = box.width * image_width
    box_height = box.height * image_height

    return x_min, y_min, box_width, box_height


def draw_sample(ax: plt.Axes, sample: Sample) -> None:
    image = Image.open(sample.image_path).convert("RGB")
    image_width, image_height = image.size

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"{sample.split}/{sample.image_path.name}\n"
        f"{image_width}x{image_height}, boxes={len(sample.boxes)}",
        fontsize=9,
    )

    for box in sample.boxes:
        x_min, y_min, box_width, box_height = box_to_pixel_coordinates(
            box=box,
            image_width=image_width,
            image_height=image_height,
        )
        color = CLASS_COLORS[box.class_id]
        label = f"{CLASS_NAMES[box.class_id]} ({box.area:.3f})"

        rectangle = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rectangle)
        ax.text(
            x_min,
            max(y_min - 4, 0),
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )


def save_grid(samples: list[Sample], title: str, output_path: Path, columns: int = 3) -> None:
    if not samples:
        print(f"Skipping {title}: no samples found")
        return

    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, sample in zip(axes_list, samples):
        draw_sample(ax, sample)

    for ax in axes_list[len(samples):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")


def take_random(samples: list[Sample], count: int) -> list[Sample]:
    if len(samples) <= count:
        return samples

    return random.sample(samples, count)


def main() -> None:
    random.seed(RANDOM_SEED)
    train_samples = collect_samples("train")

    empty_label_samples = [sample for sample in train_samples if len(sample.boxes) == 0]
    fire_only_samples = [
        sample
        for sample in train_samples
        if sample.boxes and {box.class_id for box in sample.boxes} == {0}
    ]
    smoke_only_samples = [
        sample
        for sample in train_samples
        if sample.boxes and {box.class_id for box in sample.boxes} == {1}
    ]
    fire_and_smoke_samples = [
        sample
        for sample in train_samples
        if {box.class_id for box in sample.boxes} == {0, 1}
    ]
    small_box_samples = [
        sample
        for sample in train_samples
        if any(box.area < SMALL_BOX_AREA_THRESHOLD for box in sample.boxes)
    ]
    multi_box_samples = [sample for sample in train_samples if len(sample.boxes) >= 2]

    save_grid(take_random(fire_only_samples, 6), "Fire-only train samples", OUTPUT_DIR / "fire_only.png")
    save_grid(take_random(smoke_only_samples, 6), "Smoke-only train samples", OUTPUT_DIR / "smoke_only.png")
    save_grid(take_random(fire_and_smoke_samples, 6), "Fire and smoke train samples", OUTPUT_DIR / "fire_and_smoke.png")
    save_grid(take_random(small_box_samples, 6), "Small bounding box train samples", OUTPUT_DIR / "small_boxes.png")
    save_grid(take_random(multi_box_samples, 6), "Multiple bounding box train samples", OUTPUT_DIR / "multiple_boxes.png")
    save_grid(take_random(empty_label_samples, 6), "Empty-label train samples", OUTPUT_DIR / "empty_labels.png")


if __name__ == "__main__":
    main()

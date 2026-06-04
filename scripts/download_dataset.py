from pathlib import Path

import kagglehub


DATASET_HANDLE = "pengbo00/home-fire-dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(RAW_DATA_DIR),
    )

    print(f"Dataset downloaded to: {dataset_path}")


if __name__ == "__main__":
    main()

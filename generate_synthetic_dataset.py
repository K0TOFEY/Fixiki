"""Generate a reproducible, entirely fictional quality-control dataset."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_FOLDER = ROOT / "synthetic_test_data"
FEATURES_PATH = OUTPUT_FOLDER / "features.data"
LABELS_PATH = OUTPUT_FOLDER / "labels.data"
ROW_COUNT = 300
SENSOR_COUNT = 590
RANDOM_SEED = 20261127
SOURCE_ROW_START = 9_000_000_000_000_000
MODEL_SENSOR_INDICES = [59, 103, 247, 519, 477, 205, 129, 33, 341, 21]


def generate_dataset(output_folder=OUTPUT_FOLDER):
    """Write fake sensor readings and labels using the project's input formats."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    labels = np.where(np.arange(ROW_COUNT) % 5 == 0, 1, -1)
    centers = rng.uniform(-100, 1000, SENSOR_COUNT)
    scales = rng.uniform(0.2, 30, SENSOR_COUNT)
    readings = rng.normal(centers, scales, size=(ROW_COUNT, SENSOR_COUNT))

    for offset, sensor_index in enumerate(MODEL_SENSOR_INDICES):
        direction = 1 if offset % 2 == 0 else -1
        readings[labels == 1, sensor_index] += direction * 3.5 * scales[sensor_index]

    missing = rng.random(readings.shape) < 0.005
    readings[missing] = np.nan
    features_path = output_folder / "features.data"
    labels_path = output_folder / "labels.data"
    with features_path.open("w", encoding="utf-8") as features_file:
        for row in readings:
            features_file.write(
                " ".join("NaN" if np.isnan(value) else f"{value:.4f}" for value in row)
                + "\n"
            )

    start = datetime(2000, 1, 1, 8, 0)
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index, label in enumerate(labels):
            timestamp = start + timedelta(hours=2 * index)
            labels_file.write(
                f'{int(label)} "{timestamp.strftime("%d/%m/%Y %H:%M:%S")}"\n'
            )

    return features_path, labels_path


if __name__ == "__main__":
    features_file, labels_file = generate_dataset()
    print(f"Создано вымышленных строк: {ROW_COUNT}; датчиков: {SENSOR_COUNT}.")
    print(f"Файлы: {features_file} и {labels_file}.")

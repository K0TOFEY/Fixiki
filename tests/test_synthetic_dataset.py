from pathlib import Path

import functions
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_dataset_matches_sensor_schema_and_contains_fake_signal():
    sensors, labels = functions.load_inputs(
        ROOT / "synthetic_test_data" / "features.data",
        ROOT / "synthetic_test_data" / "labels.data",
    )
    defects = labels["label"] == 1
    _, original_times = functions.read_labels(ROOT / "labels.data")

    assert sensors.shape == (300, 590)
    assert int(defects.sum()) == 60
    assert labels["datetime"].min() > original_times.max()
    assert labels["datetime"].min() == original_times.max() + pd.Timedelta(hours=2)
    assert (
        sensors.loc[defects, "sensor_059"].mean()
        > sensors.loc[~defects, "sensor_059"].mean()
    )
    assert (
        sensors.loc[defects, "sensor_103"].mean()
        < sensors.loc[~defects, "sensor_103"].mean()
    )

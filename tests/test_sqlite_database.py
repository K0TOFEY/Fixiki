from pathlib import Path

import database


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_rows_append_to_sqlite_and_original_data_can_be_restored(tmp_path):
    database_file = tmp_path / "quality.sqlite3"
    original_features = ROOT / "features.data"
    original_labels = ROOT / "labels.data"
    synthetic_features = ROOT / "synthetic_test_data" / "features.data"
    synthetic_labels = ROOT / "synthetic_test_data" / "labels.data"

    original_rows, sensor_count = database.import_dataset(
        original_features, original_labels, database_file
    )
    assert original_rows == 1567
    assert sensor_count == 590

    database.append_dataset(
        synthetic_features,
        synthetic_labels,
        source_row_start=9_000_000_000_000_000,
        database_file=database_file,
    )
    status = database.get_dataset_status(database_file)
    loaded_sensors, loaded_labels = database.load_dataset(database_file)
    assert status["row_count"] == 1867
    assert loaded_sensors.shape == (1867, 590)
    _, original_times = database.functions.read_labels(original_labels)
    assert loaded_labels["datetime"].iloc[-1] > original_times.max()

    database.import_dataset(original_features, original_labels, database_file)
    restored_status = database.get_dataset_status(database_file)
    assert restored_status["row_count"] == original_rows
    assert restored_status["max_source_row"] == original_rows

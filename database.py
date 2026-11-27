"""SQLite storage for the quality-control dataset."""

import os
import sqlite3
from pathlib import Path

import pandas as pd

import functions

ROOT = Path(__file__).resolve().parent
TABLE_NAME = "products"


def database_path(path=None):
    """Return the configured SQLite database path."""
    if path is not None:
        return Path(path)
    return Path(os.getenv("DATABASE_PATH", ROOT / "data" / "quality.sqlite3"))


def database_available(path=None):
    """Report whether a SQLite file with the expected table is present."""
    target = database_path(path)
    if not target.is_file():
        return False
    with sqlite3.connect(target, timeout=5) as connection:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()
    return row is not None


def _sensor_columns(count):
    return [f"sensor_{index:03d}" for index in range(count)]


def _quoted(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _ensure_schema(connection, sensor_columns):
    columns = [
        '"source_row" INTEGER PRIMARY KEY',
        '"datetime" TEXT NOT NULL',
        '"label" INTEGER NOT NULL CHECK (label IN (-1, 1))',
        *(_quoted(name) + " REAL" for name in sensor_columns),
    ]
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {_quoted(TABLE_NAME)} ({', '.join(columns)})"
    )
    actual_columns = [
        (row[1], row[2].lower()) for row in connection.execute(
            f"PRAGMA table_info({_quoted(TABLE_NAME)})"
        )
    ]
    expected_columns = [
        ("source_row", "integer"),
        ("datetime", "text"),
        ("label", "integer"),
        *((name, "real") for name in sensor_columns),
    ]
    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Table {TABLE_NAME!r} exists with an unexpected schema; "
            "dataset import was stopped without changing its rows."
        )


def _read_input_files(features_path, labels_path):
    sensors = functions.read_features(features_path)
    labels, times = functions.read_labels(labels_path)
    if len(sensors) != len(labels):
        raise ValueError("Число строк датчиков и меток не совпадает.")
    if sensors.empty:
        raise ValueError("Набор данных пуст.")
    if not set(labels.unique()).issubset({-1, 1}):
        raise ValueError("Допустимые метки: 1 (брак) и -1 (годное).")
    return sensors, labels, times


def _row_values(sensors, labels, times, source_row_start):
    for row_index, (sensor_row, label, timestamp) in enumerate(
        zip(sensors.itertuples(index=False, name=None), labels, times),
        start=source_row_start,
    ):
        yield (
            row_index,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            int(label),
            *(
                None if pd.isna(value) else float(value)
                for value in sensor_row
            ),
        )


def _insert_rows(connection, sensors, labels, times, source_row_start):
    sensor_columns = _sensor_columns(sensors.shape[1])
    columns = ["source_row", "datetime", "label", *sensor_columns]
    insert_statement = (
        f"INSERT INTO {_quoted(TABLE_NAME)} "
        f"({', '.join(map(_quoted, columns))}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    connection.executemany(
        insert_statement,
        _row_values(sensors, labels, times, source_row_start),
    )
    return len(labels), len(sensor_columns)


def import_dataset(features_path, labels_path, database_file=None):
    """Atomically replace all database rows with a dataset from source files."""
    sensors, labels, times = _read_input_files(features_path, labels_path)
    target = database_path(database_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target, timeout=10) as connection:
        connection.execute("PRAGMA busy_timeout = 10000")
        _ensure_schema(connection, _sensor_columns(sensors.shape[1]))
        connection.execute(f"DELETE FROM {_quoted(TABLE_NAME)}")
        result = _insert_rows(connection, sensors, labels, times, 1)

    return result


def append_dataset(features_path, labels_path, source_row_start, database_file=None):
    """Append rows using reserved source-row IDs without replacing existing data."""
    if source_row_start < 1:
        raise ValueError("source_row_start must be a positive integer.")
    sensors, labels, times = _read_input_files(features_path, labels_path)
    target = database_path(database_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target, timeout=10) as connection:
        connection.execute("PRAGMA busy_timeout = 10000")
        _ensure_schema(connection, _sensor_columns(sensors.shape[1]))
        occupied = connection.execute(
            f"SELECT COUNT(*) FROM {_quoted(TABLE_NAME)} "
            "WHERE source_row >= ? AND source_row < ?",
            (source_row_start, source_row_start + len(labels)),
        ).fetchone()[0]
        if occupied:
            raise RuntimeError(
                "Reserved source_row range is already in use; no rows were added."
            )
        result = _insert_rows(connection, sensors, labels, times, source_row_start)

    return result


def load_dataset(database_file=None):
    """Read the dataset in chronological order for the Flask dashboard."""
    target = database_path(database_file)
    if not target.is_file():
        raise RuntimeError(f"SQLite database not found: {target}")

    with sqlite3.connect(target, timeout=10) as connection:
        connection.execute("PRAGMA busy_timeout = 10000")
        actual_columns = [
            row[1] for row in connection.execute(
                f"PRAGMA table_info({_quoted(TABLE_NAME)})"
            )
        ]
        if not actual_columns:
            raise RuntimeError(f"Table {TABLE_NAME!r} does not exist in {target}.")
        sensor_columns = [
            name for name in actual_columns
            if name not in {"source_row", "datetime", "label"}
        ]
        if not sensor_columns:
            raise RuntimeError(f"Table {TABLE_NAME!r} has no sensor columns.")
        selected_columns = ["source_row", "datetime", "label", *sensor_columns]
        rows = connection.execute(
            f"SELECT {', '.join(map(_quoted, selected_columns))} "
            f"FROM {_quoted(TABLE_NAME)} ORDER BY datetime, source_row"
        ).fetchall()

    if not rows:
        raise RuntimeError(f"Table {TABLE_NAME!r} contains no dataset rows.")
    labels = pd.DataFrame({
        "label": [row[2] for row in rows],
        "datetime": pd.to_datetime([row[1] for row in rows]),
    })
    sensors = pd.DataFrame([row[3:] for row in rows], columns=sensor_columns)
    return sensors, labels


def get_dataset_status(database_file=None):
    """Return a compact marker that changes when rows are added to the dataset."""
    target = database_path(database_file)
    if not target.is_file():
        raise RuntimeError(f"SQLite database not found: {target}")

    with sqlite3.connect(target, timeout=5) as connection:
        connection.execute("PRAGMA busy_timeout = 5000")
        row = connection.execute(
            f"SELECT COUNT(*), MAX(source_row), MAX(datetime) "
            f"FROM {_quoted(TABLE_NAME)}"
        ).fetchone()

    row_count, max_source_row, max_datetime = row
    return {
        "row_count": row_count,
        "max_source_row": max_source_row,
        "max_datetime": max_datetime,
    }


def main():
    row_count, sensor_count = import_dataset(
        ROOT / "features.data",
        ROOT / "labels.data",
    )
    print(f"Загружено строк: {row_count}; датчиков: {sensor_count}.")


if __name__ == "__main__":
    main()

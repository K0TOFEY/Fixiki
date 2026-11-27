"""PostgreSQL storage for the quality-control dataset."""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

import functions

TABLE_NAME = "quality_products"


def _sensor_columns(count):
    return [f"sensor_{index:03d}" for index in range(count)]


def _expected_columns(sensor_columns):
    return [
        ("source_row", "bigint"),
        ("datetime", "timestamp without time zone"),
        ("label", "smallint"),
        *((name, "double precision") for name in sensor_columns),
    ]


def _ensure_schema(cursor, sensor_columns):
    definitions = [
        sql.SQL("{} BIGINT PRIMARY KEY").format(sql.Identifier("source_row")),
        sql.SQL("{} TIMESTAMP WITHOUT TIME ZONE NOT NULL").format(sql.Identifier("datetime")),
        sql.SQL("{} SMALLINT NOT NULL CHECK (label IN (-1, 1))").format(sql.Identifier("label")),
        *(
            sql.SQL("{} DOUBLE PRECISION").format(sql.Identifier(name))
            for name in sensor_columns
        ),
    ]
    cursor.execute(
        sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(TABLE_NAME), sql.SQL(", ").join(definitions)
        )
    )
    cursor.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = %s
        ORDER BY ordinal_position
        """,
        (TABLE_NAME,),
    )
    actual_columns = cursor.fetchall()
    expected_columns = _expected_columns(sensor_columns)
    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Table {TABLE_NAME!r} exists with an unexpected schema; "
            "dataset import was stopped without changing its rows."
        )


def import_dataset(features_path, labels_path, database_url=None):
    """Replace the PostgreSQL dataset with the matching pair of source files."""
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before importing the dataset.")

    sensors = functions.read_features(features_path)
    labels, times = functions.read_labels(labels_path)
    if len(sensors) != len(labels):
        raise ValueError("Число строк датчиков и меток не совпадает.")
    if sensors.empty:
        raise ValueError("Набор данных пуст.")
    if not set(labels.unique()).issubset({-1, 1}):
        raise ValueError("Допустимые метки: 1 (брак) и -1 (годное).")

    sensor_columns = _sensor_columns(sensors.shape[1])
    columns = ["source_row", "datetime", "label", *sensor_columns]
    values = [
        (
            row_index,
            timestamp.to_pydatetime(),
            int(label),
            *(
                None if pd.isna(value) else float(value)
                for value in sensor_row
            ),
        )
        for row_index, (sensor_row, label, timestamp) in enumerate(
            zip(sensors.itertuples(index=False, name=None), labels, times), start=1
        )
    ]

    with psycopg2.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            _ensure_schema(cursor, sensor_columns)
            cursor.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(TABLE_NAME)))
            insert_statement = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                sql.Identifier(TABLE_NAME),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
            ).as_string(connection)
            execute_values(cursor, insert_statement, values, page_size=25)

    return len(values), len(sensor_columns)


def append_dataset(features_path, labels_path, source_row_start, database_url=None):
    """Append a dataset while assigning source-row IDs from a reserved range."""
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before importing the dataset.")
    if source_row_start < 1:
        raise ValueError("source_row_start must be a positive integer.")

    sensors = functions.read_features(features_path)
    labels, times = functions.read_labels(labels_path)
    if len(sensors) != len(labels):
        raise ValueError("Число строк датчиков и меток не совпадает.")
    if sensors.empty:
        raise ValueError("Набор данных пуст.")
    if not set(labels.unique()).issubset({-1, 1}):
        raise ValueError("Допустимые метки: 1 (брак) и -1 (годное).")

    sensor_columns = _sensor_columns(sensors.shape[1])
    columns = ["source_row", "datetime", "label", *sensor_columns]
    values = [
        (
            row_index,
            timestamp.to_pydatetime(),
            int(label),
            *(
                None if pd.isna(value) else float(value)
                for value in sensor_row
            ),
        )
        for row_index, (sensor_row, label, timestamp) in enumerate(
            zip(sensors.itertuples(index=False, name=None), labels, times),
            start=source_row_start,
        )
    ]

    with psycopg2.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            _ensure_schema(cursor, sensor_columns)
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {} WHERE source_row >= %s AND source_row < %s").format(
                    sql.Identifier(TABLE_NAME)
                ),
                (source_row_start, source_row_start + len(values)),
            )
            if cursor.fetchone()[0]:
                raise RuntimeError(
                    "Reserved source_row range is already in use; no rows were added."
                )
            insert_statement = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                sql.Identifier(TABLE_NAME),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
            ).as_string(connection)
            execute_values(cursor, insert_statement, values, page_size=25)

    return len(values), len(sensor_columns)


def load_dataset(database_url=None):
    """Read the dataset in chronological order for the Flask dashboard."""
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before loading the PostgreSQL dataset.")

    with psycopg2.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                  AND column_name NOT IN ('source_row', 'datetime', 'label')
                ORDER BY ordinal_position
                """,
                (TABLE_NAME,),
            )
            sensor_columns = [row[0] for row in cursor.fetchall()]
            if not sensor_columns:
                raise RuntimeError(f"Table {TABLE_NAME!r} has no sensor columns.")
            selected_columns = ["source_row", "datetime", "label", *sensor_columns]
            cursor.execute(
                sql.SQL("SELECT {} FROM {} ORDER BY datetime, source_row").format(
                    sql.SQL(", ").join(map(sql.Identifier, selected_columns)),
                    sql.Identifier(TABLE_NAME),
                )
            )
            rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(f"Table {TABLE_NAME!r} contains no dataset rows.")
    labels = pd.DataFrame({
        "label": [row[2] for row in rows],
        "datetime": [row[1] for row in rows],
    })
    sensors = pd.DataFrame([row[3:] for row in rows], columns=sensor_columns)
    return sensors, labels


def get_dataset_status(database_url=None):
    """Return a compact marker that changes when rows are added to the dataset."""
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before checking the PostgreSQL dataset.")

    with psycopg2.connect(database_url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT COUNT(*), MAX(source_row), MAX(datetime) FROM {}"
                ).format(sql.Identifier(TABLE_NAME))
            )
            row_count, max_source_row, max_datetime = cursor.fetchone()

    return {
        "row_count": row_count,
        "max_source_row": max_source_row,
        "max_datetime": max_datetime.isoformat() if max_datetime else None,
    }


def main():
    root = Path(__file__).resolve().parent
    row_count, sensor_count = import_dataset(
        root / "features.data",
        root / "labels.data",
    )
    print(f"Загружено строк: {row_count}; датчиков: {sensor_count}.")


if __name__ == "__main__":
    main()

"""Append the fictional test dataset to the local SQLite database."""

from dotenv import load_dotenv

import database
from generate_synthetic_dataset import (
    FEATURES_PATH,
    LABELS_PATH,
    SOURCE_ROW_START,
    generate_dataset,
)

ROOT = FEATURES_PATH.parent.parent


def main():
    load_dotenv(ROOT / ".env")
    generate_dataset()

    confirmation = input(
        "Добавить 300 вымышленных строк в таблицу products SQLite? "
        "Введите APPEND TEST DATA для подтверждения: "
    )
    if confirmation != "APPEND TEST DATA":
        print("Загрузка отменена.")
        return

    row_count, sensor_count = database.append_dataset(
        FEATURES_PATH,
        LABELS_PATH,
        source_row_start=SOURCE_ROW_START,
    )
    print(
        f"Добавлено вымышленных строк: {row_count}; датчиков: {sensor_count}. "
        "Дашборд должен показать уведомление в течение 10 секунд."
    )


if __name__ == "__main__":
    main()

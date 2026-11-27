"""Restore the original local dataset to the configured PostgreSQL database."""

from pathlib import Path

from dotenv import load_dotenv

import database

ROOT = Path(__file__).resolve().parent


def main():
    load_dotenv(ROOT / ".env")
    confirmation = input(
        "Заменить ВСЮ таблицу quality_products исходными features.data и labels.data? "
        "Будут удалены тестовые и любые другие новые строки. "
        "Введите RESTORE ORIGINAL для подтверждения: "
    )
    if confirmation != "RESTORE ORIGINAL":
        print("Восстановление отменено.")
        return

    row_count, sensor_count = database.import_dataset(
        ROOT / "features.data",
        ROOT / "labels.data",
    )
    print(
        f"Восстановлена исходная база: {row_count} строк; "
        f"{sensor_count} датчиков."
    )


if __name__ == "__main__":
    main()

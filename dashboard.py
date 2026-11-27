"""Flask-сервис для просмотра контроля качества изделий."""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

import charts
import database
import functions

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA_FOLDER = ROOT / "data"
MAX_UPLOAD_MB = 100  # Максимальный общий размер загружаемых файлов.
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
RECENT_COUNT = 20  # Размер окна актуальных датчиков.
HEALTH_WINDOW = 50  # Размер окна для контроля пропусков датчиков.
SILENT_LIMIT = 0.80  # Свыше 80% пустых строк — датчик почти не передаёт данные.
MIN_PERIOD_ITEMS = 30  # Минимум изделий для пересчёта выбранного периода.
STATE = {"data": None, "message": "Загрузите данные на странице загрузки.", "views": {}}
DB_LOADED_SNAPSHOT = None
DATABASE_ENABLED = False
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)


def load_state(features_path=None, labels_path=None, csv_path=None):
    """Читает данные, пересчитывает показатели и сохраняет их в общей памяти."""
    sensors, labels = functions.load_inputs(features_path, labels_path, csv_path)
    result = functions.calculate(sensors, labels)
    STATE["data"] = result
    STATE["views"] = {}
    STATE["message"] = ""


def load_initial_data():
    """При запуске читает данные из data/, а для прежней папки проекта — из корня."""
    global DATABASE_ENABLED, DB_LOADED_SNAPSHOT
    if database.database_available():
        snapshot = database.get_dataset_status()
        sensors, labels = database.load_dataset()
        result = functions.calculate(sensors, labels)
        STATE["data"] = result
        STATE["views"] = {}
        STATE["message"] = ""
        DB_LOADED_SNAPSHOT = snapshot
        DATABASE_ENABLED = True
        return

    DATA_FOLDER.mkdir(exist_ok=True)
    csv_path = DATA_FOLDER / "input.csv"
    features_path = DATA_FOLDER / "features.data"
    labels_path = DATA_FOLDER / "labels.data"
    if csv_path.exists():
        load_state(csv_path=csv_path)
    elif features_path.exists() and labels_path.exists():
        load_state(features_path=features_path, labels_path=labels_path)
    elif (ROOT / "features.data").exists() and (ROOT / "labels.data").exists():
        load_state(ROOT / "features.data", ROOT / "labels.data")


def table_html(frame):
    """Преобразует таблицу в безопасный HTML для шаблона."""
    display_frame = frame.copy()
    numeric_columns = display_frame.select_dtypes(include=["number"]).columns
    display_frame[numeric_columns] = display_frame[numeric_columns].round(2)
    return display_frame.to_html(index=False, border=0, classes="data-table",
                                 na_rep="не найдено", escape=True,
                                 float_format=lambda value: f"{value:.2f}")


def zone_table_html(frame, show_zone=True):
    """Показывает датчики в порядке зон и окрашивает строки таблицы."""
    colors = {
        "красная": "#fae4e4",
        "жёлтая": "#fff3d4",
        "зелёная": "#e4f3e9",
        "Нет данных": "#edf2f5",
    }

    display_frame = frame if show_zone else frame.drop(columns=["Зона"])

    def row_color(row):
        zone = frame.loc[row.name, "Зона"]
        style = "background-color: " + colors.get(zone, "white")
        return [style] * len(row)

    formatters = {}
    for column in display_frame.columns:
        if pd.api.types.is_numeric_dtype(display_frame[column]):
            if "доля брака" in column.lower():
                formatters[column] = lambda value: "" if pd.isna(value) else f"{value:.2%}"
            else:
                formatters[column] = lambda value: "" if pd.isna(value) else f"{value:.2f}"

    return (display_frame.style.apply(row_color, axis=1)
            .format(formatters, na_rep="не найдено", escape="html")
            .hide(axis="index").to_html(table_attributes='class="data-table"'))


def selected_dates(data):
    """Возвращает даты фильтра или весь период исходных данных."""
    times = data["labels"]["datetime"]
    first = times.min().date().isoformat()
    last = times.max().date().isoformat()
    start = session.get("period_start", first)
    end = session.get("period_end", last)
    if start < first or end > last or start > end:
        return first, last
    return start, end


def calculate_period(data, start, end):
    """Пересчитывает показатели только по изделиям внутри выбранных дат."""
    times = data["labels"]["datetime"]
    first_time = pd.Timestamp(start)
    last_time = pd.Timestamp(end) + pd.Timedelta(days=1)
    keep = (times >= first_time) & (times < last_time)
    count = int(keep.sum())
    if count < MIN_PERIOD_ITEMS:
        raise ValueError(f"В диапазоне только {count} изделий. Выберите не меньше {MIN_PERIOD_ITEMS}.")
    sensors = data["sensors"].loc[keep].reset_index(drop=True)
    labels = data["labels"].loc[keep].reset_index(drop=True)
    split = int(len(labels) * functions.TRAIN_SHARE)
    if len(labels["label"].iloc[:split].unique()) < 2:
        raise ValueError("В первых 80% выбранного диапазона должны быть и годные изделия, и брак.")
    result = functions.calculate(sensors, labels)
    if not result["keys"]:
        raise ValueError("Для выбранного диапазона не получилось рассчитать границы датчиков. Выберите другой период.")
    return result


def current_data():
    """Возвращает расчёты для выбранного периода и сохраняет их в кэше."""
    data = STATE["data"]
    if not data:
        return None
    start, end = selected_dates(data)
    first = data["labels"]["datetime"].min().date().isoformat()
    last = data["labels"]["datetime"].max().date().isoformat()
    if start == first and end == last:
        return data
    key = (start, end)
    if key not in STATE["views"]:
        try:
            if len(STATE["views"]) >= 3:
                STATE["views"].clear()
            STATE["views"][key] = calculate_period(data, start, end)
        except (ValueError, IndexError, KeyError, ZeroDivisionError) as error:
            session.pop("period_start", None)
            session.pop("period_end", None)
            flash(f"Не удалось применить сохранённый диапазон: {error}")
            return data
    return STATE["views"][key]


@app.context_processor
def add_period_to_page():
    """Передаёт выбранный период в общий шаблон страниц."""
    data = current_data()
    if not data:
        return {
            "active_period": None,
            "attention_zone_class": "zone-green",
            "database_enabled": DATABASE_ENABLED,
        }
    return {
        "active_period": selected_dates(STATE["data"]),
        "attention_zone_class": latest_attention_class(data),
        "database_enabled": DATABASE_ENABLED,
    }


def latest_attention_class(data):
    """Выбирает цвет вкладки по самой опасной зоне последнего изделия."""
    zones = set(latest_sensor_readings(data, len(data["bad"]) - 1)["Зона"])
    if "красная" in zones:
        return "zone-red"
    if "жёлтая" in zones:
        return "zone-yellow"
    return "zone-green"


def selected_period(data):
    """Возвращает выбранный период роста или первый доступный период."""
    periods = data["late"]["table"]
    if periods.empty:
        return None, 0
    number = request.args.get("period", "0")
    if request.method == "POST":
        number = request.form.get("period", "0")
    try:
        index = int(number)
    except ValueError:
        index = 0
    index = min(max(index, 0), len(periods) - 1)
    return periods.iloc[index], index


def period_summary(data, period):
    """Собирает короткое описание периода и самого раннего оперативного сигнала."""
    start = period["Начало"]
    end = period["Конец"]
    times = data["labels"]["datetime"]
    selected = (times >= start) & (times <= end)
    count = int(selected.sum())
    defects = int(data["bad"][selected.to_numpy()].sum())
    summary = {
        "start": start,
        "end": end,
        "items": count,
        "defects": defects,
        "rate": defects / count if count else 0,
        "baseline": data["ordinary"],
        "method": None,
    }
    signals = [(name, period[name]) for name in ["CUSUM", "5 из 20"]
               if pd.notna(period[name])]
    if signals:
        method, signal_time = min(signals, key=lambda item: item[1])
        summary["method"] = method
        summary["signal_time"] = signal_time
        summary["hours_to_signal"] = max(0, (signal_time - start).total_seconds() / 3600)
        weekly_time = period["Неделя"]
        if pd.notna(weekly_time) and weekly_time > signal_time:
            summary["days_before_weekly"] = (weekly_time - signal_time).total_seconds() / 86400
    return summary


def signal_basis(data, period):
    """Возвращает основания датчиков для каждого сработавшего способа обнаружения."""
    methods = ["День", "Неделя", "Месяц", "CUSUM", "5 из 20"]
    output = []
    times = data["labels"]["datetime"]
    for method in methods:
        date = period[method]
        if pd.notna(date):
            index = int(np.searchsorted(times.to_numpy(), np.datetime64(date), side="left"))
            index = min(index, len(times) - 1)
            basis = functions.make_attention(data, index)
            output.append({"name": method, "date": date, "empty": basis.empty,
                           "table": table_html(basis),
                           "chart": charts.signal_basis_chart(basis)})
    return output


def burst_sensor_candidates(data, period):
    """Ранжирует датчики по изменению вклада модели в периоде роста."""
    risk = data["model"]
    if not risk["available"]:
        return "", risk["reason"]
    times = data["labels"]["datetime"]
    period_mask = (times >= period["Начало"]) & (times <= period["Конец"])
    reference_mask = (np.arange(len(data["bad"])) < data["split"]) & (data["bad"] == 0)
    if not period_mask.any() or not reference_mask.any():
        return "", "Недостаточно изделий для сравнения с опорной историей."
    contributions = risk["contributions"]
    difference = contributions.loc[period_mask].mean() - contributions.loc[reference_mask].mean()
    result = pd.DataFrame({
        "Датчик": difference.index,
        "Изменение вклада в балл модели": difference.values,
    })
    result = result.sort_values("Изменение вклада в балл модели", ascending=False).head(5)
    result["Изменение вклада в балл модели"] = result["Изменение вклада в балл модели"].round(2)
    return table_html(result.reset_index(drop=True)), ""


def save_uploads(files, password):
    """Проверяет типы и пароли, сохраняет корректную загрузку и пересчитывает данные."""
    required_password = os.getenv("UPLOAD_PASSWORD")
    if required_password and password != required_password:
        return "Пароль загрузки неверный."
    csv_file = files.get("csv_file")
    feature_file = files.get("features_file")
    label_file = files.get("labels_file")
    if csv_file and (feature_file or label_file):
        return "Загрузите либо один CSV, либо пару файлов датчиков и меток."
    if not valid_uploads(csv_file, feature_file, label_file):
        return "Проверьте расширения: CSV либо features.data и labels.data."
    return process_uploads(csv_file, feature_file, label_file)


def valid_uploads(csv_file, feature_file, label_file):
    """Проверяет наличие нужной пары и допустимые расширения."""
    if csv_file:
        return csv_file.filename.lower().endswith(".csv")
    if not feature_file or not label_file:
        return False
    return feature_file.filename.lower().endswith(".data") and label_file.filename.lower().endswith(".data")


def process_uploads(csv_file, feature_file, label_file):
    """Сначала проверяет временные файлы и только после расчёта заменяет текущий набор."""
    DATA_FOLDER.mkdir(exist_ok=True)
    temp_csv = DATA_FOLDER / "upload.tmp"
    temp_features = DATA_FOLDER / "features.upload.tmp"
    temp_labels = DATA_FOLDER / "labels.upload.tmp"
    try:
        if csv_file:
            csv_file.save(temp_csv)
            load_state(csv_path=temp_csv)
            finish_upload(temp_csv, DATA_FOLDER / "input.csv", [DATA_FOLDER / "features.data", DATA_FOLDER / "labels.data"])
        else:
            feature_file.save(temp_features)
            label_file.save(temp_labels)
            load_state(features_path=temp_features, labels_path=temp_labels)
            finish_upload(temp_features, DATA_FOLDER / "features.data", [])
            finish_upload(temp_labels, DATA_FOLDER / "labels.data", [DATA_FOLDER / "input.csv"])
    except (ValueError, OSError, IndexError):
        remove_temp_uploads([temp_csv, temp_features, temp_labels])
        raise
    return "Данные загружены и расчёты обновлены."


def remove_temp_uploads(paths):
    """Удаляет только временные файлы этой загрузки после ошибки."""
    for path in paths:
        if path.exists():
            path.unlink()


def finish_upload(source, destination, remove_paths):
    """Перемещает проверенный временный файл на постоянное место."""
    source.replace(destination)
    for path in remove_paths:
        if path.exists():
            path.unlink()


@app.route("/", methods=["GET", "POST"])
def home():
    """Показывает короткое объяснение системы и статус данных."""
    source = STATE["data"]
    if request.method == "POST" and source:
        if request.form.get("reset"):
            session.pop("period_start", None)
            session.pop("period_end", None)
            flash("Показан весь период данных.")
            return redirect(url_for("home"))
        start = request.form.get("start", "")
        end = request.form.get("end", "")
        first = source["labels"]["datetime"].min().date().isoformat()
        last = source["labels"]["datetime"].max().date().isoformat()
        if not start or not end or start < first or end > last or start > end:
            flash("Проверьте даты: они должны входить в период загруженных данных.")
            return redirect(url_for("home"))
        try:
            result = calculate_period(source, start, end)
        except (ValueError, IndexError, KeyError, ZeroDivisionError) as error:
            flash(str(error))
            return redirect(url_for("home"))
        key = (start, end)
        if len(STATE["views"]) >= 3:
            STATE["views"].clear()
        STATE["views"][key] = result
        session["period_start"] = start
        session["period_end"] = end
        flash(f"Выбран диапазон {start} — {end}: {len(result['sensors'])} изделий.")
        return redirect(url_for("home"))

    data = current_data()
    summary = {}
    period = None
    if data:
        start, end = selected_dates(source)
        source_times = source["labels"]["datetime"]
        summary = {
            "rows": len(data["sensors"]),
            "bad": int(data["bad"].sum()),
            "bad_rate": float(data["bad"].mean()),
            "first_date": data["labels"]["datetime"].min(),
            "last_date": data["labels"]["datetime"].max(),
            "periods": len(data["late"]["table"]),
            "sensor_count": data["sensors"].shape[1],
            "kept_sensors": data["clean"].shape[1],
            "missing_before": float(data["sensors"].isna().to_numpy().mean()),
            "missing_after": float(data["clean"].isna().to_numpy().mean()),
            "data_min": source_times.min().date().isoformat(),
            "data_max": source_times.max().date().isoformat(),
        }
        period = {"start": start, "end": end}
        cleanup = charts.data_charts(data)
    else:
        cleanup = []
    return render_template("home.html", summary=summary, period=period,
                           shewhart=charts.p_chart(data) if data else "",
                           cleanup=cleanup,
                           message=STATE["message"])


@app.route("/late", methods=["GET", "POST"])
def late_page():
    """Сравнивает сроки обнаружения роста на разных временных масштабах."""
    data = current_data()
    if request.method == "POST":
        return redirect(url_for("late_page", period=request.form.get("period", "0")))
    if not data:
        return render_template("late.html", message=STATE["message"])
    period, index = selected_period(data)
    scales = charts.period_scale_charts(data, period) if period is not None else ""
    bases = signal_basis(data, period) if period is not None else []
    candidates, candidates_reason = (
        burst_sensor_candidates(data, period) if period is not None else ("", "")
    )
    table = table_html(data["late"]["table"])
    summary = period_summary(data, period) if period is not None else None
    return render_template("late.html", data=data, period=period, index=index,
                           periods=data["late"]["table"], summary=summary,
                           table=table, scales=scales,
                           rolling=charts.p_chart(data), bases=bases,
                           candidates=candidates, candidates_reason=candidates_reason,
                           message=STATE["message"])


@app.route("/data")
def data_page():
    """Перенаправляет старый адрес статистики набора на главную страницу."""
    return redirect(url_for("home"))


@app.route("/sensors")
def sensors_page():
    """Показывает рейтинги датчиков и ранговые связи."""
    data = current_data()
    if not data:
        return render_template("sensors.html", message=STATE["message"])
    mann, forest, heat = charts.sensor_charts(data)
    medians = charts.time_charts(data)
    return render_template("sensors.html", data=data, charts=[mann, forest, heat, medians],
                           message=STATE["message"])


@app.route("/sensor", methods=["GET", "POST"])
def sensor_page():
    """Показывает распределение датчика и долю брака по группам значений."""
    data = current_data()
    if not data:
        return render_template("sensor.html", message=STATE["message"])
    name = request.values.get("sensor", data["keys"][0])
    if name not in data["keys"]:
        name = data["keys"][0]
    histogram, bars, weekly, groups = charts.sensor_detail_charts(data, name)
    return render_template("sensor.html", data=data, sensor=name, histogram=histogram,
                           bars=bars, weekly=weekly, table=table_html(groups),
                           message=STATE["message"])


@app.route("/limits", methods=["GET", "POST"])
def limits_page():
    """Показывает распределение выбранного датчика и границы 3σ."""
    data = current_data()
    if not data:
        return render_template("limits.html", message=STATE["message"])
    name = request.form.get("sensor", data["keys"][0]) if request.method == "POST" else request.args.get("sensor", data["keys"][0])
    if name not in data["keys"]:
        name = data["keys"][0]
    border = data["borders"][data["borders"]["Датчик"] == name]
    row = border.iloc[0]
    deviation = row["Направление"] * (data["clean"][name] - row["Центр"]) / row["Сигма"]
    history = []
    for zone, cutoff in [("Жёлтая", 2), ("Красная", 3)]:
        selected = deviation >= cutoff
        count = int(selected.sum())
        defects = int(data["bad"][selected.to_numpy()].sum())
        low, high = functions.wilson_interval(defects, count)
        history.append([zone, count, defects, defects / count if count else np.nan, low, high])
    history = pd.DataFrame(history, columns=["Зона", "Изделий", "Брак", "Доля брака", "Нижняя 95%", "Верхняя 95%"])
    return render_template("limits.html", data=data, sensor=name, chart=charts.boundary_chart(data, name),
                           border=table_html(border), history=table_html(history),
                           message=STATE["message"])


@app.route("/time")
def time_page():
    """Показывает недельную связь брака с пропусками и здоровье передачи."""
    data = current_data()
    if not data:
        return render_template("time.html", message=STATE["message"])
    weekly, heat = charts.when_where_charts(data)
    health = sensor_health_summary(data)
    health_display = health.copy()
    for column in ["Обычная доля пропусков", "Жёлтая граница", "Красная граница",
                   "Пропуски за последние 50"]:
        health_display[column] = health_display[column].map(lambda value: f"{value:.2%}")
    return render_template("time.html", data=data,
                           charts=[weekly, heat,
                                   charts.health_chart(health) if not health.empty else ""],
                           table=table_html(data["weekly"]),
                           health=zone_table_html(health_display) if not health.empty else "",
                           health_empty=health.empty,
                           window=min(HEALTH_WINDOW, len(data["bad"])),
                           message=STATE["message"])


@app.route("/attention", methods=["GET", "POST"])
def attention_page():
    """Показывает зоны датчиков последнего изделия и отдельный сигнал окна."""
    data = current_data()
    if not data:
        return render_template("attention.html", message=STATE["message"])
    default_latest_index = len(data["bad"]) - 1
    latest_index, latest_index_error = selected_index(
        request.form.get("latest_index"), default_latest_index, len(data["bad"])
    )
    last_readings = latest_sensor_readings(data, latest_index)
    yellow_count = int((last_readings["Зона"] == "жёлтая").sum())
    red_count = int((last_readings["Зона"] == "красная").sum())
    missing_count = int((last_readings["Зона"] == "Нет данных").sum())
    defect_indices = np.flatnonzero(data["bad"])
    default_index = int(defect_indices[-1]) if len(defect_indices) else len(data["bad"]) - 1
    index, index_error = selected_index(request.form.get("index"), default_index, len(data["bad"]))
    if data["bad"][index] != 1:
        index = default_index
        index_error = True
    options = [{"index": int(item), "date": data["labels"].loc[item, "datetime"]}
               for item in defect_indices]
    defects = defect_details(data, index)
    recent_rate = float(data["bad"][-RECENT_COUNT:].mean())
    limit = data["ordinary"] + 3 * np.sqrt(data["ordinary"] * (1 - data["ordinary"]) / RECENT_COUNT)
    last_time = data["labels"].loc[latest_index, "datetime"]
    return render_template("attention.html", data=data,
                           latest=zone_table_html(last_readings, show_zone=False),
                           last_time=last_time,
                           latest_index=latest_index,
                           total_rows=len(data["bad"]),
                           latest_label="Брак" if data["bad"][latest_index] else "Годное",
                           latest_index_error=latest_index_error,
                           yellow_count=yellow_count, red_count=red_count,
                           missing_count=missing_count,
                           defects=zone_table_html(defects, show_zone=False),
                           index=index,
                           defect_indices=options,
                           growth_signal=recent_rate > limit, recent_rate=recent_rate,
                           control_limit=limit,
                           index_error=index_error,
                           message=STATE["message"])


def latest_sensor_readings(data, index):
    """Сравнивает все ключевые датчики одного изделия с историческими границами."""
    rows = []
    for name in data["keys"]:
        value = data["clean"].loc[index, name]
        border = data["borders"].loc[data["borders"]["Датчик"] == name].iloc[0]
        if pd.isna(value):
            rows.append([name, "Нет данных", np.nan, np.nan, np.nan, 0, np.nan])
            continue
        deviation = border["Направление"] * (value - border["Центр"]) / border["Сигма"]
        if deviation >= 3:
            zone = "красная"
            boundary = border["Красная"]
        elif deviation >= 2:
            zone = "жёлтая"
            boundary = border["Жёлтая"]
        else:
            zone = "зелёная"
            boundary = border["Жёлтая"]
        over_yellow = border["Направление"] * (data["clean"][name] - border["Центр"]) / border["Сигма"] >= 2
        count = int(over_yellow.sum())
        rate = float(data["bad"][over_yellow.to_numpy()].mean()) if count else np.nan
        rows.append([name, zone, value, boundary, deviation, count, rate])
    result = pd.DataFrame(rows, columns=["Датчик", "Зона", "Значение", "Ближайшая граница",
                                        "Отклонение, σ", "Историческое N", "Историческая доля брака"])
    order = {"красная": 0, "жёлтая": 1, "зелёная": 2, "Нет данных": 3}
    result["Порядок зоны"] = result["Зона"].map(order)
    return (result.sort_values(["Порядок зоны", "Отклонение, σ"],
                               ascending=[True, False], na_position="last")
            .drop(columns="Порядок зоны").reset_index(drop=True))


def sensor_health_summary(data):
    """Сравнивает пропуски последних изделий с личной нормой каждого датчика."""
    rows = []
    window = min(HEALTH_WINDOW, len(data["bad"]))
    split = data["split"]
    for name in data["keys"]:
        training_missing = data["clean"][name].iloc[:split].isna()
        normal = float(training_missing.mean())
        standard_error = np.sqrt(normal * (1 - normal) / window)
        yellow = min(1.0, normal + 2 * standard_error)
        red = min(1.0, normal + 3 * standard_error)
        recent_missing = data["clean"][name].tail(window).isna()
        rate = float(recent_missing.mean())
        if rate > red:
            zone = "красная"
        elif rate > yellow:
            zone = "жёлтая"
        else:
            zone = "зелёная"
        rows.append([name, zone, normal, yellow, red, rate, int(recent_missing.sum())])
    result = pd.DataFrame(rows, columns=["Датчик", "Зона", "Обычная доля пропусков",
                                        "Жёлтая граница", "Красная граница",
                                        "Пропуски за последние 50", "Пустых в окне"])
    zone_order = {"красная": 0, "жёлтая": 1, "зелёная": 2}
    result["Порядок зоны"] = result["Зона"].map(zone_order)
    return (result.sort_values(["Порядок зоны", "Пропуски за последние 50"],
                               ascending=[True, False])
            .loc[lambda frame: frame["Пустых в окне"] > 0]
            .drop(columns="Порядок зоны").reset_index(drop=True))


def manual_check(data):
    """Проверяет заполненные вручную датчики и возвращает зоны с историческим контекстом."""
    rows = []
    for name in data["keys"]:
        text = request.form.get("value_" + name, "").strip()
        if text:
            try:
                value = float(text)
                if not np.isfinite(value):
                    raise ValueError("Значение должно быть конечным числом.")
                rows.append(functions.check_manual_value(data, name, value))
            except ValueError:
                rows.append([name, text, "Ошибка: введите число", "", "", ""])
    return pd.DataFrame(rows, columns=["Датчик", "Значение", "Статус", "Отклонение, σ", "Доля брака", "Изделий в истории"])


def selected_index(value, default, total):
    """Проверяет номер изделия и сообщает, если его пришлось заменить."""
    try:
        index = int(value) if value is not None else default
    except ValueError:
        return default, True
    if index < 0 or index >= total:
        return default, True
    return index, False


@app.route("/risk", methods=["GET", "POST"])
def risk_page():
    """Показывает временную проверку модели и оценку выбранного изделия."""
    data = current_data()
    if not data:
        return render_template("risk.html", message=STATE["message"])
    risk = data["model"]
    if not risk["available"]:
        return render_template("risk.html", data=data, risk=risk, message=STATE["message"])
    index, index_error = selected_index(request.form.get("index"), len(data["bad"]) - 1, len(data["bad"]))
    contribution = risk["contributions"].iloc[index].rename("Вклад").reset_index()
    contribution.columns = ["Датчик", "Вклад"]
    roc, bars = charts.risk_charts(risk, contribution)
    missing = data["clean"][risk["sensors"]].tail(RECENT_COUNT).isna().mean()
    silent = list(missing[missing > SILENT_LIMIT].index)
    return render_template("risk.html", data=data, risk=risk, index=index,
                           level=risk["levels"][index], roc=roc, contributions=bars,
                           summary=table_html(risk["summary"]), silent=silent,
                           index_error=index_error,
                           message=STATE["message"])


@app.route("/testing")
def testing_page():
    """Сравнивает алгоритмы градиентного бустинга на выбранном периоде."""
    data = current_data()
    if not data:
        return render_template("testing.html", message=STATE["message"])
    comparison = data.get("model_comparison")
    if comparison is None:
        comparison = functions.make_model_comparison(
            data["clean"], data["bad"], data["split"], data["model_features"]
        )
        data["model_comparison"] = comparison
    return render_template("testing.html", data=data, comparison=comparison,
                           message=STATE["message"])


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    """Показывает форму загрузки и сохраняет прошедшие проверку данные."""
    global DB_LOADED_SNAPSHOT, DATABASE_ENABLED
    message = request.args.get("message", "")
    data = current_data()
    manual = ""
    manual_submitted = False
    if request.method == "POST":
        if request.form.get("mode") == "reload_database":
            try:
                snapshot = database.get_dataset_status()
                sensors, labels = database.load_dataset()
                result = functions.calculate(sensors, labels)
            except (sqlite3.Error, RuntimeError, ValueError):
                app.logger.exception("Не удалось обновить данные из SQLite.")
                flash("Не удалось загрузить данные из БД. Проверьте подключение и повторите попытку.")
            else:
                STATE["data"] = result
                STATE["views"] = {}
                STATE["message"] = ""
                DB_LOADED_SNAPSHOT = snapshot
                DATABASE_ENABLED = True
                session.pop("period_start", None)
                session.pop("period_end", None)
                flash(f"Данные из БД загружены: {len(labels)} строк.")
            return redirect(url_for("upload_page"))

        if request.form.get("mode") == "manual":
            manual_submitted = True
            if data:
                result = manual_check(data)
                if len(result):
                    manual = table_html(result)
            else:
                message = "Сначала загрузите данные, чтобы сравнить показания с историей."
            return render_template(
                "upload.html", message=message, data=data, manual=manual,
                manual_submitted=manual_submitted,
                password_required=bool(os.getenv("UPLOAD_PASSWORD")),
            )
        try:
            message = save_uploads(request.files, request.form.get("password", ""))
        except (ValueError, OSError, IndexError) as error:
            message = "Не удалось прочитать данные: " + str(error)
        return redirect(url_for("upload_page", message=message))
    return render_template("upload.html", message=message, data=data, manual=manual,
                           manual_submitted=manual_submitted,
                           password_required=bool(os.getenv("UPLOAD_PASSWORD")))


@app.route("/database/status")
def database_status():
    """Checks whether the SQLite database has rows newer than the loaded dataset."""
    if not DATABASE_ENABLED:
        return jsonify(enabled=False)
    try:
        current_snapshot = database.get_dataset_status()
    except (sqlite3.Error, RuntimeError):
        app.logger.exception("Не удалось проверить обновления SQLite.")
        return jsonify(enabled=True, error="Не удалось проверить обновления базы данных."), 503

    loaded_snapshot = DB_LOADED_SNAPSHOT
    current_max_row = current_snapshot["max_source_row"]
    loaded_max_row = loaded_snapshot["max_source_row"] if loaded_snapshot else None
    has_new_data = bool(
        loaded_snapshot
        and (
            current_snapshot["row_count"] > loaded_snapshot["row_count"]
            or (
                current_max_row is not None
                and (loaded_max_row is None or current_max_row > loaded_max_row)
            )
            or (
                current_snapshot["max_datetime"]
                and (
                    not loaded_snapshot["max_datetime"]
                    or current_snapshot["max_datetime"] > loaded_snapshot["max_datetime"]
                )
            )
        )
    )
    return jsonify(enabled=True, has_new_data=has_new_data)


@app.route("/health")
def health():
    """Возвращает короткий ответ для проверки доступности сервера."""
    return "ok"


def defect_details(data, index):
    """Возвращает датчики выбранного изделия, которые вышли за жёлтую границу."""
    bad_indices = np.flatnonzero(data["bad"])
    if index not in bad_indices:
        return pd.DataFrame(columns=["Датчик", "Зона", "Значение", "Граница", "Сигмы",
                                     "Доля брака", "Изделий", "Номер изделия"])
    return functions.make_attention(data, index)


def upload_too_large(error):
    """Показывает понятное сообщение при превышении лимита размера."""
    return render_template("upload.html", message=f"Файл превышает лимит {MAX_UPLOAD_MB} МБ."), 413


app.register_error_handler(413, upload_too_large)

try:
    load_initial_data()
except (sqlite3.Error, ValueError, OSError, IndexError, RuntimeError) as error:
    if isinstance(error, sqlite3.Error):
        app.logger.exception("Не удалось загрузить данные при запуске.")
    STATE["message"] = "Не удалось загрузить данные: " + str(error)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)

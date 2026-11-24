from io import BytesIO

import dashboard


def test_health_returns_ok():
    response = dashboard.app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"


def test_dashboard_pages_return_success():
    client = dashboard.app.test_client()
    for path in ["/", "/late", "/sensors", "/sensor", "/limits", "/time", "/attention", "/risk", "/testing", "/upload"]:
        assert client.get(path).status_code == 200
    assert client.get("/data", follow_redirects=True).status_code == 200
    assert client.get("/static/plotly.min.js").status_code == 200


def test_sensor_cleanup_chart_shows_unique_value_limit():
    chart, _ = dashboard.charts.data_charts(dashboard.STATE["data"])
    assert r"0\u20131" in chart
    assert r"\u0443\u043d\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0445" in chart


def test_spearman_heatmap_is_tall_enough_to_read_sensor_labels():
    _, _, heatmap = dashboard.charts.sensor_charts(dashboard.STATE["data"])
    sensor_count = len(dashboard.STATE["data"]["keys"]) + 1
    expected_height = max(760, 28 * sensor_count + 180)

    assert f'"height":{expected_height}' in heatmap
    assert '"tickangle":-45' in heatmap
    assert '"size":11' in heatmap


def test_sidebar_shows_every_page_without_collapsible_menu():
    page = dashboard.app.test_client().get("/").get_data(as_text=True)
    for label in ["Главная", "Обрати внимание",
                  "Датчики и связи", "Датчик подробно", "Когда и где",
                  "Оценка модели", "Загрузка данных"]:
        assert f">{label}</a>" in page
    assert ">Разбор всплеска</a>" not in page
    assert "<details>" not in page
    assert "Данные и целевая метка" not in page
    assert "Границы 3σ" not in page
    assert page.index(">Главная</a>") < page.index(">Обрати внимание</a>")
    assert 'class="active" href="/" aria-current="page"' in page


def test_attention_tab_color_follows_highest_zone(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        dashboard,
        "latest_sensor_readings",
        lambda data, index: pd.DataFrame({"Зона": ["жёлтая", "красная"]}),
    )
    assert dashboard.latest_attention_class({"bad": [0]}) == "zone-red"


def test_attention_tab_starts_yellow_for_current_last_product():
    page = dashboard.app.test_client().get("/attention").get_data(as_text=True)
    assert 'class="attention-status zone-yellow active"' in page


def test_every_sidebar_page_marks_current_route_active():
    client = dashboard.app.test_client()
    pages = [
        ("/sensors", "sensors_page", "Датчики и связи"),
        ("/sensor", "sensor_page", "Датчик подробно"),
        ("/time", "time_page", "Когда и где"),
        ("/risk", "risk_page", "Оценка модели"),
        ("/testing", "testing_page", "Тестирование"),
        ("/upload", "upload_page", "Загрузка данных"),
    ]
    for path, endpoint, label in pages:
        page = client.get(path).get_data(as_text=True)
        assert f'class="active" href="{path}" aria-current="page">{label}</a>' in page


def test_home_explains_loaded_data_and_links_to_burst_review():
    response = dashboard.app.test_client().get("/")
    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Последнее изделие в нём" in page
    assert "Это не онлайн-мониторинг" not in page
    assert "Даты ограничены загруженным файлом" not in page
    assert "Перейти к разбору всплеска" not in page
    assert "Карта Шухарта: доля брака" in page
    assert "Зелёная линия — обычный уровень" in page
    assert "Очистка датчиков" in page
    assert "Целевая метка и доля брака" not in page
    assert page.index("Очистка датчиков") < page.index("</main>")
    assert "Пустые ячейки" in page
    assert "Датчики по доле пропусков" not in page
    assert "count × доля пропусков" not in page
    assert "Удалённые датчики" not in page


def test_data_page_redirects_to_home():
    response = dashboard.app.test_client().get("/data")
    assert response.status_code == 302
    assert response.location == "/"


def test_sensor_detail_uses_histogram_not_time_scatter():
    response = dashboard.app.test_client().get("/sensor")
    page = response.get_data(as_text=True)
    histogram = dashboard.charts.sensor_detail_charts(
        dashboard.STATE["data"], dashboard.STATE["data"]["keys"][0]
    )[0]
    assert "Распределение показаний sensor_" in page
    selected_sensor = dashboard.STATE["data"]["keys"][0]
    assert page.count(f"Распределение показаний {selected_sensor}") == 1
    assert "Показания sensor_" not in page
    assert '"orientation":"h"' in page
    assert '"title":{"text":"Probability"' in histogram
    assert '"title":{"text":"\\u0417\\u043d\\u0430\\u0447\\u0435\\u043d\\u0438\\u0435 \\u0434\\u0430\\u0442\\u0447\\u0438\\u043a\\u0430"' in histogram
    assert '"orientation":"h"' in histogram
    assert "#0072b2" in histogram
    assert "#e69f00" in histogram
    assert '"barmode":"group"' in histogram
    assert '"opacity":1' in histogram
    for color in ["#00845a", "#b87900", "#c43d3d"]:
        assert color in histogram
    assert '"width":4' in histogram
    assert "центральный диапазон значений от 0,5-го до 99,5-го процентиля" in page
    assert "Типичный уровень".encode("unicode_escape").decode("ascii") in histogram
    assert "annotation_text" not in histogram
    assert "по неделям" in page
    assert "медианное значение датчика за календарную неделю" in page


def test_shewhart_chart_caps_y_axis_at_sixty_percent():
    chart = dashboard.charts.p_chart(dashboard.STATE["data"])
    assert '"range":[0,0.6]' in chart


def test_sensor_health_page_shows_control_limits_and_zone_list():
    page = dashboard.app.test_client().get("/time").get_data(as_text=True)
    assert "Количество брака и замеров по неделям" in page
    assert "Количество брака: %{y:.0f}".encode("unicode_escape").decode("ascii") in page
    assert "Количество замеров: %{y:.0f}".encode("unicode_escape").decode("ascii") in page
    assert "Контрольная карта пропусков" in page
    assert "Обычная доля пропусков" in page
    assert "Жёлтая граница" in page
    assert "Красная граница" in page
    assert "Связь количества пропусков и брака" not in page
    assert "Корреляция Спирмена" not in page
    assert "σ = √(p × (1 − p) / 50)" in page


def test_missing_defect_correlation_finds_weekly_positive_and_negative_links():
    import pandas as pd

    dates = []
    bad = []
    positive_missing = []
    negative_missing = []
    week_bad_counts = [0, 0, 1, 1, 2, 2]
    week_positive_missing = [0, 0, 1, 1, 2, 2]
    week_negative_missing = [2, 2, 1, 1, 0, 0]
    for week_index, (bad_count, positive_count, negative_count) in enumerate(
        zip(week_bad_counts, week_positive_missing, week_negative_missing)
    ):
        start = pd.Timestamp("2020-01-07") + pd.Timedelta(weeks=week_index)
        dates.extend([start, start + pd.Timedelta(days=1)])
        bad.extend([1] * bad_count + [0] * (2 - bad_count))
        positive_missing.extend([None] * positive_count + [1] * (2 - positive_count))
        negative_missing.extend([None] * negative_count + [1] * (2 - negative_count))

    data = {
        "labels": pd.DataFrame({"datetime": dates}),
        "bad": bad,
        "clean": pd.DataFrame({
            "positive": positive_missing,
            "negative": negative_missing,
        }),
    }

    _, correlations = dashboard.charts.missing_defect_correlation(data)
    values = correlations.set_index("Датчик")["Корреляция Спирмена"]
    assert values["positive"] == 1
    assert values["negative"] == -1


def test_missing_defect_correlation_chart_is_weekly_count_scatter():
    chart, _ = dashboard.charts.missing_defect_correlation(dashboard.STATE["data"])
    assert '"type":"scatter"' in chart
    assert "Количество пропущенных показаний датчиков за неделю".encode("unicode_escape").decode("ascii") in chart
    assert "Количество бракованных изделий за неделю".encode("unicode_escape").decode("ascii") in chart


def test_missing_defect_chart_compares_weekly_counts_and_reports_week_volume():
    import pandas as pd

    dates = pd.date_range("2020-01-07", periods=12, freq="D")
    data = {
        "labels": pd.DataFrame({"datetime": dates}),
        "bad": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        "clean": pd.DataFrame({
            "sensor_1": [1, 1, None, 1, None, 1, None, 1, None, 1, None, None],
            "sensor_2": [1, 1, 1, 1, 1, 1, 1, None, 1, None, None, None],
        }),
    }

    chart, _ = dashboard.charts.missing_defect_correlation(data)

    assert "Пропущенных показаний: %{x}".encode("unicode_escape").decode("ascii") in chart
    assert "Бракованных изделий: %{y}".encode("unicode_escape").decode("ascii") in chart
    assert "Всего изделий: %{customdata[1]}".encode("unicode_escape").decode("ascii") in chart
    assert "Доля пропусков: %{customdata[2]:.2%}".encode("unicode_escape").decode("ascii") in chart


def test_health_chart_shows_current_missing_rate_as_bars_and_reference_markers():
    import pandas as pd

    health = pd.DataFrame({
        "Датчик": ["sensor_001", "sensor_002"],
        "Зона": ["красная", "зелёная"],
        "Обычная доля пропусков": [0.1, 0.05],
        "Жёлтая граница": [0.2, 0.15],
        "Красная граница": [0.3, 0.25],
        "Пропуски за последние 50": [0.4, 0.0],
        "Пустых в окне": [8, 0],
    })

    chart = dashboard.charts.health_chart(health)

    assert '"type":"bar"' in chart
    assert chart.count('"type":"scatter"') >= 3
    assert "Пропуски в последних 50 замерах".encode("unicode_escape").decode("ascii") in chart
    assert "Зелёный порог".encode("unicode_escape").decode("ascii") in chart
    assert "Жёлтый порог".encode("unicode_escape").decode("ascii") in chart
    assert "Красный порог".encode("unicode_escape").decode("ascii") in chart
    assert chart.count('"symbol":"line-ew"') == 3
    assert '"x":["sensor_001","sensor_002"]' in chart
    assert "Пропущено в окне".encode("unicode_escape").decode("ascii") in chart
    assert '"color":"#2874a6"' in chart


def test_weekly_defect_chart_includes_measurement_count_on_secondary_axis():
    import pandas as pd

    data = {
        "weekly": pd.DataFrame({
            "Дата": pd.to_datetime(["2020-01-06", "2020-01-13"]),
            "Доля": [0.1, 0.2],
            "sum": [1, 4],
            "count": [10, 20],
        }),
        "heatmap": pd.DataFrame([[0.1]], index=["sensor"], columns=["week"]),
    }

    chart, _ = dashboard.charts.when_where_charts(data)

    assert "Количество брака".encode("unicode_escape").decode("ascii") in chart
    assert "Количество замеров".encode("unicode_escape").decode("ascii") in chart
    assert '"barmode":"group"' in chart
    assert '"secondary_y"' not in chart
    assert "Количество замеров: %{y:.0f}".encode("unicode_escape").decode("ascii") in chart
    assert "31.12.2019 – 06.01.2020".encode("unicode_escape").decode("ascii") in chart


def test_sensor_health_summary_sorts_red_yellow_green():
    import numpy as np
    import pandas as pd

    data = {
        "keys": ["yellow", "red", "green"],
        "clean": pd.DataFrame({
            "yellow": np.r_[[np.nan] * 4, np.zeros(76), [np.nan] * 6, np.zeros(14)],
            "red": np.r_[[np.nan] * 4, np.zeros(76), [np.nan] * 8, np.zeros(12)],
            "green": np.r_[[np.nan] * 4, np.zeros(76), [np.nan], np.zeros(19)],
        }),
        "bad": np.zeros(100, dtype=int),
        "split": 80,
    }
    result = dashboard.sensor_health_summary(data)
    assert result["Зона"].tolist() == ["красная", "жёлтая", "зелёная"]
    assert result.loc[0, "Пропуски за последние 50"] == 0.16


def test_sensor_health_summary_excludes_sensors_without_recent_missing_values():
    import numpy as np
    import pandas as pd

    data = {
        "keys": ["has_missing", "no_missing"],
        "clean": pd.DataFrame({
            "has_missing": np.r_[np.zeros(99), np.nan],
            "no_missing": np.zeros(100),
        }),
        "bad": np.zeros(100, dtype=int),
        "split": 50,
    }

    result = dashboard.sensor_health_summary(data)

    assert result["Датчик"].tolist() == ["has_missing"]


def test_time_page_explains_when_no_recent_missing_values():
    original_data = dashboard.STATE["data"]
    data = original_data.copy()
    data["clean"] = data["clean"].copy()
    data["clean"].loc[:, data["keys"]] = 0.0
    dashboard.STATE["data"] = data
    try:
        page = dashboard.app.test_client().get("/time").get_data(as_text=True)
    finally:
        dashboard.STATE["data"] = original_data

    assert "В последних 50 замерах пропусков не обнаружено." in page


def test_selected_dates_are_kept_for_analysis_pages(monkeypatch):
    monkeypatch.setattr(dashboard, "calculate_period", lambda data, start, end: data)
    client = dashboard.app.test_client()
    response = client.post("/", data={"start": "2008-07-20", "end": "2008-08-20"})
    assert response.status_code == 302

    page = client.get("/").get_data(as_text=True)
    assert "Период анализа: 2008-07-20 — 2008-08-20" in page


def test_date_filter_rejects_range_with_too_few_items():
    client = dashboard.app.test_client()
    response = client.post("/", data={"start": "2008-07-19", "end": "2008-07-19"})
    page = client.get(response.location).get_data(as_text=True)
    assert response.status_code == 302
    assert "В диапазоне только" in page


def test_burst_review_distinguishes_detection_from_prediction():
    response = dashboard.app.test_client().get("/late")
    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Ретроспективное сравнение" in page
    assert "не предсказывает брак" in page
    assert "не доказывают причину брака" in page


def test_burst_review_ranks_model_sensor_contributions():
    data = dashboard.STATE["data"]
    period = data["late"]["table"].iloc[0]

    table, reason = dashboard.burst_sensor_candidates(data, period)

    assert not reason
    assert "Датчик" in table
    assert "Изменение вклада в балл модели" in table
    assert dashboard.STATE["data"]["model"]["sensors"][0] in table


def test_risk_page_reports_recall_focused_threshold_and_held_out_metrics():
    response = dashboard.app.test_client().get("/risk")
    page = response.get_data(as_text=True)
    risk = dashboard.STATE["data"]["model"]
    metrics = risk["metrics"]

    assert response.status_code == 200
    assert "порог с приоритетом поиска брака" in page
    assert "минимальное число тревог при обнаружении строго больше половины брака" in page
    assert "валидация" not in page.lower()
    assert "Наивный ориентир для сравнения" not in page
    assert "Сравнение на последних 20%" in page
    assert "уже нельзя считать полностью независимой финальной проверкой" not in page
    assert "Сбалансированная точность" in page
    assert metrics["true_good"] + metrics["false_alarm"] + metrics["missed_defect"] + metrics["found_defect"] == metrics["count"]
    assert metrics["found_defect"] >= risk["default_metrics"]["found_defect"]
    assert risk["threshold_selection"]["recall"] > 0.5
    assert risk["threshold_selection"]["count"] == dashboard.STATE["data"]["split"] - risk["fit_end"]
    assert "Искусственных примерах" not in page


def test_testing_page_compares_boosting_models_on_the_same_held_out_metrics():
    response = dashboard.app.test_client().get("/testing")
    page = response.get_data(as_text=True)
    comparison = dashboard.STATE["data"]["model_comparison"]

    assert response.status_code == 200
    assert "Градиентный бустинг" in page
    assert "XGBoost" in page
    assert "Текущая модель — логистическая регрессия" in page
    assert "последних 20%" in page
    assert comparison["available"]
    assert len(comparison["models"]) == 5
    for model in comparison["models"]:
        assert model["available"]
        assert model["threshold_selection"]["count"] == comparison["split"] - comparison["fit_end"]
        assert model["metrics"]["count"] == len(
            dashboard.STATE["data"]["bad"]
        ) - dashboard.STATE["data"]["split"]
        assert set(model["metrics"]) >= {
            "accuracy", "balanced_accuracy", "recall", "precision", "f1",
            "false_alarm", "missed_defect", "found_defect",
        }
    undersampled = comparison["models"][2]
    assert undersampled["name"] == "Random Forest с уменьшением класса годных"
    assert undersampled["temporal_cv_folds"]
    assert undersampled["threshold_selection"]["precision"] >= undersampled["minimum_precision"]
    assert undersampled["metrics"]["precision"] < undersampled["minimum_precision"]
    assert "На последних 20% точность ниже целевого уровня" in page
    cost_sensitive = comparison["models"][3]
    assert cost_sensitive["name"] == "Cost-sensitive логистическая регрессия"
    assert cost_sensitive["threshold_selection"]["precision"] >= cost_sensitive["minimum_precision"]
    precision_policy = comparison["models"][4]
    assert "точность от 20%" in precision_policy["name"]
    assert precision_policy["threshold_selection"]["precision"] >= precision_policy["minimum_precision"]
    assert precision_policy["metrics"]["precision"] >= 0.10
    assert precision_policy["metrics"]["f1"] > dashboard.STATE["data"]["model"]["metrics"]["f1"]


def test_recall_threshold_minimizes_alerts_while_exceeding_half_recall():
    import numpy as np

    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2])

    threshold, result = dashboard.functions.recall_focused_threshold(labels, scores)

    assert threshold == 0.7
    assert result["recall"] == 0.75
    assert result["found_defect"] == 3


def test_precision_threshold_maximizes_recall_without_violating_the_floor():
    import numpy as np

    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    threshold, result = dashboard.functions.precision_constrained_threshold(
        labels, scores, minimum_precision=0.5
    )

    assert threshold == 0.7
    assert result["precision"] >= 0.5
    assert result["recall"] == 1
    assert result["found_defect"] == 3
    assert result["false_alarms"] == 0


def test_upload_rejects_wrong_extension():
    client = dashboard.app.test_client()
    response = client.post("/upload", data={"csv_file": (BytesIO(b"bad"), "input.txt")})
    page = client.get(response.location).get_data(as_text=True)
    assert response.status_code == 302
    assert "Проверьте расширения" in page


def test_upload_rejects_csv_without_required_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DATA_FOLDER", tmp_path)
    client = dashboard.app.test_client()
    response = client.post(
        "/upload",
        data={"csv_file": (BytesIO(b"wrong,value\n1,2\n"), "input.csv")},
    )
    page = client.get(response.location).get_data(as_text=True)
    assert response.status_code == 302
    assert "CSV должен содержать столбцы datetime и label" in page
    assert list(tmp_path.iterdir()) == []


def test_upload_requires_configured_password(monkeypatch):
    monkeypatch.setenv("UPLOAD_PASSWORD", "private")
    client = dashboard.app.test_client()
    response = client.post(
        "/upload",
        data={"password": "wrong", "csv_file": (BytesIO(b"data"), "input.csv")},
    )
    page = client.get(response.location).get_data(as_text=True)
    assert "Пароль загрузки неверный" in page


def test_upload_size_limit_is_configured():
    assert dashboard.app.config["MAX_CONTENT_LENGTH"] == dashboard.MAX_UPLOAD_MB * 1024 * 1024


def test_manual_sensor_entry_is_available_under_upload():
    client = dashboard.app.test_client()
    upload_page = client.get("/upload").get_data(as_text=True)
    attention_page = client.get("/attention").get_data(as_text=True)

    assert "Ввести показания датчиков вручную" in upload_page
    assert 'name="mode" value="manual"' in upload_page
    assert 'name="value_sensor_' in upload_page
    assert "Проверить показания вручную" not in attention_page


def test_manual_sensor_entry_shows_historical_context_on_upload_page():
    client = dashboard.app.test_client()
    data = dashboard.STATE["data"]
    sensor = data["keys"][0]
    border = data["borders"].loc[data["borders"]["Датчик"] == sensor].iloc[0]
    value = border["Центр"]
    response = client.post(
        "/upload",
        data={"mode": "manual", f"value_{sensor}": str(value)},
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Исторический контекст введённых значений" in page
    assert sensor in page
    assert "Обычная" in page


def test_dashboard_loaded_enterprise_data_matches_expected_counts():
    data = dashboard.STATE["data"]
    assert len(data["sensors"]) == 1567
    assert data["sensors"].shape[1] == 590
    assert int(data["bad"].sum()) == 104
    assert data["clean"].shape[1] == 446


def test_dashboard_language_does_not_make_probability_claim():
    response = dashboard.app.test_client().get("/attention")
    text = response.get_data(as_text=True).lower()
    assert "вероятность брака" not in text
    assert "прогноз брака" not in text


def test_attention_page_shows_last_product_sensor_zones():
    page = dashboard.app.test_client().get("/attention").get_data(as_text=True)
    last_index = len(dashboard.STATE["data"]["bad"]) - 1
    assert "Строка" in page
    assert "Зоны ключевых датчиков" in page
    assert "Зелёный фон — показание в обычном историческом диапазоне" in page
    assert "Жёлтый — предупредительная зона" in page
    assert "Красный — контрольная граница превышена" in page
    assert "Цвета — статистические ориентиры" not in page
    assert "<th>Зона</th>" not in page
    assert "Разбор помогает раньше заметить рост брака" not in page
    assert "Выбрана историческая строка из диапазона" not in page
    assert "Данные не поступают онлайн" not in page
    assert "Отклонение последнего изделия в σ" not in page
    assert 'name="latest_index"' in page
    assert f'max="{last_index}" value="{last_index}"' in page


def test_defect_product_card_shows_table_without_chart():
    page = dashboard.app.test_client().get("/attention").get_data(as_text=True)
    card_start = page.index("<h2>Карточка бракованного изделия</h2>")
    card_end = page.index("</section>", card_start)
    card = page[card_start:card_end]

    assert "<table" in card
    assert '<div class="chart">' not in card


def test_attention_page_can_select_another_product():
    page = dashboard.app.test_client().post(
        "/attention", data={"mode": "latest", "latest_index": "0"}
    ).get_data(as_text=True)
    assert "Строка 0 · 19.07.2008 11:55" in page
    assert "Проверяем выбранное изделие из истории" not in page
    assert "Данные не поступают онлайн" not in page


def test_sensors_page_hides_sensor_time_correlation_table():
    page = dashboard.app.test_client().get("/sensors").get_data(as_text=True)
    assert "Связь ключевых датчиков со временем" not in page
    assert "Связь со временем" not in page


def test_zone_table_is_colored_and_sorted_red_yellow_green():
    import numpy as np
    import pandas as pd

    ordered = dashboard.latest_sensor_readings(
        {
            "keys": ["green", "yellow", "red"],
            "clean": pd.DataFrame({"green": [0], "yellow": [2.5], "red": [3.5]}),
            "borders": pd.DataFrame([
                {"Датчик": name, "Центр": 0, "Сигма": 1, "Направление": 1,
                 "Жёлтая": 2, "Красная": 3}
                for name in ["green", "yellow", "red"]
            ]),
            "bad": np.array([0]),
        },
        0,
    )
    html = dashboard.zone_table_html(ordered)
    assert html.index("red") < html.index("yellow") < html.index("green")
    assert "#fae4e4" in html
    assert "#fff3d4" in html
    assert "#e4f3e9" in html


def test_zone_table_can_hide_zone_text_while_preserving_row_colors():
    import pandas as pd

    html = dashboard.zone_table_html(pd.DataFrame({
        "Датчик": ["red_sensor", "yellow_sensor"],
        "Зона": ["красная", "жёлтая"],
        "Значение": [3.2, 2.4],
    }), show_zone=False)

    assert "<th>Зона</th>" not in html
    assert "красная" not in html
    assert "#fae4e4" in html
    assert "#fff3d4" in html


def test_sensor_deciles_are_returned_in_numeric_order():
    groups = dashboard.charts.sensor_detail_charts(
        dashboard.STATE["data"], dashboard.STATE["data"]["keys"][0]
    )[3]
    numbers = [int(label) for label in groups["Дециль"]]
    assert numbers == list(range(1, len(numbers) + 1))


def test_zone_chart_uses_the_same_red_and_yellow_colors():
    import pandas as pd

    chart = dashboard.charts.signal_basis_chart(pd.DataFrame({
        "Датчик": ["sensor_red", "sensor_yellow"],
        "Зона": ["красная", "жёлтая"],
        "Сигмы": [3.2, 2.4],
    }))
    assert "#a52e32" in chart
    assert "#f2c500" in chart


def test_latest_sensor_readings_uses_one_requested_product():
    import numpy as np
    import pandas as pd

    data = {
        "keys": ["sensor_001"],
        "clean": pd.DataFrame({"sensor_001": [0.0, 2.5]}),
        "borders": pd.DataFrame([{
            "Датчик": "sensor_001",
            "Центр": 0.0,
            "Сигма": 1.0,
            "Направление": 1,
            "Жёлтая": 2.0,
            "Красная": 3.0,
        }]),
        "bad": np.array([0, 1]),
    }
    result = dashboard.latest_sensor_readings(data, 1)
    assert result.loc[0, "Зона"] == "жёлтая"
    assert result.loc[0, "Значение"] == 2.5

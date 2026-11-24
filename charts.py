"""Plotly-графики для страниц Flask."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import functions

ZONE_COLORS = {"красная": "#a52e32", "жёлтая": "#f2c500", "зелёная": "#18764b"}
ZONE_ORDER = ["красная", "жёлтая", "зелёная"]


def chart_html(chart):
    """Преобразует график в HTML-фрагмент с локальной библиотекой Plotly."""
    chart.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#25333b"},
        title={"font": {"size": 16}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=["#27645f", "#c18b32", "#57728a", "#a8433c", "#64845f"],
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#202a31"}},
        margin={"l": 64, "r": 24, "t": 70, "b": 56},
    )
    chart.update_xaxes(showline=True, linecolor="#cfd7d5", gridcolor="#e8ece9",
                       zerolinecolor="#d6ddda")
    chart.update_yaxes(showline=True, linecolor="#cfd7d5", gridcolor="#e8ece9",
                       zerolinecolor="#d6ddda")
    return chart.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )


def target_charts(data):
    """Строит столбцы годных/брака и недельный график доли брака."""
    counts = pd.DataFrame({"Изделие": ["Брак", "Годное"],
                           "Доля": [data["bad"].mean(), 1 - data["bad"].mean()]})
    bars = px.bar(counts, x="Изделие", y="Доля", color="Изделие",
                  color_discrete_map={"Брак": "#a8433c", "Годное": "#27734d"},
                  category_orders={"Изделие": ["Годное", "Брак"]},
                  labels={"Изделие": "Результат контроля", "Доля": "Доля изделий"},
                  title="Доля годных изделий и брака")
    bars.update_yaxes(tickformat=".2%", range=[0, 1])
    line = px.line(data["weekly"], x="Дата", y="Доля", markers=True,
                   title="Доля брака по неделям",
                   labels={"Дата": "Неделя", "Доля": "Доля брака"})
    line.update_yaxes(tickformat=".2%", rangemode="tozero")
    return chart_html(bars), chart_html(line)


def data_charts(data):
    """Показывает количество сохранённых/удалённых датчиков и пропуски."""
    counts = pd.DataFrame({"Группа": ["Осталось", "Пропуски", "0–1 уникальных значений"],
                           "Датчиков": [len(data["clean"].columns), len(data["remove_missing"]),
                                        len(data["remove_unique"])]})
    bars = px.bar(counts, x="Группа", y="Датчиков", color="Группа",
                  title="Результат очистки датчиков",
                  labels={"Группа": "Причина", "Датчиков": "Число датчиков"},
                  color_discrete_map={
                      "Осталось": "#27734d",
                      "Пропуски": "#c18b32",
                      "0–1 уникальных значений": "#a8433c",
                  })
    missing = px.histogram(pd.DataFrame({"Доля пропусков": data["missing"]}),
                           x="Доля пропусков", nbins=20)
    return chart_html(bars), chart_html(missing)


def sensor_charts(data):
    """Строит рейтинги двух методов и их корреляционную матрицу."""
    mann = px.bar(data["mann"], x="AUC", y="Датчик", orientation="h",
                  title="Различие показаний между классами",
                  labels={"AUC": "AUC-подобная мера различия", "Датчик": "Датчик"})
    forest = px.bar(data["forest"], x="Важность", y="Датчик", orientation="h",
                    title="Важность признаков в Random Forest",
                    labels={"Важность": "Относительная важность", "Датчик": "Датчик"})
    names = data["keys"] + ["Брак (0/1)"]
    heat = px.imshow(data["correlations"].loc[names, names], text_auto=".2f",
                     zmin=-1, zmax=1, color_continuous_scale="RdBu",
                     title="Корреляция рангов показаний и метки брака",
                     labels={"x": "Показатель", "y": "Показатель", "color": "Связь"})
    heat.update_layout(height=max(760, 28 * len(names) + 180))
    heat.update_traces(textfont={"size": 11})
    heat.update_xaxes(tickangle=-45, tickfont={"size": 11}, automargin=True)
    heat.update_yaxes(tickfont={"size": 11}, automargin=True)
    return chart_html(mann), chart_html(forest), chart_html(heat)


def time_charts(data):
    """Строит расширенную корреляцию и недельные медианы датчиков."""
    medians = px.line(data["medians"], x="Неделя", y=data["keys"][:6],
                      title="Медианные показания ключевых датчиков по неделям",
                      labels={"Неделя": "Неделя", "value": "Медианное значение",
                              "variable": "Датчик"})
    return chart_html(medians)


def boundary_chart(data, name):
    """Рисует распределения датчика и односторонние границы 2σ и 3σ."""
    frame = data["clean"][[name]].copy()
    frame["Статус"] = ["Брак" if value else "Годное" for value in data["bad"]]
    chart = px.histogram(frame, x=name, color="Статус", barmode="overlay", opacity=0.65)
    border = data["borders"][data["borders"]["Датчик"] == name].iloc[0]
    chart.add_vline(x=border["Центр"], line_color="blue", annotation_text="Типичное")
    chart.add_vline(x=border["Жёлтая"], line_color=ZONE_COLORS["жёлтая"], annotation_text="2σ")
    chart.add_vline(x=border["Красная"], line_color=ZONE_COLORS["красная"], annotation_text="3σ")
    return chart_html(chart)


def when_where_charts(data):
    """Строит недельные количества брака и замеров, а также тепловую карту."""
    weekly_data = data["weekly"]
    week_ends = pd.to_datetime(weekly_data["Дата"])
    week_labels = (
        (week_ends - pd.Timedelta(days=6)).dt.strftime("%d.%m.%Y")
        + " – "
        + week_ends.dt.strftime("%d.%m.%Y")
    )
    hover_template = (
        "Неделя: %{customdata}<br>"
        "Дата окончания недели: %{x|%d.%m.%Y}<br>"
        "%{y}<extra></extra>"
    )
    weekly = go.Figure()
    weekly.add_trace(
        go.Bar(x=week_ends, y=weekly_data["sum"], customdata=week_labels,
               name="Количество брака",
               marker={"color": "#c43d3d"},
               hovertemplate=hover_template.replace("%{y}", "Количество брака: %{y:.0f}")),
    )
    weekly.add_trace(
        go.Bar(x=week_ends, y=weekly_data["count"], customdata=week_labels,
               name="Количество замеров",
               marker={"color": "#27734d"},
               hovertemplate=hover_template.replace("%{y}", "Количество замеров: %{y:.0f}")),
    )
    weekly.update_layout(barmode="group")
    weekly.update_xaxes(title_text="Неделя")
    weekly.update_yaxes(title_text="Количество изделий")
    heat = px.imshow(data["heatmap"], text_auto=".0%", zmin=0, zmax=1, aspect="auto")
    return chart_html(weekly), chart_html(heat)


def health_chart(health):
    """Сравнивает долю пропусков в последнем окне с нормой и порогами."""
    chart = go.Figure()
    chart.add_trace(go.Bar(
        x=health["Датчик"],
        y=health["Пропуски за последние 50"],
        name="Пропуски в последних 50 замерах",
        marker_color="#2874a6",
        customdata=health[["Пустых в окне"]].to_numpy(),
        hovertemplate=(
            "Датчик: %{x}<br>Доля пропусков: %{y:.2%}"
            "<br>Пропущено в окне: %{customdata[0]}<extra></extra>"
        ),
    ))
    markers = [
        ("Зелёный порог", "Обычная доля пропусков", "#18764b"),
        ("Жёлтый порог", "Жёлтая граница", "#d99a00"),
        ("Красный порог", "Красная граница", "#c43d3d"),
    ]
    for label, column, color in markers:
        chart.add_trace(go.Scatter(
            x=health["Датчик"],
            y=health[column],
            mode="markers",
            name=label,
            marker={"color": color, "symbol": "line-ew", "size": 18,
                    "line": {"color": color, "width": 3}},
            hovertemplate=f"Датчик: %{{x}}<br>{label}: %{{y:.2%}}<extra></extra>",
        ))
    chart.update_layout(
        title="Доля пропусков в последних замерах по датчикам",
        barmode="overlay",
        height=max(480, 28 * len(health)),
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 60, "r": 30, "t": 100, "b": 130},
    )
    chart.update_xaxes(title="Датчик", tickangle=-45)
    chart.update_yaxes(title="Доля пропусков", tickformat=".2%", range=[0, 1])
    return chart_html(chart)


def health_history_chart(data):
    """Показывает недельную долю пропусков как историческую тепловую карту."""
    frame = data["clean"][data["keys"]].isna().copy()
    frame["Неделя"] = data["labels"]["datetime"].dt.to_period("W").astype(str)
    health = frame.groupby("Неделя")[data["keys"]].mean().T
    return chart_html(px.imshow(health, text_auto=".0%", zmin=0, zmax=1, aspect="auto"))


def missing_defect_correlation(data):
    """Сопоставляет недельное число пропусков с числом бракованных изделий."""
    dates = data["labels"]["datetime"]
    sensor_names = data.get("keys", list(data["clean"].columns))
    missing_per_item = data["clean"][sensor_names].isna().sum(axis=1).to_numpy()
    weekly = pd.DataFrame({
        "Неделя": dates.dt.to_period("W-MON").astype(str),
        "Брак": np.asarray(data["bad"], dtype=int),
        "Пропуски": missing_per_item,
    })
    totals = weekly.groupby("Неделя").agg(
        **{"Количество брака": ("Брак", "sum"),
           "Количество замеров": ("Брак", "size"),
           "Количество пропусков": ("Пропуски", "sum")}
    ).reset_index()
    possible_readings = totals["Количество замеров"] * len(sensor_names)
    totals["Доля брака"] = totals["Количество брака"] / totals["Количество замеров"]
    totals["Доля пропусков"] = totals["Количество пропусков"] / possible_readings.replace(0, np.nan)

    chart = go.Figure()
    customdata = totals[[
        "Неделя", "Количество замеров", "Доля пропусков", "Доля брака"
    ]].to_numpy()
    chart.add_trace(go.Scatter(
        x=totals["Количество пропусков"],
        y=totals["Количество брака"],
        customdata=customdata,
        mode="markers",
        name="Неделя",
        marker={"color": "#2874a6", "size": 10, "opacity": 0.8},
        hovertemplate=(
            "Неделя: %{customdata[0]}<br>"
            "Пропущенных показаний: %{x}<br>"
            "Бракованных изделий: %{y}<br>"
            "Всего изделий: %{customdata[1]}<br>"
            "Доля пропусков: %{customdata[2]:.2%}<br>"
            "Доля брака: %{customdata[3]:.2%}<extra></extra>"
        ),
    ))
    chart.update_xaxes(title="Количество пропущенных показаний датчиков за неделю")
    chart.update_yaxes(title="Количество бракованных изделий за неделю")

    count_correlation = np.nan
    if len(totals) >= 2 and totals["Количество пропусков"].nunique() > 1 \
            and totals["Количество брака"].nunique() > 1:
        count_correlation = totals["Количество пропусков"].corr(
            totals["Количество брака"], method="spearman"
        )
        chart.update_layout(
            title=f"Недельные количества: пропуски и брак (ρ Спирмена = {count_correlation:+.2f})"
        )
        slope, intercept = np.polyfit(
            totals["Количество пропусков"], totals["Количество брака"], 1
        )
        x_range = np.array([
            totals["Количество пропусков"].min(),
            totals["Количество пропусков"].max(),
        ])
        chart.add_trace(go.Scatter(
            x=x_range,
            y=slope * x_range + intercept,
            mode="lines",
            name="Линейный тренд",
            line={"color": "#c43d3d", "dash": "dash"},
            hoverinfo="skip",
        ))
    else:
        chart.update_layout(title="Недельные количества: пропуски и брак")

    rows = []
    weekly_defect_rate = totals.set_index("Неделя")["Доля брака"]
    missing = data["clean"][sensor_names].isna().astype(float)
    missing.index = weekly["Неделя"].to_numpy()
    weekly_missing_rate = missing.groupby(level=0).mean()
    for name in weekly_missing_rate.columns:
        paired = pd.concat(
            [weekly_defect_rate, weekly_missing_rate[name]], axis=1, join="inner"
        ).dropna()
        if len(paired) < 5 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            continue
        rows.append([name, paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"),
                     len(paired)])

    result = pd.DataFrame(rows, columns=["Датчик", "Корреляция Спирмена", "Недель"])
    result = result.sort_values(
        "Корреляция Спирмена", key=lambda values: values.abs(), ascending=False
    ).head(20)
    return chart_html(chart), result


def defect_product_chart(data, index):
    """Рисует ключевые показания выбранного бракованного изделия."""
    values = data["clean"].loc[index, data["keys"]].dropna().rename("Значение")
    frame = values.reset_index()
    chart = px.bar(
        frame, x="index", y="Значение",
        title="Показания датчиков выбранного изделия с браком",
        labels={"index": "Датчик", "Значение": "Показание"},
    )
    return chart_html(chart)


def sensor_detail_charts(data, name):
    """Строит гистограмму значений датчика и долю брака по десяти группам."""
    frame = data["clean"][[name]].copy()
    frame["Статус"] = ["Брак" if value else "Годное" for value in data["bad"]]
    visible = frame.dropna(subset=[name])
    lower, upper = visible[name].quantile([0.005, 0.995])
    displayed = visible[visible[name].between(lower, upper)]
    distribution_colors = {"Годное": "#0072b2", "Брак": "#e69f00"}
    threshold_colors = {
        "Типичный уровень": "#00845a",
        "Жёлтая граница": "#b87900",
        "Красная граница": "#c43d3d",
    }
    histogram = px.histogram(
        displayed, y=name, color="Статус", barmode="group",
        histnorm="probability", orientation="h", opacity=1, nbins=45,
        color_discrete_map=distribution_colors,
        labels={name: "Значение датчика", "probability": "Probability"},
    )
    border = data["borders"][data["borders"]["Датчик"] == name].iloc[0]
    thresholds = [
        ("Типичный уровень", border["Центр"]),
        ("Жёлтая граница", border["Жёлтая"]),
        ("Красная граница", border["Красная"]),
    ]
    for label, value in thresholds:
        color = threshold_colors[label]
        histogram.add_hline(
            y=value, line_color=color, line_width=4,
            line_dash="solid" if label == "Типичный уровень" else "dash",
        )
        histogram.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=label,
            line={
                "color": color,
                "width": 4,
                "dash": "solid" if label == "Типичный уровень" else "dash",
            },
            hoverinfo="skip",
        ))
    value_range = [lower, upper, *[value for _, value in thresholds]]
    padding = max((max(value_range) - min(value_range)) * 0.06, 1e-9)
    histogram.update_layout(
        height=500,
        bargap=0.08,
        legend={"orientation": "h", "y": 1.12, "x": 0.02, "xanchor": "left"},
        margin={"l": 75, "r": 30, "t": 105, "b": 65},
    )
    histogram.update_xaxes(title="Probability", tickformat=".2%", rangemode="tozero")
    histogram.update_yaxes(
        title="Значение датчика",
        range=[min(value_range) - padding, max(value_range) + padding],
        zeroline=False,
    )
    values = frame.dropna(subset=[name]).copy()
    values["Дециль"] = pd.qcut(values[name], 10, duplicates="drop")
    values["Брак"] = data["bad"][values.index]
    grouped = (values.groupby("Дециль", observed=True, sort=True)["Брак"]
               .agg(["mean", "count"]).reset_index())
    decile_intervals = grouped["Дециль"].tolist()
    grouped["Дециль"] = [str(index) for index in range(1, len(grouped) + 1)]
    grouped["Диапазон значений"] = [
        f"({interval.left:.2f}, {interval.right:.2f}]"
        for interval in decile_intervals
    ]
    grouped = grouped.rename(columns={"mean": "Доля брака", "count": "Изделий"})
    grouped = grouped[[
        "Дециль",
        "Диапазон значений",
        "Доля брака",
        "Изделий",
    ]]
    bars = px.bar(
        grouped, x="Дециль", y="Доля брака",
        title="Доля брака в группах значений датчика",
        labels={"Дециль": "Номер дециля", "Доля брака": "Доля брака"},
        category_orders={"Дециль": grouped["Дециль"].tolist()},
        custom_data=["Дециль", "Диапазон значений", "Изделий"],
    )
    bars.update_traces(
        hovertemplate="Дециль: %{customdata[0]}<br>Диапазон: %{customdata[1]}<br>"
                      "Доля брака: %{y:.2%}<br>Изделий: %{customdata[2]}<extra></extra>"
    )
    bars.update_yaxes(tickformat=".2%", rangemode="tozero")
    bars.update_xaxes(
        tickvals=grouped["Дециль"].tolist(),
        ticktext=[
            f"{number}  {interval}"
            for number, interval in zip(
                grouped["Дециль"], grouped["Диапазон значений"]
            )
        ],
        tickangle=-45,
    )

    weekly_values = pd.DataFrame({
        "Дата": pd.to_datetime(data["labels"]["datetime"]),
        "Показание": data["clean"][name],
    }).dropna(subset=["Показание"])
    iso_calendar = weekly_values["Дата"].dt.isocalendar()
    weekly_values["Неделя"] = (
        iso_calendar["year"].astype(str)
        + " · № "
        + iso_calendar["week"].astype(str)
    )
    weekly = (weekly_values.groupby("Неделя", sort=True)["Показание"]
              .agg(["median", "count"]).reset_index()
              .rename(columns={"median": "Медианное показание", "count": "Измерений"}))
    weekly_chart = px.bar(
        weekly, x="Неделя", y="Медианное показание",
        title=f"Медианное показание {name} по неделям",
        labels={"Неделя": "Неделя", "Медианное показание": "Медианное показание"},
        custom_data=["Измерений"],
    )
    weekly_chart.update_traces(
        hovertemplate="Неделя: %{x}<br>Медиана: %{y}<br>Измерений: %{customdata[0]}<extra></extra>"
    )
    border = data["borders"][data["borders"]["Датчик"] == name].iloc[0]
    if len(weekly):
        week_range = [weekly["Неделя"].iloc[0], weekly["Неделя"].iloc[-1]]
        weekly_chart.add_trace(go.Scatter(
            x=week_range,
            y=[border["Жёлтая"], border["Жёлтая"]],
            mode="lines",
            name="Жёлтая граница",
            line={"color": "#d99a00", "width": 2, "dash": "dash"},
            hovertemplate="Жёлтая граница: %{y:.2f}<extra></extra>",
        ))
        weekly_chart.add_trace(go.Scatter(
            x=week_range,
            y=[border["Красная"], border["Красная"]],
            mode="lines",
            name="Красная граница",
            line={"color": "#c43d3d", "width": 3, "dash": "dash"},
            hovertemplate="Красная граница: %{y:.2f}<extra></extra>",
        ))
    weekly_chart.update_layout(
        legend={"orientation": "h", "y": 1.08, "x": 0}
    )
    weekly_chart.update_xaxes(tickangle=-45)
    return chart_html(histogram), chart_html(bars), chart_html(weekly_chart), grouped


def risk_charts(risk, contributions):
    """Рисует ROC-кривую и вклады датчиков в выбранное изделие."""
    curve = px.line(
        risk["curve"], x="Ложные срабатывания", y="Найденный брак",
        title="Компромисс между найденным браком и ложными тревогами",
        labels={"Ложные срабатывания": "Доля ложных тревог среди годных",
                "Найденный брак": "Доля найденного брака"},
    )
    curve.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Случайный ориентир",
                      line={"dash": "dash", "color": "#8b969a"})
    curve.update_xaxes(tickformat=".2%", range=[0, 1])
    curve.update_yaxes(tickformat=".2%", range=[0, 1])
    bars = px.bar(contributions, x="Вклад", y="Датчик", orientation="h",
                  color="Вклад", color_continuous_scale=["#57728a", "#c9ceca", "#a8433c"],
                  color_continuous_midpoint=0,
                  labels={"Вклад": "Вклад в оценку модели", "Датчик": "Датчик"})
    bars.add_vline(x=0, line_color="#59676d", line_width=1)
    return chart_html(curve), chart_html(bars)


def p_chart(data):
    """Рисует скользящую долю брака в окне из последних изделий."""
    window = functions.SIGNAL_WINDOW
    frame = pd.DataFrame({"Дата": data["labels"]["datetime"], "Брак": data["bad"]})
    frame["Доля"] = frame["Брак"].rolling(window, min_periods=window).mean()
    rate = data["ordinary"]
    sigma = np.sqrt(rate * (1 - rate) / window)
    chart = px.line(frame, x="Дата", y="Доля", title="Карта Шухарта доли брака")
    chart.add_hline(y=rate, line_color="green", annotation_text="Обычный уровень")
    upper = rate + 3 * sigma
    lower = max(0, rate - 3 * sigma)
    chart.add_hline(y=upper, line_color="red", line_dash="dash", annotation_text="Верхняя граница")
    chart.add_hline(y=lower, line_color="red", line_dash="dash", annotation_text="Нижняя граница")
    outside = frame[frame["Доля"] > upper]
    chart.update_yaxes(title="Доля брака в окне", tickformat=".2%", range=[0, 0.6])
    chart.add_scatter(x=outside["Дата"], y=outside["Доля"], mode="markers",
                      name="Рост выше коридора", marker={"color": "red"})
    return chart_html(chart)


def period_scale_charts(data, period):
    """Показывает один период роста в дневном, недельном и месячном масштабе."""
    start = period["Начало"] - pd.Timedelta(days=35)
    end = period["Конец"] + pd.Timedelta(days=35)
    rows = []
    for frequency, name in [("D", "Дни"), ("W-MON", "Недели"), ("MS", "Месяцы")]:
        frame = pd.DataFrame({"Дата": data["labels"]["datetime"], "Брак": data["bad"]})
        frame = frame.set_index("Дата").loc[start:end].resample(frequency)["Брак"].agg(["sum", "count"])
        frame["Доля"] = frame["sum"] / frame["count"]
        frame["Порог"] = functions.binom.ppf(0.99, frame["count"], data["ordinary"]) / frame["count"]
        for date, row in frame.iterrows():
            rows.append([date, name, row["Доля"], row["Порог"]])
    values = pd.DataFrame(rows, columns=["Дата", "Масштаб", "Доля", "Порог"])
    chart = px.line(values,
                    x="Дата", y="Доля", color="Масштаб", markers=True)
    chart.add_hline(y=data["ordinary"], line_color="green", annotation_text="Обычный уровень")
    for name in values["Масштаб"].unique():
        scale = values[values["Масштаб"] == name]
        chart.add_scatter(x=scale["Дата"], y=scale["Порог"], mode="lines",
                          name="Верхняя граница — " + name,
                          line={"dash": "dash"})
    chart.update_yaxes(tickformat=".2%", rangemode="tozero")
    return chart_html(chart)


def signal_basis_chart(basis):
    """Строит столбчатый график σ-отклонений датчиков при сигнале."""
    if basis.empty:
        return ""
    chart = px.bar(basis, x="Сигмы", y="Датчик", color="Зона", orientation="h",
                   color_discrete_map=ZONE_COLORS, category_orders={"Зона": ZONE_ORDER})
    return chart_html(chart)

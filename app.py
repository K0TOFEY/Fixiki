"""Интерфейс Streamlit для простого просмотра качества изделий."""

from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

import calc

st.set_page_config(page_title="Контроль качества", layout="wide")
SECTIONS = ["О данных", "Ключевые датчики", "Датчики и время", "Красные значения",
            "Когда", "Датчик × неделя", "Датчик подробно",
            "Карточка брака", "Здоровье датчиков", "Внимание сейчас",
            "Что мы предсказываем (целевая переменная)", "Оценка риска по модели"]
HISTOGRAM_BINS = 20  # Число столбцов в гистограмме пропусков.
DECILE_COUNT = 10  # Делим значения датчика на десять групп.
RECENT_COUNT = 20  # Смотрим последние двадцать изделий.
TABLE_ROWS = 10  # В раскрытом расчёте показываем первые десять строк.
SILENT_LIMIT = 0.80  # Свыше 80% пустых строк датчик считается молчащим.
ZONE_COLORS = {"красная": "#a52e32", "жёлтая": "#f2c500", "зелёная": "#18764b"}
# Что делает: читает файлы и считает один раз; получает их байты; возвращает словарь таблиц; нужен, чтобы не повторять расчёты.
@st.cache_data
def get_calculation(sensors, labels):
    # ВАЖНО: кэшируем расчёт уже выбранного диапазона, а не элементы интерфейса.
    if labels.empty:
        raise ValueError("В выбранном периоде нет изделий. Выберите другие даты.")
    return calc.calculate(sensors, labels)
# Что делает: показывает календарь периода; получает таблицы и возвращает выбранные строки; ограничивает анализ датами файла.
def select_dataset_period(sensors, labels):
    # Календарь ограничен самой ранней и поздней датами загруженных данных.
    first_date = labels["datetime"].min().date()
    last_date = labels["datetime"].max().date()
    selected = st.sidebar.date_input("Период данных", value=(first_date, last_date),
                                     min_value=first_date, max_value=last_date,
                                     format="DD.MM.YYYY", key=f"period_{first_date}_{last_date}")
    if not isinstance(selected, tuple) or len(selected) != 2:
        st.info("Выберите начальную и конечную дату.")
        return sensors.iloc[0:0], labels.iloc[0:0]
    start_date, end_date = selected
    keep = labels["datetime"].dt.date
    keep = (keep >= start_date) & (keep <= end_date)
    return sensors.loc[keep].reset_index(drop=True), labels.loc[keep].reset_index(drop=True)
# Что делает: показывает пояснение и первые строки; получает текст и таблицу; ничего не возвращает; помогает увидеть промежуточные числа.
def show_explanation(text, table):
    with st.expander("Как это посчитано"):
        st.write(text)
        st.dataframe(table.head(TABLE_ROWS), width="stretch")
# Что делает: показывает объём и очистку; получает расчётные таблицы; ничего не возвращает; помогает проверить качество входных данных.
def show_section_1(data):
    st.title("О данных: загружены изделия и датчики")
    # Сначала показываем объём и период исходной таблицы.
    sensors = data["sensors"]
    labels = data["labels"]
    st.write(f"Строк: {len(sensors)} · датчиков: {sensors.shape[1]} · меток брака: {data['bad'].sum()} ({data['bad'].mean():.2%})")
    missing_before = sensors.isna().to_numpy().mean()
    missing_after = data["clean"].isna().to_numpy().mean()
    st.write(f"Пустые ячейки: {missing_before:.2%} до очистки и {missing_after:.2%} после.")
    st.write(f"Период: {labels['datetime'].min()} — {labels['datetime'].max()}")
    st.write(f"Удалено: {len(data['missing_removed'])} из-за пропусков и {len(data['unique_removed'])} с 0–1 уникальными значениями; осталось {data['clean'].shape[1]}.")
    left, right = st.columns(2)
    count_table = pd.DataFrame({"Группа": ["Осталось", "Пропусков >50%", "0–1 уникальных значений"], "Датчиков": [data["clean"].shape[1], len(data["missing_removed"]), len(data["unique_removed"])]})
    # Рядом показываем причины удаления и распределение пропусков.
    left.plotly_chart(px.bar(count_table, x="Группа", y="Датчиков", color="Группа"), width="stretch")
    missing_table = pd.DataFrame({"Доля пропусков": data["missing"]})
    right.plotly_chart(px.histogram(missing_table, x="Доля пропусков", nbins=HISTOGRAM_BINS), width="stretch")
    st.caption("Как читать: метка 1 считается браком (это нужно подтвердить); удаляются датчики с пропусками свыше половины и константы.")
    show_explanation("Доля пропусков считается отдельно для каждого столбца. Датчики с большим числом пропусков удаляются первыми. По умолчанию также удаляются столбцы с одним или нулём уникальных значений.", data["removed"])
# Что делает: показывает два рейтинга и корреляции; получает таблицы расчётов; ничего не возвращает; помогает найти отличающиеся показания.
def show_section_2(data):
    st.title("Ключевые датчики: два метода показывают отличия")
    left, right = st.columns(2)
    # Оба рейтинга видны одновременно для сравнения.
    left.plotly_chart(px.bar(data["mann"], x="AUC", y="Датчик", orientation="h", title="Манн—Уитни"), width="stretch")
    right.plotly_chart(px.bar(data["forest"], x="Важность", y="Датчик", orientation="h", title="Random Forest"), width="stretch")
    ranking = data["mann"].head(15)
    top_sensors = ranking["Датчик"].tolist()
    fail_mask = np.asarray(data["bad"], dtype=bool)
    figure, axes = plt.subplots(3, 5, figsize=(15, 9))
    for axis, sensor in zip(axes.ravel(), top_sensors):
        values = [
            data["clean"].loc[~fail_mask, sensor].dropna(),
            data["clean"].loc[fail_mask, sensor].dropna(),
        ]
        axis.boxplot(values, tick_labels=["норма", "брак"], showfliers=False)
        axis.set_title(sensor, fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
    for axis in axes.ravel()[len(top_sensors):]:
        axis.set_visible(False)
    figure.suptitle("Топ-15 датчиков, различающихся между нормой и браком")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    st.pyplot(figure)
    plt.close(figure)
    st.caption("Датчики выбраны по рейтингу Манна—Уитни на первых 80% периода. Коробки построены по всем изделиям выбранного периода; линия внутри коробки — медиана, границы коробки — квартили. Выбросы не показаны.")
    names = data["keys"] + ["Брак (0/1)"]
    st.plotly_chart(px.imshow(data["correlations"].loc[names, names], text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu"), width="stretch")
    st.write("В обоих списках: " + (", ".join(data["both"]) if data["both"] else "совпадений нет"))
    st.caption("Как читать: чем выше AUC или важность, тем сильнее отличие/вклад. Дрейф во времени может создавать кажущуюся связь с браком.")
    show_explanation("Манн—Уитни сравнивает значения годных и бракованных изделий в опорном периоде. Random Forest строит деревья решений и ранжирует датчики по встроенной важности. Корреляция Спирмена сравнивает ранги и использует доступные пары наблюдений.", data["mann"])
# Что делает: показывает связь датчиков со временем и браком; получает показания и метки; ничего не возвращает; помогает заметить дрейф.
def show_section_3(data):
    st.title("Датчики и время: некоторые показания могут дрейфовать")
    frame = data["clean"][data["keys"]].copy()
    # Добавляем время и метку как ещё два ряда для сравнения.
    frame["Время"] = np.arange(len(frame))
    frame["Брак (0/1)"] = data["bad"]
    correlations = frame.corr(method="spearman", min_periods=calc.MIN_CORRELATION_ROWS).fillna(0)
    st.plotly_chart(px.imshow(correlations, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu"), width="stretch")
    for name in data["keys"]:
        rho = correlations.loc[name, "Время"]
        if abs(rho) > calc.DRIFT_LIMIT:
            count = int(frame[name].notna().sum())
            st.warning(f"{name} связан со временем (ρ={rho:.2f}, N={count}); возможен дрейф.")
    st.plotly_chart(px.line(data["weekly_medians"], x="Неделя", y=data["keys"][:calc.KEY_MEDIAN_COUNT]), width="stretch")
    st.caption("Как читать: ρ выше 0,3 или ниже −0,3 — повод проверить изменение датчика во времени.")
    show_explanation("Время — номер изделия после сортировки по дате. Метка брака записана как 0 или 1. Связь со временем может объяснить кажущееся отличие датчика от брака.", correlations.reset_index())
# Что делает: показывает скользящую долю брака и контрольные границы; получает p-карту; ничего не возвращает; выделяет рост брака.
def show_section_4(data):
    st.title("Красные значения: скользящая доля брака вышла за UCL")
    result = data["p_chart"]
    frame = result["frame"]
    chart = px.line(frame, x="datetime", y="rolling_fail_rate",
                    title="p-карта: скользящая доля брака во времени",
                    labels={"datetime": "Дата и время",
                            "rolling_fail_rate": "Доля брака в окне"})
    chart.add_hline(y=result["p_bar"], line_color="green", annotation_text=f"CL = {result['p_bar']:.3f}")
    chart.add_hline(y=result["ucl"], line_color="red", line_dash="dash", annotation_text=f"UCL = {result['ucl']:.3f}")
    chart.add_hline(y=result["lcl"], line_color="red", line_dash="dash", annotation_text=f"LCL = {result['lcl']:.3f}")
    outside = frame[frame["Вне контроля"]]
    chart.add_scatter(x=outside["datetime"], y=outside["rolling_fail_rate"],
                      mode="markers", name="Вне контроля — рост брака",
                      marker={"color": "red", "size": 8})
    st.plotly_chart(chart, width="stretch")
    st.write(f"Характерный уровень p̄ = {result['p_bar']:.4f} · UCL = {result['ucl']:.4f} · LCL = {result['lcl']:.4f}")
    if len(outside):
        st.warning(f"Точек выше UCL: {len(outside)}; вне контроля {result['ooc_share']:.1%} наблюдений.")
    else:
        st.success(f"Точек выше UCL нет; вне контроля {result['ooc_share']:.1%} наблюдений.")
    st.caption(f"Как читать: доля брака считается в окне из {calc.P_CHART_WINDOW} изделий; первые {calc.P_CHART_MIN_PERIODS - 1} точек без скользящего значения. Красные точки отмечают только рост выше UCL.")
    show_explanation("Для каждой строки берётся средняя метка брака в последних 50 изделиях. p̄ — общая доля брака в выбранном периоде; сигма и контрольные границы считаются по формуле p-карты. Первые 19 точек не показываются для скользящей доли, так как задан минимум 20 изделий.", frame)
# Что делает: показывает долю брака по неделям; получает недельную таблицу; ничего не возвращает; помогает найти необычные недели.
def show_section_5(data):
    st.title("Когда: недели выше биномиального коридора отмечены вниманием")
    weekly = data["weekly"]
    # Один график показывает долю, другой — накопленное превышение.
    left, right = st.columns(2)
    chart = px.bar(weekly, x="Дата", y="Доля")
    chart.add_scatter(x=weekly["Дата"], y=weekly["Нижняя граница"], name="Нижняя граница", mode="lines", line={"color": "gray"})
    chart.add_scatter(x=weekly["Дата"], y=weekly["Верхняя граница"], name="Коридор 99%", mode="lines", line={"color": "gray"}, fill="tonexty", fillcolor="rgba(128,128,128,0.2)")
    chart.add_hline(y=data["ordinary"], line_color="green", annotation_text="Обычный уровень")
    left.plotly_chart(chart, width="stretch")
    right.plotly_chart(px.line(weekly, x="Дата", y="Лишний брак накопительно"), width="stretch")
    unusual = weekly[weekly["Необычная неделя"]]
    if len(unusual):
        st.warning("Недели выше верхней границы: " + ", ".join(unusual["Дата"].dt.strftime("%d.%m.%Y")))
        st.dataframe(unusual[["Дата", "Доля", "sum", "count"]], width="stretch")
    else:
        st.success("Недели выше верхней границы не найдены.")
    st.caption("Как читать: обычный уровень рассчитан как медиана дневных долей брака; короткий всплеск внутри недели может не найтись.")
    show_explanation("Для каждой недели биномиальное распределение даёт верхнюю границу при обычной доле. Неделя считается необычной, если её доля выше этой границы. Лишний брак — разница фактического числа и ожидаемого при обычной доле.", weekly)
# Что делает: показывает выходы за границы по неделям; получает матрицу датчик-неделя; ничего не возвращает; помогает увидеть совпадения.
def show_section_6(data):
    st.title("Датчик × неделя: показания выходили в жёлтую или красную зону")
    # Матрица содержит долю изделий, достигших хотя бы жёлтой зоны.
    st.plotly_chart(px.imshow(data["heatmap"], text_auto=".0%", zmin=0, zmax=1, aspect="auto"), width="stretch")
    st.caption("Как читать: более насыщенная ячейка означает большую долю изделий недели за границей; это не доказывает причину.")
    show_explanation("Сначала каждое значение сравнивается с границами своего датчика. Затем для каждой недели считается доля изделий в жёлтой или красной зоне. Пустые показания не считаются выходом за границу.", data["heatmap"].reset_index())
# Что делает: показывает значения датчика и брак по децилям; получает выбранный датчик; ничего не возвращает; помогает сравнить диапазоны.
def show_section_7(data):
    st.title("Датчик подробно: значения сравниваются с метками изделий")
    name = st.selectbox("Выберите датчик", data["keys"], key="detail_sensor")
    frame = data["clean"][[name]].copy()
    frame["Время"] = data["labels"]["datetime"]
    frame["Метка"] = ["Брак" if value == 1 else "Годное" for value in data["bad"]]
    frame["Брак"] = data["bad"]
    # Точки сохраняют отдельные изделия; децильная таблица упрощает сравнение.
    left, right = st.columns(2)
    chart = px.scatter(frame, x="Время", y=name, color="Метка", symbol="Метка",
                       color_discrete_map={"Брак": "red", "Годное": "blue"},
                       symbol_map={"Брак": "x", "Годное": "circle"})
    left.plotly_chart(chart, width="stretch")
    # Пропуски не входят в децильные группы.
    frame = frame.dropna(subset=[name])
    frame["Дециль"] = pd.qcut(frame[name], DECILE_COUNT, duplicates="drop").astype(str)
    deciles = frame.groupby("Дециль", observed=True)["Брак"].agg(["mean", "count"]).reset_index()
    right.plotly_chart(px.bar(deciles, x="Дециль", y="mean"), width="stretch")
    st.caption("Как читать: точки — отдельные изделия; столбцы показывают историческую долю брака по десяти диапазонам.")
    show_explanation("Изделия делятся на десять примерно равных групп по значению датчика. В каждой группе считается средняя метка 0/1. Это описывает историю и не говорит, что датчик вызвал брак.", deciles)
# Что делает: собирает выходы за границы; получает номер изделия и расчёты; возвращает таблицу; помогает разобрать брак.
def make_defect_details(data, index):
    rows = []
    for name in data["keys"]:
        value = data["clean"].loc[index, name]
        if pd.notna(value):
            zone, sigma, count, rate = calc.check_value(name, value, data)
            if zone != "зелёная":
                rows.append([name, value, zone, round(sigma, 2), count, f"{rate:.1%}"])
    detail = pd.DataFrame(rows, columns=["Датчик", "Значение", "Зона", "Сигмы", "Историческое N", "Доля брака"])
    return detail.sort_values("Сигмы", ascending=False).reset_index(drop=True)
# Что делает: показывает датчики за границами у брака; получает метки и границы; ничего не возвращает; помогает разобрать изделие.
def show_section_8(data):
    st.title("Карточка брака: значения датчиков сравниваются с историей")
    bad_indices = list(np.flatnonzero(data["bad"]))
    options = []
    for value in bad_indices:
        options.append(data["labels"].loc[value, "datetime"].strftime("%d.%m.%Y %H:%M"))
    selected = st.selectbox("Бракованное изделие", options)
    index = bad_indices[options.index(selected)]
    detail = make_defect_details(data, index)
    # График сравнивает значения всех выбранных датчиков этого изделия.
    product_values = data["clean"].loc[index, data["keys"]].dropna().rename("Значение")
    st.plotly_chart(px.bar(product_values.reset_index(), x="index", y="Значение"), width="stretch")
    if len(detail):
        st.dataframe(detail, width="stretch")
        st.warning("Зоны показывают исторические отличия; они не устанавливают причину.")
    else:
        st.success("У выбранного изделия ключевые датчики не вышли в жёлтую или красную зону.")
    st.caption("Как читать: история и число наблюдений приводятся рядом с каждым датчиком за границей.")
    show_explanation("Выберите изделие с меткой 1. Для каждого ключевого датчика значение переводится в число сигм от центра. В таблицу попадают только жёлтая и красная зоны.", detail)
# Что делает: показывает долю пропусков по неделям; получает датчики и даты; ничего не возвращает; отличает молчание от нормы.
def show_section_9(data):
    st.title("Здоровье датчиков: пропуски видны отдельно от нормы")
    frame = data["clean"][data["keys"]].copy()
    # Превращаем пустое значение в True и считаем долю по неделям.
    health = frame.isna()
    health["Неделя"] = data["labels"]["datetime"].dt.to_period("W").astype(str)
    health = health.groupby("Неделя")[data["keys"]].mean().T
    st.plotly_chart(px.imshow(health, text_auto=".0%", zmin=0, zmax=1, aspect="auto"), width="stretch")
    last_rows = data["clean"][data["keys"]].tail(RECENT_COUNT).isna().mean()
    silent = list(last_rows[last_rows > SILENT_LIMIT].index)
    if silent:
        st.error("Не передают данные (N=20, пропуски выше 80%): " + ", ".join(silent))
    else:
        st.success("Нет ключевых датчиков с пропусками более 80% в последних 20 изделиях.")
    st.caption("Как читать: 100% означает, что за эту неделю датчик не передавал показаний.")
    show_explanation("Для каждого датчика и недели берётся среднее от признака пропуска. Если больше 80% последних двадцати изделий пустые, датчик отмечается как не передающий данные. Такой датчик нельзя считать нормальным.", health.reset_index())
# Что делает: собирает последние значения; получает датчики и расчёты; возвращает таблицу предупреждений и молчащих датчиков.
def make_recent_details(data):
    rows = []
    for name in data["keys"]:
        recent = data["clean"][name].tail(RECENT_COUNT)
        if len(recent) and pd.notna(recent.iloc[-1]):
            value = recent.iloc[-1]
            zone, sigma, count, rate = calc.check_value(name, value, data)
            if zone != "зелёная":
                row_index = recent.index[-1]
                measured_at = data["labels"].loc[row_index, "datetime"].strftime("%d.%m.%Y %H:%M")
                rows.append([name, measured_at, value, zone, round(sigma, 2), count, f"{rate:.1%}"])
    columns = ["Датчик", "Дата", "Текущее", "Зона", "Сигмы", "Историческое N", "Доля брака"]
    detail = pd.DataFrame(rows, columns=columns)
    # Датчики с большим числом пропусков отдельно показываем как молчащие.
    missing_share = data["clean"][data["keys"]].tail(RECENT_COUNT).isna().mean()
    silent = []
    for name in missing_share.index:
        if missing_share[name] > SILENT_LIMIT or pd.isna(data["clean"][name].iloc[-1]):
            silent.append(name)
    return detail, silent
# Что делает: показывает последние показания и ручную форму; получает датчики и историю; ничего не возвращает; направляет внимание на проверку.
def show_section_10(data):
    st.title("Внимание сейчас: показания близки к границам из истории")
    detail, silent = make_recent_details(data)
    if len(detail):
        st.warning("Обратите внимание на датчики: " + ", ".join(detail["Датчик"]))
        st.dataframe(detail, width="stretch")
        if silent:
            st.error("Нет данных от датчиков: " + ", ".join(silent))
    else:
        if silent:
            st.error("Нет данных от датчиков: " + ", ".join(silent))
        else:
            st.success("Ключевые датчики в норме по последним изделиям. Это не гарантия годности.")
    if len(detail):
        st.plotly_chart(px.bar(detail, x="Датчик", y="Сигмы", color="Зона",
                       color_discrete_map=ZONE_COLORS), width="stretch")
    else:
        st.plotly_chart(px.histogram(data["clean"], x=data["keys"][0]), width="stretch")
    st.subheader("Проверить показания вручную")
    show_manual_form(data)
    st.caption("Как читать: «близко» означает не менее двух сигм от типичного уровня; рядом показана история, а не гарантия результата.")
    show_explanation("Последние показания сравниваются с односторонними границами, рассчитанными по годным изделиям опорного периода. Историческая доля брака считается среди всех изделий за жёлтой границей. Такие числа помогают проверить датчик, но не объясняют причину.", detail)
# Что делает: проверяет ручные значения; получает показания и границы; ничего не возвращает; добавляет исторический контекст.
def show_manual_form(data):
    for name in data["keys"]:
        # Цветное сообщение сопровождается долей брака и числом наблюдений.
        value = st.number_input(name, key="manual_" + name, value=None)
        if value is not None:
            zone, sigma, count, rate = calc.check_value(name, value, data)
            text = f"{name}: {zone} зона, отклонение {sigma:.1f}σ; в истории N={count}, брак {rate:.1%}."
            if zone == "красная":
                st.error(text)
            elif zone == "жёлтая":
                st.warning(text)
            else:
                st.success(text)


# Что делает: показывает целевую переменную; получает метки и недели; ничего не возвращает; объясняет, что считает модель.
def show_section_11(data):
    st.title("Что мы предсказываем: целевая переменная — метка изделия")
    st.warning("Метка 1 = брак, −1 = годное. Смысл меток нужно подтвердить у заказчика.")
    counts = pd.Series(data["bad"]).value_counts()
    target = pd.DataFrame({"Изделие": ["Брак", "Годное"],
                           "Доля": [counts.get(1, 0) / len(data["bad"]), counts.get(0, 0) / len(data["bad"])],
                           "Число": [counts.get(1, 0), counts.get(0, 0)]})
    target["Подпись"] = [f"{share:.1%} · N={count}" for share, count in zip(target["Доля"], target["Число"])]
    st.plotly_chart(px.bar(target, x="Изделие", y="Доля", text="Подпись",
                           color="Изделие", color_discrete_map={"Брак": "red", "Годное": "green"}),
                    width="stretch")
    st.plotly_chart(px.line(data["weekly"], x="Дата", y="Доля", markers=True,
                            labels={"Дата": "Неделя", "Доля": "Доля брака"}),
                    width="stretch")
    st.caption(f"Брак редкий ({data['bad'].mean():.1%} в выбранном периоде; 6,6% во всём наборе), поэтому accuracy как метрику не используем: она может быть высокой, даже если модель пропускает брак.")
    show_explanation("Метка 1 преобразуется в 1 (брак), а метка −1 — в 0 (годное). Доля брака показана по неделям, чтобы видеть изменение во времени. Это целевой столбец для проверки модели; его смысл необходимо подтвердить у заказчика.", target)


# Что делает: показывает проверку качества; получает модель и метки; ничего не возвращает; ставит качество рядом с оценками.
def show_risk_quality(risk, bad, split):
    test_bad = bad[split:]
    lower, upper = risk["auc_interval"]
    st.write(f"ROC-AUC = {risk['auc']:.3f} (95% интервал {lower:.3f}–{upper:.3f}); проверка N={len(test_bad)}, бракованных изделий={int(test_bad.sum())}.")
    if lower <= 0.5 or risk["auc"] < 0.70:
        st.warning("Качество слабое или неустойчивое: используйте оценку риска только как ориентир для проверки.")
    else:
        st.success("На отложенных по времени данных качество выше случайного ориентира; это не подтверждает причину брака.")
    chart = px.line(risk["curve"], x="Ложные срабатывания", y="Найденный брак",
                    title="ROC-кривая на последних 20%")
    chart.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Случайный ориентир",
                      line={"dash": "dash", "color": "gray"})
    st.plotly_chart(chart, width="stretch")
    if risk["rolling_auc"] is not None:
        low, high = risk["rolling_interval"]
        st.caption(f"Скользящая проверка по времени: ROC-AUC {risk['rolling_auc']:.3f} ({low:.3f}–{high:.3f}), N={risk['rolling_count']}; порядок строк не перемешивался.")


# Что делает: показывает доли брака по уровням; получает интервалы и размеры групп; ничего не возвращает; даёт исторический контекст.
def show_risk_summary(risk):
    summary = risk["summary"].copy()
    shares = []
    intervals = []
    for _, row in summary.iterrows():
        shares.append(f"{row['Доля брака']:.1%}" if pd.notna(row["Доля брака"]) else "нет данных")
        intervals.append(f"{row['Нижняя 95%']:.1%}–{row['Верхняя 95%']:.1%}" if row["Изделий"] else "нет данных")
    summary["Доля брака"] = shares
    summary["95% интервал"] = intervals
    st.dataframe(summary[["Уровень риска", "Доля брака", "Изделий", "95% интервал"]], width="stretch")


# Что делает: показывает уровни на последних изделиях; получает оценки и даты; ничего не возвращает; помогает выбрать запись для разбора.
def show_recent_risk(risk, labels):
    start = max(0, len(labels) - calc.RISK_REFERENCE_COUNT)
    options = [f"{index}: {labels.loc[index, 'datetime']}" for index in range(start, len(labels))]
    selected = st.selectbox("Выберите изделие для разбора", options, index=len(options) - 1)
    index = int(selected.split(":")[0])
    st.subheader(f"Уровень риска: {risk['levels'][index].lower()} · ориентир для проверки")
    st.caption(f"Уровни сравниваются с оценками последних {len(options)} изделий: повышенный — верхние 20%, высокий — верхние 5%. Это относительная категория, не вероятность.")
    contributions = risk["contributions"].iloc[index].rename("Вклад").reset_index()
    contributions.columns = ["Датчик", "Вклад"]
    st.plotly_chart(px.bar(contributions, x="Вклад", y="Датчик", orientation="h",
                           color="Вклад", color_continuous_scale=["blue", "red"],
                           title="Вклад датчиков в оценку выбранного изделия"),
                    width="stretch")


# Что делает: собирает экран оценки риска; получает качество и результаты; ничего не возвращает; не скрывает слабую проверку.
def show_section_12(data):
    st.title("Оценка риска по модели")
    risk = data["model"]
    if not risk["available"]:
        st.warning(risk["reason"])
        return
    show_risk_quality(risk, data["bad"], data["split"])
    show_risk_summary(risk)
    show_recent_risk(risk, data["labels"])
    recent_missing = data["clean"][risk["sensors"]].tail(RECENT_COUNT).isna().mean()
    silent = recent_missing[recent_missing > SILENT_LIMIT]
    if len(silent):
        st.warning("Датчики модели почти не передают данные; оценки требуют проверки: " + ", ".join(silent.index))
    else:
        st.success("В последних изделиях нет датчиков модели с пропусками выше 80%.")
    st.caption("Модель обучена на первых 80% по времени. Она обрезает крайние значения, заполняет пропуски медианой и сравнивает риск по недавнему распределению оценок.")
    show_explanation("ROC-AUC и его интервал показывают качество на последних 20%, а скользящая проверка обучает модель только на более ранних строках. Доля брака и интервал в таблице рассчитаны по отложенным изделиям, а уровни сравниваются с последними 300 оценками. Вклады показывают, какие значения датчиков сдвинули оценку выбранного изделия вверх или вниз.", risk["summary"])


# Что делает: загружает файлы и открывает экран; получает действия пользователя; ничего не возвращает; связывает меню и расчёты.
def main():
    st.sidebar.header("Загрузить данные")
    csv_file = st.sidebar.file_uploader("Единый CSV", type=["csv"])
    features_file = st.sidebar.file_uploader("features.data", key="features")
    labels_file = st.sidebar.file_uploader("labels.data", key="labels")
    section = st.sidebar.radio("Раздел", SECTIONS)
    if csv_file is None and (features_file is None or labels_file is None):
        st.info("Загрузите единый CSV или оба файла features.data и labels.data.")
        return
    try:
        sensors, labels = calc.load_data(
            BytesIO(features_file.getvalue()) if features_file else None,
            BytesIO(labels_file.getvalue()) if labels_file else None,
            BytesIO(csv_file.getvalue()) if csv_file else None)
        sensors, labels = select_dataset_period(sensors, labels)
        data = get_calculation(sensors, labels)
    except ValueError as error:
        st.error(str(error))
        return
    section_number = SECTIONS.index(section) + 1
    sections = [show_section_1, show_section_2, show_section_3, show_section_4,
                show_section_5, show_section_6, show_section_7, show_section_8,
                show_section_9, show_section_10, show_section_11, show_section_12]
    sections[section_number - 1](data)

if __name__ == "__main__":
    main()

"""Расчёты и чтение данных для Flask-дашборда."""

import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

MAX_MISSING = 0.50  # Удаляем датчик, если больше половины значений пустые.
MIN_UNIQUE = 1  # Удаляем константы.
TRAIN_SHARE = 0.80  # Первые 80% по времени — учебный период.
TOP_COUNT = 15  # Число датчиков в каждом рейтинге.
TREE_COUNT = 400  # Число деревьев в лесу.
MIN_LEAF = 5  # Минимум изделий в листе дерева.
RANDOM_SEED = 42  # Постоянное состояние генератора.
MIN_GROUP_ROWS = 3  # Минимум изделий каждого класса для сравнения.
MIN_BORDER_ROWS = 8  # Минимум годных изделий для расчёта границы.
MIN_CORRELATION_ROWS = 5  # Минимум пар для корреляции.
MAD_SCALE = 1.4826  # Переводим MAD в оценку сигмы.
MODEL_SENSORS = ["sensor_059", "sensor_103", "sensor_247", "sensor_519",
                 "sensor_477", "sensor_205", "sensor_129", "sensor_033",
                 "sensor_341", "sensor_021"]
MODEL_BOOTSTRAPS = 500  # Повторы для интервала ROC-AUC.
MODEL_LOW = 0.005  # Нижний квантиль обрезки признаков.
MODEL_HIGH = 0.995  # Верхний квантиль обрезки признаков.
RECENT_COUNT = 20  # Размер окна актуальных измерений.
SIGNAL_WINDOW = 20  # Окно раннего сигнала по доле брака.
RULE_N = 5  # Сколько браков нужно для правила N из M.
RULE_M = 20  # Размер окна правила N из M.
CUSUM_RUN_LENGTH = 300  # Целевая частота ложной тревоги CUSUM.
CUSUM_SIMULATIONS = 3000  # Повторы калибровки CUSUM.
WEEK_ALPHA = 0.01  # Граница недели соответствует коридору 99%.


def load_inputs(features_path=None, labels_path=None, csv_path=None):
    """Читает CSV либо пару файлов, проверяет метки и упорядочивает строки по времени."""
    if csv_path:
        source = pd.read_csv(csv_path)
        if "datetime" not in source or "label" not in source:
            raise ValueError("CSV должен содержать столбцы datetime и label.")
        times = pd.to_datetime(source.pop("datetime"), dayfirst=True)
        labels = pd.to_numeric(source.pop("label"))
        sensors = source.apply(pd.to_numeric, errors="coerce")
    else:
        if not features_path or not labels_path:
            raise ValueError("Нужен CSV либо оба файла датчиков и меток.")
        sensors = read_features(features_path)
        labels, times = read_labels(labels_path)
    if not set(labels.unique()).issubset({-1, 1}):
        raise ValueError("Допустимые метки: 1 (брак) и -1 (годное).")
    if len(sensors) != len(labels):
        raise ValueError("Число строк датчиков и меток не совпадает.")
    labels = pd.DataFrame({"label": labels.to_numpy(), "datetime": times})
    order = np.argsort(pd.to_datetime(labels["datetime"]).to_numpy(), kind="stable")
    return sensors.iloc[order].reset_index(drop=True), labels.iloc[order].reset_index(drop=True)


def read_features(path):
    """Читает датчики из файла без заголовка; пустые значения остаются NaN."""
    with open(path, encoding="utf-8") as source:
        rows = [line.split() for line in source.read().splitlines()]
    frame = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    frame.columns = ["sensor_" + str(index).zfill(3) for index in range(frame.shape[1])]
    return frame


def read_labels(path):
    """Читает метку и дату из каждой строки labels.data."""
    with open(path, encoding="utf-8") as source:
        rows = source.read().splitlines()
    labels = []
    times = []
    for row in rows:
        parts = row.split('"')
        labels.append(int(parts[0].strip()))
        times.append(pd.to_datetime(parts[1], format="%d/%m/%Y %H:%M:%S"))
    return pd.Series(labels), pd.Series(times)


def clean_sensors(sensors):
    """Удаляет датчики с пропусками выше 50% и с одним или нулём значений."""
    missing = sensors.isna().mean()
    unique = sensors.nunique(dropna=True)
    remove_missing = list(missing[missing > MAX_MISSING].index)
    remove_unique = [name for name in unique[unique <= MIN_UNIQUE].index if name not in remove_missing]
    removed = remove_missing + remove_unique
    clean = sensors.drop(columns=removed)
    rows = []
    for name in removed:
        reason = "Пропусков больше 50%" if name in remove_missing else "0–1 уникальных значений"
        rows.append([name, reason, missing[name], unique[name]])
    table = pd.DataFrame(rows, columns=["Датчик", "Причина", "Пропусков", "Уникальных значений"])
    return clean, missing, remove_missing, remove_unique, table


def correct_p_values(p_values):
    """Поправляет p-значения методом Бенджамини—Хохберга."""
    count, order = len(p_values), np.argsort(p_values)
    values = np.array(p_values)[order] * count / np.arange(1, count + 1)
    values = np.minimum.accumulate(values[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(values, 1)
    return adjusted


def mann_ranking(sensors, bad, split):
    """Ранжирует датчики тестом Манна—Уитни на первых 80% времени."""
    rows = []
    p_values = []
    for name in sensors.columns:
        good = sensors[name].iloc[:split][bad[:split] == 0].dropna()
        defect = sensors[name].iloc[:split][bad[:split] == 1].dropna()
        if len(good) < MIN_GROUP_ROWS or len(defect) < MIN_GROUP_ROWS:
            rows.append([name, 0.5, 1.0])
            p_values.append(1.0)
        else:
            test = mannwhitneyu(defect, good, alternative="two-sided")
            auc = test.statistic / (len(good) * len(defect))
            rows.append([name, max(auc, 1 - auc), test.pvalue])
            p_values.append(test.pvalue)
    result = pd.DataFrame(rows, columns=["Датчик", "AUC", "p"])
    result["q"] = correct_p_values(p_values)
    return result.sort_values(["AUC", "q"], ascending=[False, True]).head(TOP_COUNT)


def forest_ranking(sensors, bad, split):
    """Обучает лес на прошлом и возвращает важность признаков и AUC проверки."""
    train = sensors.iloc[:split]
    medians = train.median().fillna(0)
    model = RandomForestClassifier(n_estimators=TREE_COUNT, class_weight="balanced",
                                   min_samples_leaf=MIN_LEAF, random_state=RANDOM_SEED)
    model.fit(train.fillna(medians), bad[:split])
    ranking = pd.DataFrame({"Датчик": sensors.columns, "Важность": model.feature_importances_})
    ranking = ranking.sort_values("Важность", ascending=False).head(TOP_COUNT)
    test = sensors.iloc[split:].fillna(medians)
    score = None
    if len(test) and len(np.unique(bad[split:])) == 2:
        score = roc_auc_score(bad[split:], model.predict_proba(test)[:, 1])
    return ranking, score


def choose_direction(good, defect, center):
    """Выбирает сторону датчика, где в опорной истории чаще был брак."""
    if len(defect) < MIN_GROUP_ROWS:
        return 1
    if defect.median() != center:
        return 1 if defect.median() > center else -1
    high = (defect > center).mean() - (good > center).mean()
    low = (defect < center).mean() - (good < center).mean()
    return -1 if low > high else 1


def make_borders(sensors, bad, split):
    """Считает односторонние робастные границы 2σ и 3σ по годным изделиям."""
    rows = []
    for name in sensors.columns:
        values = sensors[name].iloc[:split]
        good = values[(bad[:split] == 0) & values.notna()]
        defect = values[(bad[:split] == 1) & values.notna()]
        if len(good) < MIN_BORDER_ROWS:
            continue
        center = good.median()
        sigma = MAD_SCALE * np.median(np.abs(good.to_numpy() - center))
        if sigma <= 0 or not np.isfinite(sigma):
            continue
        direction = choose_direction(good, defect, center)
        rows.append([name, center, sigma, direction,
                     center + direction * 2 * sigma, center + direction * 3 * sigma])
    return pd.DataFrame(rows, columns=["Датчик", "Центр", "Сигма", "Направление", "Жёлтая", "Красная"])


def make_correlations(sensors, names, bad):
    """Считает попарные корреляции Спирмена между датчиками, временем и меткой."""
    frame = sensors[names].copy()
    frame["Время"] = np.arange(len(frame))
    frame["Брак (0/1)"] = bad
    return frame.corr(method="spearman", min_periods=MIN_CORRELATION_ROWS).fillna(0)


def make_weekly_data(sensors, labels, bad, borders, names):
    """Собирает недельные доли брака, медианы датчиков и выходы за жёлтые границы."""
    frame = pd.DataFrame({"Дата": labels["datetime"], "Брак": bad}).set_index("Дата")
    weekly = frame.resample("W-MON")["Брак"].agg(["sum", "count"])
    weekly["Доля"] = weekly["sum"] / weekly["count"]
    dates = labels["datetime"].dt.to_period("W").astype(str)
    heat = sensors[names].copy()
    heat["Неделя"] = dates
    for name in names:
        border = borders[borders["Датчик"] == name].iloc[0]
        direction = border["Направление"]
        zone = direction * (heat[name] - border["Центр"]) / border["Сигма"] >= 2
        heat[name] = zone.where(heat[name].notna())
    heatmap = heat.groupby("Неделя")[names].mean().T
    medians = sensors[names[:6]].copy()
    medians["Неделя"] = dates
    return weekly.reset_index(), heatmap, medians.groupby("Неделя")[names[:6]].median().reset_index()


def model_sensors(clean):
    """Берёт эталонные десять датчиков, доступные после очистки."""
    return [name for name in MODEL_SENSORS if name in clean.columns]


def fit_logistic(sensors, bad, names, end):
    """Учится на прошлом: обрезает выбросы, заполняет пропуски и масштабирует признаки."""
    train = sensors[names].iloc[:end]
    lower = train.quantile(MODEL_LOW)
    upper = train.quantile(MODEL_HIGH)
    train = train.clip(lower=lower, upper=upper, axis="columns")
    medians = train.median().fillna(0)
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    model.fit(scaler.fit_transform(train.fillna(medians)), bad[:end])
    return {"model": model, "low": lower, "high": upper, "medians": medians, "scaler": scaler}


def score_logistic(sensors, names, fitted):
    """Возвращает относительный балл и вклад каждого датчика."""
    values = sensors[names].clip(lower=fitted["low"], upper=fitted["high"], axis="columns")
    scaled = fitted["scaler"].transform(values.fillna(fitted["medians"]))
    scores = fitted["model"].decision_function(scaled)
    contributions = pd.DataFrame(scaled * fitted["model"].coef_[0], columns=names, index=sensors.index)
    return scores, contributions


def wilson_interval(defects, count):
    """Возвращает 95% интервал Уилсона для исторической доли брака."""
    if count == 0:
        return np.nan, np.nan
    rate = defects / count
    denominator = 1 + 1.96 ** 2 / count
    center = (rate + 1.96 ** 2 / (2 * count)) / denominator
    margin = 1.96 * np.sqrt(rate * (1 - rate) / count + 1.96 ** 2 / (4 * count ** 2)) / denominator
    return max(0, center - margin), min(1, center + margin)


def auc_interval(bad, scores):
    """Считает ROC-AUC и бутстрэп-интервал, сохраняя число обоих классов."""
    if len(np.unique(bad)) != 2:
        return None, [None, None]
    good_rows = np.flatnonzero(bad == 0)
    defect_rows = np.flatnonzero(bad == 1)
    random = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(MODEL_BOOTSTRAPS):
        sample = np.r_[random.choice(good_rows, len(good_rows), replace=True),
                       random.choice(defect_rows, len(defect_rows), replace=True)]
        values.append(roc_auc_score(bad[sample], scores[sample]))
    return roc_auc_score(bad, scores), np.quantile(values, [0.025, 0.975]).tolist()


def rolling_auc(sensors, bad, names, split):
    """Проверяет несколько последовательных отрезков, не перемешивая время."""
    scores = []
    targets = []
    folds = np.array_split(np.arange(split, len(bad)), min(4, len(bad) - split))
    for fold in folds:
        if len(fold) and len(np.unique(bad[:fold[0]])) == 2:
            fitted = fit_logistic(sensors, bad, names, int(fold[0]))
            fold_scores, _ = score_logistic(sensors.iloc[fold], names, fitted)
            scores.extend(fold_scores)
            targets.extend(bad[fold])
    auc, interval = auc_interval(np.array(targets), np.array(scores)) if targets else (None, [None, None])
    return auc, interval, len(targets)


def make_model(clean, bad, split):
    """Создаёт модель, временную проверку, ROC-кривую и категории риска."""
    names = model_sensors(clean)
    if len(names) < len(MODEL_SENSORS) or len(np.unique(bad[:split])) != 2:
        return {"available": False, "reason": "Недостаточно ключевых датчиков или меток."}
    fitted = fit_logistic(clean, bad, names, split)
    scores, contributions = score_logistic(clean, names, fitted)
    if len(np.unique(bad[split:])) != 2:
        return {"available": False, "reason": "В последних 20% нет обоих классов для проверки."}
    auc, interval = auc_interval(bad[split:], scores[split:])
    false_rate, hit_rate, _ = roc_curve(bad[split:], scores[split:])
    reference = scores[-min(300, len(scores)):]
    elevated, high = np.quantile(reference, [0.80, 0.95])
    levels = ["Высокий" if score > high else "Повышенный" if score > elevated else "Обычный" for score in scores]
    rows = []
    for level in ["Обычный", "Повышенный", "Высокий"]:
        group = bad[split:][np.array([value == level for value in levels[split:]])]
        low, top = wilson_interval(int(group.sum()), len(group))
        rows.append([level, float(group.mean()) if len(group) else np.nan, len(group), low, top])
    summary = pd.DataFrame(rows, columns=["Уровень риска", "Доля брака", "Изделий", "Нижняя 95%", "Верхняя 95%"])
    rolling, rolling_interval, rolling_count = rolling_auc(clean, bad, names, split)
    return {"available": True, "sensors": names, "scores": scores, "contributions": contributions,
            "levels": levels, "summary": summary, "auc": auc, "interval": interval,
            "curve": pd.DataFrame({"Ложные срабатывания": false_rate, "Найденный брак": hit_rate}),
            "rolling_auc": rolling, "rolling_interval": rolling_interval, "rolling_count": rolling_count}


def make_cusum_threshold(rate):
    """Калибрует порог CUSUM по симуляциям с целевой частотой ложной тревоги 1/300."""
    rate = min(max(rate, 0.0001), 0.5)
    target = min(max(rate * 2, rate + 0.01), 0.5)
    random = np.random.default_rng(RANDOM_SEED)
    counts = random.binomial(1, rate, size=(CUSUM_SIMULATIONS, CUSUM_RUN_LENGTH))
    change = counts * np.log(target / rate) + (1 - counts) * np.log((1 - target) / (1 - rate))
    current = np.zeros(CUSUM_SIMULATIONS)
    maximum = np.zeros(CUSUM_SIMULATIONS)
    for column in range(CUSUM_RUN_LENGTH):
        current = np.maximum(0, current + change[:, column])
        maximum = np.maximum(maximum, current)
    return float(np.quantile(maximum, 1 - 1 / CUSUM_RUN_LENGTH)), target


def make_online_signals(bad, rate):
    """Строит CUSUM и правило 5 из 20 для последовательности изделий."""
    threshold, target = make_cusum_threshold(rate)
    increment = bad * np.log(target / rate) + (1 - bad) * np.log((1 - target) / (1 - rate))
    cusum = np.zeros(len(bad))
    alarms = np.zeros(len(bad), dtype=bool)
    total = 0
    for index in range(len(bad)):
        total = max(0, total + increment[index])
        cusum[index] = total
        if total >= threshold:
            alarms[index] = True
            total = 0
    recent_bad = pd.Series(bad).rolling(RULE_M, min_periods=RULE_M).sum().to_numpy()
    rule_alarm = recent_bad >= RULE_N
    active = False
    for index in range(len(rule_alarm)):
        if rule_alarm[index] and active:
            rule_alarm[index] = False
        elif rule_alarm[index]:
            active = True
        else:
            active = False
    return {"cusum": cusum, "cusum_alarm": alarms, "threshold": threshold,
            "rule_alarm": rule_alarm, "rule_count": recent_bad}


def make_attention(data, index):
    """Показывает ключевые датчики, их зоны и частоту брака за жёлтой границей."""
    rows = []
    for name in data["keys"]:
        value = data["clean"].loc[index, name]
        border = data["borders"][data["borders"]["Датчик"] == name].iloc[0]
        signed = border["Направление"] * (value - border["Центр"]) / border["Сигма"] if pd.notna(value) else np.nan
        if pd.notna(signed) and signed >= 2:
            mask = border["Направление"] * (data["clean"][name] - border["Центр"]) / border["Сигма"] >= 2
            count = int(mask.sum())
            defects = int(data["bad"][mask.to_numpy()].sum())
            zone = "красная" if signed >= 3 else "жёлтая"
            boundary = border["Красная"] if signed >= 3 else border["Жёлтая"]
            rows.append([name, zone, value, boundary, signed, defects / count if count else 0, count, index])
    return pd.DataFrame(rows, columns=["Датчик", "Зона", "Значение", "Граница", "Сигмы", "Доля брака", "Изделий", "Номер изделия"])


def check_manual_value(data, name, value):
    """Сравнивает ручное значение с границей и добавляет историческую частоту брака."""
    border = data["borders"][data["borders"]["Датчик"] == name].iloc[0]
    sigma = border["Направление"] * (value - border["Центр"]) / border["Сигма"]
    zone = "Высокая" if sigma >= 3 else "Повышенная" if sigma >= 2 else "Обычная"
    mask = border["Направление"] * (data["clean"][name] - border["Центр"]) / border["Сигма"] >= 2
    count = int(mask.sum())
    rate = float(data["bad"][mask.to_numpy()].mean()) if count else np.nan
    return [name, value, zone, round(sigma, 2), rate, count]


def make_growth_periods(weekly, rate):
    """Находит последовательные недели выше верхней биномиальной границы."""
    weekly = weekly.copy()
    weekly["Верхняя граница"] = binom.ppf(0.99, weekly["count"], rate) / weekly["count"]
    weekly["Выше границы"] = weekly["Доля"] > weekly["Верхняя граница"]
    periods = []
    start = None
    quiet = 0
    for index, row in weekly.iterrows():
        if row["Выше границы"] and start is None:
            start = index
        if start is not None and row["Выше границы"]:
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet == 2 or index == weekly.index[-1]:
                periods.append({"start": start, "end": index - quiet})
                start = None
                quiet = 0
    if start is not None:
        periods.append({"start": start, "end": weekly.index[-1]})
    return periods


def report_alarm(times, bad, frequency, rate, start):
    """Возвращает первую дату, когда агрегированный отчёт превысил коридор."""
    frame = pd.DataFrame({"Дата": times, "Брак": bad}).set_index("Дата")
    report = frame.resample(frequency)["Брак"].agg(["sum", "count"])
    report["end"] = report.index + pd.tseries.frequencies.to_offset(frequency)
    report["upper"] = binom.ppf(0.99, report["count"], rate)
    for _, row in report.iterrows():
        if row["end"] >= start and row["sum"] > row["upper"]:
            return row["end"]
    return None


def extra_defects(bad, times, start, detected, rate):
    """Считает превышение фактического брака над базовым до обнаружения."""
    if detected is None:
        return None
    mask = (times >= start) & (times <= detected)
    return int(bad[mask].sum() - mask.sum() * rate)


def late_detection(data):
    """Собирает шкалы и сроки дневного, недельного, месячного и раннего сигналов."""
    times = data["labels"]["datetime"]
    bad = data["bad"]
    periods = make_growth_periods(data["weekly"], data["ordinary"])
    online = make_online_signals(bad, data["ordinary"])
    rows = []
    for period in periods:
        first_week = data["weekly"].loc[period["start"], "Дата"]
        last_week = data["weekly"].loc[period["end"], "Дата"]
        start = times[(times >= first_week - pd.Timedelta(days=6)) & (times <= first_week)].min()
        end = times[(times >= last_week - pd.Timedelta(days=6)) & (times <= last_week)].max()
        starts = np.flatnonzero(times >= start)
        start_index = int(starts[0]) if len(starts) else len(bad)
        detected = {"День": report_alarm(times, bad, "D", data["ordinary"], start),
                    "Неделя": report_alarm(times, bad, "W-MON", data["ordinary"], start),
                    "Месяц": report_alarm(times, bad, "MS", data["ordinary"], start),
                    "CUSUM": first_alarm(times, online["cusum_alarm"], start_index),
                    "5 из 20": first_alarm(times, online["rule_alarm"], start_index)}
        row = {"Начало": start, "Конец": end}
        for method, date in detected.items():
            row[method] = date
            row[method + " · лишний брак"] = extra_defects(bad, times, start, date, data["ordinary"])
        rows.append(row)
    return {"periods": periods, "table": pd.DataFrame(rows), "online": online}


def first_alarm(times, alarms, start):
    """Ищет первое срабатывание после начала периода роста."""
    found = np.flatnonzero(alarms[start:])
    return times.iloc[start + int(found[0])] if len(found) else None


def calculate(sensors, labels):
    """Считает показатели, рейтинги, датчики, модель и сигналы для всех страниц."""
    bad = (labels["label"].to_numpy() == 1).astype(int)
    split = int(len(sensors) * TRAIN_SHARE)
    clean, missing, remove_missing, remove_unique, removed = clean_sensors(sensors)
    mann = mann_ranking(clean, bad, split)
    forest, forest_auc = forest_ranking(clean, bad, split)
    keys = list(dict.fromkeys(mann["Датчик"].tolist() + forest["Датчик"].tolist()))
    borders = make_borders(clean, bad, split)
    keys = [name for name in keys if name in borders["Датчик"].tolist()]
    dates = labels["datetime"].iloc[:split]
    daily = pd.DataFrame({"Дата": dates, "Брак": bad[:split]}).set_index("Дата").resample("D")["Брак"].mean()
    ordinary = float(daily.median())
    weekly, heatmap, medians = make_weekly_data(clean, labels, bad, borders, keys)
    result = {"sensors": sensors, "clean": clean, "labels": labels, "bad": bad,
              "missing": missing, "remove_missing": remove_missing, "remove_unique": remove_unique,
              "removed": removed, "mann": mann, "forest": forest, "keys": keys, "borders": borders,
              "weekly": weekly, "heatmap": heatmap, "medians": medians, "ordinary": ordinary,
              "correlations": make_correlations(clean, keys, bad), "split": split,
              "forest_auc": forest_auc, "model": make_model(clean, bad, split)}
    result["late"] = late_detection(result)
    return result

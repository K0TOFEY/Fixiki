import numpy as np
import pandas as pd

from build_dashboard import (
    _HTML,
    _read_inputs,
    _sensor_names,
    _thresholds,
    analyze,
    drifting_sensors,
    exact_binomial_bounds,
    find_excess_segments,
)


def test_finds_injected_burst_but_not_ordinary_rate():
    total = np.full(100, 10)
    ordinary = np.full(100, 1)
    assert find_excess_segments(ordinary, total, 0.1) == []

    burst = ordinary.copy()
    burst[40:50] = 8
    segments = find_excess_segments(burst, total, 0.1)
    assert segments
    assert any(s["start"] <= 40 and s["end"] >= 49 for s in segments)


def test_threshold_is_one_sided_toward_historical_defects():
    rng = np.random.default_rng(42)
    good_reference = rng.normal(0, 1, 60)
    defective_reference = rng.normal(-3, 0.2, 20)
    good_holdout = rng.normal(0, 1, 20)
    values = np.concatenate([good_reference, defective_reference, good_holdout])
    frame = pd.DataFrame({"sensor_001": values})
    labels = np.r_[np.zeros(60, dtype=int), np.ones(20, dtype=int), np.zeros(20, dtype=int)]
    threshold = _thresholds(frame, labels, 80)["sensor_001"]
    assert threshold["direction"] == -1
    assert threshold["red"] < threshold["center"]
    assert np.isclose(abs(threshold["red"] - threshold["center"]), 3 * threshold["sigma"])


def test_drift_is_flagged_by_rank_correlation_cutoff():
    assert drifting_sensors({"sensor_001": 0.42, "sensor_002": -0.5, "sensor_003": 0.3}) == [
        "sensor_001", "sensor_002"
    ]


def test_random_fluctuation_band_uses_binomial_bounds():
    lower, upper = exact_binomial_bounds(100, 0.05)
    assert 0 <= lower < 0.05 < upper < 1


def test_combined_csv_and_optional_sensor_dictionary(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame({
        "datetime": ["19/07/2008 11:55:00", "19/07/2008 12:32:00"],
        "label": [-1, 1],
        "sensor_000": [1.2, 1.4],
    }).to_csv(source, index=False)
    x, y = _read_inputs(None, None, source)
    assert list(x.columns) == ["sensor_000"]
    assert y["label"].tolist() == [-1, 1]
    assert y["datetime"].iloc[0] == pd.Timestamp("2008-07-19 11:55:00")

    dictionary = tmp_path / "sensors.json"
    dictionary.write_text('{"sensor_000":{"name":"Температура","area":"Линия 1","unit":"°C"}}')
    names = _sensor_names(list(x.columns), dictionary)
    assert names["sensor_000"]["name"] == "Температура"
    assert names["sensor_000"]["area"] == "Линия 1"


def test_analyze_reports_validation_and_finds_synthetic_red_zone():
    rng = np.random.default_rng(5)
    n = 120
    bad = np.zeros(n, dtype=int)
    bad[[8, 24, 43, 61, 78, 89, 104, 115]] = 1
    a = rng.normal(0, 1, n)
    a[bad == 1] = -3
    x = pd.DataFrame({
        "sensor_001": a,
        "sensor_002": rng.normal(2, 0.1, n),
        "sensor_003": np.arange(n, dtype=float),
    })
    labels = pd.DataFrame({
        "label": np.where(bad == 1, 1, -1),
        "datetime": pd.date_range("2020-01-01", periods=n, freq="h"),
    })
    result = analyze(x, labels, {}, unique_threshold=1)
    assert result["meta"]["n"] == n
    assert result["validation"]["rolling_n"] > 0
    assert result["thresholds"]["sensor_001"]["direction"] == -1
    assert abs(result["time_corr"]["sensor_003"]) > 0.3


def test_attention_wording_avoids_prediction_and_probability_claims():
    assert "Это подсказка для проверки оборудования, не приговор." in _HTML
    assert "вероятность брака" not in _HTML.lower()
    assert "это не прогноз" not in _HTML.lower()

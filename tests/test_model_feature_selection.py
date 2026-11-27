import numpy as np
import pandas as pd

import functions


def test_model_features_fuse_mann_whitney_and_forest_ranks_on_training_only(
    monkeypatch,
):
    sensors = [f"sensor_{index:03d}" for index in range(20)]
    bad = np.array([0, 1] * 10)
    mann_order = sensors[:15]
    forest_order = sensors[:5] + sensors[15:] + sensors[5:10]
    calls = []

    def fake_mann_ranking(clean, labels, split):
        calls.append(("mann", split))
        return pd.DataFrame({"Датчик": mann_order})

    def fake_forest_ranking(clean, labels, split, *, evaluate_auc=True):
        calls.append(("forest", split, evaluate_auc))
        return pd.DataFrame({"Датчик": forest_order}), None

    monkeypatch.setattr(functions, "mann_ranking", fake_mann_ranking)
    monkeypatch.setattr(functions, "forest_ranking", fake_forest_ranking)

    selected = functions.model_sensors(
        pd.DataFrame(columns=sensors),
        bad,
        train_end=12,
    )

    assert selected == [
        "sensor_000",
        "sensor_001",
        "sensor_002",
        "sensor_003",
        "sensor_004",
        "sensor_005",
        "sensor_006",
        "sensor_007",
        "sensor_015",
        "sensor_008",
    ]
    assert calls == [("mann", 12), ("forest", 12, False)]


def test_model_features_fall_back_when_ranked_features_underperform(
    monkeypatch,
):
    bad = np.array([0, 1] * 50)
    ranked_sensors = [f"sensor_{index:03d}" for index in range(10)]
    calls = []
    sensors = pd.DataFrame(
        {
            **{name: 1 - bad for name in ranked_sensors},
            **{name: bad for name in functions.MODEL_REFERENCE_SENSORS},
        }
    )

    def fake_mann_ranking(clean, labels, split):
        calls.append(("mann", split))
        return pd.DataFrame({"Датчик": ranked_sensors})

    def fake_forest_ranking(clean, labels, split, *, evaluate_auc=True):
        calls.append(("forest", split, evaluate_auc))
        return pd.DataFrame({"Датчик": ranked_sensors}), None

    monkeypatch.setattr(functions, "mann_ranking", fake_mann_ranking)
    monkeypatch.setattr(functions, "forest_ranking", fake_forest_ranking)
    monkeypatch.setattr(functions, "fit_logistic", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        functions,
        "score_logistic",
        lambda values, names, fitted: (values[names[0]].to_numpy(), None),
    )

    selected = functions.model_sensors(sensors, bad, train_end=100)

    assert selected == functions.MODEL_REFERENCE_SENSORS
    assert calls == [
        ("mann", 80),
        ("forest", 80, False),
    ]

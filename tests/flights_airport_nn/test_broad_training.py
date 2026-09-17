from datetime import date

import numpy as np
import torch

from flights_airport_nn import train_broad
from flights_airport_nn.compare_broad import choices, combine


class FixedRanker:
    def eval(self):
        return self

    def rank(self, x, mask):
        scores = torch.arange(x.shape[1], 0, -1, device=x.device, dtype=x.dtype)
        return scores.repeat(x.shape[0], 1)


def test_missing_candidate_is_counted_as_failure(tmp_path, monkeypatch):
    folder = tmp_path / "2026-06-03"
    folder.mkdir()
    x = np.zeros((2, 3, 2, 24), np.float16)
    x[0, 0, 0, 13] = 1
    np.savez(
        folder / "part-0000.npz",
        x=x,
        length=np.array([1, 0], np.uint8),
        count=np.array([1, 0], np.uint8),
        target=np.array([0, -1], np.int16),
        truth_type=np.array([0, 2], np.int8),
        nearest=np.array([0, -1], np.int16),
        truth=np.array(["KLAX", "KABC"]),
        flight_id=np.array(["a_1", "b_2"]),
        endpoint=np.array(["takeoff", "landing"]),
    )
    monkeypatch.setattr(train_broad, "CACHE_ROOT", tmp_path)
    result = train_broad.score(FixedRanker(), [date(2026, 6, 3)])
    assert result["correct"].tolist() == [True, False]
    assert result["nearest_correct"].tolist() == [True, False]
    assert result["metrics"]["small_airport"]["accuracy"] == 0


def test_fallback_rules_select_by_nearest_type_without_test_labels():
    base = {
        "date": np.array(["2026-06-03"] * 4),
        "flight_id": np.array(["f1", "f2", "f3", "f4"]),
        "endpoint": np.array(["takeoff"] * 4),
        "truth_type": np.array([0, 0, 2, 2]),
        "truth": np.array(["A", "A", "B", "B"]),
        "target": np.array([1, 1, 1, 1]),
        "nearest": np.array([0, 0, 0, 0]),
        "nearest_type": np.array([0, 0, 2, 2]),
    }
    old = {
        **base,
        "predicted": np.array([1, 1, 0, 0]),
        "confidence": np.array([0.99] * 4),
    }
    new = {
        **base,
        "predicted": np.array([0, 0, 1, 1]),
        "confidence": np.array([0.99] * 4),
    }
    rule = choices(old, new)
    assert rule["0"]["model"] == "old"
    assert rule["2"]["model"] == "new"
    selected, source = combine(old, new, rule)
    assert selected.tolist() == [1] * 4
    assert source.tolist() == ["old", "old", "new", "new"]

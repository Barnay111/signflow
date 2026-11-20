import random

import pytest

from lipflow.sign import NONE, CORNERS, SignReader, StreamMerger, parse_crop


class _R:  # SignReader.merge needs only `confirm`
    merge = SignReader.merge

    def __init__(self, confirm):
        self.confirm = confirm


@pytest.mark.parametrize("confirm", [1, 2, 3])
def test_stream_matches_batch_merge(confirm):
    rnd = random.Random(confirm)
    for _ in range(500):
        labels = [rnd.choice([NONE, NONE, "а", "б", "дом"]) for _ in range(rnd.randint(0, 40))]
        batch = [g for _, g, _ in _R(confirm).merge([(g, 0.9) for g in labels])]
        m = StreamMerger(confirm)
        stream = [g for g in (m.push(x) for x in labels) if g is not None]
        assert stream == batch, labels


def test_parse_crop():
    assert parse_crop(None, 100, 100) is None
    assert parse_crop("br", 1000, 500) == (600, 200, 400, 300)
    assert parse_crop("10,20,30,40", 1000, 500) == (10, 20, 30, 40)
    assert set(CORNERS) >= {"br", "bl", "tr", "tl", "right", "left"}


def test_to_text_joins_letters():
    g = [("М", 1), ("А", 1), ("М", 1), ("А", 1), ("дом / здание", 1), ("Я", 1)]
    assert SignReader.to_text(g) == "Мама дом Я"

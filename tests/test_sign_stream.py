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


def test_srt_cues_follow_signs():
    from lipflow.live import srt
    out = srt([(1.0, "М", 0.9), (1.5, "А", 0.9), (3.0, "дом / здание", 0.8), (20.0, "Я", 0.9)])
    cues = out.strip().split("\n\n")
    assert len(cues) == 4
    assert cues[0].splitlines()[1] == "00:00:01,000 --> 00:00:01,500"
    assert cues[1].splitlines()[2] == "Ма"
    assert cues[2].splitlines()[2] == "Ма дом"
    assert cues[3].splitlines()[1] == "00:00:20,000 --> 00:00:24,000"  # held 4 s at most


def test_recorder_keeps_video_in_sync(tmp_path):
    import cv2
    import numpy as np
    from lipflow.live import Recorder
    r = Recorder(str(tmp_path), (64, 48), t0=10.0)
    frame = np.zeros((48, 64, 3), np.uint8)
    for t in (10.0, 10.5, 12.0):  # an uneven camera: frames are repeated to follow the clock
        r.write(frame, t)
    files = r.finish([(9.5, "дом", 0.9), (11.0, "Я", 0.8), (30.0, "после", 0.9)])
    assert r.written == 61
    cap = cv2.VideoCapture(files[0])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 61
    txt = open(files[1], encoding="utf-8").read()
    assert txt.startswith("дом Я") and "после" not in txt
    assert "00:00:01,000" in open(files[2], encoding="utf-8").read()

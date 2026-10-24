import importlib.util
from pathlib import Path

import numpy as np
import pytest

from thesis.vad.ucf import CLASSES, parse_annotations, prompts

cv2 = pytest.importorskip("cv2")
spec = importlib.util.spec_from_file_location(
    "score_clip", Path(__file__).resolve().parents[1] / "experiments" / "ch5_ucfcrime" / "score_clip.py"
)
score_clip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_clip)


def test_parse_annotations(tmp_path):
    p = tmp_path / "ann.txt"
    p.write_text(
        "Arson011_x264.mp4  Arson  150  420  680  1267\n"
        "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n"
        "Fighting003_x264.mp4  Fighting  1820  3103  -1  -1\n"
    )
    ann = parse_annotations(p)
    assert [a.cls for a in ann] == ["Arson", "Normal", "Fighting"]
    assert ann[0].segments == [(150, 420), (680, 1267)]
    assert not ann[1].is_anomaly and ann[1].segments == []
    y = ann[2].frame_labels(2000)
    assert y[1819] == 0 and y[1820] == 1 and y[-1] == 1


def test_prompts_cover_all_classes():
    pr = prompts()
    assert list(pr) == CLASSES and len(CLASSES) == 14
    assert all(len(v) == 3 for v in pr.values())


def test_sample_frames(tmp_path):
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(40):
        writer.write(np.full((48, 64, 3), k, dtype=np.uint8))
    writer.release()
    gen = score_clip.sample_frames(path, stride=16)
    idx = []
    try:
        while True:
            k, im = next(gen)
            idx.append(k)
            assert im.size == (64, 48)
    except StopIteration as stop:
        n_frames, fps = stop.value
    assert idx == [0, 16, 32] and n_frames == 40 and fps == pytest.approx(30.0)

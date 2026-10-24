import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from thesis.vlm.client import Judgement, compose_grid, parse_judgement
from thesis.vlm.requests import clip_frames, request_id

spec = importlib.util.spec_from_file_location(
    "query_vlm", Path(__file__).resolve().parents[1] / "experiments" / "ch4_fusion" / "query_vlm.py"
)
query_vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_vlm)


def test_parse_judgement_variants():
    j = parse_judgement('```json\n{"p_anomaly": 0.8, "uncertainty": 0.1, "reason": "fell"}\n```')
    assert (j.p_anomaly, j.uncertainty) == (0.8, 0.1)
    assert parse_judgement('Sure: {"p_anomaly": 0.2, "entropy": 0.5}').uncertainty == 0.5
    with pytest.raises(ValueError):
        parse_judgement('{"p_anomaly": 1.5, "uncertainty": 0.1}')


def test_clip_frames_snaps_to_available():
    assert clip_frames(5, np.arange(1, 50), offsets=(-40, -10, 0, 10)) == [1, 5, 15]


def test_compose_grid_shape(tmp_path):
    paths = []
    for k in range(5):
        p = tmp_path / f"f{k}.png"
        Image.new("RGB", (640, 480), (k * 40, 0, 0)).save(p)
        paths.append(str(p))
    grid = compose_grid(paths, tile_width=320, columns=3)
    assert grid.size == (960, 480)


class FakeClient:
    def __init__(self):
        self.calls = 0

    def judge(self, image, prompt):
        self.calls += 1
        return Judgement(0.9 if "frames" in prompt else 0.3, 0.2, "ok", "{}")


def test_run_writes_and_resumes(tmp_path):
    seq_dir = tmp_path / "frames" / "fall-01" / "fall-01-cam0-rgb"
    seq_dir.mkdir(parents=True)
    for f in range(1, 31):
        Image.new("RGB", (64, 48)).save(seq_dir / f"fall-01-cam0-rgb-{f:03d}.png")
    reqs = [
        dict(id=request_id("fall-01", [1, 11, 21], "clip6"), seq="fall-01", frames=[1, 11, 21], variant="clip6", kind="event"),
        dict(id=request_id("fall-01", [21], "frame"), seq="fall-01", frames=[21], variant="frame", kind="event"),
    ]
    out = tmp_path / "resp.jsonl"
    client = FakeClient()
    query_vlm.run(reqs, client, tmp_path / "frames", out)
    query_vlm.run(reqs, client, tmp_path / "frames", out)
    assert client.calls == 2
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["p_anomaly"] for r in lines] == [0.9, 0.3]

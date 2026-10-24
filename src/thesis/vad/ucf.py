"""UCF-Crime test split: temporal annotations and zero-shot CLIP prompts.

The test annotation file (Temporal_Anomaly_Annotation_for_Testing_Videos.txt)
has one line per video:

    <video>.mp4  <class>  <start1>  <end1>  <start2>  <end2>

with frame indices of up to two anomalous segments (-1 if absent); normal
videos have class "Normal" and all indices -1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANOMALY_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
CLASSES = ANOMALY_CLASSES + ["Normal"]

DESCRIPTIONS = {
    "Abuse": "a person being abused or beaten by another person",
    "Arrest": "police officers arresting a person",
    "Arson": "a person setting a fire on purpose",
    "Assault": "a person attacking another person",
    "Burglary": "a burglar breaking into a building",
    "Explosion": "an explosion with fire and smoke",
    "Fighting": "people fighting each other",
    "RoadAccidents": "a road traffic accident with crashing vehicles",
    "Robbery": "an armed robbery of a person or a shop",
    "Shooting": "a person shooting a gun",
    "Shoplifting": "a person shoplifting goods in a store",
    "Stealing": "a person stealing a car or other property",
    "Vandalism": "a person vandalising and damaging property",
    "Normal": "people going about normal everyday activities",
}
TEMPLATES = [
    "a surveillance camera image of {}.",
    "CCTV footage showing {}.",
    "a security camera frame of {}.",
]


@dataclass
class VideoAnnotation:
    video: str
    cls: str
    segments: list[tuple[int, int]]

    @property
    def is_anomaly(self) -> bool:
        return self.cls != "Normal"

    def frame_labels(self, n_frames: int) -> np.ndarray:
        y = np.zeros(n_frames, dtype=np.int8)
        for s, e in self.segments:
            y[max(0, s):min(n_frames, e + 1)] = 1
        return y


def parse_annotations(path: str | Path) -> list[VideoAnnotation]:
    out = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        video, cls = parts[0], parts[1]
        s1, e1, s2, e2 = (int(v) for v in parts[2:6])
        segs = [(s, e) for s, e in ((s1, e1), (s2, e2)) if s >= 0 and e >= 0]
        out.append(VideoAnnotation(video, cls, segs))
    return out


def prompts() -> dict[str, list[str]]:
    return {c: [t.format(DESCRIPTIONS[c]) for t in TEMPLATES] for c in CLASSES}

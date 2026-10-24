"""Semantic level: a vision-language model behind an OpenAI-compatible
chat-completions API (LM Studio, vLLM, Ollama, ...).

The model is asked a semantic question (controlled activity or accident),
not a kinematic one, and returns strict JSON with its probability of an
accident and its own uncertainty.
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

from PIL import Image

PROMPT_FRAME = (
    "Is the action performed by the person in the centre of the frame a "
    "controlled sporting or everyday activity, or is it an accident? "
    'Respond with strict JSON only: {"p_anomaly": <0..1>, "uncertainty": <0..1>, '
    '"reason": "<one short sentence>"}. p_anomaly is your probability that '
    "this is a genuine accident/fall requiring help. uncertainty is your own "
    "uncertainty about that judgement (0 = certain, 1 = a coin flip)."
)

PROMPT_CLIP = (
    "The image shows {n} frames from a surveillance camera in temporal order, "
    "left to right, top to bottom, {dt:.2f} s apart. Is the action performed by the "
    "person a controlled sporting or everyday activity (for example sitting or lying "
    "down on purpose, exercising, bending), or is it an accident such as a fall? "
    'Respond with strict JSON only: {{"p_anomaly": <0..1>, "uncertainty": <0..1>, '
    '"reason": "<one short sentence>"}}. p_anomaly is your probability that this is '
    "a genuine accident/fall requiring help. uncertainty is your own uncertainty "
    "about that judgement (0 = certain, 1 = a coin flip)."
)


@dataclass
class Judgement:
    p_anomaly: float
    uncertainty: float
    reason: str
    raw: str


def compose_grid(paths: list[str], tile_width: int = 320, columns: int = 3) -> Image.Image:
    """Tile frames left-to-right, top-to-bottom at a common width."""
    images = [Image.open(p).convert("RGB") for p in paths]
    w0, h0 = images[0].size
    tile_h = round(tile_width * h0 / w0)
    images = [im.resize((tile_width, tile_h)) for im in images]
    cols = min(columns, len(images))
    rows = -(-len(images) // cols)
    grid = Image.new("RGB", (cols * tile_width, rows * tile_h), "black")
    for k, im in enumerate(images):
        grid.paste(im, ((k % cols) * tile_width, (k // cols) * tile_h))
    return grid


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_judgement(raw: str) -> Judgement:
    """Parse the JSON reply, tolerating code fences and surrounding text; the
    key 'entropy' (earlier prompt version) is accepted for 'uncertainty'."""
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
    data = json.loads(match.group(0))
    unc = data.get("uncertainty", data.get("entropy"))
    p, u = float(data["p_anomaly"]), float(unc)
    if not (0 <= p <= 1 and 0 <= u <= 1):
        raise ValueError(f"values out of [0, 1]: {data}")
    return Judgement(p, u, str(data.get("reason", "")), raw)


class VLMClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "", api_key: str = "lm-studio"):
        from openai import OpenAI  # optional dependency: pip install -e ".[vlm]"

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def models(self) -> list[str]:
        return [m.id for m in self._client.models.list().data]

    def judge(self, image: Image.Image, prompt: str, timeout: float = 120.0) -> Judgement:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": to_data_url(image)}},
                ],
            }],
            temperature=0.0,
            timeout=timeout,
        )
        return parse_judgement(response.choices[0].message.content)

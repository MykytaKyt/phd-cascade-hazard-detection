#!/usr/bin/env python3
"""Chapter 4: check that the served VLM sees the image (independent of the verification prompt).

Sends one URFD frame with the request to describe the scene only, with the text
before and after the image, and saves the answers.

Usage:
    python experiments/ch4_fusion/describe_check.py --base-url http://localhost:8000/v1 --model <id> \
        --out results/ch4/<model>/describe_check.json
"""
import argparse
import json
from pathlib import Path

from thesis.vlm.client import compose_grid, to_data_url

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Describe what the person in this image is doing and their body position, in one sentence."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--frame", default=ROOT / "data" / "frames" / "fall-01" / "fall-01-cam0-rgb" /
                        "fall-01-cam0-rgb-160.png")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="none")
    url = to_data_url(compose_grid([str(args.frame)], tile_width=640, columns=1))
    text, image = {"type": "text", "text": QUESTION}, {"type": "image_url", "image_url": {"url": url}}
    out = dict(frame=Path(args.frame).name, question=QUESTION)
    for order, content in (("text_first", [text, image]), ("image_first", [image, text])):
        r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": content}],
                                           temperature=0.0, max_tokens=80)
        out[order] = dict(answer=r.choices[0].message.content, prompt_tokens=r.usage.prompt_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local run: send the VLM requests to an OpenAI-compatible server (LM Studio).

Reads a requests JSONL (make_requests.py), builds the image of every request from
the URFD frames (a 3 x 2 grid for clips, the frame itself otherwise), queries the
model and appends one JSON line per request to the responses file. Already answered
requests are skipped, so the run can be interrupted and resumed.

Requires the URFD frames (tools/download_urfd.sh data) and `pip install -e ".[vlm]"`.

Usage:
    python experiments/ch4_fusion/query_vlm.py --list-models
    python experiments/ch4_fusion/query_vlm.py --model <id from --list-models> --kind event
    python experiments/ch4_fusion/query_vlm.py --model <id> --kind frame
"""
import argparse
import json
import time
from pathlib import Path

from thesis.kinematic.dataset import frame_number, list_sequence_frames
from thesis.vlm.client import PROMPT_CLIP, PROMPT_FRAME, VLMClient, compose_grid

ROOT = Path(__file__).resolve().parents[2]


def frame_paths(frames_dir: Path, seq: str) -> dict[int, str]:
    return {frame_number(Path(p).name): p for p in list_sequence_frames(frames_dir, seq)}


def answered(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip() and "error" not in json.loads(line)}


def run(requests: list[dict], client, frames_dir: Path, out_path: Path, fps: float = 30.0) -> None:
    done = answered(out_path)
    todo = [r for r in requests if r["id"] not in done]
    print(f"{len(requests)} requests, {len(done)} already answered, {len(todo)} to go")
    cache: dict[str, dict[int, str]] = {}
    t0 = time.time()
    with open(out_path, "a") as out:
        for k, req in enumerate(todo, 1):
            paths = cache.setdefault(req["seq"], frame_paths(frames_dir, req["seq"]))
            rec = dict(req)
            try:
                files = [paths[f] for f in req["frames"]]
                if len(files) > 1:
                    dt = (req["frames"][1] - req["frames"][0]) / fps
                    image, prompt = compose_grid(files), PROMPT_CLIP.format(n=len(files), dt=dt)
                else:
                    image, prompt = compose_grid(files, tile_width=640, columns=1), PROMPT_FRAME
                start = time.time()
                j = client.judge(image, prompt)
                rec.update(p_anomaly=j.p_anomaly, uncertainty=j.uncertainty, reason=j.reason, raw=j.raw,
                           latency_s=round(time.time() - start, 3))
            except Exception as e:  # keep going; failed requests are retried on the next run
                rec["error"] = f"{type(e).__name__}: {e}"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if k % 25 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"  {k}/{len(todo)}  {rate:.1f} s/request, ~{rate * (len(todo) - k) / 60:.0f} min left")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", default=ROOT / "results" / "vlm" / "urfd_requests.jsonl")
    parser.add_argument("--out", default=ROOT / "results" / "vlm" / "urfd_responses.jsonl")
    parser.add_argument("--frames-dir", default=ROOT / "data" / "frames")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--kind", choices=["event", "frame", "all"], default="event")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    client = VLMClient(base_url=args.base_url, model=args.model)
    if args.list_models:
        print("\n".join(client.models()))
        return
    if not args.model:
        parser.error("--model is required (see --list-models)")

    with open(args.requests) as f:
        requests = [json.loads(line) for line in f if line.strip()]
    if args.kind != "all":
        requests = [r for r in requests if r["kind"] == args.kind]
    requests = requests[: args.limit] if args.limit else requests
    run(requests, client, Path(args.frames_dir), Path(args.out))
    print(f"Responses in {args.out}; send this file back for the analysis.")


if __name__ == "__main__":
    main()

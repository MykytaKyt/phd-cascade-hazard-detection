#!/usr/bin/env python3
"""Local run: YOLOv8-Pose over the URFD frames -> DDL features, raw keypoints and
XCoM-based margins of stability.

Requires the URFD frames (tools/download_urfd.sh data) and `pip install -e ".[pose]"`.
A GPU speeds it up but is not required (--device cpu).

Writes data/features_kp.parquet (DDL features + raw pose + XCoM columns).

Usage:
    bash tools/download_urfd.sh data
    python experiments/ch3_kinematic/extract_features.py --device 0
"""
import argparse
import logging
from pathlib import Path

from thesis.kinematic.extract import extract_dataset_features
from thesis.kinematic.xcom import add_xcom_features

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data")
    parser.add_argument("--frames-dir", default=None, help="default: <data-dir>/frames")
    parser.add_argument("--out", default=None, help="default: <data-dir>/features_kp.parquet")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--leg-length", type=float, default=0.9, help="pendulum length l in metres")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_dir = Path(args.data_dir)
    frames_dir = Path(args.frames_dir) if args.frames_dir else data_dir / "frames"
    out = Path(args.out) if args.out else data_dir / "features_kp.parquet"
    device = int(args.device) if str(args.device).isdigit() else args.device

    df = extract_dataset_features(data_dir, frames_dir, model_name=args.model, device=device)
    df = add_xcom_features(df, leg_length_m=args.leg_length)
    df.to_parquet(out, index=False)
    print(f"Saved {out}: {len(df)} rows, valid={int(df['valid'].sum())}, "
          f"mos_xcom_x > 0: {(df['mos_xcom_x'] > 0).mean():.1%}, NaN: {df['mos_xcom_x'].isna().mean():.1%}")


if __name__ == "__main__":
    main()

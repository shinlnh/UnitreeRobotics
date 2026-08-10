#!/usr/bin/env python3
"""Run one real-data inference through the complete exported ApplePnP graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from applepnp_policy import DEFAULT_MODEL, ApplePnPOnnxPolicy
from torchcodec.decoders import VideoDecoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/sonic/g1_fetch_clean"
STATE_SLICES = {
    "left_leg": slice(0, 6),
    "right_leg": slice(6, 12),
    "waist": slice(12, 15),
    "left_arm": slice(15, 22),
    "right_arm": slice(22, 29),
    "left_hand": slice(29, 36),
    "right_hand": slice(36, 43),
}
MODEL_HAND_TO_WBC = np.asarray([5, 6, 3, 4, 0, 1, 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    parquet = sorted((args.dataset / "data").rglob("*.parquet"))[0]
    video = sorted((args.dataset / "videos").rglob("*.mp4"))[0]
    state = np.asarray(pd.read_parquet(parquet).iloc[0]["observation.state"], dtype=np.float32)
    frame = VideoDecoder(video, dimension_order="NHWC").get_frame_at(0).data.numpy()
    observation = {"video.ego_view": frame}
    for name, indices in STATE_SLICES.items():
        value = state[indices]
        if name.endswith("hand"):
            value = value[MODEL_HAND_TO_WBC]
        observation[f"state.{name}"] = value
    policy = ApplePnPOnnxPolicy(args.model.resolve(), seed=args.seed)
    action, info = policy.get_action(observation)
    for name, value in action.items():
        array = np.asarray(value)
        print(f"{name}: shape={array.shape}, range=[{array.min():.4f}, {array.max():.4f}]")
    print(f"ApplePnP ONNX smoke passed in {info['inference_seconds']:.3f}s")


if __name__ == "__main__":
    main()

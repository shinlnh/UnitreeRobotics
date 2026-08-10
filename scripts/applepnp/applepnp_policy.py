"""LeApp runtime adapter for NVIDIA's exported Unitree G1 ApplePnP policy."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gr00t.policy.policy import BasePolicy
from leapp import InferenceManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "checkpoints/nvidia-gr00t-n1.7-applepnp-v1/exported_leapp.yaml"
STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}
ACTION_NAMES = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "navigate_command",
    "base_height_command",
)
# GEAR uses index, middle, thumb; the exported state preprocessor expects
# thumb, middle, index. The decoded hand action is already in GEAR order.
HAND_STATE_TO_MODEL = np.asarray([4, 5, 6, 2, 3, 0, 1])


def _unbatch_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values; got shape {np.shape(value)}")
    return array


def _unbatch_image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (480, 640, 3):
        raise ValueError(f"video.ego_view must have shape (480, 640, 3); got {array.shape}")
    if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"video.ego_view must be uint8 or floating point; got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.float32)


class ApplePnPOnnxPolicy(BasePolicy):
    """Expose the LeApp graph through the standard GR00T policy protocol."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, seed: int = 0):
        super().__init__(strict=True)
        if not model_path.is_file():
            raise FileNotFoundError(f"ApplePnP model is missing: {model_path}")
        self.model_path = model_path
        self.seed = seed
        self.call_index = 0
        self.manager = InferenceManager(str(model_path))
        self.noise_device = self.manager.nodes["action_head"].device
        self.generator = torch.Generator(device=self.noise_device)
        self.generator.manual_seed(seed)

    def check_observation(self, observation: dict[str, Any]) -> None:
        _unbatch_image(observation["video.ego_view"])
        for name, size in STATE_DIMS.items():
            _unbatch_vector(observation[f"state.{name}"], size, f"state.{name}")

    def check_action(self, action: dict[str, Any]) -> None:
        if set(action) != {f"action.{name}" for name in ACTION_NAMES}:
            raise ValueError(f"unexpected ApplePnP action keys: {sorted(action)}")
        for name, value in action.items():
            array = np.asarray(value)
            if array.shape[0] != 16 or not np.isfinite(array).all():
                raise ValueError(f"invalid action chunk {name}: shape={array.shape}")

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        started = time.perf_counter()
        inputs = self.manager.get_mock_input()
        image = _unbatch_image(observation["video.ego_view"])
        inputs["preprocess_video/ego_view"] = torch.from_numpy(image)[None].to(
            self.manager.nodes["preprocess_video"].device
        )
        for name, size in STATE_DIMS.items():
            state = _unbatch_vector(observation[f"state.{name}"], size, f"state.{name}")
            if name.endswith("hand"):
                state = state[HAND_STATE_TO_MODEL]
            inputs[f"preprocess_state/{name}"] = torch.from_numpy(state.copy())[None].to(
                self.manager.nodes["preprocess_state"].device
            )
        inputs["action_head/initial_noise"] = torch.randn(
            (1, 40, 132), generator=self.generator, device=self.noise_device
        )
        with torch.inference_mode():
            outputs = self.manager.run_policy(inputs)
        action = {
            f"action.{name}": outputs[f"decode_action/{name}"][0].detach().cpu().numpy()
            for name in ACTION_NAMES
        }
        self.call_index += 1
        return action, {
            "inference_seconds": time.perf_counter() - started,
            "call_index": self.call_index,
            "model": str(self.model_path),
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        seed = self.seed if options is None else int(options.get("seed", self.seed))
        # LeApp caches outputs created under inference mode; reset those buffers
        # under the same mode so PyTorch permits the in-place zero operation.
        with torch.inference_mode():
            self.manager.reset()
        self.generator.manual_seed(seed)
        self.call_index = 0
        return {"seed": seed}

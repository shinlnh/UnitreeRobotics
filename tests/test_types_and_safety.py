from pathlib import Path

from unitree_gr00t.config import load_config
from unitree_gr00t.safety import SafetySupervisor
from unitree_gr00t.types import ActionChunk, CartesianAction

ROOT = Path(__file__).resolve().parents[1]


def test_action_chunk_matches_sonic_dimensions() -> None:
    action = CartesianAction(0.01, -0.02, 0.03, True)
    chunk = ActionChunk.from_cartesian([action] * 40)
    chunk.validate()
    assert chunk.horizon == 40
    assert len(chunk.motion_token[0]) == 64
    assert len(chunk.left_hand_joints[0]) == 7
    assert len(chunk.right_hand_joints[0]) == 7


def test_safety_clips_delta_and_workspace() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    safety = SafetySupervisor(config.safety)
    decision = safety.filter(CartesianAction(1.0, -1.0, 1.0, False), (0.99, 0.01, 0.94))
    assert decision.clipped
    assert decision.action.dx <= 0.01 + 1e-9
    assert decision.action.dy >= -0.01 - 1e-9
    assert decision.action.dz <= 0.01 + 1e-9


def test_emergency_stop_zeroes_motion() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    safety = SafetySupervisor(config.safety)
    safety.emergency_stop()
    decision = safety.filter(CartesianAction(0.03, 0.04, -0.02, True), (0.5, 0.5, 0.5))
    assert decision.action == CartesianAction(0.0, 0.0, 0.0, True)
    assert decision.reasons == ("emergency_stop",)

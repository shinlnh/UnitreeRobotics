from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from unitree_rl_groot.groot.fetch import FetchMission, FetchPhase
from unitree_rl_groot.groot.isaac_multitask import (
    BASE_HEIGHT_LIMITS_M,
    NAV_COMMAND_LIMITS,
    PINK_HAND_LIMITS,
    POLICY_HANDS_TO_PINK,
    REAL_G1_ACTION_WIDTHS,
    REAL_G1_STATE_JOINTS,
    WRIST_WORKSPACES,
    IsaacActionSafetyFilter,
    eef9d_to_pose7_xyzw,
    joint_groups,
    make_policy_observation,
    pose7_xyzw_to_eef9d,
    real_g1_actions_to_isaac,
)
from unitree_rl_groot.groot.lab_commands import LabCommandInbox, LabCommandKind, parse_lab_command
from unitree_rl_groot.groot.multitask import G1_FRUIT_TASKS, canonical_prompt, resolve_fruit_task
from unitree_rl_groot.groot.supervisor import (
    PINK_LEFT_HAND_INDICES,
    PINK_RIGHT_HAND_INDICES,
    SafeFetchSupervisor,
)

ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_is_one_four_task_contract() -> None:
    assert [task.object_name for task in G1_FRUIT_TASKS] == ["apple", "pear", "grapes", "starfruit"]
    assert len({task.dataset_directory for task in G1_FRUIT_TASKS}) == 4
    assert all("place it on the plate" in task.prompt for task in G1_FRUIT_TASKS)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Pick up the red apple and place it on the plate", "apple"),
        ("Hãy lấy quả lê vàng bỏ lên đĩa", "pear"),
        ("Lấy chùm nho xanh cho anh", "grapes"),
        ("Nhặt quả khế vàng", "starfruit"),
    ],
)
def test_resolves_english_and_vietnamese(instruction: str, expected: str) -> None:
    assert resolve_fruit_task(instruction).object_name == expected


def test_persistent_lab_command_parser_separates_control_from_missions() -> None:
    mission = parse_lab_command("Hãy lấy quả lê vàng bỏ lên đĩa")
    assert mission.kind is LabCommandKind.MISSION
    assert mission.task is not None and mission.task.object_name == "pear"
    assert parse_lab_command("status").kind is LabCommandKind.STATUS
    assert parse_lab_command("dừng").kind is LabCommandKind.STOP
    assert parse_lab_command("reset").kind is LabCommandKind.RESET
    assert parse_lab_command("thoát").kind is LabCommandKind.QUIT
    invalid = parse_lab_command("hãy lau bàn")
    assert invalid.kind is LabCommandKind.INVALID
    assert "supported object" in str(invalid.error)


def test_persistent_lab_inbox_is_fifo_and_nonblocking() -> None:
    inbox = LabCommandInbox()
    assert inbox.poll() is None
    inbox.submit("status")
    inbox.submit("Lấy táo đỏ")
    assert inbox.poll().kind is LabCommandKind.STATUS
    assert inbox.poll().kind is LabCommandKind.MISSION
    assert inbox.poll() is None


def test_rejects_unknown_and_ambiguous_requests() -> None:
    with pytest.raises(ValueError, match="supported object"):
        resolve_fruit_task("pick up the steering wheel")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_fruit_task("move the apple and pear")


def test_canonical_prompt() -> None:
    assert canonical_prompt("apple") == "Pick up the red apple and place it on the plate"
    with pytest.raises(ValueError, match="unsupported object"):
        canonical_prompt("bottle")


def test_local_base_checkpoint_has_generalist_real_g1_contract() -> None:
    raw = json.loads((ROOT / "checkpoints/nvidia-gr00t-n1.7-3b/processor_config.json").read_text())
    config = raw["processor_kwargs"]["modality_configs"]["real_g1_relative_eef_relative_joints"]
    assert config["video"] == {"delta_indices": [-20, 0], "modality_keys": ["ego_view"]}
    assert config["state"]["modality_keys"] == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
    ]
    assert config["action"]["modality_keys"][-2:] == [
        "base_height_command",
        "navigate_command",
    ]
    assert config["language"]["modality_keys"] == ["annotation.human.task_description"]


def test_isaac_bridge_preserves_eef_pose() -> None:
    angle = np.deg2rad(40.0)
    pose = np.array([0.3, -0.2, 0.8, 0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
    restored = eef9d_to_pose7_xyzw(pose7_xyzw_to_eef9d(pose))
    np.testing.assert_allclose(restored[:3], pose[:3], atol=1e-6)
    assert abs(float(np.dot(restored[3:], pose[3:]))) == pytest.approx(1.0, abs=1e-6)


def test_isaac_bridge_uses_exact_dataset_joint_order() -> None:
    names = [name for group in REAL_G1_STATE_JOINTS.values() for name in group]
    positions = np.arange(len(names), dtype=np.float32)[None]
    groups = joint_groups(names, positions)
    for group_name, expected_names in REAL_G1_STATE_JOINTS.items():
        expected = [names.index(name) for name in expected_names]
        np.testing.assert_array_equal(groups[group_name][0], expected)


def test_isaac_policy_observation_has_real_g1_shapes() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    pose = np.array([0.2, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    groups = {
        name: np.zeros((1, len(joints)), dtype=np.float32) for name, joints in REAL_G1_STATE_JOINTS.items()
    }
    observation = make_policy_observation(
        rgb_history=[image, image],
        left_wrist_pose=pose,
        right_wrist_pose=pose,
        groups=groups,
        prompt="Pick up the red apple and place it on the plate",
    )
    assert observation["video.ego_view"].shape == (1, 2, 48, 64, 3)
    assert observation["state.left_wrist_eef_9d"].shape == (1, 1, 9)
    assert observation["annotation.human.task_description"] == [
        "Pick up the red apple and place it on the plate"
    ]


def test_real_g1_actions_map_to_isaac_32d_and_reorder_hands() -> None:
    horizon = 3
    action = {}
    for name, width in REAL_G1_ACTION_WIDTHS.items():
        value = np.zeros((1, horizon, width), dtype=np.float32)
        if name.endswith("eef_9d"):
            value[..., 3] = 1.0
            value[..., 7] = 1.0
        action[f"action.{name}"] = value
    action["action.left_hand"][0] = np.arange(7, dtype=np.float32)
    action["action.right_hand"][0] = np.arange(7, 14, dtype=np.float32)
    action["action.navigate_command"][0, :, :] = [0.3, -0.1, 0.2]
    action["action.base_height_command"][0, :, 0] = 0.78

    isaac_action = real_g1_actions_to_isaac(action)
    assert isaac_action.shape == (horizon, 32)
    np.testing.assert_array_equal(isaac_action[0, 14:28], np.arange(14)[list(POLICY_HANDS_TO_PINK)])
    np.testing.assert_allclose(isaac_action[0, 28:32], [0.3, -0.1, 0.2, 0.78])


def _neutral_isaac_action() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.array([0.22, 0.14, 0.12, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right = np.array([0.22, -0.14, 0.12, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    hands = np.zeros(14, dtype=np.float32)
    action = np.concatenate((left, right, hands, np.zeros(3), np.array([0.72]))).astype(np.float32)
    return action, left, right, hands


def test_safety_filter_leaves_a_safe_hold_unchanged() -> None:
    action, left, right, hands = _neutral_isaac_action()
    filtered, report = IsaacActionSafetyFilter().filter(
        action,
        current_left_wrist=left,
        current_right_wrist=right,
        current_hands=hands,
    )
    np.testing.assert_allclose(filtered, action, atol=1e-7)
    assert not report.was_clipped


def test_safety_filter_is_immutable_and_prevents_crossed_arms() -> None:
    action, left, right, hands = _neutral_isaac_action()
    action[0:3] = [2.0, -1.0, 1.5]
    action[3:7] = [1.0, 0.0, 0.0, 0.0]
    action[7:10] = [2.0, 1.0, 1.5]
    action[10:14] = [0.0, 1.0, 0.0, 0.0]
    action[14:28] = 10.0
    action[28:31] = [4.0, -4.0, 4.0]
    action[31] = 0.2
    original = action.copy()

    filtered, report = IsaacActionSafetyFilter().filter(
        action,
        current_left_wrist=left,
        current_right_wrist=right,
        current_hands=hands,
    )

    np.testing.assert_array_equal(action, original)
    assert report.was_clipped
    assert filtered[1] >= WRIST_WORKSPACES["left"][1, 0]
    assert filtered[8] <= WRIST_WORKSPACES["right"][1, 1]
    assert np.linalg.norm(filtered[0:3] - left[:3]) <= 0.012001
    assert np.linalg.norm(filtered[7:10] - right[:3]) <= 0.012001
    assert np.all(filtered[14:28] >= PINK_HAND_LIMITS[:, 0])
    assert np.all(filtered[14:28] <= PINK_HAND_LIMITS[:, 1])
    assert np.max(np.abs(filtered[14:28] - hands)) <= 0.080001
    assert np.all(np.abs(filtered[28:31]) <= NAV_COMMAND_LIMITS)
    assert np.max(np.abs(filtered[28:31])) <= 0.050001
    assert BASE_HEIGHT_LIMITS_M[0] <= filtered[31] <= BASE_HEIGHT_LIMITS_M[1]
    assert filtered[31] == pytest.approx(0.715, abs=1e-6)
    assert report.requested_wrist_rotation_deg[0] == pytest.approx(180.0, abs=1e-4)
    assert {
        "left_wrist_workspace",
        "right_wrist_workspace",
        "left_wrist_position_slew",
        "right_wrist_position_slew",
        "left_wrist_orientation_slew",
        "right_wrist_orientation_slew",
        "hand_joint_limits",
        "hand_joint_slew",
        "navigation_limits",
        "navigation_slew",
        "base_height_limits",
        "base_height_slew",
    } <= set(report.clipped_fields)


def test_safety_filter_rate_limits_repeated_policy_targets_without_mutating_them() -> None:
    action, left, right, hands = _neutral_isaac_action()
    action[0] = 0.50
    action[7] = 0.50
    original = action.copy()
    safety = IsaacActionSafetyFilter(max_wrist_step_m=0.01)

    first, _ = safety.filter(
        action,
        current_left_wrist=left,
        current_right_wrist=right,
        current_hands=hands,
    )
    second, _ = safety.filter(
        action,
        current_left_wrist=first[0:7],
        current_right_wrist=first[7:14],
        current_hands=first[14:28],
    )

    np.testing.assert_array_equal(action, original)
    assert first[0] == pytest.approx(left[0] + 0.01, abs=1e-6)
    assert second[0] == pytest.approx(left[0] + 0.02, abs=1e-6)
    assert first[7] == pytest.approx(right[0] + 0.01, abs=1e-6)
    assert second[7] == pytest.approx(right[0] + 0.02, abs=1e-6)


def _pose(x: float, y: float, z: float = 0.72, yaw_deg: float = 0.0) -> np.ndarray:
    half_yaw = np.deg2rad(yaw_deg) / 2.0
    return np.array([x, y, z, 0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)


def test_fetch_supervisor_navigates_then_allows_only_nearest_hand() -> None:
    supervisor = SafeFetchSupervisor(
        FetchMission("yellow pear", "pickup table", "plate"), initial_object_height_m=0.77
    )
    destination = _pose(4.0, 0.0)
    far = supervisor.observe(
        robot_pose=_pose(0.0, 0.0),
        object_pose=_pose(2.0, 0.5, 0.77),
        destination_pose=destination,
    )
    assert far.phase is FetchPhase.NAVIGATE_TO_OBJECT
    assert far.active_hand == "left"
    np.testing.assert_allclose(far.manipulation_target_base, [2.0, 0.5, 0.05], atol=1e-6)
    assert far.navigation_command[0] > 0.0
    assert far.navigation_command[1] > 0.0

    action, left, right, hands = _neutral_isaac_action()
    action[0:14] += 0.2
    action[14:28] = 0.5
    original = action.copy()
    walking = supervisor.supervise_action(
        action,
        far,
        current_left_wrist=left,
        current_right_wrist=right,
        current_hands=hands,
    )
    np.testing.assert_array_equal(action, original)
    np.testing.assert_array_equal(walking[0:7], left)
    np.testing.assert_array_equal(walking[7:14], right)
    np.testing.assert_array_equal(walking[14:28], hands)
    np.testing.assert_array_equal(walking[28:31], far.navigation_command)

    near = supervisor.observe(
        robot_pose=_pose(1.4, 0.0),
        object_pose=_pose(2.0, 0.2, 0.77),
        destination_pose=destination,
    )
    assert near.phase is FetchPhase.GRASP
    assert near.active_hand == "left"
    # Simulate the bad accumulated VLA target from the regression screenshot:
    # hand above the head and far from the selected fruit.
    action[0:3] = [0.55, 0.45, 0.90]
    grasp = supervisor.supervise_action(
        action,
        near,
        current_left_wrist=left,
        current_right_wrist=right,
        current_hands=hands,
    )
    np.testing.assert_array_equal(grasp[7:14], right)
    np.testing.assert_array_equal(grasp[14 + PINK_RIGHT_HAND_INDICES], hands[PINK_RIGHT_HAND_INDICES])
    np.testing.assert_array_equal(grasp[14 + PINK_LEFT_HAND_INDICES], action[14 + PINK_LEFT_HAND_INDICES])
    target = near.manipulation_target_base
    assert target[0] - 0.20 <= grasp[0] <= target[0] + 0.20
    assert target[1] - 0.18 <= grasp[1] <= target[1] + 0.18
    assert target[2] - 0.10 <= grasp[2] <= target[2] + 0.22
    assert grasp[2] < 0.30
    assert grasp[31] == pytest.approx(0.72)


def test_fetch_supervisor_returns_unused_arm_to_neutral_pose() -> None:
    supervisor = SafeFetchSupervisor(
        FetchMission("red apple", "pickup table", "plate"), initial_object_height_m=0.75
    )
    near = supervisor.observe(
        robot_pose=_pose(0.0, 0.0),
        object_pose=_pose(0.55, 0.15, 0.75),
        destination_pose=_pose(3.0, 0.0),
    )
    action, left, right, hands = _neutral_isaac_action()
    bad_right = right.copy()
    bad_right[:3] = [0.35, -0.40, 0.50]
    neutral_right = right.copy()
    neutral_right[:3] = [0.18, -0.18, 0.08]
    neutral_hands = np.full(14, 0.1, dtype=np.float32)
    supervised = supervisor.supervise_action(
        action,
        near,
        current_left_wrist=left,
        current_right_wrist=bad_right,
        current_hands=hands,
        neutral_left_wrist=left,
        neutral_right_wrist=neutral_right,
        neutral_hands=neutral_hands,
    )
    np.testing.assert_array_equal(supervised[7:14], neutral_right)
    np.testing.assert_array_equal(
        supervised[14 + PINK_RIGHT_HAND_INDICES], neutral_hands[PINK_RIGHT_HAND_INDICES]
    )


def test_fetch_supervisor_carry_place_and_complete_transitions() -> None:
    supervisor = SafeFetchSupervisor(
        FetchMission("red apple", "pickup table", "plate"), initial_object_height_m=0.75
    )
    destination = _pose(3.0, 0.0)
    supervisor.observe(
        robot_pose=_pose(0.0, 0.0),
        object_pose=_pose(0.5, -0.1, 0.75),
        destination_pose=destination,
    )
    carry = supervisor.observe(
        robot_pose=_pose(0.0, 0.0),
        object_pose=_pose(0.5, -0.1, 0.84),
        destination_pose=destination,
    )
    assert carry.phase is FetchPhase.CARRY_TO_DESTINATION
    assert carry.navigation_command[0] > 0.0

    place = supervisor.observe(
        robot_pose=_pose(2.45, 0.0),
        object_pose=_pose(2.8, 0.0, 0.84),
        destination_pose=destination,
    )
    assert place.phase is FetchPhase.PLACE
    np.testing.assert_array_equal(place.navigation_command, np.zeros(3))

    verify = supervisor.observe(
        robot_pose=_pose(2.45, 0.0),
        object_pose=_pose(3.0, 0.0, 0.77),
        destination_pose=destination,
    )
    assert verify.phase is FetchPhase.VERIFY_DELIVERY
    complete = supervisor.observe(
        robot_pose=_pose(2.45, 0.0),
        object_pose=_pose(3.0, 0.0, 0.77),
        destination_pose=destination,
        official_success=True,
    )
    assert complete.phase is FetchPhase.COMPLETE

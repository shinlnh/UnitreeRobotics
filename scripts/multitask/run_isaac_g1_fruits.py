#!/usr/bin/env python3
"""Run language -> GR00T N1.7 REAL_G1 -> G1 whole-body control in Isaac Sim."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "source" / "unitree_rl_groot"))

from unitree_rl_groot.groot.fetch import FetchMission  # noqa: E402
from unitree_rl_groot.groot.isaac_multitask import (  # noqa: E402
    NOMINAL_BASE_HEIGHT_M,
    POLICY_HANDS_TO_PINK,
    REAL_G1_STATE_JOINTS,
    IsaacActionSafetyFilter,
    joint_groups,
    make_policy_observation,
    pose7_xyzw_to_eef9d,
    real_g1_actions_to_isaac,
)
from unitree_rl_groot.groot.lab_commands import LabCommandInbox, LabCommandKind  # noqa: E402
from unitree_rl_groot.groot.multitask import TASK_BY_OBJECT, resolve_fruit_task  # noqa: E402
from unitree_rl_groot.groot.supervisor import SafeFetchSupervisor  # noqa: E402

from scripts.sonic.policy_client import PolicyClient  # noqa: E402

DEFAULT_INITIAL_STATE_DATASET = PROJECT_ROOT / "datasets/g1_locomanip_hf/dataset_annotated_g1_locomanip.hdf5"
TARGET_XY = (-0.35, 0.45)
DISTRACTOR_XY = (
    (-0.62, 0.34),
    (-0.62, 0.56),
    (-0.08, 0.34),
)
FRUIT_CENTER_Z = {"apple": 0.755, "pear": 0.755, "grapes": 0.750, "starfruit": 0.725}
MIN_SAFE_PELVIS_Z = 0.55
MAX_SAFE_TILT_DEG = 45.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--instruction", help="English/Vietnamese request naming exactly one fruit")
    command.add_argument("--object", choices=tuple(TASK_BY_OBJECT), help="canonical target shortcut")
    parser.add_argument("--task", default="Isaac-G1-SteeringWheel-Locomanipulation")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5550)
    parser.add_argument("--policy-timeout-ms", type=int, default=120_000)
    parser.add_argument("--initial-state-dataset", type=Path, default=DEFAULT_INITIAL_STATE_DATASET)
    parser.add_argument("--demo", default="demo_1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=50.0)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--scene-smoke-steps", type=int, default=100)
    parser.add_argument("--max-wrist-step-m", type=float, default=0.012)
    parser.add_argument("--max-wrist-rotation-deg", type=float, default=7.5)
    parser.add_argument("--capture-path", type=Path, help="optional reset-frame PNG for scene inspection")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="keep Isaac open and accept repeated missions plus stop/reset/status/quit on stdin",
    )
    parser.add_argument(
        "--scene-smoke",
        action="store_true",
        help="launch the four-fruit Isaac scene and take one safe controller step without GR00T",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"], device="cpu")
    return parser.parse_args()


def _fruit_cfg(object_name: str, entity_name: str, position: tuple[float, float, float]):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    common = {
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color={
                "apple": (0.78, 0.025, 0.02),
                "pear": (0.94, 0.72, 0.03),
                "grapes": (0.10, 0.52, 0.04),
                "starfruit": (0.98, 0.78, 0.03),
            }[object_name]
        ),
        "physics_material": sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=1.0,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.16),
    }
    spawn = {
        "apple": sim_utils.SphereCfg(radius=0.055, **common),
        # The public pear demonstrations use a round yellow/orange fruit. A
        # capsule looked like a pill to the vision model and created avoidable
        # domain shift; the sphere matches its silhouette and grasp diameter.
        "pear": sim_utils.SphereCfg(radius=0.055, **common),
        "grapes": sim_utils.SphereCfg(radius=0.050, **common),
        "starfruit": sim_utils.CylinderCfg(radius=0.060, height=0.050, axis="Z", **common),
    }[object_name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{entity_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=spawn,
    )


def _install_four_fruit_scene(env_cfg, target_name: str, *, persistent: bool = False) -> dict[str, str]:
    """Bind target to ``scene.object`` and install the other three as distractors."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    entity_by_object: dict[str, str] = {}
    scene_anchor = "apple" if persistent else target_name
    positions = {scene_anchor: (*TARGET_XY, FRUIT_CENTER_Z[scene_anchor])}
    positions.update(
        (
            (name, (*xy, FRUIT_CENTER_Z[name]))
            for name, xy in zip(
                (name for name in TASK_BY_OBJECT if name != scene_anchor), DISTRACTOR_XY, strict=True
            )
        )
    )
    for object_name, position in positions.items():
        entity_name = "object" if object_name == scene_anchor else f"fruit_{object_name}"
        cfg = _fruit_cfg(object_name, entity_name.title().replace("_", ""), position)
        if entity_name == "object":
            env_cfg.scene.object = cfg
        else:
            setattr(env_cfg.scene, entity_name, cfg)
        entity_by_object[object_name] = entity_name
    # Keep the language/visual goal aligned: the drop-off table contains the
    # plate named by every official NVIDIA fruit instruction.
    env_cfg.scene.destination_plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DestinationPlate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -3.55, 0.715)),
        spawn=sim_utils.CylinderCfg(
            radius=0.18,
            height=0.025,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.74, 0.76)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    return entity_by_object


def _strengthen_neutral_posture(env_cfg) -> None:
    """Keep the waist neutral and regularize shoulder redundancy.

    The nearby tabletop is reachable by the arms. Allowing task-space IK to
    use the waist made it satisfy noisy wrist orientations by twisting the
    torso up to 30 degrees. Locomotion remains responsible for turning the
    whole robot toward targets that are outside the arm workspace.
    """

    env_cfg.actions.upper_body_ik.pink_controlled_joint_names = [
        pattern
        for pattern in env_cfg.actions.upper_body_ik.pink_controlled_joint_names
        if not pattern.startswith("waist_")
    ]

    for task_cfg in env_cfg.actions.upper_body_ik.controller.variable_input_tasks:
        if getattr(task_cfg, "controlled_joints", None):
            task_cfg.controlled_joints = [
                name for name in task_cfg.controlled_joints if not name.startswith("waist_")
            ]
            task_cfg.cost = 0.5
            task_cfg.gain = 0.12


def _neutral_relative_action(
    left_wrist: np.ndarray, right_wrist: np.ndarray, hands: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (
            left_wrist,
            right_wrist,
            hands,
            np.zeros(3, dtype=np.float32),
            np.array([NOMINAL_BASE_HEIGHT_M], dtype=np.float32),
        )
    ).astype(np.float32)


def _merge_scene_state(env, recorded_state: dict) -> dict:
    """Add project-local distractors absent from NVIDIA's one-object reset file."""

    current_state = env.scene.get_state(is_relative=False)
    merged = {category: dict(entities) for category, entities in recorded_state.items()}
    for category, entities in current_state.items():
        merged.setdefault(category, {})
        for entity_name, entity_state in entities.items():
            merged[category].setdefault(entity_name, entity_state)
    # The annotated reset supplies the official G1 articulation pose.  Its
    # steering-wheel rigid pose has a different asset origin, so all project
    # fruit keep their deterministic procedural defaults instead.
    merged["rigid_object"] = dict(current_state["rigid_object"])
    return merged


def _base_relative_wrist_poses(env) -> tuple[torch.Tensor, torch.Tensor]:
    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul

    obs = env.obs_buf["policy"]
    left_world = torch.cat((obs["left_eef_pos"], obs["left_eef_quat"]), dim=-1)
    right_world = torch.cat((obs["right_eef_pos"], obs["right_eef_quat"]), dim=-1)
    base_inv = transform_inv(env.get_base().get_pose())
    return transform_mul(base_inv, left_world), transform_mul(base_inv, right_world)


def _build_policy_input(env, history: deque[np.ndarray], prompt: str) -> dict[str, object]:
    left_pose, right_pose = _base_relative_wrist_poses(env)
    robot = env.scene["robot"]
    positions = robot.data.joint_pos.torch.detach().cpu().numpy()
    groups = joint_groups(robot.data.joint_names, positions)
    return make_policy_observation(
        rgb_history=history,
        left_wrist_pose=left_pose[0].detach().cpu().numpy(),
        right_wrist_pose=right_pose[0].detach().cpu().numpy(),
        groups=groups,
        prompt=prompt,
    )


def _world_action(env, relative_action: np.ndarray) -> torch.Tensor:
    """Compose a copied base-relative action into world coordinates.

    ``torch.as_tensor`` on a CPU NumPy view aliases its storage.  Mutating that
    tensor used to overwrite the cached GR00T action chunk, causing the base
    transform to compound every time an action sample was held for 2-3 steps.
    ``torch.tensor`` deliberately owns fresh storage.
    """

    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_mul

    action = torch.tensor(relative_action[None], dtype=torch.float32, device=env.device)
    base_pose = env.get_base().get_pose()
    action[:, 0:7] = transform_mul(base_pose, action[:, 0:7])
    action[:, 7:14] = transform_mul(base_pose, action[:, 7:14])
    return action


def _pink_hand_positions(env) -> np.ndarray:
    robot = env.scene["robot"]
    groups = joint_groups(robot.data.joint_names, robot.data.joint_pos.torch.detach().cpu().numpy())
    policy_order = np.concatenate((groups["left_hand"][0], groups["right_hand"][0]))
    return np.ascontiguousarray(policy_order[list(POLICY_HANDS_TO_PINK)], dtype=np.float32)


def _safe_real_g1_action(env) -> torch.Tensor:
    """Round-trip the current posture through REAL_G1 and hold it for one step."""

    left_pose, right_pose = _base_relative_wrist_poses(env)
    robot = env.scene["robot"]
    groups = joint_groups(robot.data.joint_names, robot.data.joint_pos.torch.detach().cpu().numpy())
    raw_action = {
        "action.left_wrist_eef_9d": pose7_xyzw_to_eef9d(left_pose.detach().cpu().numpy())[:, None],
        "action.right_wrist_eef_9d": pose7_xyzw_to_eef9d(right_pose.detach().cpu().numpy())[:, None],
        "action.left_hand": groups["left_hand"][:, None],
        "action.right_hand": groups["right_hand"][:, None],
        "action.left_arm": groups["left_arm"][:, None],
        "action.right_arm": groups["right_arm"][:, None],
        "action.waist": groups["waist"][:, None],
        "action.base_height_command": np.full((1, 1, 1), NOMINAL_BASE_HEIGHT_M, dtype=np.float32),
        "action.navigate_command": np.zeros((1, 1, 3), dtype=np.float32),
    }
    relative_action = real_g1_actions_to_isaac(raw_action)[0]
    return _world_action(env, relative_action)


def _stability_status(env) -> tuple[float, float]:
    pelvis_z = float(env.get_base().get_pose()[0, 2])
    projected_gravity = env.obs_buf["lower_body_policy"][0, 6:9]
    gravity_z = float(projected_gravity[2])
    tilt_deg = math.degrees(math.acos(float(np.clip(-gravity_z, -1.0, 1.0))))
    return pelvis_z, tilt_deg


def _has_fallen(env) -> tuple[bool, float, float]:
    pelvis_z, tilt_deg = _stability_status(env)
    return pelvis_z < MIN_SAFE_PELVIS_Z or tilt_deg > MAX_SAFE_TILT_DEG, pelvis_z, tilt_deg


def _max_abs_waist_deg(env) -> float:
    robot = env.scene["robot"]
    groups = joint_groups(robot.data.joint_names, robot.data.joint_pos.torch.detach().cpu().numpy())
    return float(np.rad2deg(np.max(np.abs(groups["waist"][0]))))


def _target_status(env, target_entity: str, destination_pose: torch.Tensor) -> str:
    target_pose = env.scene[target_entity].data.root_pose_w.torch[0]
    robot_xy = env.get_base().get_pose()[0, :2]
    robot_distance = float(torch.linalg.vector_norm(target_pose[:2] - robot_xy))
    destination_distance = float(torch.linalg.vector_norm(target_pose[:2] - destination_pose[0, :2]))
    return (
        f"robot-target={robot_distance:.2f}m target-destination={destination_distance:.2f}m "
        f"target-z={float(target_pose[2]):.2f}m"
    )


def _build_lab_command_window(inbox: LabCommandInbox):
    """Create an in-simulator command surface backed by the same safe inbox."""

    import omni.ui as ui

    command_model = ui.SimpleStringModel("")
    status_model = ui.SimpleStringModel("IDLE — choose a fruit or enter a command")

    def submit_text(text: str) -> None:
        inbox.submit(text)
        command_model.set_value("")

    def submit_field() -> None:
        text = command_model.get_value_as_string().strip()
        if text:
            submit_text(text)

    window = ui.Window(
        "Unitree Mission Center",
        width=480,
        height=250,
        visible=True,
        dock_preference=ui.DockPreference.RIGHT_TOP,
    )
    with window.frame, ui.VStack(spacing=6, height=0):
        ui.Label("Command the persistent G1 lab", height=24, style={"font_size": 18})
        ui.StringField(model=status_model, read_only=True, height=28)
        ui.StringField(model=command_model, height=30)
        ui.Button("SEND COMMAND", clicked_fn=submit_field, height=32)
        with ui.HStack(spacing=4, height=30):
            ui.Button(
                "APPLE", clicked_fn=lambda: submit_text("Pick up the red apple and place it on the plate")
            )
            ui.Button(
                "PEAR",
                clicked_fn=lambda: submit_text("Pick up the yellow pear and place it on the plate"),
            )
            ui.Button(
                "GRAPES",
                clicked_fn=lambda: submit_text("Pick up the green grapes and place it on the plate"),
            )
            ui.Button(
                "STARFRUIT",
                clicked_fn=lambda: submit_text("Pick up the yellow starfruit and place it on the plate"),
            )
        with ui.HStack(spacing=4, height=30):
            ui.Button("STATUS", clicked_fn=lambda: submit_text("status"))
            ui.Button("STOP", clicked_fn=lambda: submit_text("stop"))
            ui.Button("RESET", clicked_fn=lambda: submit_text("reset"))
            ui.Button("QUIT", clicked_fn=lambda: submit_text("quit"))
    return window, status_model


def _run_persistent_lab(
    *,
    env,
    simulation_app,
    client: PolicyClient,
    history: deque[np.ndarray],
    entity_by_object: dict[str, str],
    destination_pose: torch.Tensor,
    initial_scene_state: dict,
    initial_instruction: str | None,
    args: argparse.Namespace,
    neutral_left_wrist: np.ndarray,
    neutral_right_wrist: np.ndarray,
    neutral_hands: np.ndarray,
) -> int:
    """Run a commandable lab until the operator explicitly quits or closes Kit."""

    inbox = LabCommandInbox()
    if initial_instruction:
        inbox.submit(initial_instruction)
    inbox.start_stdin_reader()
    lab_window = None
    status_model = None
    if "kit" in (args.visualizer or []):
        lab_window, status_model = _build_lab_command_window(inbox)

    def show_status(message: str) -> None:
        if status_model is not None:
            status_model.set_value(message)

    print(
        "[LAB READY] Isaac stays open. Enter a fruit mission, or: help | status | stop | reset | quit",
        flush=True,
    )
    print("[LAB SCOPE] apple, pear, grapes, starfruit; unsupported skills are rejected", flush=True)
    show_status("IDLE — choose a fruit or enter a command")

    neutral_action = _neutral_relative_action(neutral_left_wrist, neutral_right_wrist, neutral_hands)
    safety_filter = IsaacActionSafetyFilter(
        max_wrist_step_m=args.max_wrist_step_m,
        max_wrist_rotation_deg=args.max_wrist_rotation_deg,
    )
    active_task = None
    active_instruction = ""
    target_entity = ""
    supervisor = None
    action_chunk: np.ndarray | None = None
    control_index = 0
    executed_horizon = 0
    inference_count = 0
    mission_step = 0
    stable_delivery_steps = 0
    last_phase = None
    policy_period = 1.0 / args.policy_hz
    required_stable_steps = max(1, round(0.5 / float(env.step_dt)))
    max_mission_steps = max(1, round(args.max_seconds / float(env.step_dt)))
    safety_clamps: Counter[str] = Counter()

    def clear_mission() -> None:
        nonlocal active_task, supervisor, action_chunk, control_index, last_phase
        active_task = None
        supervisor = None
        action_chunk = None
        control_index = 0
        last_phase = None
        show_status("IDLE — waiting for a command")

    def restore_lab(reason: str) -> None:
        nonlocal stable_delivery_steps, mission_step, safety_filter
        clear_mission()
        env.reset_to(
            state=initial_scene_state,
            env_ids=torch.tensor([0], device=env.device),
            is_relative=False,
        )
        history.clear()
        image = env.obs_buf["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
        history.append(image.copy())
        client.reset({"seed": args.seed})
        safety_filter = IsaacActionSafetyFilter(
            max_wrist_step_m=args.max_wrist_step_m,
            max_wrist_rotation_deg=args.max_wrist_rotation_deg,
        )
        stable_delivery_steps = 0
        mission_step = 0
        print(f"[LAB RESET] {reason}; scene restored, robot is idle", flush=True)
        show_status(f"RESET COMPLETE — {reason}")

    while simulation_app.is_running() and not simulation_app.is_exiting():
        command = inbox.poll()
        quit_requested = False
        while command is not None:
            if command.kind is LabCommandKind.HELP:
                print(
                    "[LAB HELP] Examples: 'Lấy quả táo đỏ bỏ lên đĩa', "
                    "'Pick up the green grapes and place it on the plate'. "
                    "Controls: status, stop, reset, quit.",
                    flush=True,
                )
            elif command.kind is LabCommandKind.STATUS:
                pelvis_z, tilt_deg = _stability_status(env)
                if active_task is None:
                    print(
                        f"[LAB STATUS] idle; pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg",
                        flush=True,
                    )
                else:
                    print(
                        f"[LAB STATUS] mission={active_task.object_name} phase={supervisor.phase.value} "
                        f"elapsed={mission_step * env.step_dt:.1f}s "
                        f"pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg "
                        f"{_target_status(env, target_entity, destination_pose)}",
                        flush=True,
                    )
            elif command.kind is LabCommandKind.STOP:
                clear_mission()
                print("[LAB STOP] mission cancelled; returning both arms to neutral", flush=True)
                show_status("STOPPED — returning to neutral")
            elif command.kind is LabCommandKind.RESET:
                restore_lab("operator command")
            elif command.kind is LabCommandKind.QUIT:
                quit_requested = True
            elif command.kind is LabCommandKind.INVALID:
                print(f"[LAB REJECTED] {command.error}. Type 'help' for supported commands.", flush=True)
                show_status(f"REJECTED — {command.error}")
            elif command.kind is LabCommandKind.MISSION:
                active_task = command.task
                active_instruction = command.text
                target_entity = entity_by_object[active_task.object_name]
                initial_height = float(env.scene[target_entity].data.root_pose_w.torch[0, 2])
                object_description = active_task.prompt.removeprefix("Pick up the ").split(" and place", 1)[0]
                supervisor = SafeFetchSupervisor(
                    FetchMission(object_description, "pickup table", "plate"),
                    initial_object_height_m=initial_height,
                )
                client.reset({"seed": args.seed})
                action_chunk = None
                control_index = 0
                mission_step = 0
                stable_delivery_steps = 0
                last_phase = None
                safety_clamps.clear()
                print(
                    f"[LAB MISSION] {active_instruction!r} -> target={active_task.object_name}; "
                    "this preempts any previous mission",
                    flush=True,
                )
                show_status(f"RUNNING — target={active_task.object_name}; starting executive")
            command = inbox.poll()
        if quit_requested:
            print("[LAB QUIT] closing the persistent Isaac session", flush=True)
            return 0

        left_pose_t, right_pose_t = _base_relative_wrist_poses(env)
        current_left = left_pose_t[0].detach().cpu().numpy()
        current_right = right_pose_t[0].detach().cpu().numpy()
        current_hands = _pink_hand_positions(env)

        if active_task is None:
            requested_action = neutral_action
        else:
            target_pose = env.scene[target_entity].data.root_pose_w.torch[0]
            destination_distance = float(torch.linalg.vector_norm(target_pose[:2] - destination_pose[0, :2]))
            target_speed = float(
                torch.linalg.vector_norm(env.scene[target_entity].data.root_lin_vel_w.torch[0])
            )
            delivered_now = destination_distance <= 0.35 and target_speed <= 0.08
            stable_delivery_steps = stable_delivery_steps + 1 if delivered_now else 0
            decision = supervisor.observe(
                robot_pose=env.get_base().get_pose()[0].detach().cpu().numpy(),
                object_pose=target_pose.detach().cpu().numpy(),
                destination_pose=destination_pose[0].detach().cpu().numpy(),
                official_success=stable_delivery_steps >= required_stable_steps,
            )
            if decision.phase is not last_phase:
                print(
                    f"[EXECUTIVE] target={active_task.object_name} phase={decision.phase.value} "
                    f"active_hand={decision.active_hand} transitions={decision.transitions or ('initial',)}",
                    flush=True,
                )
                action_chunk = None
                control_index = 0
                last_phase = decision.phase
                show_status(
                    f"RUNNING — target={active_task.object_name}; phase={decision.phase.value}; "
                    f"t={mission_step * env.step_dt:.1f}s"
                )

            if decision.phase.value == "complete":
                completed_target = active_task.object_name
                print(
                    f"[LAB SUCCESS] {completed_target} delivered and stable; lab remains open",
                    flush=True,
                )
                clear_mission()
                show_status(f"SUCCESS — {completed_target} delivered")
                requested_action = neutral_action
            elif decision.walking:
                # Walking phases are deterministic; do not ask the VLA for arm
                # motion that the supervisor will intentionally discard.
                requested_action = supervisor.supervise_action(
                    neutral_action,
                    decision,
                    current_left_wrist=current_left,
                    current_right_wrist=current_right,
                    current_hands=current_hands,
                    neutral_left_wrist=neutral_left_wrist,
                    neutral_right_wrist=neutral_right_wrist,
                    neutral_hands=neutral_hands,
                )
            else:
                if action_chunk is None or control_index >= math.ceil(
                    executed_horizon * policy_period / float(env.step_dt)
                ):
                    started = time.perf_counter()
                    raw_action, info = client.get_action(_build_policy_input(env, history, decision.prompt))
                    action_chunk = real_g1_actions_to_isaac(raw_action)
                    executed_horizon = min(args.execution_horizon, action_chunk.shape[0])
                    control_index = 0
                    inference_count += 1
                    latency = info.get("inference_seconds") if isinstance(info, dict) else None
                    measured = time.perf_counter() - started
                    print(
                        f"[POLICY {inference_count:03d}] target={active_task.object_name} "
                        f"execute={executed_horizon} latency="
                        f"{latency if latency is not None else measured:.2f}s",
                        flush=True,
                    )
                action_index = min(
                    int(control_index * float(env.step_dt) / policy_period), executed_horizon - 1
                )
                requested_action = supervisor.supervise_action(
                    action_chunk[action_index],
                    decision,
                    current_left_wrist=current_left,
                    current_right_wrist=current_right,
                    current_hands=current_hands,
                    neutral_left_wrist=neutral_left_wrist,
                    neutral_right_wrist=neutral_right_wrist,
                    neutral_hands=neutral_hands,
                )
                control_index += 1

        safe_relative_action, safety_report = safety_filter.filter(
            requested_action,
            current_left_wrist=current_left,
            current_right_wrist=current_right,
            current_hands=current_hands,
        )
        safety_clamps.update(safety_report.clipped_fields)
        obs, _, _, _, _ = env.step(_world_action(env, safe_relative_action))
        image = obs["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
        history.append(image.copy())

        fallen, pelvis_z, tilt_deg = _has_fallen(env)
        if fallen:
            restore_lab(f"automatic fall recovery (pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg)")
            continue
        if active_task is not None:
            mission_step += 1
            if mission_step >= max_mission_steps:
                failed_target = active_task.object_name
                failed_phase = supervisor.phase.value
                clear_mission()
                print(
                    f"[LAB TIMEOUT] target={failed_target} phase={failed_phase}; "
                    f"no physical success after {args.max_seconds:.1f}s; robot returns to idle",
                    flush=True,
                )
                show_status(f"TIMEOUT — target={failed_target}; phase={failed_phase}")

    print("[LAB CLOSED] Isaac window was closed", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    # Pink IK converts articulation tensors to NumPy internally; rollout is
    # inference-only and must never build autograd graphs around those tensors.
    torch.set_grad_enabled(False)
    initial_instruction = args.instruction or args.object
    if not args.interactive and initial_instruction is None:
        initial_instruction = "Pick up the red apple and place it on the plate"
    task = (
        TASK_BY_OBJECT[args.object]
        if args.object
        else resolve_fruit_task(initial_instruction)
        if initial_instruction
        else None
    )
    if args.max_seconds <= 0 or args.policy_hz <= 0:
        raise ValueError("--max-seconds and --policy-hz must be positive")
    if args.execution_horizon <= 0 or args.scene_smoke_steps <= 0:
        raise ValueError("--execution-horizon and --scene-smoke-steps must be positive")
    if args.warmup_seconds < 0:
        raise ValueError("--warmup-seconds must be non-negative")
    if args.interactive and args.scene_smoke:
        raise ValueError("--interactive and --scene-smoke cannot be combined")
    if not args.initial_state_dataset.is_file():
        raise FileNotFoundError(
            f"missing Isaac initial-state dataset: {args.initial_state_dataset}; "
            "run `make download-locomanip-dataset`"
        )

    if not args.scene_smoke:
        probe = PolicyClient(args.policy_host, args.policy_port, timeout_ms=2_000)
        try:
            if not probe.ping():
                raise RuntimeError(
                    f"GR00T N1.7 server is not ready at {args.policy_host}:{args.policy_port}; "
                    "run `make serve-multitask` first"
                )
        finally:
            probe.close()

    if initial_instruction:
        print(f"[COMMAND] {initial_instruction}", flush=True)
    else:
        print("[COMMAND] persistent lab starts idle and waits for terminal input", flush=True)
    if task is not None:
        print(f"[TARGET] {task.object_name}; one REAL_G1 checkpoint sees all four fruit", flush=True)
    print(
        "[PIPELINE] RGB+proprio+language -> GR00T N1.7 -> Pink IK + AGILE locomotion -> Isaac Sim",
        flush=True,
    )

    app_launcher = AppLauncher(args, enable_cameras=True)
    simulation_app = app_launcher.app
    # Isaac Sim 6 fast shutdown can terminate the interpreter from close().
    # Preserve failure by default and only opt into exit code 0 after a real
    # smoke/success predicate passes.
    exit_code = 1
    env = None
    dataset = None
    client = None
    try:
        import gymnasium as gym
        import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
        from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env_cfg.episode_length_s = 24.0 * 60.0 * 60.0 if args.interactive else args.max_seconds
        env_cfg.high_res_video = True
        env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        _strengthen_neutral_posture(env_cfg)
        if args.interactive:
            # The project executive owns per-command completion and recovery.
            # Disabling ManagerBasedRLEnv terminations prevents an invisible
            # auto-reset from ending/restarting the lab between user commands.
            for term_name in ("time_out", "object_dropping", "object_too_far", "success"):
                if hasattr(env_cfg.terminations, term_name):
                    setattr(env_cfg.terminations, term_name, None)
        scene_anchor = task.object_name if task is not None else "apple"
        entity_by_object = _install_four_fruit_scene(env_cfg, scene_anchor, persistent=args.interactive)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        print("[INIT] Isaac environment constructed", flush=True)

        dataset = HDF5DatasetFileHandler()
        dataset.open(str(args.initial_state_dataset))
        print(f"[INIT] Initial-state dataset opened: {args.initial_state_dataset.name}", flush=True)
        if args.demo not in dataset.get_episode_names():
            raise ValueError(f"unknown initial-state demo: {args.demo}")
        episode = dataset.load_episode(args.demo, env.device)
        print(f"[INIT] Loaded {args.demo}", flush=True)
        recorded_state = episode.get_initial_state()
        if recorded_state is None:
            raise ValueError(f"{args.demo} does not contain an initial Isaac scene state")
        initial_scene_state = _merge_scene_state(env, recorded_state)
        env.reset_to(
            state=initial_scene_state,
            env_ids=torch.tensor([0], device=env.device),
            is_relative=False,
        )
        print("[INIT] Robot and merged scene state reset", flush=True)
        # Validate every exact state joint before any learned action reaches the controller.
        robot = env.scene["robot"]
        groups = joint_groups(robot.data.joint_names, robot.data.joint_pos.torch.detach().cpu().numpy())
        assert {name: values.shape[1] for name, values in groups.items()} == {
            name: len(joints) for name, joints in REAL_G1_STATE_JOINTS.items()
        }
        print(
            f"[ISAAC READY] action={env.action_space.shape[-1]}D camera="
            f"{tuple(env.obs_buf['policy']['robot_pov_cam'].shape)} objects={sorted(entity_by_object)}",
            flush=True,
        )
        if args.capture_path is not None:
            from PIL import Image

            frame = env.obs_buf["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
            args.capture_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(args.capture_path)
            print(f"[CAPTURE] {args.capture_path}", flush=True)

        history_steps = max(1, round(1.0 / float(env.step_dt)))
        history: deque[np.ndarray] = deque(maxlen=history_steps + 1)
        first_image = env.obs_buf["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
        history.append(first_image.copy())

        if args.scene_smoke:
            smoke_input = _build_policy_input(env, history, initial_instruction)
            assert smoke_input["video.ego_view"].shape == (1, 2, 540, 960, 3)
            print(
                f"[SMOKE] REAL_G1 contract validated; holding balance for {args.scene_smoke_steps} steps",
                flush=True,
            )
            for smoke_step in range(args.scene_smoke_steps):
                try:
                    obs, _, _, _, _ = env.step(_safe_real_g1_action(env))
                except BaseException as exc:
                    print(f"[SMOKE ERROR] {type(exc).__name__}: {exc!r}", flush=True)
                    raise
                fallen, pelvis_z, tilt_deg = _has_fallen(env)
                if fallen:
                    raise RuntimeError(
                        f"balance smoke failed at step {smoke_step}: "
                        f"pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg"
                    )
                image = obs["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
                history.append(image.copy())
            pelvis_z, tilt_deg = _stability_status(env)
            print(
                f"[SMOKE PASS] four fruit, REAL_G1, Pink IK and AGILE stable for "
                f"{args.scene_smoke_steps} steps; pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg",
                flush=True,
            )
            exit_code = 0
            return 0

        destination_pose = env.get_end_fixture().get_pose().clone()
        warmup_steps = round(args.warmup_seconds / float(env.step_dt))
        if warmup_steps:
            print(
                f"[WARMUP] Holding nominal posture for {warmup_steps} steps ({args.warmup_seconds:.2f}s)",
                flush=True,
            )
        for warmup_step in range(warmup_steps):
            obs, _, _, _, _ = env.step(_safe_real_g1_action(env))
            fallen, pelvis_z, tilt_deg = _has_fallen(env)
            if fallen:
                print(
                    f"[SAFETY STOP] warm-up instability at step {warmup_step}: "
                    f"pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg",
                    flush=True,
                )
                return 1
            image = obs["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
            history.append(image.copy())
        pelvis_z, tilt_deg = _stability_status(env)
        print(f"[BALANCE READY] pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg", flush=True)
        min_pelvis_z = pelvis_z
        max_tilt_deg = tilt_deg

        neutral_left_t, neutral_right_t = _base_relative_wrist_poses(env)
        neutral_left = neutral_left_t[0].detach().cpu().numpy().copy()
        neutral_right = neutral_right_t[0].detach().cpu().numpy().copy()
        neutral_hands = _pink_hand_positions(env).copy()
        print(
            "[POSTURE READY] warm-up pose captured as neutral; "
            f"left_xyz={np.round(neutral_left[:3], 3).tolist()} "
            f"right_xyz={np.round(neutral_right[:3], 3).tolist()} "
            f"waist={_max_abs_waist_deg(env):.1f}deg",
            flush=True,
        )
        max_left_wrist_z = float(neutral_left[2])
        max_right_wrist_z = float(neutral_right[2])
        max_waist_deg = _max_abs_waist_deg(env)

        client = PolicyClient(args.policy_host, args.policy_port, timeout_ms=args.policy_timeout_ms)
        client.reset({"seed": args.seed})
        if args.interactive:
            result = _run_persistent_lab(
                env=env,
                simulation_app=simulation_app,
                client=client,
                history=history,
                entity_by_object=entity_by_object,
                destination_pose=destination_pose,
                initial_scene_state=initial_scene_state,
                initial_instruction=initial_instruction,
                args=args,
                neutral_left_wrist=neutral_left,
                neutral_right_wrist=neutral_right,
                neutral_hands=neutral_hands,
            )
            exit_code = result
            return result

        assert task is not None
        target_entity = entity_by_object[task.object_name]
        initial_object_height = float(env.scene[target_entity].data.root_pose_w.torch[0, 2])
        object_description = task.prompt.removeprefix("Pick up the ").split(" and place", 1)[0]
        supervisor = SafeFetchSupervisor(
            FetchMission(object_description, "pickup table", "plate"),
            initial_object_height_m=initial_object_height,
        )
        safety_filter = IsaacActionSafetyFilter(
            max_wrist_step_m=args.max_wrist_step_m,
            max_wrist_rotation_deg=args.max_wrist_rotation_deg,
        )
        safety_clamps: Counter[str] = Counter()
        max_steps = round(args.max_seconds / float(env.step_dt))
        action_chunk: np.ndarray | None = None
        control_index = 0
        executed_horizon = 0
        inference_count = 0
        policy_period = 1.0 / args.policy_hz
        last_phase = None

        for step in range(max_steps):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                break
            official_success = bool(env.termination_manager.get_term("success")[0])
            decision = supervisor.observe(
                robot_pose=env.get_base().get_pose()[0].detach().cpu().numpy(),
                object_pose=env.scene[target_entity].data.root_pose_w.torch[0].detach().cpu().numpy(),
                destination_pose=destination_pose[0].detach().cpu().numpy(),
                official_success=official_success,
            )
            if decision.phase is not last_phase:
                print(
                    f"[EXECUTIVE] phase={decision.phase.value} active_hand={decision.active_hand} "
                    f"transitions={decision.transitions or ('initial',)} prompt={decision.prompt!r}",
                    flush=True,
                )
                action_chunk = None
                control_index = 0
                last_phase = decision.phase
            if action_chunk is None or control_index >= math.ceil(
                executed_horizon * policy_period / float(env.step_dt)
            ):
                inference_started = time.perf_counter()
                raw_action, info = client.get_action(_build_policy_input(env, history, decision.prompt))
                action_chunk = real_g1_actions_to_isaac(raw_action)
                executed_horizon = min(args.execution_horizon, action_chunk.shape[0])
                control_index = 0
                inference_count += 1
                latency = info.get("inference_seconds") if isinstance(info, dict) else None
                measured = time.perf_counter() - inference_started
                print(
                    f"[POLICY {inference_count:03d}] horizon={action_chunk.shape[0]} "
                    f"execute={executed_horizon} latency={latency if latency is not None else measured:.2f}s "
                    f"{_target_status(env, entity_by_object[task.object_name], destination_pose)}",
                    flush=True,
                )

            action_index = min(int(control_index * float(env.step_dt) / policy_period), executed_horizon - 1)
            left_pose, right_pose = _base_relative_wrist_poses(env)
            current_hands = _pink_hand_positions(env)
            max_left_wrist_z = max(max_left_wrist_z, float(left_pose[0, 2]))
            max_right_wrist_z = max(max_right_wrist_z, float(right_pose[0, 2]))
            max_waist_deg = max(max_waist_deg, _max_abs_waist_deg(env))
            supervised_action = supervisor.supervise_action(
                action_chunk[action_index],
                decision,
                current_left_wrist=left_pose[0].detach().cpu().numpy(),
                current_right_wrist=right_pose[0].detach().cpu().numpy(),
                current_hands=current_hands,
                neutral_left_wrist=neutral_left,
                neutral_right_wrist=neutral_right,
                neutral_hands=neutral_hands,
            )
            safe_relative_action, safety_report = safety_filter.filter(
                supervised_action,
                current_left_wrist=left_pose[0].detach().cpu().numpy(),
                current_right_wrist=right_pose[0].detach().cpu().numpy(),
                current_hands=current_hands,
            )
            safety_clamps.update(safety_report.clipped_fields)
            if safety_report.was_clipped and step % 25 == 0:
                print(
                    f"[SAFETY] clipped={','.join(safety_report.clipped_fields)} "
                    f"wrist_step={safety_report.requested_wrist_step_m}->"
                    f"{safety_report.applied_wrist_step_m} "
                    f"height={safety_report.applied_base_height_m:.3f}m",
                    flush=True,
                )
            action = _world_action(env, safe_relative_action)
            obs, _, terminated, timed_out, _ = env.step(action)
            control_index += 1
            image = obs["policy"]["robot_pov_cam"][0].detach().cpu().numpy().astype(np.uint8)
            history.append(image.copy())

            fallen, pelvis_z, tilt_deg = _has_fallen(env)
            min_pelvis_z = min(min_pelvis_z, pelvis_z)
            max_tilt_deg = max(max_tilt_deg, tilt_deg)
            if fallen:
                print(
                    f"[SAFETY STOP] fall detected at {step * env.step_dt:.2f}s: "
                    f"pelvis_z={pelvis_z:.3f}m tilt={tilt_deg:.1f}deg; "
                    f"wrist_z_max=({max_left_wrist_z:.3f},{max_right_wrist_z:.3f})m "
                    f"waist_max={max_waist_deg:.1f}deg clamps={dict(safety_clamps)}",
                    flush=True,
                )
                return 1

            official_success = bool(env.termination_manager.get_term("success")[0])
            if official_success:
                print(
                    f"[SUCCESS] {task.object_name} placed on the destination table at "
                    f"{step * env.step_dt:.1f}s simulated time; "
                    f"min_pelvis_z={min_pelvis_z:.3f}m max_tilt={max_tilt_deg:.1f}deg "
                    f"wrist_z_max=({max_left_wrist_z:.3f},{max_right_wrist_z:.3f})m "
                    f"waist_max={max_waist_deg:.1f}deg",
                    flush=True,
                )
                exit_code = 0
                return 0
            if bool(terminated[0]):
                active = [
                    name for name, value in env.termination_manager.get_active_iterable_terms(0) if value[0]
                ]
                print(f"[FAILED] Isaac termination: {active}", flush=True)
                return 1
            if bool(timed_out[0]):
                break

        print(
            f"[FAILED] no physical success in {args.max_seconds:.1f}s; "
            f"phase={supervisor.phase.value}; min_pelvis_z={min_pelvis_z:.3f}m "
            f"max_tilt={max_tilt_deg:.1f}deg; "
            f"wrist_z_max=({max_left_wrist_z:.3f},{max_right_wrist_z:.3f})m "
            f"waist_max={max_waist_deg:.1f}deg safety_clamps={dict(safety_clamps)}",
            flush=True,
        )
        return 1
    finally:
        if client is not None:
            client.close()
        if dataset is not None:
            dataset.close()
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

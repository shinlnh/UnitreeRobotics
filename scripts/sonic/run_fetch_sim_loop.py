#!/usr/bin/env python3
"""Run the SONIC MuJoCo bridge in the project fetch-object scene."""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import MethodType
from typing import Any

import mujoco
import numpy as np
import tyro
import zmq
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCENE = PROJECT_ROOT / "assets/sonic/g1_fetch_scene.xml"
SONIC_ROOT = PROJECT_ROOT / ".deps/GR00T-WholeBodyControl"
G1_MODEL = SONIC_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
G1_MESHES = G1_MODEL.parent / "meshes"
TEMPLATE_INCLUDE = (
    "../../.deps/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)

OBJECT_PROFILES = {
    "apple": ("sphere", "0.055", "0.85 0.05 0.04 1", 0.055, 0.14),
    "red_soda_can": ("cylinder", "0.035 0.075", "0.85 0.04 0.03 1", 0.075, 0.18),
    "blue_plastic_bottle": ("capsule", "0.032 0.065", "0.05 0.25 0.90 1", 0.097, 0.12),
    "green_foam_block": ("box", "0.055 0.042 0.042", "0.08 0.75 0.16 1", 0.042, 0.08),
    "yellow_plastic_cup": ("cylinder", "0.045 0.065", "0.95 0.78 0.04 1", 0.065, 0.09),
    "small_plush_toy": ("ellipsoid", "0.060 0.050 0.070", "0.75 0.30 0.65 1", 0.070, 0.10),
    "cardboard_snack_box": ("box", "0.045 0.028 0.080", "0.62 0.38 0.14 1", 0.080, 0.16),
}
SOURCE_PROFILES = {
    "table": (0.62, "0.34 0.28 0.035", "0 0 -0.325", "0.08 0.08 0.29"),
    "low_shelf": (0.42, "0.30 0.24 0.030", "0 0 -0.225", "0.08 0.08 0.19"),
    "pickup_tray": (0.20, "0.24 0.19 0.020", "0 0 -0.105", "0.10 0.10 0.085"),
}
DESTINATION_PROFILES = {"green_plate", "delivery_bin", "return_tray", "marked_table_area"}


@dataclass
class FetchSimConfig(SimLoopConfig):
    object_profile: str = "apple"
    """Object geometry profile in the fetch curriculum."""

    source_profile: str = "table"
    """Source fixture profile: table, low_shelf, or pickup_tray."""

    destination_profile: str = "green_plate"
    """Destination fixture profile."""

    randomize_scene: bool = True
    """Randomize object pose, mass, friction, color, and light after each reset."""

    scene_seed: int = 0
    """Randomization seed; zero draws a fresh non-deterministic seed."""

    telemetry_port: int = 5590
    """ZMQ PUB port for physics-grounded oracle and success telemetry."""

    telemetry_hz: float = 50.0
    """Physics telemetry publication rate."""


def _append_box(body: ET.Element, name: str, pos: str, size: str, rgba: str) -> None:
    ET.SubElement(
        body,
        "geom",
        {"name": name, "type": "box", "pos": pos, "size": size, "rgba": rgba},
    )


def materialize_scene(
    directory: Path,
    *,
    object_profile: str = "apple",
    source_profile: str = "table",
    destination_profile: str = "green_plate",
) -> Path:
    """Make MuJoCo asset paths absolute without modifying the pinned SONIC checkout."""
    if object_profile not in OBJECT_PROFILES:
        raise ValueError(f"unknown object profile {object_profile!r}; choose {sorted(OBJECT_PROFILES)}")
    if source_profile not in SOURCE_PROFILES:
        raise ValueError(f"unknown source profile {source_profile!r}; choose {sorted(SOURCE_PROFILES)}")
    if destination_profile not in DESTINATION_PROFILES:
        raise ValueError(
            f"unknown destination profile {destination_profile!r}; choose {sorted(DESTINATION_PROFILES)}"
        )
    robot_xml = G1_MODEL.read_text(encoding="utf-8")
    robot_xml = robot_xml.replace('meshdir="meshes"', f'meshdir="{G1_MESHES}"', 1)
    robot_copy = directory / "g1_29dof_with_hand.xml"
    robot_copy.write_text(robot_xml, encoding="utf-8")

    scene_xml = FETCH_SCENE.read_text(encoding="utf-8")
    scene_xml = scene_xml.replace(TEMPLATE_INCLUDE, str(robot_copy), 1)
    root = ET.fromstring(scene_xml)

    source = root.find(".//body[@name='source_fixture']")
    source_top = root.find(".//geom[@name='source_table_top']")
    source_leg = root.find(".//geom[@name='source_table_leg']")
    fetch_object = root.find(".//body[@name='fetch_object']")
    object_geom = root.find(".//geom[@name='fetch_object_collision']")
    destination = root.find(".//body[@name='destination_table']")
    destination_goal = root.find(".//geom[@name='plate']")
    success_zone = root.find(".//site[@name='delivery_success_zone']")
    required = (source, source_top, source_leg, fetch_object, object_geom, destination, destination_goal)
    if any(element is None for element in required):
        raise ValueError("fetch scene template is missing a required named element")

    source_z, top_size, leg_pos, leg_size = SOURCE_PROFILES[source_profile]
    source.set("pos", f"0.95 0.36 {source_z}")
    source_top.set("size", top_size)
    source_leg.set("pos", leg_pos)
    source_leg.set("size", leg_size)

    geom_type, geom_size, rgba, half_height, mass = OBJECT_PROFILES[object_profile]
    object_geom.set("type", geom_type)
    object_geom.set("size", geom_size)
    object_geom.set("rgba", rgba)
    object_geom.set("mass", str(mass))
    object_geom.attrib.pop("material", None)
    source_half_thickness = float(top_size.split()[2])
    fetch_object.set("pos", f"0.82 0.36 {source_z + source_half_thickness + half_height}")
    stem = fetch_object.find("./geom[@name='apple_stem']")
    if stem is not None and object_profile != "apple":
        fetch_object.remove(stem)

    destination_goal.attrib.clear()
    destination_goal.set("name", "destination_goal")
    destination_goal.set("friction", "1.1 0.01 0.001")
    goal_rgba = "0.10 0.80 0.20 1"
    if destination_profile == "green_plate":
        destination_goal.attrib.update(
            {"type": "cylinder", "pos": "0 0 0.047", "size": "0.15 0.012", "rgba": goal_rgba}
        )
    elif destination_profile == "marked_table_area":
        destination_goal.attrib.update(
            {"type": "box", "pos": "0 0 0.041", "size": "0.16 0.13 0.006", "rgba": "0.1 0.9 0.2 0.55"}
        )
    else:
        destination_goal.attrib.update(
            {"type": "box", "pos": "0 0 0.050", "size": "0.16 0.13 0.015", "rgba": goal_rgba}
        )
        wall_height = "0.06" if destination_profile == "delivery_bin" else "0.025"
        _append_box(destination, "goal_wall_front", "0.17 0 0.10", f"0.01 0.14 {wall_height}", goal_rgba)
        _append_box(destination, "goal_wall_back", "-0.17 0 0.10", f"0.01 0.14 {wall_height}", goal_rgba)
        _append_box(destination, "goal_wall_left", "0 0.14 0.10", f"0.16 0.01 {wall_height}", goal_rgba)
        _append_box(destination, "goal_wall_right", "0 -0.14 0.10", f"0.16 0.01 {wall_height}", goal_rgba)
    if success_zone is not None:
        success_zone.set("pos", "0 0 0.085")
        success_zone.set("size", "0.12 0.008")

    scene_copy = directory / "g1_fetch_scene.xml"
    ET.ElementTree(root).write(scene_copy, encoding="unicode")
    return scene_copy


def install_reset_randomization(simulator: Any, *, seed: int, enabled: bool) -> None:
    """Randomize the object and light whenever the fetch scene resets."""
    environment = simulator.sim_env
    model = environment.mj_model
    data = environment.mj_data
    object_joint = model.joint("fetch_object_freejoint").id
    qpos_address = int(model.jnt_qposadr[object_joint])
    object_body = model.body("fetch_object").id
    object_geom = model.geom("fetch_object_collision").id
    base_qpos = model.qpos0[qpos_address : qpos_address + 7].copy()
    base_mass = float(model.body_mass[object_body])
    base_rgba = model.geom_rgba[object_geom].copy()
    rng = np.random.default_rng(None if seed == 0 else seed)

    def randomized_reset(_environment: Any) -> None:
        mujoco.mj_resetData(model, data)
        if enabled:
            data.qpos[qpos_address : qpos_address + 3] = base_qpos[:3] + np.array(
                [rng.uniform(-0.08, 0.08), rng.uniform(-0.07, 0.07), 0.0]
            )
            yaw = rng.uniform(-np.pi, np.pi)
            data.qpos[qpos_address + 3 : qpos_address + 7] = np.array(
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            )
            model.body_mass[object_body] = base_mass * rng.uniform(0.8, 1.2)
            model.geom_friction[object_geom] = np.array(
                [rng.uniform(0.75, 1.35), rng.uniform(0.008, 0.025), rng.uniform(0.001, 0.004)]
            )
            model.geom_rgba[object_geom, :3] = np.clip(
                base_rgba[:3] * rng.uniform(0.82, 1.18, size=3), 0.0, 1.0
            )
            if model.nlight:
                model.light_pos[0, :2] = rng.uniform(-0.8, 0.8, size=2)
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)

    environment.reset = MethodType(randomized_reset, environment)
    environment.reset()


def install_keyboard_reset(simulator: Any, *, host: str = "localhost", port: int = 5580) -> None:
    """Apply an ``r`` reset at the simulation-step boundary."""
    reset_requested = Event()
    environment = simulator.sim_env
    original_step = environment.sim_step

    def step_with_reset() -> None:
        if reset_requested.is_set():
            environment.reset()
            reset_requested.clear()
            print("[FetchScene] Reset and randomized scene.")
        original_step()

    environment.sim_step = step_with_reset

    def listen() -> None:
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect(f"tcp://{host}:{port}")
        subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        try:
            while True:
                if subscriber.recv_string() == "r":
                    reset_requested.set()
        finally:
            subscriber.close(linger=0)
            context.term()

    Thread(target=listen, name="fetch-reset-listener", daemon=True).start()


def install_fetch_telemetry(simulator: Any, *, port: int, frequency: float) -> None:
    """Publish scene state and physical success evidence at the simulation boundary."""
    if frequency <= 0.0:
        raise ValueError("telemetry frequency must be greater than zero")
    environment = simulator.sim_env
    model = environment.mj_model
    data = environment.mj_data
    original_step = environment.sim_step

    pelvis = model.body("pelvis").id
    fetch_object = model.body("fetch_object").id
    object_geom = model.geom("fetch_object_collision").id
    source_geom = model.geom("source_table_top").id
    destination_geom = model.geom("destination_goal").id
    destination_site = model.site("delivery_success_zone").id
    left_wrist = model.body("left_wrist_yaw_link").id
    right_wrist = model.body("right_wrist_yaw_link").id

    def is_descendant(body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 2)
    publisher.bind(f"tcp://*:{port}")
    publish_period = 1.0 / frequency
    next_publish_time = 0.0
    last_sim_time = float("-inf")

    def vertical_half_extent(geom_id: int) -> float:
        geom_type = int(model.geom_type[geom_id])
        if geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        }:
            return float(model.geom_size[geom_id, 1])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(model.geom_size[geom_id, 0])
        return float(model.geom_size[geom_id, 2])

    def step_with_telemetry() -> None:
        nonlocal last_sim_time, next_publish_time
        original_step()
        if data.time < last_sim_time:
            next_publish_time = 0.0
        last_sim_time = float(data.time)
        if data.time + 1e-9 < next_publish_time:
            return
        next_publish_time = data.time + publish_period

        left_contacts = 0
        right_contacts = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == object_geom:
                other_body = int(model.geom_bodyid[geom2])
            elif geom2 == object_geom:
                other_body = int(model.geom_bodyid[geom1])
            else:
                continue
            left_contacts += int(is_descendant(other_body, left_wrist))
            right_contacts += int(is_descendant(other_body, right_wrist))

        object_position = data.xpos[fetch_object].copy()
        object_velocity = data.cvel[fetch_object, 3:6].copy()
        goal_position = data.site_xpos[destination_site].copy()
        source_surface_z = float(data.geom_xpos[source_geom, 2] + vertical_half_extent(source_geom))
        goal_surface_z = float(data.geom_xpos[destination_geom, 2] + vertical_half_extent(destination_geom))
        goal_distance_xy = float(np.linalg.norm(object_position[:2] - goal_position[:2]))
        object_speed = float(np.linalg.norm(object_velocity))
        lifted = bool(object_position[2] > source_surface_z + 0.12)
        in_goal = bool(
            goal_distance_xy <= float(model.site_size[destination_site, 0])
            and goal_surface_z - 0.02 <= object_position[2] <= goal_surface_z + 0.30
        )
        released = left_contacts + right_contacts == 0
        delivered = bool(in_goal and released and object_speed < 0.20)

        payload = {
            "sim_time": float(data.time),
            "pelvis_position": data.xpos[pelvis].tolist(),
            "pelvis_quaternion_wxyz": data.xquat[pelvis].tolist(),
            "pelvis_up_z": float(data.xmat[pelvis].reshape(3, 3)[2, 2]),
            "left_wrist_position": data.xpos[left_wrist].tolist(),
            "right_wrist_position": data.xpos[right_wrist].tolist(),
            "object_position": object_position.tolist(),
            "object_quaternion_wxyz": data.xquat[fetch_object].tolist(),
            "object_speed": object_speed,
            "source_surface_z": source_surface_z,
            "goal_position": goal_position.tolist(),
            "goal_surface_z": goal_surface_z,
            "goal_distance_xy": goal_distance_xy,
            "left_object_contacts": left_contacts,
            "right_object_contacts": right_contacts,
            "lifted": lifted,
            "in_goal": in_goal,
            "released": released,
            "delivered": delivered,
            "robot_upright": bool(data.xpos[pelvis, 2] > 0.55 and data.xmat[pelvis, 8] > 0.55),
        }
        with suppress(zmq.Again):
            publisher.send_string(json.dumps(payload, separators=(",", ":")), flags=zmq.NOBLOCK)

    environment.sim_step = step_with_telemetry
    print(f"[FetchTelemetry] Publishing physics evidence on tcp://*:{port} at {frequency:g} Hz")


def main(config: FetchSimConfig) -> None:
    if config.enable_image_publish and not config.enable_offscreen:
        raise ValueError("--enable-image-publish requires --enable-offscreen")
    if not FETCH_SCENE.is_file():
        raise FileNotFoundError(FETCH_SCENE)

    with tempfile.TemporaryDirectory(prefix="unitree-fetch-scene-") as temporary_directory:
        scene = materialize_scene(
            Path(temporary_directory),
            object_profile=config.object_profile,
            source_profile=config.source_profile,
            destination_profile=config.destination_profile,
        )
        wbc_config: dict[str, Any] = config.load_wbc_yaml()
        wbc_config["ROBOT_SCENE"] = str(scene)
        wbc_config["ENV_NAME"] = "g1_fetch_apple_to_plate"
        simulator = SimulatorFactory.create_simulator(
            config=wbc_config,
            env_name="default",
            onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
            offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
            enable_image_publish=config.enable_image_publish,
            robot_model=instantiate_g1_robot_model(),
        )
        install_reset_randomization(simulator, seed=config.scene_seed, enabled=config.randomize_scene)
        install_keyboard_reset(simulator)
        install_fetch_telemetry(
            simulator,
            port=config.telemetry_port,
            frequency=config.telemetry_hz,
        )
        SimulatorFactory.start_simulator(
            simulator,
            as_thread=False,
            enable_image_publish=config.enable_image_publish,
            mp_start_method=config.mp_start_method,
            camera_port=config.camera_port,
        )


if __name__ == "__main__":
    main(tyro.cli(FetchSimConfig))

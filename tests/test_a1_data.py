import xml.etree.ElementTree as ET

import numpy as np

from unitree_gr00t.a1_data import is_noop, model_action, mujoco_autolimits_xml


def test_robocerebra_noop_rule_preserves_gripper_transitions() -> None:
    previous = np.zeros(7)
    assert is_noop(np.zeros(7), previous, np)
    gripper_transition = np.zeros(7)
    gripper_transition[-1] = 1
    assert not is_noop(gripper_transition, previous, np)


def test_model_action_maps_libero_gripper_sign() -> None:
    closing = model_action(np.array([0, 0, 0, 0, 0, 0, -1]), np)
    opening = model_action(np.array([0, 0, 0, 0, 0, 0, 1]), np)
    assert closing[-1] == 1
    assert opening[-1] == 0


def test_mujoco_autolimits_compatibility_for_legacy_ranged_joint() -> None:
    legacy = (
        '<mujoco><compiler angle="radian"/><worldbody><joint range="0 1"/></worldbody></mujoco>'
    )
    normalized = ET.fromstring(mujoco_autolimits_xml(legacy))
    assert normalized.find("compiler").get("autolimits") == "true"

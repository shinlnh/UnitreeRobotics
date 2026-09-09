import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_frontier_prepare import (
    controllable_goal_states,
    first_false_to_true,
    goal_matches_instruction,
    portable_model_xml,
    valid_anchor_frames,
)


def test_controllable_goals_exclude_fixture_and_init_invariants() -> None:
    parsed = {
        "fixtures": {"rack": ["rack_1"]},
        "goal_state": [
            ["in", "bowl_1", "rack_1_top_region"],
            ["on", "rack_1", "table_rack_init_region"],
            ["on", "mug_1", "table_mug_init_region"],
        ],
    }
    goals = controllable_goal_states(parsed, ["bowl_1", "mug_1"])
    assert goals == (("in", "bowl_1", "rack_1_top_region"),)


def test_goal_language_is_association_not_outcome_label() -> None:
    goal = ("in", "red_coffee_mug_1", "rack_1_top_region")
    assert goal_matches_instruction(goal, "Place the red coffee mug on the rack")
    assert not goal_matches_instruction(goal, "Place the plate on the rack")


def test_frontier_scan_refines_first_exact_physical_hit() -> None:
    values = np.zeros(40, dtype=np.bool_)
    values[23:] = True
    assert (
        first_false_to_true(start=3, end=35, stride=8, evaluate=lambda index: bool(values[index]))
        == 23
    )


def test_frontier_rejects_already_achieved_goal_and_invalid_interval() -> None:
    assert first_false_to_true(start=2, end=8, stride=3, evaluate=lambda _: True) is None
    with pytest.raises(OursContractError, match="interval"):
        first_false_to_true(start=2, end=2, stride=1, evaluate=lambda _: False)


def test_anchor_frames_are_deduplicated_bounded_and_sorted() -> None:
    assert valid_anchor_frames(frontier=100, subgoal_start=70, offsets=(32, 16, 16, 8)) == (84, 92)


def test_demo_xml_assets_are_relocated_from_collection_machine(tmp_path) -> None:
    xml = """<mujoco><asset>
      <mesh name="robot" file="/old/env/site-packages/robosuite/models/r.stl"/>
      <texture name="scene" file="/Users/me/NEW_LIBERO/libero/libero/assets/t.png"/>
    </asset><worldbody><body><joint actuatorfrclimited="false" range="0 1"/></body></worldbody></mujoco>"""
    relocated = portable_model_xml(
        xml,
        libero_package_root=tmp_path / "LIBERO" / "libero" / "libero",
        robosuite_package_root=tmp_path / "robosuite",
    )
    assert str(tmp_path / "robosuite/models/r.stl") in relocated
    assert str(tmp_path / "LIBERO/libero/libero/assets/t.png") in relocated
    assert "actuatorfrclimited" not in relocated
    assert 'limited="true"' in relocated

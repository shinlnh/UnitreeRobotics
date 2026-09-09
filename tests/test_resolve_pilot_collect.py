import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_pilot_collect import (
    PILOT_MILESTONES,
    canonical_baseline_sources,
    generate_centered_recovery_programs,
    generate_recovery_programs,
    newly_reached_predicates,
    physical_reachability,
    program_crb_targets,
)


def test_recovery_programs_are_deterministic_bounded_and_ordered() -> None:
    kwargs = {
        "count": 3,
        "depth": 2,
        "code_dimension": 4,
        "prefix_length": 8,
        "minimum_angle": 0.05,
        "maximum_angle": 0.2,
        "seed": 41007,
        "np": np,
    }
    first = generate_recovery_programs(**kwargs)
    second = generate_recovery_programs(**kwargs)
    assert len(first) == 3
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.steering_codes, right.steering_codes)
        assert left.prefix_lengths == (8, 8)
        assert left.subgoal_offsets == (0, 0)
        norms = np.linalg.norm(left.steering_codes, axis=1)
        assert np.all(norms >= 0.05)
        assert np.all(norms <= 0.2)


def test_physical_reachability_never_uses_efficiency_costs() -> None:
    target = physical_reachability(
        predicates_before={"cup": [True, False], "plate": [False]},
        predicates_after={"cup": [True, True], "plate": [False]},
        completed_before=1,
        completed_after=2,
        final_success=False,
        np=np,
    )
    assert np.array_equal(target, np.asarray([1.0, 1.0, 0.0]))
    assert newly_reached_predicates(
        {"cup": [True, False], "plate": [False]},
        {"cup": [True, True], "plate": [False]},
    ) == ("cup:1",)


def test_program_crb_requires_every_step_to_beat_its_deletion_and_b() -> None:
    recovery = np.asarray([1.0, 1.0, 0.0])
    deletions = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    baselines = np.zeros((2, len(PILOT_MILESTONES)), dtype=np.float32)
    per_slot, program = program_crb_targets(recovery, deletions, baselines, np=np)
    assert np.array_equal(per_slot[0], np.asarray([1.0, 1.0, 0.0]))
    assert np.array_equal(per_slot[1], np.asarray([1.0, 0.0, 0.0]))
    assert np.array_equal(program, np.asarray([1.0, 0.0, 0.0]))


def test_invalid_program_geometry_is_rejected() -> None:
    with pytest.raises(OursContractError, match="configuration"):
        generate_recovery_programs(
            count=1,
            depth=2,
            code_dimension=4,
            prefix_length=8,
            minimum_angle=0.3,
            maximum_angle=0.2,
            seed=1,
            np=np,
        )
    with pytest.raises(OursContractError, match="offsets"):
        generate_recovery_programs(
            count=1,
            depth=2,
            code_dimension=4,
            prefix_length=8,
            minimum_angle=0.1,
            maximum_angle=0.2,
            seed=1,
            np=np,
            subgoal_offsets=(-1,),
        )


def test_canonical_baselines_reuse_identical_estimands() -> None:
    assert canonical_baseline_sources(1) == (("deletion", 0),)
    assert canonical_baseline_sources(2) == (
        ("shared_baseline", 0),
        ("deletion", 1),
    )
    assert canonical_baseline_sources(4) == (
        ("shared_baseline", 0),
        ("handoff", 1),
        ("handoff", 2),
        ("deletion", 3),
    )
    with pytest.raises(OursContractError, match="depth"):
        canonical_baseline_sources(0)


def test_centered_proposals_keep_the_registered_center_and_bounds() -> None:
    center = np.asarray([[0.1, -0.1, 0.05, -0.05]], dtype=np.float32)
    programs = generate_centered_recovery_programs(
        count=8,
        center_codes=center,
        standard_deviation=0.02,
        prefix_length=6,
        minimum_angle=0.01,
        maximum_angle=0.25,
        seed=71007,
        np=np,
    )
    assert np.allclose(programs[0].steering_codes, center)
    assert all(program.prefix_lengths == (6,) for program in programs)
    norms = np.asarray([np.linalg.norm(program.steering_codes[0]) for program in programs])
    assert np.all((norms >= 0.01) & (norms <= 0.25))


def test_centered_proposals_keep_exact_center_then_search_locally() -> None:
    center = np.asarray([[0.1, 0.0, 0.0, 0.0], [0.0, -0.1, 0.0, 0.0]])
    programs = generate_centered_recovery_programs(
        count=4,
        center_codes=center,
        standard_deviation=0.02,
        prefix_length=6,
        frozen_prefix=1,
        minimum_angle=0.01,
        maximum_angle=0.2,
        seed=17,
        np=np,
    )
    assert np.allclose(programs[0].steering_codes, center)
    assert programs[0].prefix_lengths == (6, 6)
    assert all(np.allclose(program.steering_codes[0], center[0]) for program in programs)
    assert any(not np.array_equal(program.steering_codes, center) for program in programs[1:])
    for program in programs:
        norms = np.linalg.norm(program.steering_codes, axis=1)
        assert np.all(norms >= 0.01)
        assert np.all(norms <= 0.2)

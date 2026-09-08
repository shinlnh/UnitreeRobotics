import numpy as np

from unitree_gr00t.ours_policy import SelectiveConsensusRecovery


def test_selective_consensus_reobserves_then_executes_medoid_nonstop_prefix() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=2,
        failure_threshold=0.5,
    )
    chunk_a = np.ones((16, 7), dtype=np.float32)
    chunk_b = np.full((16, 7), 2.0, dtype=np.float32)
    scores_a = np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32)
    scores_b = np.asarray([5.0, 4.0, 1.0] + [0.0] * 14, dtype=np.float32)
    valid = np.asarray([True, True, True] + [False] * 14)

    first = controller.decide(
        candidate=0,
        action_chunk=chunk_a,
        scores=scores_a,
        valid=valid,
        completion_probability=0.1,
        progress_probability=0.2,
        failure_probability=0.9,
        np=np,
    )
    assert first.suppress_stop_confirmation
    assert first.option == "REOBSERVE"

    second = controller.decide(
        candidate=0,
        action_chunk=chunk_b,
        scores=scores_b,
        valid=valid,
        completion_probability=0.2,
        progress_probability=0.3,
        failure_probability=0.9,
        np=np,
    )
    assert not second.suppress_stop_confirmation
    assert second.option == "CONSENSUS_PREFIX"
    assert second.candidate > 0
    assert second.hypothesis_count == 2


def test_low_failure_stop_uses_one_prefix_without_extra_policy_call() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=4,
        failure_threshold=0.5,
    )
    chunk = np.ones((16, 7), dtype=np.float32)
    scores = np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32)
    valid = np.asarray([True, True, True] + [False] * 14)
    directive = controller.decide(
        candidate=0,
        action_chunk=chunk,
        scores=scores,
        valid=valid,
        completion_probability=0.1,
        progress_probability=0.2,
        failure_probability=0.1,
        np=np,
    )
    assert directive.option == "CONSENSUS_PREFIX"
    assert directive.candidate == 2
    assert directive.hypothesis_count == 1
    assert not directive.recovery_triggered


def test_selective_consensus_accepts_action_and_high_confidence_stop() -> None:
    controller = SelectiveConsensusRecovery(completion_threshold=0.8, consensus_hypotheses=4)
    chunk = np.zeros((16, 7), dtype=np.float32)
    scores = np.zeros(17, dtype=np.float32)
    valid = np.ones(17, dtype=np.bool_)
    action = controller.decide(
        candidate=8,
        action_chunk=chunk,
        scores=scores,
        valid=valid,
        completion_probability=0.1,
        progress_probability=0.2,
        np=np,
    )
    stop = controller.decide(
        candidate=0,
        action_chunk=chunk,
        scores=scores,
        valid=valid,
        completion_probability=0.9,
        progress_probability=0.9,
        np=np,
    )
    assert action.option == "ACCEPT_B"
    assert stop.option == "ADVANCE"


def test_stagnation_boundary_advances_without_reading_outcome() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.99,
        consensus_hypotheses=4,
        force_boundary_steps=150,
    )
    directive = controller.decide(
        candidate=8,
        action_chunk=np.zeros((16, 7), dtype=np.float32),
        scores=np.zeros(17, dtype=np.float32),
        valid=np.ones(17, dtype=np.bool_),
        completion_probability=0.01,
        progress_probability=0.01,
        subgoal_elapsed_steps=150,
        np=np,
    )
    assert directive.candidate == 0
    assert directive.option == "ADVANCE"
    assert not directive.suppress_stop_confirmation

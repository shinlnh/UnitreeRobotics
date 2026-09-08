import numpy as np

from unitree_gr00t.ours_policy import SelectiveConsensusRecovery


def test_selective_consensus_reobserves_then_executes_medoid_nonstop_prefix() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=2,
        failure_threshold=0.5,
        consensus_cooldown_decisions=2,
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

    cooldown = controller.decide(
        candidate=0,
        action_chunk=chunk_a,
        scores=scores_a,
        valid=valid,
        completion_probability=0.1,
        progress_probability=0.2,
        failure_probability=0.9,
        np=np,
    )
    assert cooldown.hypothesis_count == 1
    assert not cooldown.recovery_triggered


def test_low_failure_stop_preserves_b_without_recovery() -> None:
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
    assert directive.option == "ACCEPT_B"
    assert directive.candidate == 0
    assert directive.hypothesis_count == 1
    assert not directive.recovery_triggered


def test_single_hypothesis_is_a_bounded_recovery_intervention() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        failure_threshold=0.5,
    )
    directive = controller.decide(
        candidate=0,
        action_chunk=np.ones((16, 7), dtype=np.float32),
        scores=np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        valid=np.asarray([True, True, True] + [False] * 14),
        completion_probability=0.1,
        progress_probability=0.2,
        failure_probability=0.9,
        np=np,
    )
    assert directive.option == "CONSENSUS_PREFIX"
    assert directive.hypothesis_count == 1
    assert directive.recovery_triggered

    exhausted = controller.decide(
        candidate=0,
        action_chunk=np.ones((16, 7), dtype=np.float32),
        scores=np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        valid=np.asarray([True, True, True] + [False] * 14),
        completion_probability=0.1,
        progress_probability=0.2,
        failure_probability=0.9,
        np=np,
    )
    assert exhausted.candidate == 0
    assert exhausted.option == "ADVANCE"
    assert not exhausted.recovery_triggered


def test_recovery_budget_resets_only_when_subgoal_changes() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        max_recovery_attempts=1,
    )
    kwargs = {
        "candidate": 0,
        "action_chunk": np.ones((16, 7), dtype=np.float32),
        "scores": np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        "valid": np.asarray([True, True, True] + [False] * 14),
        "completion_probability": 0.1,
        "progress_probability": 0.2,
        "failure_probability": 0.9,
        "np": np,
    }
    assert controller.decide(subgoal_index=0, **kwargs).recovery_triggered
    assert not controller.decide(subgoal_index=0, **kwargs).recovery_triggered
    assert controller.decide(subgoal_index=1, **kwargs).recovery_triggered


def test_failure_onset_guard_defers_early_recovery() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        min_recovery_elapsed_steps=75,
    )
    kwargs = {
        "candidate": 0,
        "action_chunk": np.ones((16, 7), dtype=np.float32),
        "scores": np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        "valid": np.asarray([True, True, True] + [False] * 14),
        "completion_probability": 0.1,
        "progress_probability": 0.2,
        "failure_probability": 0.9,
        "np": np,
    }
    early = controller.decide(subgoal_elapsed_steps=74, **kwargs)
    onset = controller.decide(subgoal_elapsed_steps=75, **kwargs)
    assert early.candidate == 0
    assert early.option == "ACCEPT_B"
    assert not early.recovery_triggered
    assert onset.candidate > 0
    assert onset.recovery_triggered


def test_learned_option_values_veto_harmful_recovery() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        use_option_values=True,
        option_value_margin=0.1,
    )
    kwargs = {
        "candidate": 0,
        "action_chunk": np.ones((16, 7), dtype=np.float32),
        "scores": np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        "valid": np.asarray([True, True, True] + [False] * 14),
        "completion_probability": 0.1,
        "progress_probability": 0.2,
        "failure_probability": 0.9,
        "np": np,
    }
    accept = controller.decide(
        option_values=np.asarray([2.0, 0.0, 0.0, -1.0, 1.0, 0.5]),
        **kwargs,
    )
    assert accept.candidate == 0
    assert not accept.recovery_triggered

    recover = controller.decide(
        option_values=np.asarray([0.0, 0.0, 2.0, -1.0, 0.0, 1.0]),
        **kwargs,
    )
    assert recover.option == "RETRY_CURRENT"
    assert recover.recovery_triggered
    assert recover.apply_subgoal_transition
    assert recover.subgoal_delta == 0
    assert recover.reanchor


def test_learned_option_values_apply_backtrack_and_confirmed_advance() -> None:
    kwargs = {
        "candidate": 0,
        "action_chunk": np.ones((16, 7), dtype=np.float32),
        "scores": np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        "valid": np.asarray([True, True, True] + [False] * 14),
        "completion_probability": 0.1,
        "progress_probability": 0.2,
        "failure_probability": 0.9,
        "subgoal_elapsed_steps": 75,
        "np": np,
    }
    backtrack = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        use_option_values=True,
    ).decide(
        subgoal_index=2,
        option_values=np.asarray([0.0, -1.0, 1.0, 3.0, 2.0, -2.0]),
        **kwargs,
    )
    assert backtrack.option == "BACKTRACK_ONE"
    assert backtrack.apply_subgoal_transition
    assert backtrack.subgoal_delta == -1

    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=1,
        use_option_values=True,
    )
    values = np.asarray([0.0, -2.0, -1.0, -3.0, 3.0, -4.0])
    unconfirmed = controller.decide(
        subgoal_index=1,
        stop_pending=False,
        option_values=values,
        **kwargs,
    )
    assert unconfirmed.option == "ACCEPT_B"
    assert not unconfirmed.apply_subgoal_transition
    confirmed = controller.decide(
        subgoal_index=1,
        stop_pending=True,
        option_values=values,
        **kwargs,
    )
    assert confirmed.option == "ADVANCE"
    assert confirmed.apply_subgoal_transition
    assert confirmed.subgoal_delta == 1


def test_consensus_finishes_after_nonstop_followup_hypotheses() -> None:
    controller = SelectiveConsensusRecovery(
        completion_threshold=0.8,
        consensus_hypotheses=4,
    )
    kwargs = {
        "action_chunk": np.ones((16, 7), dtype=np.float32),
        "scores": np.asarray([4.0, 1.0, 3.0] + [0.0] * 14, dtype=np.float32),
        "valid": np.asarray([True, True, True] + [False] * 14),
        "completion_probability": 0.1,
        "progress_probability": 0.2,
        "failure_probability": 0.9,
        "subgoal_elapsed_steps": 75,
        "np": np,
    }
    first = controller.decide(candidate=0, **kwargs)
    second = controller.decide(candidate=2, **kwargs)
    third = controller.decide(candidate=2, **kwargs)
    fourth = controller.decide(candidate=2, **kwargs)
    assert first.suppress_stop_confirmation
    assert second.suppress_stop_confirmation
    assert third.suppress_stop_confirmation
    assert fourth.option == "CONSENSUS_PREFIX"
    assert fourth.candidate == 2
    assert fourth.hypothesis_count == 4


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

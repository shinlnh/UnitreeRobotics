from __future__ import annotations

import numpy as np
import pytest
from unitree_rl_groot.groot.locomanipulation import N15MsgSerializer, parse_commanded_fetch


def test_vietnamese_command_is_admitted_for_installed_skill() -> None:
    mission = parse_commanded_fetch("Hãy lấy vô lăng và mang tới bàn giao")

    assert mission.object_name == "steering wheel"
    assert mission.destination == "drop-off table"


def test_unsupported_object_is_rejected_instead_of_faked() -> None:
    with pytest.raises(ValueError, match="only supports"):
        parse_commanded_fetch("Hãy lấy chai nước và mang tới bàn")


def test_n15_serializer_round_trips_numeric_arrays_without_pickle() -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded = N15MsgSerializer.from_bytes(N15MsgSerializer.to_bytes({"value": value}))

    np.testing.assert_array_equal(decoded["value"], value)


def test_n15_serializer_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match="object-dtype"):
        N15MsgSerializer.to_bytes({"value": np.asarray([object()], dtype=object)})

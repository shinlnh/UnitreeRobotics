from __future__ import annotations

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack")
pytest.importorskip("msgpack_numpy")
zmq = pytest.importorskip("zmq")

from unitree_rl_groot.groot.client import GrootPolicyClient, MsgSerializer  # noqa: E402


def test_msgpack_numpy_round_trip() -> None:
    expected = {
        "video": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "state": np.array([1.0, 2.0], dtype=np.float32),
    }
    actual = MsgSerializer.from_bytes(MsgSerializer.to_bytes(expected))
    np.testing.assert_array_equal(actual["video"], expected["video"])
    np.testing.assert_array_equal(actual["state"], expected["state"])


def test_serializer_forbids_pickle_backed_object_arrays() -> None:
    with pytest.raises(TypeError, match="object-dtype"):
        MsgSerializer.to_bytes(np.array([object()], dtype=object))


def test_serializer_rejects_forged_object_array_payload() -> None:
    forged = msgpack.packb({b"nd": True, b"kind": b"O", b"shape": [1], b"data": b"not-a-pickle"})
    with pytest.raises(ValueError, match="pickle-bearing"):
        MsgSerializer.from_bytes(forged)


def test_policy_client_uses_native_endpoint_contract() -> None:
    class ProtocolSocket:
        def send(self, payload: bytes) -> None:
            request = MsgSerializer.from_bytes(payload)
            if request["endpoint"] == "ping":
                self.response = {"status": "ok"}
            elif request["endpoint"] == "reset":
                self.response = {}
            else:
                assert request["endpoint"] == "get_action"
                assert "observation" in request["data"]
                self.response = [{"navigate_command": np.zeros((1, 4, 3), dtype=np.float32)}, {}]

        def recv(self) -> bytes:
            return MsgSerializer.to_bytes(self.response)

    client = GrootPolicyClient.__new__(GrootPolicyClient)
    client.host = "127.0.0.1"
    client.port = 5555
    client.api_token = None
    client._closed = False
    client._socket = ProtocolSocket()
    assert client.ping()
    assert client.reset() == {}
    action, info = client.get_action({"video": {}, "state": {}, "language": {}})
    assert action["navigate_command"].shape == (1, 4, 3)
    assert info == {}

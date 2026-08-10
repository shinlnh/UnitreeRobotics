"""Contracts for the Isaac Lab G1 commanded locomanipulation demo.

The NVIDIA N1.5 inference service predates the N1.7 wire protocol used by
``GrootPolicyClient``.  This module deliberately keeps its compatibility
client Torch-free so Isaac Sim and the VLA can run in isolated environments.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import zmq


@dataclass(frozen=True)
class CommandedFetchMission:
    """A command admitted by the currently supported Isaac Sim scene."""

    instruction: str
    object_name: str = "steering wheel"
    source: str = "pickup table"
    destination: str = "drop-off table"


_OBJECT_ALIASES = ("steering wheel", "wheel", "vo lang", "vô lăng")
_FETCH_VERBS = ("pick", "fetch", "bring", "move", "take", "lay", "nhat", "lấy", "mang", "đem")


def parse_commanded_fetch(instruction: str) -> CommandedFetchMission:
    """Validate a natural-language command against the installed skill.

    This is an admission gate, not an LLM parser.  The published checkpoint is
    trained for one steering-wheel pick/navigate/place skill, so unsupported
    objects are rejected instead of silently pretending to understand them.
    """

    normalized = " ".join(str(instruction).strip().lower().split())
    if not normalized:
        raise ValueError("instruction must not be empty")
    if not any(alias in normalized for alias in _OBJECT_ALIASES):
        raise ValueError("installed Isaac Sim skill only supports the steering wheel (Vietnamese: 'vô lăng')")
    if not any(verb in normalized for verb in _FETCH_VERBS):
        raise ValueError("instruction must request a pick/fetch/bring action")
    return CommandedFetchMission(instruction=" ".join(str(instruction).strip().split()))


class N15MsgSerializer:
    """Safe implementation of GR00T N1.5's ndarray msgpack protocol."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=N15MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=N15MsgSerializer._decode, raw=False)

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                raise TypeError("object-dtype ndarrays are forbidden")
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if hasattr(obj, "model_dump_json"):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        raise TypeError(f"cannot serialize {type(obj).__name__}")

    @staticmethod
    def _decode(obj: dict[str, Any]) -> Any:
        if obj.get("__ndarray_class__"):
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if obj.get("__ModalityConfig_class__"):
            return json.loads(obj["as_json"])
        return obj


class GrootN15PolicyClient:
    """Synchronous client for the isolated GR00T N1.5 policy process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, *, timeout_ms: int = 120_000):
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._closed = False
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        if self._closed or self._socket is None:
            raise RuntimeError("client is closed")
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(N15MsgSerializer.to_bytes(request))
            response = N15MsgSerializer.from_bytes(self._socket.recv())
        except zmq.error.Again:
            self._connect()
            raise TimeoutError(f"GR00T N1.5 server timed out at {self.host}:{self.port}") from None
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T N1.5 server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            response = self.call("ping")
        except (RuntimeError, TimeoutError, zmq.error.ZMQError):
            return False
        return isinstance(response, dict) and response.get("status") == "ok"

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self.call("get_action", observation)
        if not isinstance(response, dict):
            raise TypeError(f"unexpected N1.5 response: {type(response).__name__}")
        result = {str(key): np.asarray(value) for key, value in response.items()}
        if not result or not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("GR00T returned an empty or non-finite action")
        return result

    def set_seed(self, seed: int) -> None:
        response = self.call("set_seed", {"seed": int(seed)})
        if not isinstance(response, dict) or response.get("seed") != int(seed):
            raise RuntimeError(f"GR00T server rejected policy seed {seed}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> GrootN15PolicyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

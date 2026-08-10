"""Lightweight, pickle-free client for the pinned GR00T PolicyServer protocol."""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class MessageSerializer:
    """Wire-compatible numeric ndarray serialization with no pickle path."""

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.dtype.kind == "O":
            raise TypeError("object-dtype arrays are forbidden")
        return mnp.encode(value)

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict):
            nd_marker = value.get(b"nd", value.get("nd"))
            kind = value.get(b"kind", value.get("kind"))
            if nd_marker and kind in (b"O", "O"):
                raise ValueError("pickle-bearing object-dtype ndarray payload is forbidden")
        return mnp.decode(value)

    @classmethod
    def to_bytes(cls, value: Any) -> bytes:
        return msgpack.packb(value, default=cls._encode)

    @classmethod
    def from_bytes(cls, value: bytes) -> Any:
        hook = functools.partial(cls._decode)
        return msgpack.unpackb(value, object_hook=hook, raw=False)


class PolicyClient:
    """Small subset of NVIDIA's PolicyClient used by SONIC inference."""

    def __init__(self, host: str = "localhost", port: int = 5550, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.socket: zmq.Socket | None = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, *, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        assert self.socket is not None
        try:
            self.socket.send(MessageSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        response = MessageSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T PolicyServer error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
        except zmq.error.ZMQError:
            self._init_socket()
            return False
        return True

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise ValueError("GR00T PolicyServer returned an invalid action response")
        return response[0], response[1]

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

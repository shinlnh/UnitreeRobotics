"""Thin, Torch-free client for the native GR00T N1.7 policy server.

The wire protocol follows ``gr00t.policy.server_client`` at the commit pinned
in ``configs/setup/versions.env``. Keeping this client free of GR00T and Torch
allows Isaac Lab 3.0 (Torch 2.11) and GR00T N1.7 (Torch 2.9) to coexist.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class MsgSerializer:
    """GR00T-compatible msgpack serializer with object arrays forbidden."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=lambda value: value)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        hook = functools.partial(MsgSerializer._safe_decode, chain=lambda value: value)
        return msgpack.unpackb(data, object_hook=hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain=None) -> Any:
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError("object-dtype ndarrays are forbidden because msgpack-numpy would use pickle")
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain=None) -> Any:
        if isinstance(obj, dict):
            nd_marker = obj.get(b"nd", obj.get("nd"))
            kind = obj.get(b"kind", obj.get("kind"))
            if nd_marker and kind in (b"O", "O"):
                raise ValueError("refusing pickle-bearing object-dtype ndarray payload")
        return mnp.decode(obj, chain=chain)


class GrootPolicyClient:
    """Synchronous native PolicyServer client with timeout recovery."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        *,
        timeout_ms: int = 30_000,
        api_token: str | None = None,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._context = zmq.Context()
        self._closed = False
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        if self._closed:
            raise RuntimeError("client is closed")
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self._socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self._socket.recv())
        except zmq.error.Again:
            self._connect()
            raise TimeoutError(f"GR00T server timed out at {self.host}:{self.port}") from None
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            response = self.call("ping")
        except (TimeoutError, RuntimeError, zmq.error.ZMQError):
            return False
        return isinstance(response, dict) and response.get("status") == "ok"

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call("get_action", {"observation": observation, "options": options})
        if not isinstance(response, list | tuple) or len(response) != 2:
            raise TypeError(f"unexpected GR00T get_action response: {type(response).__name__}")
        return response[0], response[1]

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call("reset", {"options": options})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> GrootPolicyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

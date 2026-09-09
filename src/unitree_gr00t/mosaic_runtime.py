"""Training-only runtime controls for paired MOSAIC counterfactual execution."""

from __future__ import annotations

import copy
import hashlib
import math
import pickle
from dataclasses import dataclass
from typing import Any

from .ours import OursContractError


@dataclass(frozen=True)
class RandomTapeSnapshot:
    """Global and evaluator-owned randomness at one absolute physical step."""

    python_state: Any
    numpy_state: Any
    dynamic_state: Any
    torch_cpu_state: Any
    torch_cuda_states: tuple[Any, ...]
    absolute_step: int


def capture_random_tape(
    *,
    dynamic_state: Any,
    absolute_step: int,
    random_module: Any,
    np: Any,
    torch: Any | None = None,
) -> RandomTapeSnapshot:
    """Capture randomness that the old physical-only snapshots omitted."""

    if absolute_step < 0:
        raise OursContractError("MOSAIC random-tape step cannot be negative")
    torch_cpu = None
    torch_cuda: tuple[Any, ...] = ()
    if torch is not None:
        torch_cpu = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            torch_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return RandomTapeSnapshot(
        python_state=copy.deepcopy(random_module.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        dynamic_state=copy.deepcopy(dynamic_state),
        torch_cpu_state=torch_cpu,
        torch_cuda_states=torch_cuda,
        absolute_step=absolute_step,
    )


def restore_random_tape(
    snapshot: RandomTapeSnapshot,
    *,
    random_module: Any,
    np: Any,
    torch: Any | None = None,
) -> tuple[Any, int]:
    """Restore a tape and return evaluator-owned dynamic state plus step."""

    if snapshot.absolute_step < 0:
        raise OursContractError("MOSAIC random-tape snapshot is invalid")
    random_module.setstate(copy.deepcopy(snapshot.python_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_state))
    if snapshot.torch_cpu_state is not None:
        if torch is None:
            raise OursContractError("MOSAIC torch RNG state requires torch during restore")
        torch.set_rng_state(snapshot.torch_cpu_state.clone())
        if snapshot.torch_cuda_states:
            if not torch.cuda.is_available():
                raise OursContractError("MOSAIC CUDA RNG state cannot be restored")
            torch.cuda.set_rng_state_all([state.clone() for state in snapshot.torch_cuda_states])
    return copy.deepcopy(snapshot.dynamic_state), snapshot.absolute_step


def random_tape_sha256(snapshot: RandomTapeSnapshot, *, np: Any) -> str:
    """Hash the full exogenous tape for compact arm-contract manifests."""

    digest = hashlib.sha256()
    digest.update(pickle.dumps(snapshot.python_state, protocol=5))
    digest.update(pickle.dumps(snapshot.numpy_state, protocol=5))
    digest.update(pickle.dumps(snapshot.dynamic_state, protocol=5))
    digest.update(str(snapshot.absolute_step).encode())
    for state in (snapshot.torch_cpu_state, *snapshot.torch_cuda_states):
        if state is None:
            continue
        array = np.ascontiguousarray(state.cpu().numpy())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class InitialFlowNoiseSteering:
    """Replace exactly one GR00T action head's initial flow-noise direction.

    The upstream model samples its initial action noise inside
    ``get_action_with_features`` and exposes no option for overriding it. This
    one-shot pre-hook intercepts only the first timestep-zero call to the action
    encoder. It preserves the sampled norm and leaves an all-zero code exactly
    unchanged. All model weights remain frozen.
    """

    def __init__(
        self,
        action_head: Any,
        *,
        basis: Any,
        code: Any,
        maximum_angle: float,
    ) -> None:
        if not math.isfinite(maximum_angle) or not 0.0 < maximum_angle <= math.pi / 2:
            raise OursContractError("MOSAIC flow-noise maximum angle is invalid")
        self._basis = basis
        self._code = code
        self._maximum_angle = float(maximum_angle)
        self._consumed = False
        self.baseline_noise: Any | None = None
        self.steered_noise: Any | None = None
        encoder = getattr(action_head, "action_encoder", None)
        if encoder is None or not hasattr(encoder, "register_forward_pre_hook"):
            raise OursContractError("MOSAIC action head has no hookable action encoder")
        self._handle = encoder.register_forward_pre_hook(self._steer)

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _steer(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if self._consumed:
            return None
        if len(inputs) < 3:
            raise OursContractError("MOSAIC action-encoder input contract changed")
        actions, timesteps = inputs[0], inputs[1]
        if actions.ndim != 3 or timesteps.ndim != 1 or len(timesteps) != len(actions):
            raise OursContractError("MOSAIC initial flow-noise tensor shape is invalid")
        if not bool((timesteps == 0).all().item()):
            return None

        torch = __import__("torch")
        batch = actions.shape[0]
        flat = actions.detach().float().reshape(batch, -1)
        basis = torch.as_tensor(self._basis, dtype=torch.float32, device=actions.device)
        code = torch.as_tensor(self._code, dtype=torch.float32, device=actions.device)
        if code.ndim == 1:
            code = code.unsqueeze(0).expand(batch, -1)
        if (
            basis.ndim != 2
            or basis.shape[0] != flat.shape[1]
            or code.shape != (batch, basis.shape[1])
            or not bool(torch.isfinite(basis).all().item())
            or not bool(torch.isfinite(code).all().item())
        ):
            raise OursContractError("MOSAIC flow-noise basis or code shape is invalid")
        angles = torch.linalg.vector_norm(code, dim=1)
        if bool((angles > self._maximum_angle + 1e-7).any().item()):
            raise OursContractError("MOSAIC flow-noise code exceeds its trust region")

        self.baseline_noise = actions.detach().clone()
        if bool((angles == 0).all().item()):
            self.steered_noise = actions.detach().clone()
            self._consumed = True
            return None

        baseline_norm = torch.linalg.vector_norm(flat, dim=1)
        if bool((baseline_norm <= 0).any().item()):
            raise OursContractError("MOSAIC sampled a zero initial flow-noise vector")
        unit = flat / baseline_norm[:, None]
        tangent = code @ basis.T
        tangent = tangent - (tangent * unit).sum(dim=1, keepdim=True) * unit
        tangent_norm = torch.linalg.vector_norm(tangent, dim=1)
        nonzero = angles > 0
        if bool((tangent_norm[nonzero] <= 1e-10).any().item()):
            raise OursContractError("MOSAIC steering direction is parallel to flow noise")
        tangent_unit = tangent / tangent_norm.clamp_min(1e-10)[:, None]
        moved = baseline_norm[:, None] * (
            torch.cos(angles)[:, None] * unit
            + torch.sin(angles)[:, None] * tangent_unit
        )
        moved = torch.where(nonzero[:, None], moved, flat)
        steered = moved.reshape_as(actions).to(dtype=actions.dtype)
        self.steered_noise = steered.detach().clone()
        self._consumed = True
        return (steered, *inputs[1:])

    def assert_consumed(self) -> None:
        if not self._consumed:
            raise OursContractError("MOSAIC flow-noise hook did not observe timestep zero")

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> InitialFlowNoiseSteering:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _flow_noise_sha256(value: Any, *, np: Any) -> str:
    array = np.asarray(value.detach().float().cpu().numpy(), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_mosaic_steering_policy(
    selector_policy: Any,
    *,
    code_dimension: int,
    basis_seed: int,
    maximum_angle: float,
) -> Any:
    """Wrap exact B with one registered continuous initial-noise intervention.

    This server-side wrapper is training-only. The request supplies a steering
    code and an explicit prefix; the underlying B policy still supplies the
    frozen action generator, context, and selector metadata.
    """

    import numpy as np
    from gr00t.policy.policy import PolicyWrapper

    from .recovery_curvature import orthogonal_steering_basis

    base_policy = getattr(selector_policy, "policy", None)
    model = getattr(base_policy, "model", None)
    action_head = getattr(model, "action_head", None)
    action_dimension = int(getattr(action_head, "action_dim", 0))
    action_horizon = int(getattr(getattr(action_head, "config", None), "action_horizon", 0))
    ambient_dimension = action_dimension * action_horizon
    if (
        action_head is None
        or ambient_dimension < 1
        or code_dimension < 1
        or code_dimension > ambient_dimension
        or basis_seed < 0
    ):
        raise OursContractError("MOSAIC steering-policy dimensions are invalid")
    basis = orthogonal_steering_basis(
        ambient_dimension,
        code_dimension,
        basis_seed,
        np=np,
    )
    basis_sha256 = hashlib.sha256(np.asarray(basis, dtype="<f4").tobytes()).hexdigest()

    class MosaicSteeringPolicy(PolicyWrapper):
        def check_observation(self, observation: dict[str, Any]) -> None:
            selector_policy.check_observation(observation)

        def check_action(self, action: dict[str, Any]) -> None:
            selector_policy.check_action(action)

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            options = options or {}
            request = options.get("mosaic")
            if not isinstance(request, dict):
                raise OursContractError("MOSAIC policy request is missing mosaic options")
            code = np.asarray(request.get("steering_code"), dtype=np.float32)
            prefix_length = request.get("prefix_length")
            program_id = request.get("program_id")
            rule_index = request.get("rule_index")
            if (
                code.shape != (code_dimension,)
                or not np.isfinite(code).all()
                or not isinstance(prefix_length, int)
                or not 1 <= prefix_length <= action_horizon
                or not isinstance(program_id, str)
                or not program_id
                or not isinstance(rule_index, int)
                or rule_index < 0
            ):
                raise OursContractError("MOSAIC policy request is invalid")
            with InitialFlowNoiseSteering(
                action_head,
                basis=basis,
                code=code,
                maximum_angle=maximum_angle,
            ) as steering:
                action, info = selector_policy._get_action(observation, options)
                steering.assert_consumed()
            assert steering.baseline_noise is not None
            assert steering.steered_noise is not None
            baseline_norm = float(steering.baseline_noise.detach().float().norm().cpu())
            steered_norm = float(steering.steered_noise.detach().float().norm().cpu())
            metadata = {
                "program_id": program_id,
                "rule_index": rule_index,
                "steering_code": code.tolist(),
                "prefix_length": prefix_length,
                "maximum_angle": maximum_angle,
                "basis_seed": basis_seed,
                "basis_sha256": basis_sha256,
                "baseline_noise_sha256": _flow_noise_sha256(
                    steering.baseline_noise, np=np
                ),
                "steered_noise_sha256": _flow_noise_sha256(
                    steering.steered_noise, np=np
                ),
                "baseline_noise_norm": baseline_norm,
                "steered_noise_norm": steered_norm,
            }
            return action, dict(info) | {"mosaic": metadata}

    return MosaicSteeringPolicy(selector_policy)

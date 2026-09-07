import json
from pathlib import Path

import numpy as np

from unitree_gr00t.a0_eval import _clean_partial_trace, _reset_environment


def test_clean_partial_trace_keeps_only_completed_episode(tmp_path: Path) -> None:
    trace = tmp_path / "decisions.jsonl"
    frame_root = tmp_path / "frames" / "Ideal" / "case1"
    frame_root.mkdir(parents=True)
    kept_frame = frame_root / "trial-00-call-00001.npz"
    dropped_frame = frame_root / "trial-01-call-00001.npz"
    kept_frame.write_bytes(b"kept")
    dropped_frame.write_bytes(b"partial")
    rows = [
        {
            "task_type": "Ideal",
            "case": "case1",
            "trial": 0,
            "frame_bundle": str(kept_frame.relative_to(tmp_path)),
        },
        {
            "task_type": "Ideal",
            "case": "case1",
            "trial": 1,
            "frame_bundle": str(dropped_frame.relative_to(tmp_path)),
        },
    ]
    trace.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")

    _clean_partial_trace(tmp_path, {("Ideal", "case1", 0)})

    remaining = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [row["trial"] for row in remaining] == [0]
    assert kept_frame.is_file()
    assert not dropped_frame.exists()


def test_missing_optional_init_uses_seeded_environment_reset(tmp_path: Path) -> None:
    class FakeEnvironment:
        seeded = None
        resets = 0

        def seed(self, value: int) -> None:
            self.seeded = value

        def reset(self) -> None:
            self.resets += 1

        def _get_observations(self) -> dict[str, str]:
            return {"state": "reset"}

    environment = FakeEnvironment()
    observation, source = _reset_environment(environment, tmp_path / "missing.init", 17, np)

    assert observation == {"state": "reset"}
    assert source == "seeded_environment_reset"
    assert environment.seeded == 17
    assert environment.resets == 1

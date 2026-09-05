import json
from pathlib import Path

from unitree_gr00t.reporting import write_report
from unitree_gr00t.types import EpisodeResult


def test_report_writes_self_contained_artifacts(tmp_path: Path) -> None:
    result = EpisodeResult(
        task_id="pick_red_cube",
        instruction="Pick up the red cube",
        seed=1,
        success=True,
        steps=12,
        policy_calls=3,
        safety_clips=1,
        elapsed_seconds=0.01,
        final_objects={
            "red_cube": {"kind": "cube", "color": "#f00", "x": 0.5, "y": 0.5},
        },
        events=["grasp:red_cube"],
    )
    summary = write_report([result], tmp_path)
    assert summary["success_rate"] == 1.0
    assert (tmp_path / "report.html").is_file()
    assert "Generalist robotics smoke report" in (tmp_path / "report.html").read_text()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["episodes"] == 1

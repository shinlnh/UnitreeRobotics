import json
from pathlib import Path

from unitree_gr00t import a1


def test_source_audit_tracks_pinned_exclusions_and_overlap(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    usable = source / "scene" / "case1"
    missing = source / "scene" / "case2"
    usable.mkdir(parents=True)
    missing.mkdir(parents=True)
    (usable / "demo.hdf5").touch()
    (usable / "selected.bddl").touch()
    (usable / "extra.bddl").touch()
    manifest = source / "trainingset.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "scene": "scene",
                    "case": "case1",
                    "task_description": "Task: source instruction",
                },
                {
                    "scene": "scene",
                    "case": "case2",
                    "task_description": "Task: missing demonstration",
                },
            ]
        )
    )
    benchmark = tmp_path / "benchmark" / "Ideal" / "case1"
    benchmark.mkdir(parents=True)
    (benchmark / "task_description.txt").write_text(
        "Task: unseen benchmark\nStep: do something\n[0, 1]\n"
    )
    monkeypatch.setattr(
        a1,
        "_demonstration_metadata",
        lambda _: ("source instruction", "selected.bddl"),
    )

    audit = a1.audit_training_source(
        source,
        manifest,
        tmp_path / "benchmark",
        expected_manifest_sha256=a1.sha256_file(manifest),
        expected_manifest_rows=2,
        expected_usable_episodes=1,
    )

    assert audit.valid
    assert audit.missing_demonstrations == ("scene/case2",)
    assert audit.resolved_ambiguous_bddl_cases == ("scene/case1:selected.bddl",)
    assert audit.exact_prompt_overlaps == ()


def test_normalize_prompt_is_case_and_punctuation_insensitive() -> None:
    assert a1.normalize_prompt("Pick-up the Mug!") == "pick up the mug"

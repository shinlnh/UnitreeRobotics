#!/usr/bin/env python3
"""Compile every procedural SONIC fetch-scene profile with MuJoCo."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
from run_fetch_sim_loop import (
    DESTINATION_PROFILES,
    OBJECT_PROFILES,
    SOURCE_PROFILES,
    materialize_scene,
)


def main() -> int:
    validated = 0
    for object_profile in OBJECT_PROFILES:
        for source_profile in SOURCE_PROFILES:
            for destination_profile in DESTINATION_PROFILES:
                with tempfile.TemporaryDirectory(prefix="validate-fetch-") as directory:
                    scene = materialize_scene(
                        Path(directory),
                        object_profile=object_profile,
                        source_profile=source_profile,
                        destination_profile=destination_profile,
                    )
                    model = mujoco.MjModel.from_xml_path(str(scene))
                    model.body("fetch_object")
                    model.geom("destination_goal")
                    validated += 1
    print(
        json.dumps(
            {
                "objects": len(OBJECT_PROFILES),
                "sources": len(SOURCE_PROFILES),
                "destinations": len(DESTINATION_PROFILES),
                "validated_scenes": validated,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

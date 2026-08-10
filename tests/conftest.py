"""Make the external Isaac Lab package importable without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "source" / "unitree_rl_groot"
sys.path.insert(0, str(PACKAGE_ROOT))

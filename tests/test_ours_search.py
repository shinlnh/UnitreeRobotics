from pathlib import Path

import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_search import build_r0_registry


def test_r0_registry_fails_closed_when_registered_variant_is_missing(tmp_path: Path) -> None:
    with pytest.raises(OursContractError, match="expected 1 R0 variants"):
        build_r0_registry(tmp_path, expected_variants=1)

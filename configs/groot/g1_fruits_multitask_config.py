"""GR00T N1.7 modality contract for the official 43-DoF Unitree G1 dataset."""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

JOINT_GROUPS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]

# The source demonstrations run at 20 Hz.  Forty actions therefore represent a
# two-second chunk, matching the N1.7 default action horizon.
config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["rs_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=JOINT_GROUPS),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=JOINT_GROUPS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# This project owns NEW_EMBODIMENT.  Rebinding makes repeat imports safe in
# notebooks, tests and distributed workers.
MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

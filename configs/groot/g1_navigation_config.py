"""GR00T N1.7 modality config for language-conditioned G1 base navigation."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

g1_navigation_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["proprio"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["navigate_command"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_navigation_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

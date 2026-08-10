"""GR00T N1.5 modality contract for Isaac Lab's G1 locomanipulation task.

This is kept project-local so setup never overwrites the pinned GR00T checkout.
It mirrors the data config shipped with Isaac Lab 3.0.
"""

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.model.transforms import GR00TTransform


class G1LocomanipulationSDGDataConfig:
    """Camera, state and 32D action modalities used by NVIDIA's checkpoint."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_hand_pose",
        "state.right_hand_pose",
        "state.left_hand_joint_positions",
        "state.right_hand_joint_positions",
        "state.object_pose",
        "state.goal_pose",
        "state.end_fixture_pose",
    ]
    action_keys = [
        "action.left_hand_pose",
        "action.right_hand_pose",
        "action.left_hand_joint_positions",
        "action.right_hand_joint_positions",
        "action.base_velocity",
        "action.base_height",
    ]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
        )

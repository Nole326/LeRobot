"""Predict an absolute EE action and segment boundary from synchronized observations."""

import torch

from .act_policy import ACTInputs
from .ee_command import unpack
from .skill_execution import SkillExecution


class EEPolicy(ACTInputs):
    def __init__(self, checkpoint, device="cuda"):
        super().__init__(checkpoint, device)
        self.execution = SkillExecution({"control": self.control})
        self.reset_scene()

    def reset_scene(self):
        self.model.reset()
        self.execution.new_scene()

    @torch.inference_mode()
    def predict(self, front, wrist, joint_motor):
        """Return xyz, rotation vector, gripper radians and a skill boundary for this decision."""
        batch = self.observation(front, wrist, joint_motor)
        chunk = self.model.predict_action_chunk(batch)
        value, boundary = self.execution.next(chunk, self.stats["amu"], self.stats["astd"])
        action, _ = unpack(value.cpu().numpy())
        if boundary:
            self.model.reset()
        return action, boundary

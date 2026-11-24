"""Load the RGB/edge cohort and the placement-and-return policy."""

import json
from pathlib import Path

import numpy as np

from .rgb_residual import ACTResidualPolicy
from .residual import CENTER, SCALE, make_policy


class RGBController:
    def __init__(self, base, directory, encoder):
        self.base = base
        self.residual = ACTResidualPolicy.from_safetensors(
            str(directory / "model.safetensors"),
            edge_aware=encoder == "edges",
            device=base.device,
        )

    def reset(self):
        self.base.reset_scene()
        self.residual.reset()

    def act(self, front, wrist, joints):
        target = self.base.predict(front, wrist, joints)
        return self.residual(front, wrist, joints, target).cpu().numpy()


class PlacementController:
    def __init__(self, base, directory):
        from ..sac.history.vision import PlacementDetector
        from ..sac.history.release import ReleasePlacementTrigger

        self.base = base
        self.inner = make_policy(
            base.model,
            base.stats,
            directory / "model.safetensors",
            base.device,
            1,
            placement=True,
        )
        self.detector_type, self.trigger_type = (
            PlacementDetector,
            ReleasePlacementTrigger,
        )
        self.reset()

    def reset(self):
        self.inner.reset()
        self.detector = self.detector_type()
        self.trigger = self.trigger_type()
        self.steps = self.start = 0
        self.returning = None

    def act(self, front, wrist, joints):
        from ..sac.history.home import ReturnHome, at_home

        q = np.asarray(joints, np.float32).reshape(6)
        count = self.detector.update(front, wrist, q)["count"]
        if self.returning is None:
            notice = self.trigger.update(float(q[-1]), count)
            if notice is not None and self.steps:
                self.release_step, self.expected = self.steps, notice
                self.returning = ReturnHome(q)
        if self.returning is not None:
            returned = at_home(q) and count == self.expected
            if returned or self.steps - self.release_step >= 270:
                if not returned:
                    self.trigger.confirmed = self.expected - 1
                self.returning = None
                self.base.reset_scene()
                self.inner.previous_q = None
                self.start = self.steps
        elif self.steps - self.start >= 1200:
            self.base.reset_scene()
            self.inner.previous_q = None
            self.start = self.steps
        if self.returning is None:
            target = self.inner.act(front, wrist, q)
        else:
            target = self.returning.action(q)
            self.inner.last_action = (target - CENTER) / SCALE
        self.steps += 2
        return target


def qpos_controller(base, directory, fraction):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf8"))
    expected_base = config.get("base_subfolder")
    if (
        expected_base
        and int(expected_base.rsplit("/", 1)[-1].removesuffix("k")) * 1000
        != base.control["optimizer_updates"]
    ):
        raise ValueError("The selected ACT checkpoint does not match the residual base")
    if (
        config.get("encoder") in {"rgb", "edges", "dual_rgb_resblocks"}
        and "action_mode" not in config
    ):
        if fraction != 1:
            raise ValueError("The released historical evaluation uses full amplitude")
        if config["encoder"] == "dual_rgb_resblocks":
            return PlacementController(base, directory)
        return RGBController(base, directory, config["encoder"])
    return make_policy(
        base.model, base.stats, directory / "model.safetensors", base.device, fraction
    )

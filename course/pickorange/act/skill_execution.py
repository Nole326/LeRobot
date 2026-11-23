"""Decode ACT chunks at 30 Hz and reset temporal aggregation between single-orange skills."""

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler


class SkillExecution:
    def __init__(self, checkpoint):
        cfg = checkpoint["control"]
        if cfg["mode"] != "absolute_ee_robot_base" or cfg["action_hz"] != 30:
            raise ValueError("Expected an absolute EE ACT with 30 Hz commands")
        self.single = cfg["skill_end"]
        if self.single and cfg.get("skill_end_encoding") != "logit":
            raise ValueError("The single-orange policy must provide a segment-end logit")
        self.ensemble = ACTTemporalEnsembler(cfg["ensemble"], cfg["chunk"])
        self.new_scene()

    def new_scene(self):
        self.ensemble.reset()
        self.high = 0
        self.armed = True

    def next(self, chunk, mean, std):
        """Take one predicted chunk, returning one EE action and a skill-boundary event."""
        if tuple(chunk.shape) != (1, self.ensemble.chunk_size, 11 if self.single else 10):
            raise ValueError("Unexpected EE chunk shape")
        values = self.ensemble.update(chunk)[0] * std + mean
        if not self.single:
            return values, False
        # Only the motion is averaged. Classify the boundary from the newest
        # observation so an older chunk cannot delay or repeat a skill reset.
        values[-1] = chunk[0, 0, -1] * std[-1] + mean[-1]
        # The training label is positive for the last two decisions. Latch until
        # the learned score falls again so a stationary arm cannot trigger twice.
        high = bool(values[-1] >= 0)
        if not high:
            self.armed = True
        self.high = self.high + 1 if high else 0
        finished = self.armed and self.high >= 2
        if finished:
            self.ensemble.reset()
            self.high = 0
            self.armed = False
        return values, finished

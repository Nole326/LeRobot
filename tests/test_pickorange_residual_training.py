"""Training contracts for the published PickOrange residual experiments."""

import importlib.util
import json
from pathlib import Path
import unittest

from course.pickorange.sac.reward import BANDS, SceneLedger


class TrainingProfiles(unittest.TestCase):
    def test_act_profiles_only_change_control_specific_fields(self):
        folder = Path(__file__).resolve().parents[1] / "course/pickorange/sac"
        q, ee = (
            json.loads((folder / name).read_text()) for name in ("qpos.json", "ee.json")
        )
        self.assertEqual(
            {k for k in q if q[k] != ee[k]},
            {"base", "base_updates", "control", "gripper_effort"},
        )
        self.assertEqual(
            (
                q["batch_size"],
                q["critic_warmup"],
                q["actor_updates"],
                q["replay_capacity"],
            ),
            (128, 10000, 100000, 150000),
        )

    def test_terminal_score_bands(self):
        row = dict(
            step=0,
            inside=[False] * 3,
            oranges=[[0.2, 0, 0.02]] * 3,
            plate=[0, 0, 0],
            audit_tcp=[0, 0, 0.1],
            q_rad=[0] * 6,
        )
        ledger = SceneLedger(row)
        for count, (low, high) in enumerate(BANDS):
            ledger.stable = [i < count for i in range(3)]
            score = sum(ledger.components(800).values())
            self.assertLessEqual(low, score)
            self.assertLessEqual(score, high)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_groot_critic_warmup_and_actor_update(self):
        import torch
        from course.pickorange.groot_sac.sac_core import SAC, discount

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1704)
            model = SAC(32 * 1152 + 12, 96, visual_shape=(32, 1152), privileged_dim=30)
            s = torch.randn(32, 32 * 1152 + 12)
            a = torch.zeros(32, 96)
            p = torch.randn(32, 30)
            batch = (s, a, torch.ones(32), s, torch.full((32,), 0.99), a, a)
            bound = torch.full((1, 96), 0.02)
            before = [v.detach().clone() for v in model.actor.parameters()]
            critic = [v.detach().clone() for v in model.q.parameters()]
            model.update(
                batch, bound, critic_only=True, privileged=p, next_privileged=p
            )
            self.assertTrue(
                all(torch.equal(v, b) for v, b in zip(model.actor.parameters(), before))
            )
            self.assertTrue(
                any(not torch.equal(v, b) for v, b in zip(model.q.parameters(), critic))
            )
            model.update(batch, bound, privileged=p, next_privileged=p)
            self.assertTrue(
                any(
                    not torch.equal(v, b)
                    for v, b in zip(model.actor.parameters(), before)
                )
            )
            self.assertEqual((model.actor_updates, model.critic_updates), (1, 2))
            self.assertAlmostEqual(discount(16, False, True), 0.99)
            self.assertEqual(discount(16, True, False), 0)

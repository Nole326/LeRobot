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

    def test_named_evaluation_profiles_select_the_reported_bounds(self):
        from unittest.mock import patch
        from course.pickorange.act import evaluate

        for profile, folder, fraction in (
            ("ACT-2-80k", None, 1),
            ("Residual-5-80k-gripper", "Residual-5", 0),
            ("Residual-6-56k-quarter", "Residual-6", 0.25),
            ("Residual-8-56k-quarter", "Residual-8", 1),
        ):
            with (
                self.subTest(profile=profile),
                patch.object(evaluate, "run") as run,
                patch(
                    "sys.argv", ["evaluate", "--profile", profile, "--out", "result"]
                ),
            ):
                evaluate.main()
                args = run.call_args.args[0]
                self.assertEqual(args.arm_fraction, fraction)
                self.assertEqual(
                    args.residual.parent.name if args.residual else None, folder
                )
                self.assertEqual((args.first_seed, args.episodes), (731000, 50))

    def test_historical_evaluation_protocols(self):
        from unittest.mock import patch
        from course.pickorange.act import evaluate

        for profile, base, synced, effort, slots in (
            ("Residual-1-24k", "80k", True, "dynamic", None),
            ("Residual-2-24k", "80k", True, "dynamic", None),
            ("Residual-3-26k", "80k", False, "dynamic", 1),
            ("Residual-4-100k", "100k", True, "fixed-10Nm", 1),
            ("Residual-5-90k-full", "100k", True, "fixed-10Nm", 2),
        ):
            with (
                self.subTest(profile=profile),
                patch.object(evaluate, "run") as run,
                patch(
                    "sys.argv", ["evaluate", "--profile", profile, "--out", "result"]
                ),
            ):
                evaluate.main()
                args = run.call_args.args[0]
                self.assertEqual(
                    (
                        args.base.name,
                        args.synchronized_reset,
                        args.gripper_effort,
                        args.required_slots,
                        args.arm_fraction,
                    ),
                    (base, synced, effort, slots, 1),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
        "requires torch and safetensors",
    )
    def test_historical_critic_warmup_and_joint_update(self):
        import torch
        from course.pickorange.sac.history.learner import Learner
        from course.pickorange.sac.learner import SpatialEncoder

        with torch.random.fork_rng(devices=[]):
            for edge, placement in ((False, False), (True, False), (False, True)):
                torch.manual_seed(1704)
                model = Learner(
                    edge_aware=edge,
                    vectorized_augment=True,
                    encoder=SpatialEncoder() if placement else None,
                    device="cpu",
                )
                h, w = (96, 128) if placement else (120, 160)
                obs = (
                    torch.randint(256, (2, 2, 3, h, w), dtype=torch.uint8),
                    torch.zeros(2, 18),
                    torch.zeros(2, 6),
                )
                batch = (
                    obs,
                    torch.zeros(2, 6),
                    torch.ones(2),
                    obs,
                    torch.full((2,), 0.99),
                )
                actor = model.actor.mean[-1].weight.detach().clone()
                critic = model.q.heads[0].network[-1].weight.detach().clone()
                model.update(batch, critic_only=True, fixed_base_target=True)
                self.assertTrue(torch.equal(actor, model.actor.mean[-1].weight))
                self.assertFalse(
                    torch.equal(critic, model.q.heads[0].network[-1].weight)
                )
                model.update(batch)
                self.assertFalse(torch.equal(actor, model.actor.mean[-1].weight))
                self.assertEqual((model.updates, model.q_updates), (1, 2))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "requires torch and safetensors")
    def test_ee_residual_requires_segmented_80k_base(self):
        from course.pickorange.act.residual import validate_ee_residual_base
        control = dict(mode="absolute_ee_robot_base", optimizer_updates=80000, skill_end=True)
        validate_ee_residual_base(control)
        for change in (dict(skill_end=False), dict(optimizer_updates=100000), dict(mode="stride2")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_ee_residual_base(control | change)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_pixel_replay_storage_preserves_identical_samples_after_wraparound(self):
        import numpy as np
        import torch
        from course.pickorange.sac.replay import DeviceReplay

        # Force host-only, mixed and device-only sampling on a CPU runner.
        for gpu_slots in (0, 3, 10):
            replay = DeviceReplay(6, gpu_slots=gpu_slots, device="cpu")
            numeric = np.zeros(18, dtype=np.float32)
            previous = None
            for step in range(15):
                action = np.full(6, step, dtype=np.float32)
                current = replay.observe(torch.full((2, 3, 96, 128), step, dtype=torch.uint8), numeric, action)
                if previous is not None:
                    replay.append(previous, action, float(step), current, 0.99)
                previous = current
            indices = np.random.default_rng(2).integers(6, size=12)
            before, actions, rewards, after, _ = replay.sample(np.random.default_rng(2), 12)
            np.testing.assert_array_equal(rewards.numpy(), indices + 9)
            torch.testing.assert_close(actions[:, 0], rewards)
            torch.testing.assert_close(after[2][:, 0], rewards)
            torch.testing.assert_close(before[0][:, 0, 0, 0, 0].float(), rewards - 1)
            torch.testing.assert_close(after[0][:, 1, 2, 95, 127].float(), rewards)

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

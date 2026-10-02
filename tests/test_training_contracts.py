"""Optional torch/NumPy tests; lightweight source CI needs neither package."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None
if AVAILABLE:
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'course/scripts'))
    from training_contracts import FinalObservationBuffer, capture_rng, restore_rng


@unittest.skipUnless(AVAILABLE, 'Optional application environment: torch and NumPy required')
class TrainingContractsTest(unittest.TestCase):
    def test_rng_caches_and_private_streams(self):
        ng = {'env': np.random.default_rng(47)}
        tg = {'sampler': torch.Generator().manual_seed(53)}
        random.seed(41)
        np.random.seed(43)
        torch.manual_seed(59)
        random.gauss(0, 1)
        np.random.normal()
        saved = capture_rng(ng, tg)

        def draw():
            return (random.gauss(0, 1), float(np.random.normal()), torch.rand(5).tolist(),
                    ng['env'].normal(size=5).tolist(), torch.randperm(9, generator=tg['sampler']).tolist())

        expected = draw()
        restore_rng(saved, ng, tg)
        self.assertEqual(expected, draw())

    def test_private_registry_mismatch_rejected(self):
        saved = capture_rng({'env': np.random.default_rng(1)}, {})
        with self.assertRaises(ValueError):
            restore_rng(saved, {}, {})

    def test_terminal_rows_and_timeout_bootstrap(self):
        capture = FinalObservationBuffer()
        terminal = {'state': torch.tensor([[1.], [2.], [3.]]), 'rgb': torch.ones(3, 2, 2, 3, dtype=torch.uint8)}
        capture.capture(terminal, [1, 0])
        returned = {k: torch.zeros_like(v) for k, v in terminal.items()}
        nxt, bootstrap = capture.consume(returned, torch.tensor([False, True, False]), torch.tensor([True, False, False]))
        self.assertEqual(nxt['state'].flatten().tolist(), [1., 2., 0.])
        self.assertEqual(bootstrap.tolist(), [True, False, True])
        self.assertFalse(returned['rgb'].any())
        self.assertIsNone(capture.pending)

    def test_missing_terminal_observation_rejected(self):
        with self.assertRaises(RuntimeError):
            FinalObservationBuffer().consume({'state': torch.zeros(1, 3)}, torch.tensor([False]), torch.tensor([True]))

    def test_stale_capture_rejected(self):
        capture = FinalObservationBuffer()
        obs = {'state': torch.zeros(1, 3)}
        capture.capture(obs, [0])
        with self.assertRaises(RuntimeError):
            capture.consume(obs, torch.tensor([False]), torch.tensor([False]))


if __name__ == '__main__':
    unittest.main()

"""Keep raw uint8 replay frames on the learner GPU and retain the ACT image path."""

from collections import deque

import numpy as np
import torch


class DeviceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.records = deque()
        self.obs = {}
        self.refs = {}
        self.serial = 0
        self.images = torch.empty(
            (capacity + 4, 2, 3, 96, 128), device="cuda", dtype=torch.uint8
        )
        self.slots = len(self.images)

    def observe(self, image, numeric, base):
        i = self.serial
        self.serial += 1
        assert i - self.slots not in self.obs
        self.images[i % self.slots].copy_(torch.as_tensor(image))
        self.obs[i] = (numeric.copy(), base.copy())
        self.refs[i] = 0
        return i

    def append(self, s, a, r, ns, d):
        self.records.append((s, a.copy(), r, ns, d))
        for k in (s, ns):
            self.refs[k] += 1
        if len(self.records) > self.capacity:
            old = self.records.popleft()
            for k in (old[0], old[3]):
                self.refs[k] -= 1
                if not self.refs[k]:
                    del self.refs[k]
                    del self.obs[k]

    def sample(self, rng, n=128):
        rows = [self.records[int(i)] for i in rng.integers(len(self.records), size=n)]

        def batch(values):
            return torch.as_tensor(np.asarray(values), device="cuda")

        def states(column):
            ids = [row[column] for row in rows]
            values = [self.obs[i] for i in ids]
            return (
                self.images[batch([i % self.slots for i in ids])],
                batch([v[0] for v in values]),
                batch([v[1] for v in values]),
            )

        return (
            states(0),
            batch([r[1] for r in rows]),
            batch([r[2] for r in rows]).float(),
            states(3),
            batch([r[4] for r in rows]).float(),
        )

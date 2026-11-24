"""Reference-count adjacent transitions; keep uint8 camera frames in memory."""

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
        self.slots = capacity + 4
        self.gpu_slots = min(self.slots, 80000)
        self.images = torch.empty(
            (self.gpu_slots, 2, 3, 96, 128), device="cuda", dtype=torch.uint8
        )
        self.host = torch.empty(
            (self.slots - self.gpu_slots, 2, 3, 96, 128),
            device="cpu",
            dtype=torch.uint8,
        )

    def observe(self, image, numeric, base):
        i = self.serial
        self.serial += 1
        assert i - self.slots not in self.obs
        slot = i % self.slots
        target = (
            self.images[slot]
            if slot < self.gpu_slots
            else self.host[slot - self.gpu_slots]
        )
        target.copy_(torch.as_tensor(image))
        self.obs[i] = (numeric.copy(), base.copy())
        self.refs[i] = 0
        return i

    def sample(self, rng, n=128):
        rows = [self.records[int(i)] for i in rng.integers(len(self.records), size=n)]

        def batch(values):
            return torch.as_tensor(np.asarray(values), device="cuda")

        def states(column):
            ids = [row[column] for row in rows]
            values = [self.obs[i] for i in ids]
            slots = np.asarray(ids) % self.slots
            remote = slots >= self.gpu_slots
            if not remote.any():
                images = self.images[batch(slots)]
            elif remote.all():
                images = self.host[torch.from_numpy(slots - self.gpu_slots)].to("cuda")
            else:
                images = torch.empty(
                    (len(ids), *self.images.shape[1:]), device="cuda", dtype=torch.uint8
                )
                images[batch(np.flatnonzero(~remote))] = self.images[
                    batch(slots[~remote])
                ]
                images[batch(np.flatnonzero(remote))] = self.host[
                    torch.from_numpy(slots[remote] - self.gpu_slots)
                ].to("cuda")
            return images, batch([v[0] for v in values]), batch([v[1] for v in values])

        return (
            states(0),
            batch([r[1] for r in rows]),
            batch([r[2] for r in rows]).float(),
            states(3),
            batch([r[4] for r in rows]).float(),
        )

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

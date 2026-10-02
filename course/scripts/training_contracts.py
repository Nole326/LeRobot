"""Independent training contracts; no simulator or upstream monkey patches.

RNG coverage is explicit: callers must register private streams and save their
sampler/replay state separately at the same quiescent checkpoint boundary.
This module does not serialize a simulator, DataLoader workers or ACT caches.
"""
import copy
import random

import numpy as np
import torch


def capture_rng(numpy_streams, torch_streams, *, include_cuda=False):
    if include_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA RNG requested but CUDA unavailable')
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        'cuda': torch.cuda.get_rng_state_all() if include_cuda else None,
        'numpy_private': {k: copy.deepcopy(v.bit_generator.state) for k, v in numpy_streams.items()},
        'torch_private': {k: v.get_state().clone() for k, v in torch_streams.items()},
    }


def restore_rng(state, numpy_streams, torch_streams):
    if set(state['numpy_private']) != set(numpy_streams) or set(state['torch_private']) != set(torch_streams):
        raise ValueError('Private RNG registry differs from checkpoint')
    if state['cuda'] is not None and len(state['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA device count differs from checkpoint')
    for key, stream in numpy_streams.items():
        if state['numpy_private'][key]['bit_generator'] != stream.bit_generator.state['bit_generator']:
            raise ValueError('NumPy private generator type differs')
    # Call after model/optimizer/loader construction and loading, before sampling.
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    for key, stream in numpy_streams.items():
        stream.bit_generator.state = copy.deepcopy(state['numpy_private'][key])
    for key, stream in torch_streams.items():
        stream.set_state(state['torch_private'][key])


class FinalObservationBuffer:
    """Capture policy-only tensors BEFORE auto-reset; consume once per step.

    The simulator recorder must capture fresh terminal RGB and state for the
    provided environment rows. Returned reset observations remain untouched for
    the next policy call; replay gets a separate corrected next-observation.
    This class alone cannot verify a camera's physical timestamp.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.pending = None

    def capture(self, policy_observation, env_ids):
        if self.pending is not None:
            raise RuntimeError('Unconsumed final observation')
        if not policy_observation:
            raise ValueError('Empty policy observation')
        first = next(iter(policy_observation.values()))
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=first.device)
        if ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel():
            raise ValueError('Invalid terminal environment IDs')
        n = first.shape[0]
        if bool(((ids < 0) | (ids >= n)).any()):
            raise ValueError('Terminal environment ID out of bounds')
        rows = {}
        for key, value in policy_observation.items():
            if value.shape[0] != n or value.device != first.device:
                raise ValueError('Inconsistent observation batch')
            rows[key] = value[ids].clone()
        self.pending = (ids.clone(), rows)

    def consume(self, returned_policy_observation, terminated, truncated):
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise ValueError('Termination flags must be boolean')
        if terminated.ndim != 1 or terminated.shape != truncated.shape:
            raise ValueError('Termination flag shape mismatch')
        done = terminated | truncated
        result = {k: v.clone() for k, v in returned_policy_observation.items()}
        if not result or any(v.shape[0] != done.numel() or v.device != done.device for v in result.values()):
            raise ValueError('Observation/flag batch or device mismatch')
        if not bool(done.any()):
            if self.pending is not None:
                raise RuntimeError('Stale final observation without a done flag')
            return result, ~terminated
        if self.pending is None:
            raise RuntimeError('Missing pre-reset final observation')
        ids, rows = self.pending
        if not torch.equal(ids.sort().values, done.nonzero().flatten()) or set(rows) != set(result):
            raise ValueError('Final observation rows/keys mismatch')
        for key, value in rows.items():
            if value.shape[1:] != result[key].shape[1:] or value.dtype != result[key].dtype:
                raise ValueError('Final observation shape/dtype mismatch')
            result[key][ids] = value
        self.clear()
        # Timeout bootstraps from terminal observation; real termination does not.
        return result, ~terminated

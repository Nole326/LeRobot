"""Experimental Evo-1 dual-view adapter for local, unpublished checkpoints.

Requires the Evo-1 code tree (``lerobot.policies.evo1``) and a local LeRobot
dataset snapshot that provides the training normalisation statistics. The
adapter speaks the same NPZ request/response contract as the other PickOrange
services so inputs and outputs can be audited identically.

It is deliberately *not* wired to a benchmark protocol: the task checkpoint is
not distributed with this repository, so no pinned revision can be claimed.
Use it for local integration only and report results as experimental.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

from common import load_protocol, publish, rng_state, save_arrays, sha256, timing

WEIGHT_FILES = ('config.json', 'model.safetensors')


def verify_weights(model):
    if not model.is_dir():
        raise ValueError('Use a directory holding the exported policy')
    missing = [name for name in WEIGHT_FILES if not (model / name).is_file()]
    if missing:
        raise ValueError(f'Missing policy files: {missing}')
    identity = {name: sha256(model / name) for name in WEIGHT_FILES}
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    if config.get('type') != 'evo1':
        raise ValueError('Expected an Evo-1 policy config')
    if config.get('chunk_size') != 50 or config.get('n_action_steps') != 50:
        raise ValueError('This adapter only supports the 50-step action chunk')
    return identity


def verify_code(root):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('Use an unmodified policy code tree')
    return head


def load_policy(model, code_root, dataset_root, device):
    if code_root is not None:
        sys.path.insert(0, str(code_root))
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.evo1.modeling_evo1 import EVO1Policy
    from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(model), local_files_only=True)
    cfg.device = str(device)
    policy = EVO1Policy.from_pretrained(str(model), config=cfg, local_files_only=True, strict=True)
    policy.eval()
    dataset = LeRobotDataset(repo_id='local/pickorange', root=str(dataset_root), video_backend='pyav')
    preprocess, postprocess = make_evo1_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    return policy, preprocess, postprocess, torch


def predict(loaded, data, instruction):
    policy, preprocess, postprocess, torch = loaded
    q = np.asarray(data['joint_motor'], dtype=np.float32).reshape(1, 6)
    views = {}
    for cam in ('front', 'wrist'):
        image = np.asarray(data[cam][0])
        if image.shape != (480, 640, 3):
            raise ValueError('Expected two 480x640 RGB views')
        if image.dtype != np.uint8:
            raise ValueError('Views must be uint8 RGB')
        views[cam] = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0)
    batch = {'observation.images.front': views['front'], 'observation.images.wrist': views['wrist'],
             'observation.state': torch.from_numpy(q), 'task': [instruction]}
    with torch.inference_mode():
        actions = postprocess(policy.predict_action_chunk(preprocess(batch)))[0].float().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float64).reshape(50, 6)
    if not np.isfinite(actions).all():
        raise ValueError('Non-finite model output')
    return actions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--code-root', type=Path)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--wall-seconds', type=float, default=3600)
    p.add_argument('--run-service', action='store_true')
    p.add_argument('--allow-unpinned-code', action='store_true')
    a = p.parse_args(argv)
    cfg = load_protocol(a.protocol)
    if not a.run_service or not np.isfinite(a.wall_seconds) or a.wall_seconds <= 0:
        p.error('Explicit --run-service and a positive wall budget are required')
    if cfg['action_hz'] != 30 or cfg['chunk_length'] != 50:
        p.error('Pick a 30 Hz / 50-step protocol so the action clock matches the checkpoint')
    if a.code_root is not None and not a.allow_unpinned_code:
        commit = verify_code(a.code_root.resolve())
    elif a.code_root is not None:
        commit = 'unpinned'
    else:
        p.error('Pass --code-root, or --allow-unpinned-code to record an unpinned import path')
        commit = 'unpinned'
    out = a.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'ipc').mkdir()
    started = time.monotonic()
    try:
        identity = verify_weights(a.model.resolve())
        policy, preprocess, postprocess, torch = load_policy(
            a.model.resolve(), a.code_root.resolve() if a.code_root else None,
            a.dataset_root.resolve(), a.device)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.save(rng_state(torch, random), out / 'model-rng-initial.pt')
        publish(out / 'service-ready.json', dict(status='model_loaded', protocol=cfg,
            experimental=True, policy_sha256=identity, code_commit=commit,
            dataset_root=str(a.dataset_root.resolve()), service_sha256=sha256(__file__),
            input_keys=['front', 'wrist', 'joint_motor', 'task_text']))
        hold, executed, horizon = timing(cfg)
        query = 0
        while time.monotonic()-started < a.wall_seconds and not (out / 'STOP').exists():
            request = out / 'ipc' / f'request-{query:06d}.npz'
            if not request.exists():
                time.sleep(.1)
                continue
            with np.load(request, allow_pickle=False) as data:
                begin = time.monotonic()
                actions = predict((policy, preprocess, postprocess, torch), data, cfg['instruction'])
                if str(a.device).startswith('cuda'):
                    torch.cuda.synchronize()
            temporary = out / 'model-rng-latest.new'
            torch.save(rng_state(torch, random), temporary)
            temporary.replace(out / 'model-rng-latest.pt')
            save_arrays(out / 'ipc' / f'response-{query:06d}.npz', actions_motor=actions,
                        inference_sec=np.asarray(time.monotonic()-begin), query_id=np.asarray(query),
                        model_rng_sha256=np.asarray(sha256(out / 'model-rng-latest.pt')))
            query += 1
            publish(out / 'service-progress.json', dict(requests=query))
        publish(out / 'service-summary.json', dict(status='exited', requests=query,
            hold_ticks=hold, executed_ticks=executed, physics_horizon=horizon))
    except BaseException:
        publish(out / 'service-error.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

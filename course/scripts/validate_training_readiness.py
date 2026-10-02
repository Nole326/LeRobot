"""Bounded CPU acceptance checks, not a training run or performance evaluation.

Run in an authorized application environment with CUDA_VISIBLE_DEVICES empty.
Outputs are private diagnostics. Only load checkpoints created by this script.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from training_contracts import FinalObservationBuffer, capture_rng, restore_rng


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_digest(value):
    """Stable tensor/state digest for independent-process comparison."""
    h = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            h.update(str((str(obj.dtype), tuple(obj.shape))).encode())
            h.update(obj.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(repr(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(type(obj).__name__.encode())
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


def final_observation_checks():
    buffer = FinalObservationBuffer()
    pre = {'state': torch.arange(12.).reshape(4, 3), 'rgb': torch.arange(48, dtype=torch.uint8).reshape(4, 3, 2, 2)}
    reset = {k: torch.zeros_like(v) for k, v in pre.items()}
    term, trunc = torch.tensor([False, True, False, True]), torch.tensor([True, False, False, True])
    buffer.capture(pre, [3, 0, 1])
    snapshot = {k: v.clone() for k, v in pre.items()}
    for v in pre.values():
        v.zero_()  # capture must own storage, not alias simulator buffers
    fixed, bootstrap = buffer.consume(reset, term, trunc)
    for k in fixed:
        assert torch.equal(fixed[k][[0, 1, 3]], snapshot[k][[0, 1, 3]])
        assert torch.equal(fixed[k][2], reset[k][2])
        assert not reset[k].any()
    assert bootstrap.tolist() == [True, False, True, False]
    rejected = 0
    for case in ('missing', 'stale', 'duplicate', 'wrong_rows'):
        buffer.clear()
        try:
            if case == 'missing':
                buffer.consume(reset, term, trunc)
            elif case == 'duplicate':
                buffer.capture(reset, [0, 0])
            elif case == 'stale':
                buffer.capture(reset, [0])
                buffer.consume(reset, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
            else:
                buffer.capture(reset, [0])
                buffer.consume(reset, term, trunc)
        except (ValueError, RuntimeError):
            rejected += 1
    assert rejected == 4
    return {'status': 'passed', 'vector_envs': 4, 'negative_cases_rejected': rejected,
            'covers': ['mixed timeout/termination', 'RGB/state row mapping', 'no alias', 'one-shot consume'],
            'real_simulator_camera_timing_validated': False}


def upstream_rng_audit():
    from lerobot.utils.random_utils import serialize_rng_state, deserialize_rng_state
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    random.gauss(0, 1)
    np.random.normal()  # each leaves an odd cached Gaussian
    saved = serialize_rng_state()
    expected = (random.gauss(0, 1), float(np.random.normal()), torch.rand(4))
    deserialize_rng_state(saved)
    actual = (random.gauss(0, 1), float(np.random.normal()), torch.rand(4))
    return {'python_gaussian_exact': expected[0] == actual[0],
            'numpy_gaussian_exact': expected[1] == actual[1],
            'numpy_absolute_error': abs(expected[1] - actual[1]),
            'torch_cpu_exact': torch.equal(expected[2], actual[2]),
            'scope': 'pinned upstream serializer; private streams/sampler not included'}


def run(args, out):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    assert not torch.cuda.is_available(), 'This validation must not allocate GPU resources'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = {'kind': 'CPU-readiness-validation-not-formal-training', 'final_observation': final_observation_checks(),
              'upstream_rng_audit': upstream_rng_audit(), 'cuda_validated': False}
    root = Path(args.dataset_root)
    dataset = LeRobotDataset(repo_id='local/' + root.name, root=root,
                             delta_timestamps={'action': [0., .1, .2, .3]})
    assert len(dataset) == 20 and dataset.meta.total_episodes == 1
    items = [dataset[i] for i in range(len(dataset))]
    for i, item in enumerate(items):
        assert item['action_is_pad'].tolist() == [i + j >= len(items) for j in range(4)]
    # A one-episode controlled motion sample cannot form a valid train/val split.
    report['data'] = {'frames': len(items), 'episodes': 1, 'all_padding_masks_correct': True,
                      'successful_expert_data': False, 'episode_disjoint_train_val_split_available': False}
    stats = {}
    for key in ('observation.state', 'action'):
        values = torch.stack([x[key] if key != 'action' else x[key][0] for x in items])
        stats[key] = {'mean': values.mean(0), 'std': values.std(0, unbiased=False).clamp_min(1e-3)}
    cfg = ACTConfig(input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        'observation.images.front': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        'observation.images.wrist': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        pretrained_backbone_weights=None, chunk_size=4, n_action_steps=1, dim_model=128,
        n_heads=4, dim_feedforward=256, n_encoder_layers=1, n_decoder_layers=1,
        n_vae_encoder_layers=1, device='cpu')

    def make_model():
        model = ACTPolicy(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.9)
        return model, opt, sched

    def batch_at(indices, normalization):
        selected = [items[int(i)] for i in indices]
        batch = {k: torch.stack([x[k] for x in selected]) for k in
                 ('observation.state', 'action', 'action_is_pad', 'observation.images.front', 'observation.images.wrist')}
        for key in ('observation.state', 'action'):
            batch[key] = (batch[key] - normalization[key]['mean']) / normalization[key]['std']
        for key in ('observation.images.front', 'observation.images.wrist'):
            batch[key] = torch.nn.functional.interpolate(batch[key], size=(64, 64), mode='bilinear', align_corners=False)
            batch[key] = (batch[key] - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        return batch

    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)
    ng = {'environment_placeholder': np.random.default_rng(271)}
    tg = {'sampler': torch.Generator().manual_seed(271)}
    model, opt, scheduler = make_model()
    permutation = torch.randperm(len(items), generator=tg['sampler'])
    cursor = 0

    def update(model, opt, scheduler, indices, normalization):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = model(batch_at(indices, normalization))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        scheduler.step()
        return loss.item()

    if args.fresh_resume_from:
        source = Path(args.fresh_resume_from)
        marker = json.loads((source / 'checkpoint.complete.json').read_text())
        assert digest(source / 'checkpoint.pt') == marker['sha256']
        saved = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
        reference = json.loads((source / 'fresh-process-reference.json').read_text())
        assert saved['step'] == marker['step'] == 2
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        for rel, sha in saved['dataset_files'].items():
            assert digest(root / rel) == sha
        restore_rng(saved['rng'], ng, tg)
        fresh_probe = {'py': [random.random(), random.gauss(0, 1)],
                       'np': [float(np.random.normal()), float(np.random.random())],
                       'torch': torch.rand(4).tolist(), 'private_np': ng['environment_placeholder'].random(4).tolist(),
                       'private_torch': torch.randperm(20, generator=tg['sampler']).tolist()}
        assert fresh_probe == reference['probe']
        selected = saved['permutation'][saved['cursor']:saved['cursor']+2]
        assert selected.tolist() == reference['indices']
        fresh_loss = update(model, opt, scheduler, selected, saved['normalization'])
        assert fresh_loss == reference['loss']
        assert tree_digest(model.state_dict()) == reference['model']
        assert tree_digest(opt.state_dict()) == reference['optimizer']
        assert tree_digest(scheduler.state_dict()) == reference['scheduler']
        return {'kind': 'fresh-process-CPU-checkpoint-resume', 'exact_match': True,
                'optimizer_calls': 1, 'cuda_validated': False}

    for _ in range(2):
        update(model, opt, scheduler, permutation[cursor:cursor+2], stats)
        cursor += 2
    random.gauss(0, 1)
    np.random.normal()
    checkpoint = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                  'step': 2, 'permutation': permutation, 'cursor': cursor, 'normalization': stats,
                  'rng': capture_rng(ng, tg), 'config': cfg.to_dict() if hasattr(cfg, 'to_dict') else str(cfg),
                  'dataset_files': {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    part = out / 'checkpoint.pt.partial'
    save_start = time.monotonic()
    with part.open('xb') as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    cp = out / 'checkpoint.pt'
    part.rename(cp)
    cp_sha = digest(cp)
    (out / 'checkpoint.complete.json').write_text(json.dumps({'sha256': cp_sha, 'step': 2}))
    save_s = time.monotonic() - save_start

    def probe():
        return {'py': [random.random(), random.gauss(0, 1)], 'np': [float(np.random.normal()), float(np.random.random())],
                'torch': torch.rand(4).tolist(), 'private_np': ng['environment_placeholder'].random(4).tolist(),
                'private_torch': torch.randperm(20, generator=tg['sampler']).tolist()}

    expected_probe = probe()
    indices = permutation[cursor:cursor+2]
    expected_loss = update(model, opt, scheduler, indices, stats)
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    del model, opt, scheduler, checkpoint
    # Constructor consumes randomness; restore only after constructing/loading.
    restored, restored_opt, restored_sched = make_model()
    assert digest(cp) == cp_sha
    loaded = torch.load(cp, map_location='cpu', weights_only=False)  # own local artifact only
    restored.load_state_dict(loaded['model'])
    restored_opt.load_state_dict(loaded['optimizer'])
    restored_sched.load_state_dict(loaded['scheduler'])
    assert loaded['step'] == 2 and loaded['cursor'] == cursor
    for rel, sha in loaded['dataset_files'].items():
        assert digest(root / rel) == sha
    restore_rng(loaded['rng'], ng, tg)
    actual_probe = probe()
    assert actual_probe == expected_probe
    actual_indices = loaded['permutation'][loaded['cursor']:loaded['cursor']+2]
    assert torch.equal(actual_indices, indices)
    actual_loss = update(restored, restored_opt, restored_sched, actual_indices, loaded['normalization'])
    assert actual_loss == expected_loss
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected_model.items())

    def same_tree(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
        return a == b

    assert same_tree(expected_opt, restored_opt.state_dict())
    assert same_tree(expected_scheduler, restored_sched.state_dict())
    reference = {'probe': expected_probe, 'indices': indices.tolist(), 'loss': expected_loss,
                 'model': tree_digest(expected_model), 'optimizer': tree_digest(expected_opt),
                 'scheduler': tree_digest(expected_scheduler)}
    (out / 'fresh-process-reference.json').write_text(json.dumps(reference, indent=2))
    del expected_model, expected_opt, loaded
    subprocess.run([sys.executable, __file__, '--dataset-root', str(root), '--output', str(out / 'fresh-process'),
                    '--fresh-resume-from', str(out)], check=True, timeout=180)
    fresh = json.loads((out / 'fresh-process/summary.json').read_text())
    assert fresh['status'] == 'passed' and fresh['exact_match']
    report['checkpoint'] = {'status': 'passed', 'same_next_batch_rng_loss_weights_optimizer_scheduler': True,
                            'sha256': cp_sha, 'bytes': cp.stat().st_size, 'save_and_hash_s': save_s,
                            'saved_step': 2, 'physical_optimizer_calls': 5, 'data_loader_workers': 0,
                            'fresh_process_exact_match': True, 'fresh_process_peak_rss_kib': fresh['peak_rss_kib'],
                            'private_env_stream_is_placeholder': True, 'simulation_state_saved': False}
    # Padded targets must not affect predictions/loss when VAE and dropout use the same RNG.
    batch = batch_at([18, 19], stats)
    changed = {k: v.clone() for k, v in batch.items()}
    changed['action'][changed['action_is_pad']] += 1000
    rng = capture_rng(ng, tg)
    restored.train()
    loss_a, info_a = restored(batch)
    restore_rng(rng, ng, tg)
    loss_b, info_b = restored(changed)
    padding_error = abs(loss_a.item()-loss_b.item())
    assert padding_error <= 1e-5
    restored.eval()
    restored.reset()
    with torch.no_grad():
        action = restored.select_action(batch)
    restored.reset()
    with torch.no_grad():
        repeated = restored.select_action(batch)
    assert torch.equal(action, repeated) and torch.isfinite(action).all()
    report['ACT'] = {'padded_target_loss_difference': padding_error, 'padding_invariance': True,
                     'padding_reduction': 'upstream mean over full chunk, not valid-element mean',
                     'post_reset_inference_repeatable': True, 'actions_applied_to_simulator': False,
                     'parameters': sum(p.numel() for p in restored.parameters())}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--fresh-resume-from', default=None)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        result = run(args, out)
        result.update(status='passed', elapsed_s=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (out/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        (out/'error.txt').write_text(traceback.format_exc())
        raise

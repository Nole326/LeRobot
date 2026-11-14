"""Local PickOrange dual-view inference, defaulting to GR00T N1.5."""
import argparse
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from common import (BACKEND_REVISION, DEFAULT_PROTOCOL, load_protocol, predict_groot,
                    publish, rng_state, save_arrays, sha256, verify_backend)


def load_policy(model, backend_root, device):
    verify_backend(backend_root, BACKEND_REVISION)
    sys.path.insert(0, str(backend_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'liftcube/src'))
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.policy import Gr00tPolicy
    from starter.models.gr00t import load_groot_weights
    cfg = load_data_config('so100_dualcam')
    if list(cfg.video_keys) != ['video.front', 'video.wrist'] or len(cfg.action_indices) != 16:
        raise ValueError('Pinned dual-camera / 16-action recipe required')
    class CheckedPolicy(Gr00tPolicy):
        def _load_model(self, model_path):
            self.model = load_groot_weights(GR00T_N1_5, model_path, self.device)
    return CheckedPolicy(model_path=str(model), embodiment_tag='new_embodiment',
        modality_config=cfg.modality_config(), modality_transform=cfg.transform(),
        denoising_steps=4, device=device)


def verify_model(model):
    # Only this exact task export is eligible for the historical protocol label.
    expected = {
        'config.json': '6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526',
        'model.safetensors.index.json': 'bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746',
        'experiment_cfg/metadata.json': 'e86d145bc30f17969002ecac5f4b3e256717e63a88486b2cb58a37d7c0281fda',
        'model-00001-of-00002.safetensors': '0e65895a7b6c052e58b679bdd68bff54a68694a81f38e23d3e03a92059b3600b',
        'model-00002-of-00002.safetensors': '7803043990838e3296f54b518c9e81987216f532660d25208d130afb4d34c403'}
    for name, digest in expected.items():
        if sha256(model / name) != digest:
            raise ValueError(f'Checkpoint identity mismatch: {name}')
    return expected


def main(argv=None, *, model_loader=load_policy, model_verifier=verify_model,
         predict=predict_groot, model_revision=None, default_protocol=DEFAULT_PROTOCOL):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=default_protocol)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--backend-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--wall-seconds', type=float, default=3600)
    p.add_argument('--run-service', action='store_true')
    a = p.parse_args(argv)
    cfg = load_protocol(a.protocol)
    from common import MODEL_REVISION
    if cfg['model_revision'] != (model_revision or MODEL_REVISION):
        p.error('Protocol and model entry point differ')
    if not a.run_service or not np.isfinite(a.wall_seconds) or a.wall_seconds <= 0:
        p.error('Explicit --run-service and a positive wall budget are required')
    import os
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        p.error('Pre-cache backbone/tokenizer assets and set HF_HUB_OFFLINE=1 before loading')
    out = a.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'ipc').mkdir()
    started = time.monotonic()
    try:
        identity = model_verifier(a.model.resolve())
        import torch
        policy = model_loader(a.model.resolve(), a.backend_root.resolve(), a.device)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.save(rng_state(torch, random), out / 'model-rng-initial.pt')
        ready = dict(status='model_loaded', protocol=cfg, model_sha256=identity,
                     service_sha256=sha256(__file__),
                     input_keys=['front', 'wrist', 'joint_motor', 'task_text'])
        if cfg.get('experimental'):
            ready['experimental'] = True
        publish(out / 'service-ready.json', ready)
        query = 0
        while time.monotonic()-started < a.wall_seconds and not (out / 'STOP').exists():
            request = out / 'ipc' / f'request-{query:06d}.npz'
            if not request.exists():
                time.sleep(.1)
                continue
            with np.load(request, allow_pickle=False) as data:
                begin = time.monotonic()
                with torch.inference_mode():
                    actions = predict(policy, data, cfg['instruction'])
                if actions.shape != (cfg['chunk_length'], 6) or not np.isfinite(actions).all():
                    raise ValueError('Decoded policy chunk differs from protocol')
                if str(a.device).startswith('cuda'):
                    torch.cuda.synchronize(a.device)
            # RNG is durable before the response becomes visible to the evaluator.
            temporary = out / 'model-rng-latest.new'
            torch.save(rng_state(torch, random), temporary)
            temporary.replace(out / 'model-rng-latest.pt')
            save_arrays(out / 'ipc' / f'response-{query:06d}.npz', actions_motor=actions,
                        inference_sec=np.asarray(time.monotonic()-begin), query_id=np.asarray(query),
                        model_rng_sha256=np.asarray(sha256(out / 'model-rng-latest.pt')))
            query += 1
            publish(out / 'service-progress.json', dict(requests=query))
        publish(out / 'service-summary.json', dict(status='exited', requests=query))
    except BaseException:
        publish(out / 'service-error.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

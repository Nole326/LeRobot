"""Experimental PickOrange inference with the Evo-1 project's native LeRobot backend."""
import json
from pathlib import Path
import sys

import numpy as np

from common import EVO1_BACKEND, EVO1_PROTOCOL, EVO1_REVISION, lerobot_batch, sha256, verify_backend
from service import main as run_service

VIEWS = ('front', 'wrist')
EXPORT_FILES = ('config.json', 'model.safetensors', 'meta/stats.json')
NORM_MODES = {'VISUAL': 'IDENTITY', 'STATE': 'MIN_MAX', 'ACTION': 'MIN_MAX'}


def verify_model(model):
    missing = [name for name in EXPORT_FILES if not (model / name).is_file()]
    if missing:
        raise ValueError(f'Missing Evo-1 checkpoint files: {missing}')
    return {name: sha256(model / name) for name in EXPORT_FILES}


def validate_config(cfg):
    if (cfg.n_obs_steps != 1 or cfg.chunk_size != 50 or cfg.n_action_steps != 50
            or cfg.num_inference_timesteps != 32 or cfg.empty_cameras != 0 or cfg.max_views < 2
            or min(cfg.max_state_dim, cfg.max_action_dim) < 6
            or cfg.task_field != 'task' or cfg.embodiment_id_field is not None
            or tuple(cfg.image_features) != tuple(f'observation.images.{cam}' for cam in VIEWS)
            or any(tuple(feature.shape) != (3, 480, 640) for feature in cfg.image_features.values())
            or tuple(getattr(cfg.input_features.get('observation.state'), 'shape', ())) != (6,)
            or tuple(getattr(cfg.output_features.get('action'), 'shape', ())) != (6,)):
        raise ValueError('Evo-1 checkpoint config differs from the PickOrange protocol')
    if any(cfg.normalization_mapping.get(key) != mode for key, mode in NORM_MODES.items()):
        raise ValueError('Expected Evo-1 MIN_MAX state/action and native visual preprocessing')


def load_stats(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    stats = {}
    for key in ('observation.state', 'action'):
        fields = raw.get(key, {})
        low = np.asarray(fields.get('min'), dtype=np.float32)
        high = np.asarray(fields.get('max'), dtype=np.float32)
        if any(x.shape != (6,) or not np.isfinite(x).all() for x in (low, high)):
            raise ValueError(f'Expected six finite min/max statistics for {key}')
        if (high < low).any():
            raise ValueError(f'Invalid normalization range for {key}')
        stats[key] = {'min': low, 'max': high}
    return stats


def load_policy(model, backend_root, device):
    verify_backend(backend_root, EVO1_BACKEND)
    source = backend_root / 'evo1_lerobot'
    sys.path.insert(0, str(source))
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.evo1 import modeling_evo1
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config
    from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors

    if not Path(modeling_evo1.__file__).resolve().is_relative_to(source):
        raise ValueError('Evo-1 import escaped the selected backend')
    cfg = PreTrainedConfig.from_pretrained(str(model), local_files_only=True)
    if not isinstance(cfg, Evo1Config):
        raise ValueError('Expected an Evo-1 checkpoint')
    validate_config(cfg)
    stats = load_stats(model / 'meta/stats.json')
    cfg.device = str(device)
    policy = modeling_evo1.EVO1Policy.from_pretrained(
        str(model), config=cfg, local_files_only=True, strict=True)
    pre, post = make_evo1_pre_post_processors(cfg, dataset_stats=stats)
    return policy, pre, post


def check_action_chunk(actions):
    if actions.shape != (50, 6) or not np.isfinite(actions).all():
        raise ValueError('Expected a finite decoded Evo-1 chunk [50,6]')
    return actions


def predict(loaded, data, instruction):
    policy, pre, post = loaded
    batch = lerobot_batch(data, instruction, VIEWS)
    policy.reset()
    actions = post(policy.predict_action_chunk(pre(batch)).float())
    if actions.shape != (1, 50, 6):
        raise ValueError('Expected a finite decoded Evo-1 chunk [1,50,6]')
    return check_action_chunk(actions.detach().cpu().numpy()[0])


def main(argv=None):
    run_service(argv, model_loader=load_policy, model_verifier=verify_model,
                predict=predict, model_revision=EVO1_REVISION, default_protocol=EVO1_PROTOCOL)


if __name__ == '__main__':
    main()

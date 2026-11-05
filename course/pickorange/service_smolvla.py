"""PickOrange SmolVLA with both real views and the export's saved statistics."""
import dataclasses
import json
import os
from pathlib import Path
import sys

from candidate_assets import SMOL, verify_files
from common import SMOL_BACKEND, SMOL_REVISION, lerobot_batch, verify_backend
from service import main


def verify_model(model):
    return verify_files(model, SMOL)


def compatible_config(raw, fields):
    """Remove only known disabled export options absent in the pinned backend."""
    raw = dict(raw)
    if raw.pop('type') != 'smolvla':
        raise ValueError('Expected SmolVLA task export')
    disabled = {'use_peft': False, 'compile_model': False, 'compile_mode': 'max-autotune'}
    for key in set(raw) - set(fields):
        if key not in disabled or type(raw[key]) is not type(disabled[key]) or raw[key] != disabled[key]:
            raise ValueError(f'Unsupported export option: {key}')
        del raw[key]
    return raw


def load_policy(model, backend_root, device):
    verify_backend(backend_root, SMOL_BACKEND)
    sys.path.insert(0, str(backend_root / 'src'))
    import draccus
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    import lerobot.policies.smolvla.processor_smolvla  # Register saved processor types.
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
                                             policy_action_to_transition, transition_to_policy_action)
    from safetensors.torch import load_model
    if not Path(modeling_smolvla.__file__).resolve().is_relative_to(backend_root):
        raise ValueError('SmolVLA import escaped the selected backend')
    raw = compatible_config(json.loads((model/'config.json').read_text()),
                            {f.name for f in dataclasses.fields(SmolVLAConfig)})
    cfg = draccus.decode(SmolVLAConfig, raw)
    if (cfg.n_obs_steps != 1 or cfg.chunk_size != 50 or cfg.n_action_steps != 50
            or cfg.num_steps != 10 or cfg.empty_cameras != 0
            or cfg.adapt_to_pi_aloha or cfg.use_delta_joint_actions_aloha
            or cfg.rtc_config is not None
            or tuple(cfg.image_features) != ('observation.images.front', 'observation.images.wrist')
            or cfg.input_features['observation.state'].shape != (6,)
            or cfg.output_features['action'].shape != (6,)):
        raise ValueError('SmolVLA task interface differs')
    vlm = Path(os.environ['SMOLVLA_VLM_DIR']).resolve(strict=True)
    if not vlm.is_dir():
        raise ValueError('Use a local SmolVLM config/tokenizer directory')
    cfg.device, cfg.load_vlm_weights, cfg.vlm_model_name = 'cpu', False, str(vlm)
    policy = modeling_smolvla.SmolVLAPolicy(cfg)
    load_model(policy, model/'model.safetensors', strict=True, device='cpu')
    policy.eval().to(device)
    pre = PolicyProcessorPipeline.from_pretrained(str(model), config_filename='policy_preprocessor.json',
        overrides={'tokenizer_processor': {'tokenizer_name': str(vlm)},
                   'device_processor': {'device': str(device)}},
        to_transition=batch_to_transition, to_output=transition_to_batch, local_files_only=True)
    post = PolicyProcessorPipeline.from_pretrained(str(model), config_filename='policy_postprocessor.json',
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action, local_files_only=True)
    return policy, pre, post


def predict(loaded, data, instruction):
    policy, pre, post = loaded
    batch = lerobot_batch(data, instruction, ('front', 'wrist'))
    # Each request supplies the only required observation. Discard all old queues.
    policy.reset()
    actions = post(policy.predict_action_chunk(pre(batch)).float())
    if actions.shape != (1, 50, 6):
        raise ValueError('Expected decoded SmolVLA chunk [1,50,6]')
    return actions.detach().cpu().numpy()[0]


if __name__ == '__main__':
    main(model_loader=load_policy, model_verifier=verify_model,
         predict=predict, model_revision=SMOL_REVISION)

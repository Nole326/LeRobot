"""Run the pinned dual-camera N1.6 PickOrange export via its native decoder."""
import json
from pathlib import Path
import sys

import numpy as np

from candidate_assets import N16, verify_files
from common import N16_BACKEND, N16_REVISION, validate_request, verify_backend
from service import main


def verify_model(model):
    digests = verify_files(model, N16)
    config = json.loads((model/'config.json').read_text())
    processor = json.loads((model/'processor_config.json').read_text())['processor_kwargs']
    modalities = processor['modality_configs']['new_embodiment']
    for key, keys, indices in (
        ('video', ['front', 'wrist'], [0]),
        ('state', ['single_arm', 'gripper'], [0]),
        ('action', ['single_arm', 'gripper'], list(range(16))),
        ('language', ['annotation.human.task_description'], [0]),
    ):
        if modalities[key]['modality_keys'] != keys or modalities[key]['delta_indices'] != indices:
            raise ValueError(f'N1.6 {key} layout differs')
    if ([c['rep'] for c in modalities['action']['action_configs']] != ['RELATIVE', 'ABSOLUTE']
            or [c['type'] for c in modalities['action']['action_configs']] != ['NON_EEF']*2
            or config['model_type'] != 'Gr00tN1d6' or not config['use_relative_action']
            or config['action_horizon'] != 50 or config['num_inference_timesteps'] != 4):
        raise ValueError('Wrong N1.6 action layout or denoising recipe')
    return digests


def observation(data, instruction):
    q = validate_request(data).astype(np.float32)
    return {
        'video': {cam: data[cam][:, None].copy() for cam in ('front', 'wrist')},
        'state': {'single_arm': q[:, None, :5], 'gripper': q[:, None, 5:]},
        'language': {'annotation.human.task_description': [[instruction]]},
    }


def predict(policy, data, instruction):
    actions, _ = policy.get_action(observation(data, instruction))
    arm, jaw = actions['single_arm'], actions['gripper']
    if arm.shape != (1, 16, 5) or jaw.shape != (1, 16, 1):
        raise ValueError('Wrong decoded N1.6 action layout')
    # The native processor has already undone relative-arm normalization.
    return np.concatenate([arm[0], jaw[0]], axis=1)


def load_policy(model, backend_root, device):
    verify_backend(backend_root, N16_BACKEND)
    import torch
    sys.path.insert(0, str(backend_root))
    import gr00t.model  # registers the fixed architecture and local processor
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.policy import BasePolicy
    from transformers import AutoModel, AutoProcessor
    if not Path(gr00t.model.__file__).resolve().is_relative_to(backend_root):
        raise ValueError('N1.6 backend import escaped the selected root')

    class CheckedPolicy(Gr00tPolicy):
        def __init__(self):
            BasePolicy.__init__(self, strict=True)
            native, info = AutoModel.from_pretrained(model, local_files_only=True,
                output_loading_info=True, use_safetensors=True, trust_remote_code=False)
            if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'Incomplete N1.6 model load: {info}')
            self.model = native.eval().to(device=device, dtype=torch.bfloat16)
            self.processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
            self.processor.eval()
            self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
            self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
            self.collate_fn = self.processor.collator
            self.language_key, = self.modality_configs['language'].modality_keys
    return CheckedPolicy()


if __name__ == '__main__':
    main(model_loader=load_policy, model_verifier=verify_model,
         predict=predict, model_revision=N16_REVISION)

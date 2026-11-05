"""Separate LeRobot PI05 task checkpoint with its own saved processors."""
import json
import os
from pathlib import Path
import sys

from candidate_assets import PI05, verify_files
from common import PI05_BACKEND, PI05_REVISION, lerobot_batch, verify_backend
from service import main

VIEWS = ('wrist', 'front')  # Order from this checkpoint's input_features.
JOINT_KEYS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
              'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')


def verify_model(model):
    digests = verify_files(model, PI05)
    config = json.loads((model/'config.json').read_text())
    keys = config['input_features']
    views = tuple(k for k in keys if k.startswith('observation.images.'))
    if (config['type'] != 'pi05' or config['use_relative_actions']
            or config['chunk_size'] != 50 or config['num_inference_steps'] != 10
            or tuple(config['action_feature_names']) != JOINT_KEYS
            or views != tuple('observation.images.' + cam for cam in VIEWS)
            or keys['observation.state']['shape'] != [6]
            or config['output_features']['action']['shape'] != [6]):
        raise ValueError('PI05 task checkpoint contract differs')
    return digests


def processors(model, device):
    from lerobot.processor import (PolicyProcessorPipeline, policy_action_to_transition,
                                   transition_to_policy_action)
    tokenizer = Path(os.environ['PI05_TOKENIZER_DIR']).resolve(strict=True)
    if not tokenizer.is_dir():
        raise ValueError('Use the locally authorized PaliGemma tokenizer')
    pre = PolicyProcessorPipeline.from_pretrained(str(model), config_filename='policy_preprocessor.json',
        overrides={'tokenizer_processor': {'tokenizer_name': str(tokenizer)},
                   'device_processor': {'device': str(device)}}, local_files_only=True)
    post = PolicyProcessorPipeline.from_pretrained(str(model), config_filename='policy_postprocessor.json',
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action, local_files_only=True)
    return pre, post


def load_policy(model, backend_root, device):
    verify_backend(backend_root, PI05_BACKEND)
    sys.path.insert(0, str(backend_root / 'src'))
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import modeling_pi05
    from safetensors.torch import load_model
    if not Path(modeling_pi05.__file__).resolve().is_relative_to(backend_root):
        raise ValueError('PI05 backend import escaped the selected root')
    cfg = PreTrainedConfig.from_pretrained(str(model), local_files_only=True)
    cfg.device = 'cpu'
    policy = modeling_pi05.PI05Policy(cfg)
    # Native from_pretrained can return random/incomplete weights after errors.
    load_model(policy, model/'model.safetensors', strict=True, device='cpu')
    policy.eval().to(device)
    return policy, *processors(model, device)


def predict(loaded, data, instruction):
    policy, preprocess, postprocess = loaded
    batch = lerobot_batch(data, instruction, VIEWS)
    result = postprocess(policy.predict_action_chunk(preprocess(batch)).float())
    if result.shape != (1, 50, 6):
        raise ValueError('Wrong decoded PI05 action layout')
    return result.detach().cpu().numpy()[0]


if __name__ == '__main__':
    main(model_loader=load_policy, model_verifier=verify_model,
         predict=predict, model_revision=PI05_REVISION)

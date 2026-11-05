"""PickOrange dual-view OFT boundary; no public trained package is assumed."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from common import sha256, validate_request

BACKEND_REVISION = 'ac34367e52d52f967fa00b7bfef1ff3d2c9e253e'
VIEWS = ('front', 'wrist')


def read_package(manifest):
    manifest = Path(manifest).resolve()
    cfg = json.loads(manifest.read_text(encoding='utf-8'))
    contract = dict(task_id='LeIsaac-SO101-PickOrange-v0', trained_camera_keys=list(VIEWS),
                    state_units='so101_motor', action_units='so101_motor',
                    action_representation='absolute_joint_targets', normalization='bounds', chunk_size=30)
    if any(cfg.get(k) != v for k, v in contract.items()) or type(cfg.get('center_crop')) is not bool:
        raise ValueError('OFT package must declare the trained PickOrange dual-view motor contract')
    if any(not isinstance(cfg.get(k), str) or not cfg[k].strip()
           for k in ('model_path', 'source', 'revision', 'unnorm_key')):
        raise ValueError('OFT package source, revision, model path and statistics key are required')
    model = (manifest.parent / cfg['model_path']).resolve(strict=True)
    stats = model / 'dataset_statistics.json'
    files = []
    for key in ('action_head', 'proprio_projector'):
        file = (manifest.parent / cfg[key]['path']).resolve()
        if not file.is_file() or sha256(file) != cfg[key]['sha256']:
            raise ValueError(f'Missing or unverified trained OFT {key}')
        files.append(file)
    if not stats.is_file() or sha256(stats) != cfg.get('statistics_sha256'):
        raise ValueError('Missing or unverified checkpoint-specific OFT statistics')
    return cfg, model, files


class PickOrangeOFT:
    def __init__(self, model, processor, head, projector, predict_action, *, unnorm_key, center_crop):
        if head is None or projector is None or type(center_crop) is not bool:
            raise ValueError('OFT requires trained action/proprio modules and an explicit crop setting')
        self.model, self.processor = model, processor
        self.head, self.projector, self.predict_action = head, projector, predict_action
        self.cfg = SimpleNamespace(num_images_in_input=2, use_proprio=True, unnorm_key=unnorm_key,
                                   center_crop=center_crop, use_film=False)

    def predict(self, data, instruction):
        q = validate_request(data).astype(np.float32)
        payload = dict(full_image=data[VIEWS[0]][0].copy(),
                       additional_images=[data[VIEWS[1]][0].copy()], state=q[0])
        actions = self.predict_action(self.cfg, self.model, self.processor, payload, instruction,
                                      action_head=self.head, proprio_projector=self.projector,
                                      use_film=False)
        # Native OFT decodes its own BOUNDS statistics, yielding absolute motor targets.
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.shape != (30, 6) or not np.isfinite(chunk).all():
            raise ValueError('Expected finite native OFT [30,6] motor targets')
        return chunk

    @classmethod
    def load(cls, manifest, backend_root, device='cpu'):
        cfg, model, files = read_package(manifest)
        root = Path(backend_root).resolve(strict=True)
        if (subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
                != BACKEND_REVISION or subprocess.check_output(
                    ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()):
            raise ValueError('Use the unmodified pinned SO101 OFT backend')
        sys.path[:0] = [str(root), str(Path(__file__).resolve().parents[1] / 'liftcube/src')]
        import torch
        from starter.models.openvla_oft import check_statistics, load_local_processor
        from starter.models.backend_io import require_complete_load
        from prismatic.vla import constants
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from experiments.robot import openvla_utils

        key = cfg['unnorm_key']
        stats = json.loads((model / 'dataset_statistics.json').read_text(encoding='utf-8'))
        check_statistics(stats[key])
        if (constants.ACTION_DIM, constants.PROPRIO_DIM, constants.NUM_ACTIONS_CHUNK) != (6, 6, 30):
            raise ValueError('Select --robot_platform SO101 before importing the pinned OFT backend')
        if constants.ACTION_PROPRIO_NORMALIZATION_TYPE.value != 'bounds':
            raise ValueError('This OFT interface requires native BOUNDS decoding')
        if str(openvla_utils.DEVICE) != str(torch.device(device)):
            raise ValueError('Use an isolated OFT backend matching the requested device')
        processor = load_local_processor(model)
        vla, info = OpenVLAForActionPrediction.from_pretrained(
            str(model), local_files_only=True, torch_dtype=torch.bfloat16, output_loading_info=True)
        require_complete_load(info)
        vla.eval().requires_grad_(False).to(device)
        vla.norm_stats = stats
        vla.vision_backbone.set_num_images_in_input(2)
        head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=6)
        projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=6)
        for component, path in zip((head, projector), files):
            if path.suffix == '.safetensors':
                from safetensors.torch import load_file
                weights = load_file(str(path), device='cpu')
            else:
                weights = torch.load(path, map_location='cpu', weights_only=True)
            component.load_state_dict(weights, strict=True)
            component.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        return cls(vla, processor, head, projector, openvla_utils.get_vla_action,
                   unnorm_key=key, center_crop=cfg['center_crop'])

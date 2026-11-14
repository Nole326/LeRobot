"""PickOrange's explicit development protocol and local, non-pickle IPC."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

DEFAULT_PROTOCOL = Path(__file__).with_name('protocols') / 'groot-native-assisted.json'
MODEL_REVISION = '10f20142f96f0b9ef139aa90855a73e28a83ca74'
BACKEND_REVISION = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'
N16_REVISION = '4e59743e6c9c37b59c3f0350d3ad20b04b5f6d08'
N16_BACKEND = 'ead52833afbbf4243f8cd5e7664f48a94de03b19'
PI05_REVISION = 'bf4a13ed9d2ce17a96be5d82939191fa7ccf9ed5'
PI05_BACKEND = '30da8e687a6dfc617fcd94afc367ac7071c376ce'
SMOL_REVISION = '71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d'
SMOL_BACKEND = '58f70b6bd370864139a3795ac3497a9eae8c42d5'
EVO1_REVISION = 'fb177e4add5c7d803715be3af721fb08b1fb3f0f'
EVO1_INSTRUCTION = 'Pick three oranges and place them on the plate, then return the arm to rest.'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_protocol(cfg):
    required = dict(name='pickorange-groot-n15-native-assisted-v1', task_id='LeIsaac-SO101-PickOrange-v0',
        instruction='Pick three oranges and put them into the plate, then reset the arm to rest state.',
        model_repo='LightwheelAI/leisaac-pick-orange-v0', model_revision=MODEL_REVISION,
        backend_revision=BACKEND_REVISION, physics_hz=60, action_hz=60,
        chunk_length=16, execute_steps=16, denoising_steps=4, episode_seconds=60,
        dynamic_gripper_effort=True, software_action_clipping=False,
        robot_gravity_enabled=True, formal_course_evaluation=False, automatic_bc_admission=False)
    if cfg.get('name') == 'pickorange-groot-n16-native-assisted-v1':
        required.update(name=cfg['name'], model_repo='wsagi/GR00T-N1.6-PickOrange',
                        model_revision=N16_REVISION, backend_revision=N16_BACKEND)
    elif cfg.get('name') == 'pickorange-pi05-native-assisted-30hz-v1':
        required.update(name=cfg['name'], model_repo='Aikwed/pi05_pick_orange_full',
                        model_revision=PI05_REVISION, backend_revision=PI05_BACKEND,
                        action_hz=30, chunk_length=50, denoising_steps=10)
    elif cfg.get('name') == 'pickorange-smolvla-native-assisted-30hz-v1':
        required.update(name=cfg['name'], model_repo='edge-inference/smolvla-so101-pick-orange',
                        model_revision=SMOL_REVISION, backend_revision=SMOL_BACKEND,
                        action_hz=30, chunk_length=50, denoising_steps=10)
    elif cfg.get('name') == 'pickorange-evo1-native-assisted-30hz-v1':
        required.update(name=cfg['name'], model_repo='MINT-SJTU/Evo-1', instruction=EVO1_INSTRUCTION,
                        model_revision=EVO1_REVISION, backend_revision=EVO1_REVISION,
                        action_hz=30, chunk_length=50, execute_steps=50, denoising_steps=32)
    for key, value in required.items():
        if type(cfg.get(key)) is not type(value) or cfg[key] != value:
            raise ValueError(f'Unsupported protocol value: {key}')
    seeds = cfg.get('scene_seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct integer scene seeds are required')
    if seeds != list(range(17, 117)):
        raise ValueError('This protocol freezes seeds 17..116; use --episodes only for a smoke prefix')
    if not isinstance(cfg.get('instruction'), str) or not cfg['instruction'].strip():
        raise ValueError('Task instruction is required')
    return cfg


def timing(cfg):
    hold = cfg['physics_hz'] // cfg['action_hz']
    return hold, hold * cfg['execute_steps'], cfg['physics_hz'] * cfg['episode_seconds']


def verify_backend(root, revision):
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if actual != revision or dirty:
        raise ValueError('Use the unmodified pinned model backend')


def close_resources(writer, env, app):
    # SimulationApp.close() may exit the process: flush curves before shutdown.
    try:
        if writer is not None:
            writer.close()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix('.new')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.new')
    with temporary.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def wait_file(path, timeout, error=None, stopped=None):
    end = time.monotonic() + timeout
    while not Path(path).exists():
        if error is not None and Path(error).exists():
            raise RuntimeError(f'Peer failed; see {Path(error).name}')
        if stopped is not None and Path(stopped).exists():
            raise RuntimeError('Peer stopped before producing the requested file')
        if time.monotonic() >= end:
            raise TimeoutError(Path(path).name)
        time.sleep(.1)


def validate_request(data):
    if set(data) != {'front', 'wrist', 'joint_motor'}:
        raise ValueError('Model input whitelist is front, wrist and measured joints only')
    q = np.asarray(data['joint_motor'])
    if q.shape != (1, 6) or not np.isfinite(q).all():
        raise ValueError('Expected finite ordered [1,6] motor feedback')
    for cam in ('front', 'wrist'):
        image = np.asarray(data[cam])
        if image.shape != (1, 480, 640, 3) or image.dtype != np.uint8:
            raise ValueError('Expected real [1,480,640,3] uint8 RGB views')
    return q.astype(np.float64, copy=True)


def lerobot_batch(data, instruction, views):
    """Real dual RGB and measured joints only; native processors handle batching."""
    import torch
    q = validate_request(data)
    if set(views) != {'front', 'wrist'} or len(views) != 2:
        raise ValueError('Both trained camera slots must be real views')
    batch = {'observation.state': torch.tensor(q[0], dtype=torch.float32), 'task': instruction}
    for cam in views:
        batch['observation.images.'+cam] = torch.from_numpy(data[cam][0].copy()).permute(2,0,1).float()/255
    return batch


def predict_groot(policy, data, instruction):
    q = validate_request(data)
    payload = {'state.single_arm': q[:, :5], 'state.gripper': q[:, 5:],
               'annotation.human.task_description': [instruction]}
    payload.update({'video.' + cam: np.asarray(data[cam]).copy() for cam in ('front', 'wrist')})
    result = policy.get_action(payload)
    arm, jaw = np.asarray(result['action.single_arm']), np.asarray(result['action.gripper'])
    if arm.shape != (16, 5) or jaw.shape != (16, 1):
        raise ValueError('Expected native GR00T action chunks [16,5] and [16,1]')
    actions = np.concatenate([arm, jaw], axis=1)
    if not np.isfinite(actions).all():
        raise ValueError('Non-finite model output')
    # Deliberately no software clamp: this protocol retains native physical stops.
    return actions


def rng_state(torch, random):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

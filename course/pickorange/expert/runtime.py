"""Public CLI boundary; checks and help do not initialize the simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
USD_SHA256 = '64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58'


def check_integrity():
    manifest = json.loads((HERE/'integrity.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        if name.endswith('teacher.py') or name.endswith('.json') or name == 'joint_contract.py':
            actual = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f'Control script/calibration differs: {name}')
    return dict(integrity_verified=True,
                check_scope='source_only', horizon_seconds=55,
                physics_hz=60, action_hz=30, actor_inputs=['front_rgb', 'wrist_rgb', 'joint_state'])


def prepare_args(replay=False):
    parser = argparse.ArgumentParser(description='Independent action-trajectory replay verification' if replay else 'Privileged state control script')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-simulation', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', help='Explicit scene seeds for reproducible evaluation')
    parser.add_argument('--min-free-mib', type=int, default=6144)
    if replay:
        parser.add_argument('--source', type=Path)
    # Parse our arguments before importing the simulator, including --help.
    preliminary, _ = parser.parse_known_args()
    if preliminary.check:
        print(json.dumps(check_integrity(), indent=2))
        return None, None
    if not preliminary.run_simulation:
        parser.error('Use --check or explicitly pass --run-simulation')
    if preliminary.out is None or (replay and preliminary.source is None):
        parser.error('--out and, for replay, --source are required')
    if not preliminary.seeds:
        parser.error('--seeds is required for simulation')
    if len(set(preliminary.seeds)) != len(preliminary.seeds) or preliminary.min_free_mib < 0:
        parser.error('Seeds must be unique; memory reserve must be nonnegative')
    check_integrity()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != 'cuda:0' or not args.enable_cameras:
        parser.error('Use --device cuda:0 --enable_cameras with one assigned visible GPU')
    args.out.mkdir(parents=True, exist_ok=False)
    # Keep the control modules first; reuse the public evaluator and common I/O.
    sys.path[1:1] = [str(HERE.parent), str(HERE.parents[1]/'scripts')]
    return args, AppLauncher

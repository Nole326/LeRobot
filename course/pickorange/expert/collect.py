"""Bounded sequential collection; control execution and replay verification each use a fresh process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

from runtime import HERE, check_integrity


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def qualified(result):
    videos = result.get('videos', {})
    return (result.get('success') is True and 0 < result['seconds'] <= 55
            and result.get('privileged_actor_inputs') is False
            and set(videos) == {'front', 'wrist'}
            and all(v['decoded_exact'] and v['frames'] > 0 for v in videos.values())
            and videos['front']['frames'] == videos['wrist']['frames'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-simulation', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=400)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_integrity(), indent=2))
        return
    if not args.run_simulation or args.out is None or not 0 < args.count <= args.max_attempts:
        parser.error('Require --run-simulation --out and 0 < count <= max-attempts')
    check_integrity()
    args.out.mkdir(parents=True, exist_ok=False)
    accepted = []
    completed = []
    try:
        for seed in range(args.start_seed, args.start_seed+args.max_attempts):
            if shutil.disk_usage(args.out).free < 4 * 1024**3:
                raise RuntimeError('Less than 4 GiB storage remains; preserve completed data')
            teacher, replay = [args.out/f'{kind}-{seed}' for kind in ('teacher', 'replay')]
            for script, output, extra in [
                ('verify_scenes.py', teacher, []),
                ('replay_success.py', replay, ['--source', str(teacher)])
            ]:
                command = [sys.executable, str(HERE/script), '--run-simulation', '--out', str(output),
                           '--seeds', str(seed), '--headless', '--enable_cameras', '--device', 'cuda:0', *extra]
                with (args.out/f'{script}-{seed}.log').open('w') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                if script == 'verify_scenes.py' and not read(teacher/'result.json')['passed']:
                    break
            teacher_ok = read(teacher/'result.json')['passed']
            result = read(replay/f'seed-{seed}'/'result.json') if teacher_ok else None
            admitted = qualified(result) if result is not None else False
            if admitted:
                accepted.append(dict(seed=seed, path=str(replay/f'seed-{seed}')))
            completed.append(dict(seed=seed, teacher_success=teacher_ok,
                                  replay_success=None if result is None else result['success'], admitted=admitted))
            publish(args.out/'dataset.json', dict(episodes=accepted, count=len(accepted),
                    requested=args.count, attempts=completed))
            publish(args.out/'status.json', dict(status='running', accepted=len(accepted), attempts=len(completed)))
            if len(accepted) == args.count:
                publish(args.out/'status.json', dict(status='completed', accepted=len(accepted)))
                return
        raise RuntimeError('Attempt budget exhausted; retain partial dataset and failure evidence')
    except BaseException as exc:
        publish(args.out/'status.json', dict(status='failed', accepted=len(accepted), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

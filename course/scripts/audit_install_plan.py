import json
from pathlib import Path
plan = json.loads(Path('/workspace/course/logs/bc-pip-plan.json').read_text())
names = {entry['metadata']['name'].lower().replace('_', '-') for entry in plan['install']}
assert not ({'torch', 'torchvision', 'torchaudio'} & names)
assert not any(name.startswith('nvidia-') for name in names)
constraints = [f"{entry['metadata']['name']}=={entry['metadata']['version']}" for entry in plan['install']]
Path('/workspace/course/bc-resolved.txt').write_text('\n'.join(sorted(constraints)) + '\n')
print(json.dumps({'packages_to_install': len(names), 'core_torch_cuda_replacement': False, 'names': sorted(names)}, indent=2))

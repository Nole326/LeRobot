import json
from pathlib import Path
import torch
assert torch.cuda.device_count() == 1
a = torch.arange(1024, device='cuda', dtype=torch.float32)
result = a @ a
assert torch.isfinite(result)
torch.cuda.synchronize()
report = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'devices': torch.cuda.device_count(), 'result': float(result)}
Path('/workspace/course/logs/cuda-smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

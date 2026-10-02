"""CPU analysis: a loose error bound alone cannot prove XY command tracking."""
import json
from pathlib import Path
import numpy as np

root = Path('/workspace/course/logs/ee-settle-20261002')
rows = json.loads((root/'trajectory.json').read_text())
movement = [r for r in rows if r['phase'] >= 0]
actual = np.asarray([r['actual_xyz_base_m'] for r in movement])
target = np.asarray([r['target_xyz_base_m'] for r in movement])
actual_range = np.ptp(actual[:, :2], axis=0)
target_range = np.ptp(target[:, :2], axis=0)
ratio = actual_range / target_range
report = {'kind': 'post-settle-axis-response-check',
          'target_xy_range_mm': (target_range*1000).tolist(),
          'actual_xy_range_mm': (actual_range*1000).tolist(),
          'range_ratio_xy': ratio.tolist(),
          'response_ratio_test_range': [0.8, 1.2],
          'both_axes_response_pass': bool(((ratio >= .8) & (ratio <= 1.2)).all()),
          'production_controller_ready': False,
          'interpretation': 'Settles safely, but X hardly follows; do not infer grasp accuracy from a 10mm bound on 2.4mm commands'}
(root/'axis-response.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

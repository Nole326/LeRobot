"""CPU-only unit tests of the validation adapter; no simulation or training."""
import json
from pathlib import Path
import numpy as np
from joint_contract import USD_LIMITS_DEG, MOTOR_LIMITS, rad_to_motor, motor_to_rad, JOINTS

values = np.deg2rad(np.stack([USD_LIMITS_DEG[:, 0], USD_LIMITS_DEG.mean(axis=1), USD_LIMITS_DEG[:, 1]]))
mapped = rad_to_motor(values)
expected = np.stack([MOTOR_LIMITS[:, 0], MOTOR_LIMITS.mean(axis=1), MOTOR_LIMITS[:, 1]])
assert np.allclose(mapped, expected, atol=2e-5)
recovered = motor_to_rad(mapped)
assert np.allclose(values, recovered, atol=1e-6)
invalid_cases = 0
for invalid in (np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)):
    try:
        rad_to_motor(invalid)
    except ValueError:
        invalid_cases += 1
assert invalid_cases == 3
report = {'status': 'passed', 'kind': 'CPU-unit-conversion-only', 'joint_order': JOINTS,
          'roundtrip_max_error_rad': float(np.max(np.abs(values-recovered))),
          'invalid_cases_rejected': invalid_cases, 'formal_training': False}
Path('/workspace/course/logs/joint-contract-cpu-20261002.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

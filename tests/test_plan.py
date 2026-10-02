"""Pure metadata checks: no Torch, simulation, devices or remote services."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_plan import inspect, JOINTS


def specified_plan():
    # Synthetic values, not a task prescription or verified configuration.
    plan = tomllib.loads((ROOT / 'configs/experiment-plan.toml').read_text(encoding='utf-8'))
    plan['policy'].update(observation_schema_id='test-obs-v1', task_condition='test instruction',
                          chunk_size=4, execute_steps=1)
    plan['control'].update(action_schema_id='test-action-v1', representation='absolute_joint',
                           frame='joint', components=JOINTS[:], units=['rad'] * 6,
                           lower=[-1.] * 6, upper=[1.] * 6, physics_dt_s=1/60,
                           physics_steps_per_action=6, policy_hz=10.)
    plan['data'].update(action_schema_id='test-action-v1', observation_schema_id='test-obs-v1')
    plan['training']['budget'] = 5
    plan['evaluation'].update(protocol_id='test-only', max_decisions=10, seed_manifest_sha256='a' * 64)
    return plan


class PlanTest(unittest.TestCase):
    def test_template_is_incomplete_not_invalid(self):
        p = tomllib.loads((ROOT / 'configs/experiment-plan.toml').read_text(encoding='utf-8'))
        report = inspect(p)
        self.assertEqual(report['errors'], [])
        self.assertIn('control.action_schema_id', report['pending'])

    def test_specified_plan_metadata_only(self):
        self.assertEqual(inspect(specified_plan()), {'errors': [], 'pending': []})

    def test_unknown_or_missing_keys(self):
        for section in ('policy', 'control', 'data', 'training', 'evaluation'):
            with self.subTest(section=section):
                p = specified_plan()
                p[section]['typo'] = 1
                self.assertTrue(inspect(p)['errors'])
        self.assertTrue(inspect({})['errors'])

    def test_schema_mismatch(self):
        for key in ('action_schema_id', 'observation_schema_id'):
            p = specified_plan()
            p['data'][key] = 'different'
            self.assertTrue(inspect(p)['errors'])

    def test_no_truth_state_alias(self):
        p = specified_plan()
        p['policy']['proprio_names'].append('object_position')
        self.assertTrue(inspect(p)['errors'])

    def test_joints_and_units_cannot_be_reordered_or_guessed(self):
        for key, value in (('components', list(reversed(JOINTS))), ('units', ['normalized'] * 6),
                           ('frame', 'table'), ('lower', [0.])):
            p = specified_plan()
            p['control'][key] = value
            self.assertTrue(inspect(p)['errors'])

    def test_delta_units(self):
        p = specified_plan()
        p['control'].update(representation='delta_ee', frame='table', components=['dx', 'gripper'],
                            units=['m', 'normalized'], lower=[-.01, 0.], upper=[.01, 1.])
        self.assertFalse(inspect(p)['errors'])
        p['control']['units'][0] = 'rad'
        self.assertTrue(inspect(p)['errors'])

    def test_timing_mismatch(self):
        p = specified_plan()
        p['control']['policy_hz'] = 20
        self.assertTrue(inspect(p)['errors'])

    def test_invalid_number_or_bound(self):
        for value in (float('nan'), float('inf'), True, '10', -1):
            p = specified_plan()
            p['control']['policy_hz'] = value
            self.assertTrue(inspect(p)['errors'])
        p['control']['lower'] = p['control']['upper']
        self.assertTrue(inspect(p)['errors'])

    def test_chunk_execution(self):
        p = specified_plan()
        p['policy']['execute_steps'] = 5
        self.assertTrue(inspect(p)['errors'])

    def test_split_and_budget_units(self):
        for section, key, value in (('data', 'split_unit', 'frame'),
                                     ('data', 'statistics_split', 'all'),
                                     ('training', 'phase', 'online_rl'),
                                     ('training', 'budget', True),
                                     ('evaluation', 'seed_manifest_sha256', '123')):
            p = specified_plan()
            p[section][key] = value
            self.assertTrue(inspect(p)['errors'])

    def test_bad_types_never_crash(self):
        original = specified_plan()
        for section, values in original.items():
            if not isinstance(values, dict):
                continue
            for key in values:
                for bad in (None, {}, [[]], True):
                    p = copy.deepcopy(original)
                    p[section][key] = bad
                    with self.subTest(section=section, key=key, value=bad):
                        self.assertTrue(inspect(p)['errors'])

    def test_cli_no_optional_imports(self):
        # -S excludes site-packages: the command must still run.
        result = subprocess.run([sys.executable, '-S', '-B', str(ROOT/'scripts/inspect_plan.py'),
                                 str(ROOT/'configs/experiment-plan.toml'), '--print-config'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['configuration_status'], 'incomplete')
        self.assertIn('NOT ASSESSED', report['training_readiness'])

    def test_invalid_toml_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.toml'
            path.write_text('[unclosed', encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/inspect_plan.py'), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()

"""CPU publication boundary checks; no simulator or policy loading."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]/'course/pickorange/expert'


class ExpertBoundaryTests(unittest.TestCase):
    def test_failed_capture_releases_encoder_and_decoder(self):
        command = '''
import io
import subprocess
from unittest.mock import Mock
from rgb_video import RGBVideo
encoder = Mock(stdin=io.BytesIO(), stdout=None)
encoder.poll.return_value = None
encoder.wait.side_effect = [subprocess.TimeoutExpired('encoder', 10), 0]
decoder = Mock(stdin=None, stdout=io.BytesIO())
decoder.poll.return_value = 0
video = RGBVideo.__new__(RGBVideo)
video.process, video.decoder = encoder, decoder
video.abort()
encoder.terminate.assert_called_once()
encoder.kill.assert_called_once()
decoder.terminate.assert_not_called()
assert encoder.stdin.closed and decoder.stdout.closed
encoder.poll.return_value = 0
video.abort()
encoder.terminate.assert_called_once()
'''
        subprocess.run([sys.executable, '-B', '-c', command], cwd=HERE, check=True)

    def test_check_and_help_without_simulator(self):
        for name in ['verify_scenes.py', 'replay_success.py', 'collect.py']:
            checked = subprocess.run([sys.executable, '-B', str(HERE/name), '--check'],
                                     text=True, capture_output=True, check=True)
            self.assertTrue(json.loads(checked.stdout)['integrity_verified'])
            subprocess.run([sys.executable, '-B', str(HERE/name), '--help'],
                           capture_output=True, check=True)

    def test_rejects_execution_without_opt_in(self):
        result = subprocess.run([sys.executable, '-B', str(HERE/'verify_scenes.py')],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--run-simulation', result.stderr)

    def test_requires_success_and_both_verified_camera_streams(self):
        # Keep generic module names from this standalone tool out of other tests.
        command = '''
from collect import qualified
base = dict(success=True, seconds=54, privileged_actor_inputs=False,
            videos={k: dict(decoded_exact=True, frames=1620) for k in ('front','wrist')})
assert qualified(base)
assert not qualified(dict(base, success=False))
assert not qualified(dict(base, seconds=56))
assert not qualified(dict(base, privileged_actor_inputs=True))
assert not qualified(dict(base, videos={'front': base['videos']['front']}))
assert not qualified(dict(base, videos=dict(base['videos'], wrist=dict(decoded_exact=False,frames=1620))))
assert not qualified(dict(base, videos=dict(base['videos'], wrist=dict(decoded_exact=True,frames=1619))))
'''
        subprocess.run([sys.executable, '-B', '-c', command], cwd=HERE, check=True)


if __name__ == '__main__':
    unittest.main()

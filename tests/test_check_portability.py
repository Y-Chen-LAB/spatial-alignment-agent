"""Check the documented rebase command against Linux task paths."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_portability.py'


class PortabilityTests(unittest.TestCase):
    def test_linux_rebase_is_dry_by_default_and_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / 'benchmarks/spatial_alignment/tasks/example/task.toml'
            task.parent.mkdir(parents=True)
            original = ('slices = ["a=/users/ychen5/p2b-data/paper2bench-data/slices/a.h5ad", '
                        '"b=benchmarks/spatial_alignment/compiled_data/b.h5ad"]\n')
            task.write_text(original)
            args = [sys.executable, str(SCRIPT), '--data-root', '/scratch/inputs']
            dry = subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)
            self.assertIn('would rewrite 1 task files', dry.stdout)
            self.assertEqual(task.read_text(), original)
            subprocess.run(args + ['--apply'], cwd=root, check=True, capture_output=True)
            self.assertEqual(task.read_text(), original.replace(
                '/users/ychen5/p2b-data/paper2bench-data', '/scratch/inputs'))

"""Mac application workers must execute Python, rather than opening another GUI."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shared.external_process import normalize_slicer_python_launch
from shared.process_group import grouped_launch_command


class MacPythonWorkerTests(unittest.TestCase):
    def test_grouped_worker_executes_bundled_python_with_space_in_path(self):
        with tempfile.TemporaryDirectory(prefix='Slicer worker ') as folder:
            contents = Path(folder) / 'Slicer.app/Contents'
            (contents / 'MacOS').mkdir(parents=True)
            (contents / 'bin').mkdir()
            python = contents / 'bin/PythonSlicer'
            python.symlink_to(sys.executable)
            app = contents / 'MacOS/Slicer'
            app.write_text('This GUI executable must never run')
            code = 'import json,sys; print(json.dumps(sys.argv[1:]))'
            program, args, _grouped = grouped_launch_command(str(app), ['--launch', 'PythonSlicer', '-c', code, 'value with spaces'])
            result = subprocess.run([program, *args], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), ['value with spaces'])

    def test_missing_mac_python_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            app = Path(folder) / 'Slicer.app/Contents/MacOS/Slicer'
            with self.assertRaises(FileNotFoundError):
                normalize_slicer_python_launch(str(app), ['--launch', 'PythonSlicer', '-u', 'worker.py'])

    def test_ctk_and_non_worker_launches_keep_arguments(self):
        for app, args in [('/opt/Slicer/Slicer', ['--launch', 'PythonSlicer', 'worker.py']),
                          ('C:/Slicer/Slicer.exe', ['--launch', 'PythonSlicer', 'worker.py']),
                          ('/Applications/Slicer.app/Contents/MacOS/Slicer', ['--python-script', 'scene.py'])]:
            self.assertEqual(normalize_slicer_python_launch(app, args), (app, args))

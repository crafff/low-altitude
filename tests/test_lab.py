"""Isolated fixtures only. Never derive a cleanup target from program output."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

LAB = Path(__file__).resolve().parents[1] / "tools/lab.py"
SPEC = importlib.util.spec_from_file_location("lab_under_test", LAB)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LabTests(unittest.TestCase):
    def setUp(self):
        # Python owns this random temporary fixture; child stdout is never used here.
        self.temporary = tempfile.TemporaryDirectory(prefix="trc-lab-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.project.mkdir()
        (self.project / "src").mkdir()
        self.marker = self.project / "src/marker.txt"
        self.marker.write_text("DO NOT CHANGE")
        (self.project / "src/marker.json").write_text('{"keep": true}')
        self.external = Path(self.temporary.name) / "outside.txt"
        self.external.write_text("OUTSIDE")

    def invoke(self, code, *extra):
        call = subprocess.run([sys.executable, "-B", str(LAB), "--project", str(self.project), "run", "--label", "fixture", "--seconds", "5", "--disk-mib", "8", *extra, "--", "/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=15)
        result = json.loads(call.stdout) if call.stdout.strip() else None
        return call, result

    def test_source_readonly_output_writable_and_private(self):
        for private in ("legacy", "human", ".git", ".codex"):
            (self.project / private).mkdir()
            (self.project / private / "private-marker").write_text("PRIVATE")
        code = '''
import errno,json,os
from pathlib import Path
p=Path('/workspace/src/marker.json')
assert json.loads(p.read_text()) == {'keep': True}
for action in (lambda:p.write_text('bad'), lambda:p.unlink(), lambda:p.rename('/workspace/src/moved.json')):
    try: action()
    except OSError as e: assert e.errno in (errno.EROFS,errno.EACCES,errno.EPERM)
    else: raise AssertionError('source mutation allowed')
assert not Path('/workspace/legacy').exists()
assert not Path('/workspace/.git').exists()
assert not Path('/workspace/human').exists()
assert not Path('/workspace/.codex').exists()
assert 'TRC_TEST_PRIVATE_ENV' not in os.environ
assert not Path('/output/../status.json').exists()
Path('/output/result.json').write_text('{"ok":true}')
'''
        with mock.patch.dict(os.environ, {"TRC_TEST_PRIVATE_ENV": "DO_NOT_INHERIT"}):
            call, result = self.invoke(code)
        self.assertEqual(call.returncode, 0, call.stderr + str(result))
        directory = self.project / "runs" / result["id"]
        self.assertEqual(json.loads((directory / "artifacts/result.json").read_text()), {"ok": True})
        self.assertEqual(self.marker.read_text(), "DO NOT CHANGE")
        self.assertEqual(self.external.read_text(), "OUTSIDE")

    def test_absolute_outside_and_symlink_escape_rejected(self):
        code = f'''
from pathlib import Path
assert not Path({str(self.external)!r}).exists()
p=Path('/output/link'); p.symlink_to('/workspace/src/marker.json')
try: p.write_text('bad')
except OSError: pass
else: raise AssertionError('symlink bypass')
try: Path('/output/../escape.txt').write_text('bad')
except OSError: pass
else: raise AssertionError('parent escape')
'''
        call, result = self.invoke(code)
        self.assertEqual(call.returncode, 0, call.stderr + str(result))
        self.assertEqual(self.external.read_text(), "OUTSIDE")

    def test_separator_empty_stdout_and_dot_are_not_cleanup_paths(self):
        for text in ("", ".", "..", "/workspace"):
            call, result = self.invoke(f"print({text!r})")
            self.assertEqual(call.returncode, 0, call.stderr)
            self.assertEqual(result["state"], "succeeded")
            self.assertEqual(self.marker.read_text(), "DO NOT CHANGE")

    def test_nonzero_and_missing_executable_terminal_state(self):
        call, result = self.invoke("raise SystemExit(7)")
        self.assertEqual(call.returncode, 7)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["reason"], "nonzero_exit")
        call = subprocess.run([sys.executable, "-B", str(LAB), "--project", str(self.project), "run", "--label", "missing", "--", "/not/an/executable"], text=True, capture_output=True, timeout=10)
        self.assertNotEqual(call.returncode, 0)
        self.assertEqual(json.loads(call.stdout)["state"], "failed")
        self.assertEqual(json.loads(call.stdout)["reason"], "nonzero_exit")

    def test_runtime_is_explicitly_readable_but_readonly(self):
        dependencies = Path(self.temporary.name) / "dependencies"
        dependencies.mkdir()
        marker = dependencies / "dependency.txt"
        marker.write_text("DEPENDENCY")
        code = f'''from pathlib import Path
p=Path({str(marker)!r})
assert p.read_text()=='DEPENDENCY'
try: p.write_text('bad')
except OSError: pass
else: raise AssertionError('runtime writable')
'''
        call, result = self.invoke(code, "--runtime", str(dependencies))
        self.assertEqual(call.returncode, 0, call.stderr + str(result))
        self.assertEqual(marker.read_text(), "DEPENDENCY")

    def test_single_running_job(self):
        first = subprocess.Popen([sys.executable, "-B", str(LAB), "--project", str(self.project), "run", "--label", "first", "--seconds", "3", "--", "/usr/bin/python3", "-c", "import time;time.sleep(1)"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 2
            while not list((self.project / "runs").glob("*/status.json")):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            call, result = self.invoke("raise RuntimeError('must not start')")
            self.assertNotEqual(call.returncode, 0)
            self.assertIsNone(result)
            self.assertIn("another lab run", call.stderr)
        finally:
            out, err = first.communicate(timeout=6)
        self.assertEqual(first.returncode, 0, out + err)

    def test_timeout_reaps_background_child(self):
        code = "import subprocess,time;subprocess.Popen(['/usr/bin/python3','-c',\"import time;from pathlib import Path;time.sleep(2);Path('/output/late').write_text('bad')\"]);time.sleep(10)"
        call, result = self.invoke(code, "--seconds", "0.3")
        self.assertNotEqual(call.returncode, 0)
        self.assertEqual(result["reason"], "timeout")
        time.sleep(2.2)
        self.assertFalse((self.project / "runs" / result["id"] / "artifacts/late").exists())

    def test_normal_exit_reaps_background_child(self):
        code = "import subprocess;subprocess.Popen(['/usr/bin/python3','-c',\"import time;from pathlib import Path;time.sleep(1);Path('/output/late').write_text('bad')\"])"
        call, result = self.invoke(code)
        self.assertEqual(call.returncode, 0, call.stderr)
        time.sleep(1.2)
        self.assertFalse((self.project / "runs" / result["id"] / "artifacts/late").exists())

    def test_total_output_checked_even_after_fast_exit(self):
        call, result = self.invoke("from pathlib import Path;Path('/output/a').write_bytes(b'a'*700000);Path('/output/b').write_bytes(b'b'*700000)", "--disk-mib", "1")
        self.assertNotEqual(call.returncode, 0)
        self.assertEqual(result["reason"], "output_limit")

    def test_missing_backend_no_workload_or_output_creation(self):
        args = mock.Mock(command=["--", "/bin/true"], label="fixture", seconds=1, disk_mib=1, memory_mib=128, stage="dev", config=None, input=[], runtime=[], gpu=False)
        with mock.patch.object(MODULE.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "no unsafe fallback"):
                MODULE.run(args, self.project)
        self.assertFalse((self.project / "runs").exists())

    def test_rejected_backend_fails_without_payload(self):
        args = mock.Mock(command=["--", "/bin/true"], label="rejected", seconds=1, disk_mib=1, memory_mib=128, stage="dev", config=None, input=[], runtime=[], gpu=False)
        with mock.patch.object(MODULE.shutil, "which", return_value="/bin/false"):
            self.assertNotEqual(MODULE.run(args, self.project), 0)
        statuses = list((self.project / "runs").glob("*/status.json"))
        self.assertEqual(json.loads(statuses[0].read_text())["state"], "failed")

    def test_symlink_run_root_rejected_before_launch(self):
        target = Path(self.temporary.name) / "elsewhere"
        target.mkdir()
        (self.project / "runs").symlink_to(target, target_is_directory=True)
        call, _ = self.invoke("raise RuntimeError('must not start')")
        self.assertNotEqual(call.returncode, 0)
        self.assertEqual(list(target.iterdir()), [])

    def test_invalid_command_label_and_config(self):
        for tail in (["--label", "../escape", "--", "/bin/true"], ["--label", "valid", "--"], ["--label", "valid", "--stage", "train", "--", "/bin/true"]):
            call = subprocess.run([sys.executable, "-B", str(LAB), "--project", str(self.project), "run", *tail], capture_output=True, text=True)
            self.assertNotEqual(call.returncode, 0)
        self.assertFalse((self.project / "runs").exists())

    def test_config_is_snapshotted_and_readable(self):
        (self.project / "configs").mkdir()
        (self.project / "configs/example.json").write_text('{"seed":42}')
        call, result = self.invoke("import json;from pathlib import Path;assert json.loads(Path('configs/example.json').read_text())['seed']==42", "--config", "configs/example.json")
        self.assertEqual(call.returncode, 0, call.stderr)
        manifest = json.loads((self.project / "runs" / result["id"] / "manifest.json").read_text())
        self.assertIn("configs/example.json", [v["path"] for v in manifest["source_snapshot"]])

    def test_nested_limits_never_raise_inherited_hard_limit(self):
        self.assertEqual(MODULE.inherited_cap(2048, 256), 256)
        self.assertEqual(MODULE.inherited_cap(128, 256), 128)
        self.assertEqual(MODULE.inherited_cap(2048, MODULE.resource.RLIM_INFINITY), 2048)

    def test_runtime_rejects_selected_project_ancestors_and_private_paths(self):
        private = self.project / "human"
        private.mkdir()
        hidden = self.project / ".private"
        hidden.mkdir()
        for runtime in (self.project, self.project.parent, private, hidden):
            call, result = self.invoke("raise RuntimeError('must not start')", "--runtime", str(runtime))
            self.assertNotEqual(call.returncode, 0)
            self.assertIsNone(result)
            self.assertIn("dedicated dependency directory", call.stderr)
        self.assertFalse((self.project / "runs").exists())

    def test_snapshot_preserves_assets_and_executable_mode(self):
        (self.project / "src/large.txt").write_bytes(b'x' * (1024**2 + 1))
        (self.project / "src/fixture.bin").write_bytes(b'\x00\xff')
        script = self.project / "src/probe.py"
        script.write_text("#!/usr/bin/python3\nfrom pathlib import Path\nassert Path('src/marker.txt').read_text() == 'DO NOT CHANGE'\nassert Path('src/large.txt').stat().st_size == 1024**2+1\nassert Path('src/fixture.bin').read_bytes() == bytes([0,255])\nPath('/output/ok').touch()\n")
        script.chmod(0o755)
        call, result = self.invoke("import subprocess;subprocess.run(['./src/probe.py'],check=True)")
        self.assertEqual(call.returncode, 0, call.stderr + str(result))
        directory = self.project / "runs" / result["id"]
        self.assertTrue((directory / "artifacts/ok").is_file())
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(next(p['mode'] for p in manifest['source_snapshot'] if p['path'] == 'src/probe.py'), '0o755')

    def test_snapshot_fails_instead_of_silently_omitting_assets(self):
        large = self.project / "src/large.bin"
        with large.open('wb') as handle:
            handle.truncate(65 * 1024**2)
        call, result = self.invoke("raise RuntimeError('must not start')")
        self.assertNotEqual(call.returncode, 0)
        self.assertIn("scope exceeds", result["error"])
        # Resize only this test-owned fixture, not any path supplied by a child.
        large.write_bytes(b'small')
        (self.project / "src/link").symlink_to(self.external)
        call, result = self.invoke("raise RuntimeError('must not start')")
        self.assertNotEqual(call.returncode, 0)
        self.assertIn("regular file", result["error"])

    def test_private_explicit_inputs_are_rejected(self):
        (self.project / ".codex").mkdir()
        (self.project / ".codex/private.json").write_text('{}')
        (self.project / ".env.json").write_text('{}')
        for name in (".codex/private.json", ".env.json"):
            call, result = self.invoke("raise RuntimeError('must not start')", "--input", name)
            self.assertNotEqual(call.returncode, 0)
            self.assertIsNone(result)
        self.assertFalse((self.project / "runs").exists())

    def test_source_traversal_errors_are_not_silently_ignored(self):
        with mock.patch.object(MODULE.os, 'scandir', side_effect=PermissionError('fixture denied')):
            with self.assertRaisesRegex(PermissionError, 'fixture denied'):
                MODULE.scoped_files(self.project, ('src',), source=True)
            self.assertEqual(MODULE.scoped_files(self.project, ('src',)), [])

    def test_doctor_separates_broken_links_from_missing_local_artifacts(self):
        for name in ("paper", "tasks"):
            (self.project / name).mkdir()
        for name in ("README.md", "AGENTS.md", "paper/PLAN.md", "tasks/current.md"):
            (self.project / name).write_text("fixture")
        (self.project / "paper/NOW.md").write_text("[任务](../tasks/current.md)\n[旧结果](../runs/missing/log.txt)\n[断链](absent.md)\n[网站](https://example.com/page)\n")
        command = [sys.executable, "-B", str(LAB), "--project", str(self.project), "doctor"]
        call = subprocess.run(command, capture_output=True, text=True, timeout=5)
        result = json.loads(call.stdout)
        self.assertEqual(call.returncode, 1)
        self.assertEqual(result['active_task'], 'tasks/current.md')
        self.assertEqual(result['issues'], ['paper/NOW.md: absent.md'])
        self.assertEqual(result['local_missing'], ['paper/NOW.md: ../runs/missing/log.txt'])
        (self.project / "paper/absent.md").write_text("restored fixture")
        call = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(call.returncode, 0, call.stdout + call.stderr)
        self.assertFalse((self.project / ".cache").exists())

    def test_search_excludes_legacy_private_and_symlinks(self):
        for name in ("paper", "legacy", "human"):
            (self.project / name).mkdir()
            (self.project / name / "entry.md").write_text("needle " + name)
        (self.project / "paper/link.md").symlink_to(self.project / "human/entry.md")
        (self.project / "paper/.private.md").write_text("needle secret")
        (self.project / "AGENTS.md").write_text("needle contract")
        call = subprocess.run([sys.executable, "-B", str(LAB), "--project", str(self.project), "search", "needle"], text=True, capture_output=True)
        self.assertIn("needle paper", call.stdout)
        self.assertNotIn("needle human", call.stdout)
        self.assertNotIn("needle legacy", call.stdout)
        self.assertNotIn("needle secret", call.stdout)
        self.assertIn("needle contract", call.stdout)


if __name__ == "__main__":
    unittest.main()

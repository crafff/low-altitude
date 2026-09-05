#!/usr/bin/env python3
"""Small research helper: isolated execution, status and bounded text search.

No delete/clean command and no unsandboxed execution fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit
import uuid

DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
CODE_ROOTS = ("tools", "src", "tests", "configs")
TEXT_ROOTS = ("README.md", "AGENTS.md", "paper", "tasks", "docs", *CODE_ROOTS)
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".csv", ".txt", ".sh", ".ini", ".cfg", ".lock"}
SKIP = {".git", ".codex", ".agents", "legacy", "human", "runs", "data", "models", ".venv", "__pycache__", ".cache"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def plain_path(path):
    """Check original path components before resolving away symbolic links."""
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"symbolic link is not allowed: {item}")
    return path


def project_path(value):
    path = plain_path(value)
    if path in {Path("/"), Path("/tmp"), Path("/home"), Path.home()} or not path.is_dir():
        raise ValueError("project must be a specific existing project directory")
    return path


def project_file(project, name):
    relative = Path(name)
    if relative.is_absolute() or any(p in SKIP or p.startswith(".") for p in relative.parts):
        raise ValueError("input must be a project-relative, non-private, non-legacy file")
    path = plain_path(project / relative)
    if not path.is_file():
        raise ValueError(f"input file is missing: {name}")
    return path


def put_json(path, value, *, update=False):
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if update:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as out:
            out.write(payload)
        os.replace(temporary, path)
    else:
        with path.open("x") as out:
            out.write(payload)


def scoped_files(project, roots=TEXT_ROOTS, *, source=False):
    """Search small text files; snapshot every ordinary non-private source asset."""
    def walk_error(error):
        if source:
            raise error

    files = []
    total = 0
    maximum = (64 if source else 32) * 1024**2
    for name in roots:
        base = plain_path(project / name)
        if not base.exists():
            continue
        entries = [(base.parent, [], [base.name])] if base.is_file() else os.walk(base, followlinks=False, onerror=walk_error)
        for current, directories, names in entries:
            directories[:] = sorted(d for d in directories if d not in SKIP and not d.startswith("."))
            if source and any((Path(current) / d).is_symlink() for d in directories):
                raise ValueError(f"source directory contains a symbolic link: {current}")
            directories[:] = [d for d in directories if not (Path(current) / d).is_symlink()]
            for name in sorted(names):
                if name.startswith(".") or Path(name).suffix in {".pyc", ".pyo"}:
                    continue
                path = Path(current) / name
                if source and (path.is_symlink() or not path.is_file()):
                    raise ValueError(f"source asset must be a regular file: {path}")
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
                if not source and (path.suffix not in TEXT_SUFFIXES or size > 1024**2):
                    continue
                files.append(path)
                total += size
                if len(files) > 2000 or total > maximum:
                    raise ValueError("text/source scope exceeds the small-project limit; use explicit read-only dependencies for large assets")
    return sorted(set(files))


def snapshot(project, destination, extras):
    sources = set(scoped_files(project, CODE_ROOTS, source=True))
    sources.update(project_file(project, item) for item in extras)
    if sum(p.stat().st_size for p in sources) > 64 * 1024**2:
        raise ValueError("source snapshot exceeds 64 MiB; use a declared read-only runtime for dependencies")
    records = []
    for source in sorted(sources):
        source = plain_path(source)
        before = source.stat()
        relative = source.relative_to(project)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_mode) != (after.st_size, after.st_mtime_ns, after.st_mode):
            raise ValueError(f"source changed while snapshotting: {relative}")
        digest = sha(target)
        if digest != sha(source):
            raise ValueError(f"source changed while snapshotting: {relative}")
        target.chmod(0o644 | (before.st_mode & 0o111))
        records.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": digest, "mode": oct(stat.S_IMODE(target.stat().st_mode))})
    return records


def runtime_paths(project, names):
    paths = []
    for name in names:
        path = plain_path(name)
        if (not path.is_dir() or len(path.parts) < 4
                or path in {Path.home(), DEFAULT_PROJECT, project}
                or path in project.parents or path in DEFAULT_PROJECT.parents
                or any((p in SKIP or p.startswith(".")) and p != ".venv" for p in path.parts)
                or any(path == mount or mount in path.parents for mount in (Path("/workspace"), Path("/output")))):
            raise ValueError("runtime must be a dedicated dependency directory, not a private directory, project root or project ancestor")
        paths.append(str(path))
    return paths


def sandbox_command(bwrap, source, artifacts, scratch, command, runtimes=(), gpu=False):
    args = [bwrap, "--unshare-all", "--new-session", "--die-with-parent", "--cap-drop", "ALL", "--clearenv"]
    # A fresh filesystem, not a read-only view of the whole user's home.
    for name in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(name).exists():
            args += ["--ro-bind", name, name]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--dir", "/run"]
    args += ["--ro-bind", str(source), "/workspace", "--bind", str(artifacts), "/output", "--bind", str(scratch), "/tmp"]
    for runtime in runtimes:
        args += ["--ro-bind", runtime, runtime]
    if gpu:
        devices = [p for p in Path("/dev").glob("nvidia*") if re.fullmatch(r"nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools)", p.name) and stat.S_ISCHR(p.stat().st_mode)]
        if not any(re.fullmatch(r"nvidia[0-9]+", p.name) for p in devices):
            raise ValueError("GPU devices are not visible here; use an approved GPU-capable environment, not a silent CPU fallback")
        for device in sorted(devices):
            args += ["--dev-bind", str(device), str(device)]
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "PYTHONPATH": "/workspace/src", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "LAB_RUN_DIR": "/output", "TMPDIR": "/tmp",
        "XDG_CACHE_HOME": "/tmp/cache", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    for key, value in environment.items():
        args += ["--setenv", key, value]
    return args + ["--remount-ro", "/", "--chdir", "/workspace", "--", *command]


def tree_size(path):
    total = 0
    for root, directories, names in os.walk(path, followlinks=False):
        directories[:] = [d for d in directories if not (Path(root) / d).is_symlink()]
        for name in names:
            try:
                info = (Path(root) / name).lstat()
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
            except FileNotFoundError:
                pass
    return total


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        except ProcessLookupError:
            process.wait(timeout=2)


def inherited_cap(requested, hard_limit):
    return requested if hard_limit == resource.RLIM_INFINITY else min(requested, hard_limit)


def run(args, project):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not command[0].strip():
        raise ValueError("provide a command after --")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", args.label):
        raise ValueError("label must be a short name, never a path")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 86400:
        raise ValueError("seconds must be in (0, 86400]")
    if not 1 <= args.disk_mib <= 51200 or not 128 <= args.memory_mib <= 131072:
        raise ValueError("invalid output or per-process memory limit")
    if args.stage in {"train", "eval"} and not args.config:
        raise ValueError("train/eval requires a configuration file")
    runtimes = runtime_paths(project, args.runtime)
    extras = list(args.input)
    if args.config:
        json.loads(project_file(project, args.config).read_text())
        extras.append(args.config)
    for item in extras:
        project_file(project, item)
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise ValueError("bubblewrap is unavailable; workload was not started (no unsafe fallback)")
    cache = plain_path(project / ".cache")
    cache.mkdir(exist_ok=True)
    lock_path = plain_path(cache / "lab-run.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another lab run is active") from None
        runs = plain_path(project / "runs")
        runs.mkdir(exist_ok=True)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.label + "-" + uuid.uuid4().hex[:8]
        directory = runs / identifier
        directory.mkdir(mode=0o700)
        for part in ("source", "artifacts", "scratch"):
            (directory / part).mkdir(mode=0o700)
        status = {"id": identifier, "stage": args.stage, "state": "running", "started_at": stamp(), "pid": None}
        put_json(directory / "status.json", status)
        process = None
        code, reason, error = 1, "launch_error", None
        budget = args.disk_mib * 1024**2
        file_cap = inherited_cap(budget, resource.getrlimit(resource.RLIMIT_FSIZE)[1])
        memory_cap = inherited_cap(args.memory_mib * 1024**2, resource.getrlimit(resource.RLIMIT_AS)[1])
        try:
            assets = snapshot(project, directory / "source", extras)
            launch = sandbox_command(bwrap, directory / "source", directory / "artifacts", directory / "scratch", command, runtimes, args.gpu)
            head = subprocess.run(["git", "-c", "gc.auto=0", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True).stdout.strip() or None
            put_json(directory / "manifest.json", {
                "schema": "lab.run.v3", "id": identifier, "stage": args.stage,
                "created_at": stamp(), "command": command, "sandbox_command": launch,
                "config": args.config, "source_snapshot": assets, "git_head": head,
                "source_of_truth": "source/ snapshot, not the potentially dirty Git HEAD",
                "seconds": args.seconds, "output_monitor_bytes": budget,
                "per_file_limit_bytes": file_cap, "requested_per_process_memory_mib": args.memory_mib,
                "effective_per_process_address_space_bytes": memory_cap,
                "limits_note": "Output+logs+scratch are polled and may overshoot; no hard aggregate disk/RAM quota. Private /dev/shm uses memory.",
                "python_supervisor": sys.version, "runtime_readonly": runtimes,
                "gpu_requested": args.gpu, "network": "none",
            })
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_cap, file_cap))
                resource.setrlimit(resource.RLIMIT_AS, (memory_cap, memory_cap))
            with (directory / "log.txt").open("xb") as log:
                process = subprocess.Popen(launch, cwd=directory, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, preexec_fn=limits)
                status["pid"] = process.pid
                put_json(directory / "status.json", status, update=True)
                deadline = time.monotonic() + args.seconds
                reason = "completed"
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        reason = "timeout"
                        break
                    if tree_size(directory / "artifacts") + tree_size(directory / "scratch") + log.tell() > budget:
                        reason = "output_limit"
                        break
                    time.sleep(0.05)
                if reason != "completed":
                    stop(process)
                code = process.wait()
                if code != 0 and reason == "completed":
                    reason = "nonzero_exit"
            output_bytes = sum(tree_size(directory / part) for part in ("artifacts", "scratch")) + (directory / "log.txt").stat().st_size
            if output_bytes > budget:
                reason = "output_limit"
        except KeyboardInterrupt:
            reason, code = "interrupted", 130
        except Exception as exc:
            reason = "supervisor_error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                stop(process)
            success = reason == "completed" and code == 0 and error is None
            status.update(state="succeeded" if success else "failed", finished_at=stamp(), reason=reason, exit_code=code, error=error)
            put_json(directory / "status.json", status, update=True)
        print(json.dumps({**status, "run_dir": str(directory)}, ensure_ascii=False))
        return 0 if success else (code if 1 <= code <= 125 else 1)


def doctor(project):
    """Read-only entry/link checks, not a simulator or sandbox workload."""
    issues, local_missing = [], []
    for name in ("README.md", "AGENTS.md", "paper/NOW.md", "paper/PLAN.md"):
        if not plain_path(project / name).is_file():
            issues.append(f"missing entry: {name}")
    active_task = None
    for path in scoped_files(project):
        if path.suffix != ".md":
            continue
        for match in re.finditer(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text()):
            link = match.group(1).strip().strip("<>")
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = Path(os.path.abspath(path.parent / unquote(url.path)))
            try:
                relative = target.relative_to(project)
            except ValueError:
                issues.append(f"{path.relative_to(project)}: link outside project: {link}")
                continue
            if path == project / "paper/NOW.md" and relative.parts[:1] == ("tasks",) and active_task is None:
                active_task = str(relative)
            # Do not inspect private targets or follow symbolic links.
            if any(p in {"human", ".git", ".codex", ".agents"} for p in relative.parts):
                issues.append(f"{path.relative_to(project)}: private link: {link}")
                continue
            if plain_path(target).exists():
                continue
            local = ((relative.parts[:1] in [("runs",), ("data",), ("models",), ("legacy",)]
                      and relative != Path("legacy/README.md"))
                     or relative.parts[:3] == ("resources", "literature", "local"))
            entry = f"{path.relative_to(project)}: {link}"
            (local_missing if local else issues).append(entry)
    if active_task is None:
        issues.append("paper/NOW.md must link to the current task")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        issues.append("bubblewrap is unavailable; workloads cannot start")
    result = {"python": sys.version.split()[0], "bubblewrap": bwrap,
              "active_task": active_task, "issues": sorted(set(issues)),
              "local_missing": sorted(set(local_missing)),
              "note": "Entry/link checks only; no network/anchor checks, CUDA, BlueSky or sandbox execution. Local-only materials need separate backup."}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if issues else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    commands = parser.add_subparsers(dest="action", required=True)
    cmd = commands.add_parser("run")
    cmd.add_argument("--label", required=True)
    cmd.add_argument("--stage", choices=("system", "dev", "train", "eval", "analysis"), default="dev")
    cmd.add_argument("--seconds", type=float, default=900)
    cmd.add_argument("--disk-mib", type=int, default=2048)
    cmd.add_argument("--memory-mib", type=int, default=8192)
    cmd.add_argument("--config")
    cmd.add_argument("--input", action="append", default=[])
    cmd.add_argument("--runtime", action="append", default=[])
    cmd.add_argument("--gpu", action="store_true")
    cmd.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("doctor")
    commands.add_parser("status")
    cmd = commands.add_parser("search")
    cmd.add_argument("query")
    cmd.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    try:
        project = project_path(args.project)
        if args.action == "run":
            return run(args, project)
        if args.action == "doctor":
            return doctor(project)
        elif args.action == "status":
            now = project / "paper/NOW.md"
            if now.is_file():
                print(now.read_text())
            runs = plain_path(project / "runs")
            if runs.exists():
                for path in sorted(runs.glob("*/status.json"))[-5:]:
                    path = plain_path(path)
                    value = json.loads(path.read_text())
                    if value.get("state") == "running":
                        value["liveness"] = "not_verified; inspect supervisor before relaunch"
                    print(json.dumps(value, ensure_ascii=False))
        else:
            if not args.query or not 1 <= args.limit <= 200:
                raise ValueError("search requires nonempty text and limit 1..200")
            count = 0
            for path in scoped_files(project):
                for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if args.query.casefold() in line.casefold():
                        print(f"{path.relative_to(project)}:{number}:{line[:700]}")
                        count += 1
                        if count >= args.limit:
                            return 0
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"lab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

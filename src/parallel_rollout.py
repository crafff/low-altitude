"""Persistent CPU collectors for synchronous complete-episode PPO batches.

Only this parent's spawned children exchange pickle messages. They contain
ordinary NumPy arrays, never torch shared-memory handles. The parent owns PPO,
checkpoint counters, and the decision to commit an entirely returned batch.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import resource
import threading
import time


MAX_EPISODE_BYTES = 64 * 1024 * 1024
MAX_WEIGHTS_BYTES = 8 * 1024 * 1024
_CHILD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
_CHILD_ENV["CUDA_VISIBLE_DEVICES"] = ""


class RolloutWorkerError(RuntimeError):
    """A failed or invalid batch; no samples from this batch may be committed."""


def _deadline(deadline):
    from paper_train import CollectionCutoff
    if time.perf_counter() >= deadline:
        raise CollectionCutoff("parallel rollout batch discarded at wall-clock cutoff")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        # Protocol 5 can write PickleBuffer directly; it has the buffer protocol
        # but no len(). nbytes also counts multidimensional views correctly.
        if self.tell() + memoryview(value).nbytes > self.limit:
            raise RolloutWorkerError("rollout payload exceeds its in-memory byte limit")
        return super().write(value)


def _encode(message, limit=MAX_EPISODE_BYTES):
    buffer = _LimitedBuffer(limit)
    pickle.dump(message, buffer, protocol=5)
    return buffer.getvalue()


def sample_sha256(samples):
    """Canonical identity of every scalar and the exact NumPy row payload."""
    import numpy as np
    digest = hashlib.sha256()
    for row in samples:
        for key in sorted(row):
            value = row[key]
            header = {"key": key}
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise RolloutWorkerError("object arrays cannot identify rollout samples")
                header.update(dtype=value.dtype.str, shape=list(value.shape))
                content = value.tobytes(order="C")
            else:
                header["scalar"] = value
                content = b""
            encoded = json.dumps(header, sort_keys=True, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        digest.update(b"\x00END_ROW\x00")
    return digest.hexdigest()


@contextmanager
def _spawn_environment():
    # Spawn imports the main module before entering _worker: these must already
    # be inherited when NumPy/torch/BLAS are imported in the fresh interpreter.
    previous = {key: os.environ.get(key) for key in _CHILD_ENV}
    os.environ.update(_CHILD_ENV)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _BoundedEnvironment:
    """Cap accumulated rows before another environment decision is collected.

    The allowance includes array storage and a conservative 2 KiB per row for
    dictionaries, scalars, list references, and trajectory/GAE bookkeeping.
    Encoded messages also have an independent hard byte cap. These are payload
    limits, not a hard process RSS limit (model and BlueSky memory are separate).
    """
    def __init__(self, env, observations, limit):
        self.env, self.current, self.limit = env, observations, limit
        self.estimated_bytes = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        self.estimated_bytes += sum(2048 + sum(obs[key].nbytes for key in
            ("own", "intruders", "action_mask")) for obs in self.current.values())
        if self.estimated_bytes > self.limit:
            raise RolloutWorkerError("episode rows exceed the in-memory byte allowance")
        result = self.env.step(actions)
        self.current = result[0]
        return result


def _worker(connection, worker_id, cpu_id, config, environment_factory):
    identity = {"worker_id": worker_id}
    try:
        started = time.perf_counter()
        os.environ.update(_CHILD_ENV)
        os.sched_setaffinity(0, {cpu_id})
        import numpy as np
        import torch
        from paper_train import CollectionCutoff, collect_episode
        from shared_ppo import SharedActorCritic

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = SharedActorCritic(config["ppo_cfg"])
        if environment_factory is None:
            from paper_environment import PaperEnvironment, load_environment_config
            ecfg, parts = load_environment_config(config["environment_config_path"])
            env = PaperEnvironment(ecfg, parts)
        else:
            # Test-only, top-level picklable factory; production always uses the
            # real environment above and the same collect_episode function.
            env = environment_factory(config["environment_config_path"])
        connection.send_bytes(_encode(dict(identity, kind="ready", pid=os.getpid(),
            cpu_id=cpu_id, init_seconds=time.perf_counter()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
        while True:
            request = pickle.loads(connection.recv_bytes(MAX_WEIGHTS_BYTES))
            if request.get("kind") == "close":
                return
            if request.get("kind") != "collect":
                raise RolloutWorkerError("unknown parent command")
            identity = {key: request[key] for key in (
                "worker_id", "batch_id", "policy_version", "policy_sha256", "episode_index")}
            if identity["worker_id"] != worker_id:
                raise RolloutWorkerError("parent dispatched another worker's episode")
            _deadline(request["deadline"])
            load_started = time.perf_counter()
            weight_blob = request["weights"]
            if hashlib.sha256(weight_blob).hexdigest() != identity["policy_sha256"]:
                raise RolloutWorkerError("frozen policy digest mismatch")
            weights = pickle.loads(weight_blob)
            model.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()}, strict=True)
            weight_seconds = time.perf_counter()-load_started
            del weights, weight_blob, request["weights"]
            index = identity["episode_index"]
            global_seed = config["training_seed"] + index
            action_seed = config["sampling_seed"] + index
            scenario_seed = config["scenario_seed_start"] + index
            random.seed(global_seed)
            np.random.seed(global_seed % (2**32))
            torch.manual_seed(global_seed)
            sampling = torch.Generator(device="cpu").manual_seed(action_seed)
            rollout_started = time.perf_counter()
            observations = env.reset(scenario_seed)
            guarded = _BoundedEnvironment(env, observations, config["max_episode_bytes"])
            samples, summary = collect_episode(guarded, model, sampling, config["ppo_cfg"],
                observations=observations, deadline=request["deadline"])
            _deadline(request["deadline"])
            timing = {"weight_load_seconds": weight_seconds,
                      "rollout_wall_seconds": time.perf_counter()-rollout_started,
                      "estimated_row_bytes": guarded.estimated_bytes,
                      "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
            hash_started = time.perf_counter()
            samples_digest = sample_sha256(samples)
            timing["sample_hash_seconds"] = time.perf_counter()-hash_started
            serialization_started = time.perf_counter()
            payload = _encode(dict(identity, kind="result", samples=samples,
                summary=summary, scenario_seed=scenario_seed, action_seed=action_seed,
                global_seed=global_seed, sample_sha256=samples_digest, timing=timing),
                config["max_episode_bytes"])
            serialization_seconds = time.perf_counter()-serialization_started
            _deadline(request["deadline"])
            send_started = time.perf_counter()
            connection.send_bytes(payload)
            # The receipt measures the complete send, including pipe blocking;
            # returning a batch requires both its full payload and this receipt.
            connection.send_bytes(_encode(dict(identity, kind="sent", payload_bytes=len(payload),
                serialization_seconds=serialization_seconds,
                send_seconds=time.perf_counter()-send_started)))
            del payload, samples, summary, guarded, observations
    except BaseException as error:
        try:
            from paper_train import CollectionCutoff
            connection.send_bytes(_encode(dict(identity, kind="error",
                cutoff=isinstance(error, CollectionCutoff),
                error=f"{type(error).__name__}: {str(error)[:2000]}")))
        except BaseException:
            pass
    finally:
        connection.close()


def _reader(connection, worker_id, inbox, stopping, limit):
    # recv_bytes can block after receiving a header. A reader per private pipe
    # lets the controller enforce a deadline even if a child stalls mid-send.
    try:
        while not stopping.is_set():
            started = time.perf_counter()
            data = connection.recv_bytes(limit)
            event = (worker_id, data, time.perf_counter()-started, None)
            while not stopping.is_set():
                try:
                    inbox.put(event, timeout=.1)
                    break
                except queue.Full:
                    continue
            del data, event
    except (EOFError, OSError) as error:
        if not stopping.is_set():
            try:
                inbox.put((worker_id, None, 0., repr(error)), timeout=.1)
            except queue.Full:
                pass


def _sender(connection, payload, worker_id, completed):
    try:
        connection.send_bytes(payload)
        completed.put((worker_id, None))
    except (EOFError, OSError) as error:
        completed.put((worker_id, repr(error)))


def validate_episode_result(message, expected, *, seen):
    """Validate one trusted-child result; injectable protocol seam for tests.

    Identity validation precedes exposing any rows. Full batch completion is
    additionally checked by collect(), including one send receipt per worker.
    """
    import numpy as np
    if not isinstance(message, dict) or message.get("kind") != "result":
        raise RolloutWorkerError("expected a complete episode result")
    if any(message.get(key) != value for key, value in expected.items()):
        raise RolloutWorkerError("stale or mismatched rollout result identity")
    index = message["episode_index"]
    if index in seen:
        raise RolloutWorkerError("duplicate episode result")
    samples, summary = message.get("samples"), message.get("summary")
    if (not isinstance(samples, list) or not samples or not isinstance(summary, dict)
            or summary.get("completed_population") is not True
            or summary.get("rollout_samples") != len(samples)):
        raise RolloutWorkerError("partial or empty episode result")
    previous, terminal = {}, set()
    for row in samples:
        if not isinstance(row, dict):
            raise RolloutWorkerError("invalid sample row")
        for key, shape in (("own", (7,)), ("action_mask", (60,))):
            array = row.get(key)
            if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.hasobject:
                raise RolloutWorkerError("samples must carry full numeric NumPy observations")
        neighbors = row.get("intruders")
        if (not isinstance(neighbors, np.ndarray) or neighbors.ndim != 2
                or neighbors.shape[1] != 10 or neighbors.dtype.hasobject):
            raise RolloutWorkerError("invalid intruder observation payload")
        if not np.isfinite(row["own"]).all() or not np.isfinite(neighbors).all():
            raise RolloutWorkerError("nonfinite observation payload")
        action, mask = row.get("action"), row["action_mask"]
        if (type(action) is not int or not 0 <= action < 60
                or mask.dtype != np.bool_ or not mask[action]):
            raise RolloutWorkerError("invalid sampled action/mask")
        for key in ("old_log_prob", "value", "reward", "advantage", "return"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                raise RolloutWorkerError("nonfinite or missing PPO sample scalar")
        acid, decision = row.get("aircraft_id"), row.get("decision_index")
        if (not isinstance(acid, str) or type(decision) is not int or decision < 0
                or type(row.get("terminated")) is not bool or acid in terminal
                or decision <= previous.get(acid, -1)):
            raise RolloutWorkerError("partial, duplicate or unordered aircraft trajectory")
        previous[acid] = decision
        if row["terminated"]:
            terminal.add(acid)
    if set(previous) != terminal or summary.get("rollout_aircraft") != len(previous):
        raise RolloutWorkerError("partial aircraft trajectory in completed episode")
    if message.get("sample_sha256") != sample_sha256(samples):
        raise RolloutWorkerError("full rollout sample digest mismatch")
    seen.add(index)


class ParallelRolloutPool:
    """One complete episode per persistent worker, one frozen policy per call.

    Deadlines are absolute time.perf_counter values. cpu_ids is explicit and
    must be inside the launcher's allowed affinity. No affinity fallback or GPU
    work is performed. environment_factory is solely a test injection point.
    """
    def __init__(self, environment_config_path, ppo_cfg, *, workers, cpu_ids,
                 scenario_seed_start, sampling_seed, training_seed, startup_deadline,
                 environment_factory=None, max_episode_bytes=MAX_EPISODE_BYTES):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("workers must be an integer in [1, 4]")
        if (len(cpu_ids) != workers or len(set(cpu_ids)) != workers
                or any(type(cpu) is not int or cpu < 0 for cpu in cpu_ids)):
            raise ValueError("one distinct nonnegative cpu_id is required per worker")
        if not set(cpu_ids).issubset(os.sched_getaffinity(0)):
            raise ValueError("requested rollout CPUs are outside the launcher's affinity")
        if ppo_cfg.get("device", "cpu") != "cpu" or ppo_cfg.get("torch_num_threads", 1) != 1:
            raise ValueError("parallel rollout requires CPU and one torch thread")
        for name, value in (("scenario_seed_start", scenario_seed_start),
                            ("sampling_seed", sampling_seed), ("training_seed", training_seed)):
            _integer(value, name)
        if not math.isfinite(startup_deadline):
            raise ValueError("startup_deadline must be finite")
        if type(max_episode_bytes) is not int or not 4096 <= max_episode_bytes <= MAX_EPISODE_BYTES:
            raise ValueError("episode payload cap must be between 4 KiB and 64 MiB")
        self.workers = workers
        self.config = dict(environment_config_path=str(environment_config_path),
            ppo_cfg=copy.deepcopy(ppo_cfg), scenario_seed_start=scenario_seed_start,
            sampling_seed=sampling_seed, training_seed=training_seed,
            max_episode_bytes=max_episode_bytes)
        self.processes, self.connections, self.readers, self.senders = [], [], [], []
        self.inbox, self.stopping = queue.Queue(maxsize=2*workers), threading.Event()
        self.closed, self.batch_id = False, 0
        self.initialized = []
        started = time.perf_counter()
        try:
            _deadline(startup_deadline)
            context = mp.get_context("spawn")
            for worker_id, cpu_id in enumerate(cpu_ids):
                parent, child = context.Pipe(duplex=True)
                self.connections.append(parent)
                process = context.Process(target=_worker,
                    args=(child, worker_id, cpu_id, self.config, environment_factory),
                    name=f"paper-rollout-{worker_id}")
                self.processes.append(process)
                try:
                    with _spawn_environment():
                        process.start()
                finally:
                    child.close()
                reader = threading.Thread(target=_reader,
                    args=(parent, worker_id, self.inbox, self.stopping, max_episode_bytes),
                    daemon=True, name=f"paper-rollout-reader-{worker_id}")
                self.readers.append(reader)
                reader.start()
            ready = {}
            while len(ready) < workers:
                worker_id, message, _ = self._receive(startup_deadline)
                if (message.get("kind") != "ready" or message.get("worker_id") != worker_id
                        or worker_id in ready or message.get("cpu_id") != cpu_ids[worker_id]
                        or message.get("pid") != self.processes[worker_id].pid):
                    raise RolloutWorkerError("invalid or duplicate worker initialization")
                ready[worker_id] = message
            self.initialized = [ready[index] for index in range(workers)]
            self.startup_seconds = time.perf_counter()-started
        except BaseException:
            self.close()
            raise

    def _receive(self, deadline):
        from paper_train import CollectionCutoff
        while True:
            _deadline(deadline)
            try:
                worker_id, data, read_seconds, error = self.inbox.get(
                    timeout=min(.1, max(0., deadline-time.perf_counter())))
            except queue.Empty:
                if any(p.pid is not None and p.exitcode is not None for p in self.processes):
                    raise RolloutWorkerError("rollout worker exited before completing its batch")
                continue
            if error is not None:
                raise RolloutWorkerError(f"rollout worker {worker_id} pipe failed: {error}")
            decode_started = time.perf_counter()
            # Trusted data exclusively from this object's own spawned child.
            message = pickle.loads(data)
            received_bytes = len(data)
            del data
            _deadline(deadline)
            if not isinstance(message, dict):
                raise RolloutWorkerError("invalid worker message")
            if message.get("kind") == "error":
                detail = f"rollout worker {worker_id}: {message.get('error')}"
                if message.get("cutoff"):
                    raise CollectionCutoff(detail)
                raise RolloutWorkerError(detail)
            return worker_id, message, {"received_bytes": received_bytes,
                "decode_seconds": time.perf_counter()-decode_started,
                "reader_wait_and_receive_seconds": read_seconds}

    def collect(self, weights, episode_indices, *, policy_version, deadline):
        """Return all ordered rows/summaries, or discard everything and close."""
        if self.closed:
            raise RolloutWorkerError("rollout pool is closed")
        started = time.perf_counter()
        try:
            import numpy as np
            import torch
            _integer(policy_version, "policy_version")
            indices = list(episode_indices)
            for index in indices:
                _integer(index, "episode_index")
            if len(indices) != self.workers or len(set(indices)) != len(indices):
                raise ValueError("each batch requires exactly one unique episode per worker")
            if not math.isfinite(deadline):
                raise ValueError("deadline must be finite")
            _deadline(deadline)
            if not self.inbox.empty():
                raise RolloutWorkerError("unexpected pending or duplicate worker output")
            snapshot = {}
            for key, value in weights.items():
                if not isinstance(key, str):
                    raise ValueError("state dict keys must be strings")
                if isinstance(value, torch.Tensor):
                    if value.device.type != "cpu":
                        raise ValueError("frozen rollout weights must already be on CPU")
                    value = value.detach().numpy()
                array = np.array(value, copy=True)
                if array.dtype.hasobject or not np.isfinite(array).all():
                    raise ValueError("frozen rollout weights must be finite numeric arrays")
                snapshot[key] = array
            weights_blob = _encode(snapshot, MAX_WEIGHTS_BYTES-4096)
            del snapshot
            digest = hashlib.sha256(weights_blob).hexdigest()
            self.batch_id += 1
            expected = {}
            dispatch_started = time.perf_counter()
            sent = queue.Queue(maxsize=self.workers)
            self.senders = []
            for worker_id, index in enumerate(indices):
                _deadline(deadline)
                identity = dict(worker_id=worker_id, batch_id=self.batch_id,
                    policy_version=policy_version, policy_sha256=digest, episode_index=index)
                expected[worker_id] = identity
                payload = _encode(dict(identity, kind="collect", weights=weights_blob,
                    deadline=deadline), MAX_WEIGHTS_BYTES)
                sender = threading.Thread(target=_sender,
                    args=(self.connections[worker_id], payload, worker_id, sent), daemon=True,
                    name=f"paper-rollout-sender-{worker_id}")
                self.senders.append(sender)
                sender.start()
            for _ in indices:
                while True:
                    _deadline(deadline)
                    try:
                        worker_id, error = sent.get(timeout=min(.1, max(0., deadline-time.perf_counter())))
                        break
                    except queue.Empty:
                        continue
                if error is not None:
                    raise RolloutWorkerError(f"rollout worker {worker_id} dispatch failed: {error}")
            dispatch_seconds = time.perf_counter()-dispatch_started
            del weights_blob, payload
            results, receipts, seen = {}, {}, set()
            while len(results) < self.workers or len(receipts) < self.workers:
                worker_id, message, receive_timing = self._receive(deadline)
                identity = expected[worker_id]
                if any(message.get(key) != value for key, value in identity.items()):
                    raise RolloutWorkerError("stale or mismatched rollout message identity")
                if message.get("kind") == "result":
                    validation_started = time.perf_counter()
                    validate_episode_result(message, identity, seen=seen)
                    index = identity["episode_index"]
                    seeds = {"scenario_seed": self.config["scenario_seed_start"]+index,
                             "action_seed": self.config["sampling_seed"]+index,
                             "global_seed": self.config["training_seed"]+index}
                    if any(message.get(key) != value for key, value in seeds.items()):
                        raise RolloutWorkerError("episode RNG or scenario seed mismatch")
                    message["timing"].update(receive_timing)
                    message["timing"]["parent_validation_seconds"] = time.perf_counter()-validation_started
                    results[worker_id] = message
                elif message.get("kind") == "sent":
                    if worker_id in receipts or worker_id not in results:
                        raise RolloutWorkerError("duplicate or premature payload receipt")
                    if message.get("payload_bytes") != results[worker_id]["timing"]["received_bytes"]:
                        raise RolloutWorkerError("partial rollout payload transfer")
                    receipts[worker_id] = message
                else:
                    raise RolloutWorkerError("unexpected rollout message")
            _deadline(deadline)
            if seen != set(indices):
                raise RolloutWorkerError("incomplete episode batch")
            if not self.inbox.empty():
                raise RolloutWorkerError("unexpected pending or duplicate worker output")
            ordered = sorted(results.values(), key=lambda result: result["episode_index"])
            samples, episodes = [], []
            for result in ordered:
                samples.extend(result["samples"])
                receipt = receipts[result["worker_id"]]
                result["timing"].update({key: receipt[key] for key in
                    ("payload_bytes", "serialization_seconds", "send_seconds")})
                episodes.append(dict(summary=result["summary"], **{key: result[key] for key in
                    ("worker_id", "batch_id", "policy_version", "policy_sha256", "episode_index",
                     "scenario_seed", "action_seed", "global_seed", "sample_sha256", "timing")}))
            wall_seconds = time.perf_counter()-started
            return dict(samples=samples, episodes=episodes, batch_id=self.batch_id,
                policy_version=policy_version, policy_sha256=digest,
                timing=dict(wall_seconds=wall_seconds, dispatch_seconds=dispatch_seconds,
                    startup_seconds=self.startup_seconds, initialized=self.initialized,
                    payload_bytes=sum(row["timing"]["payload_bytes"] for row in episodes),
                    samples_per_second=len(samples)/wall_seconds))
        except BaseException:
            self.close()
            raise

    def close(self):
        """Reap only this pool's Process objects; idempotent after any failure."""
        if self.closed:
            return
        self.closed = True
        self.stopping.set()
        for process in self.processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=2.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.)
        for connection in self.connections:
            connection.close()
        for reader in [*self.readers, *self.senders]:
            reader.join(timeout=.5)
        while True:
            try:
                self.inbox.get_nowait()
            except queue.Empty:
                break

    def __enter__(self):
        if self.closed:
            raise RolloutWorkerError("rollout pool is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

"""Bounded hardware timing of real paper rollouts; launch only through lab.

This is a throughput diagnostic, not convergence training. CUDA sampling is
explicitly brought back to the existing CPU random stream. Parallel workers
collect frozen-policy episodes only; they do not change the training runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import statistics
import subprocess
import threading
import time

import numpy as np
import torch

import paper_train
from paper_environment import PaperEnvironment, load_environment_config
from shared_ppo import SharedActorCritic, collate, update

OUT = Path(os.environ['LAB_RUN_DIR'])
SEED = 920001
MODEL_SEED = 920101
SAMPLE_SEED = 920201
SHUFFLE_SEED = 920301
ENV_CONFIG = 'configs/paper_environment_shared_navigation.json'
CORES = [12, 13, 14, 15]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def gpu_snapshot():
    def query(fields, apps=False):
        flag = '--query-compute-apps=' if apps else '--query-gpu='
        return subprocess.check_output(['nvidia-smi', flag + fields,
                                        '--format=csv,noheader,nounits'],
                                       text=True, timeout=5).strip()
    raw = query('memory.total,memory.used,memory.free,utilization.gpu')
    rows = raw.splitlines()
    if len(rows) != 1:
        raise RuntimeError('benchmark requires exactly one visible GPU')
    total, used, free, utilization = [int(x.strip()) for x in rows[0].split(',')]
    apps = []
    for line in query('pid,used_gpu_memory', True).splitlines():
        pid, memory = line.split(',')
        apps.append({'pid': int(pid), 'used_mib': int(memory)})
    return dict(time=time.time(), total_mib=total, used_mib=used, free_mib=free,
                utilization_percent=utilization, apps=apps)


class GPUWatch:
    """Fail closed on missing protected GPU jobs/headroom; exit OUR process only.

    Fraction cap covers the torch allocator, not context/library allocations.
    Driver queries see host compute PIDs even inside the private PID namespace.
    No signals are ever sent to queried PIDs.
    """
    def __init__(self):
        self.stop = threading.Event()
        self.path = OUT / 'gpu-telemetry.jsonl'
        row = self.check()
        if row['free_mib'] < 3000:
            raise RuntimeError('less than 3000 MiB free before CUDA initialization')
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        try:
            torch.cuda.memory.set_per_process_memory_fraction(256 / row['total_mib'])
            self.check()  # Context initialization also allocates outside torch.
        except Exception:
            self.stop.set()
            self.thread.join(timeout=6)
            raise

    def check(self):
        row = gpu_snapshot()
        row['own_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        with self.path.open('a') as out:
            out.write(json.dumps(row) + '\n')
        if not {3059945, 3062529} <= {a['pid'] for a in row['apps']}:
            raise RuntimeError('a protected GPU PID is absent; stop our diagnostic')
        if row['free_mib'] < 2048 or row['own_peak_rss_mib'] > 2048:
            raise RuntimeError('GPU headroom or own RSS floor exceeded')
        return row

    def loop(self):
        while not self.stop.wait(1):
            try:
                self.check()
            except Exception as error:
                try:
                    write('guard-stop.json', {'error': repr(error)})
                finally:
                    os._exit(74)  # This benchmark process only; lab reaps it.

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        self.check()


def config(device='cpu', threads=1):
    cfg = json.loads(Path('configs/paper_ppo.json').read_text())
    cfg.update(device=device, torch_num_threads=threads)
    return cfg


def model_and_optimizer(cfg, weights=None):
    torch.manual_seed(MODEL_SEED)
    model = SharedActorCritic(cfg)
    if weights is not None:
        model.load_state_dict(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 betas=cfg['adam_betas'], eps=cfg['adam_eps'],
                                 weight_decay=cfg['weight_decay'], foreach=False, fused=False)
    return model, optimizer


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_tree(v) for v in value]
    return value


def diff_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((diff_tree(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((diff_tree(x, y) for x, y in zip(a, b)), default=0.)
    assert a == b
    return 0.


def select(model, observations, generator, *, deterministic=False):
    ids = sorted(observations)
    if not ids:
        return {}, {}, {}
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, values = model(*collate([observations[k] for k in ids], device=device))
        logits, values = logits.cpu(), values.cpu()
        logs = torch.log_softmax(logits, dim=-1)
        actions = (logits.argmax(-1) if deterministic else
                   torch.multinomial(logs.exp(), 1, generator=generator).squeeze(-1))
        chosen = logs.gather(1, actions[:, None]).squeeze(-1)
    return ({k: int(actions[i]) for i, k in enumerate(ids)},
            {k: float(chosen[i]) for i, k in enumerate(ids)},
            {k: float(values[i]) for i, k in enumerate(ids)})


def episode(model, cfg, seed=SEED, env=None):
    started = time.perf_counter()
    if env is None:
        ecfg, parts = load_environment_config(ENV_CONFIG)
        env = PaperEnvironment(ecfg, parts)
    observations = env.reset(seed)
    setup = time.perf_counter() - started
    timing = {'selection_seconds': 0., 'step_seconds': 0., 'gae_seconds': 0., 'decisions': 0}
    old_select, old_gae, old_step = paper_train.select_actions, paper_train.compute_gae, env.step
    def timed_select(*args, **kwargs):
        t = time.perf_counter()
        result = select(*args, **kwargs)
        timing['selection_seconds'] += time.perf_counter() - t
        timing['decisions'] += 1
        return result
    def timed_step(*args, **kwargs):
        t = time.perf_counter()
        result = old_step(*args, **kwargs)
        timing['step_seconds'] += time.perf_counter() - t
        return result
    def timed_gae(*args, **kwargs):
        t = time.perf_counter()
        result = old_gae(*args, **kwargs)
        timing['gae_seconds'] += time.perf_counter() - t
        return result
    paper_train.select_actions, paper_train.compute_gae, env.step = timed_select, timed_gae, timed_step
    try:
        samples, summary = paper_train.collect_episode(
            env, model, torch.Generator().manual_seed(SAMPLE_SEED + seed - SEED), cfg,
            observations=observations, deadline=time.perf_counter() + 120)
    finally:
        paper_train.select_actions, paper_train.compute_gae, env.step = old_select, old_gae, old_step
    digest = hashlib.sha256()
    for row in samples:
        digest.update(json.dumps([row['aircraft_id'], row['decision_index'], row['action'],
                                  row['reward'], row['terminated']], separators=(',', ':')).encode())
    return samples, dict(summary=summary, setup_seconds=setup, **timing,
                         action_reward_trace_sha256=digest.hexdigest())


def fixture():
    cfg = config()
    model, optimizer = model_and_optimizer(cfg)
    initial = cpu_tree(model.state_dict())
    samples, record = episode(model, cfg)
    groups = Counter(s['decision_index'] for s in samples)
    busiest = max(groups, key=groups.get)
    inference = [s for s in samples if s['decision_index'] == busiest]
    with torch.no_grad():
        logits, values = model(*collate(inference))
    metrics = update(model, optimizer, samples, cfg, generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    payload = cpu_tree(dict(weights=initial, samples=samples, inference=inference,
                           ref_logits=logits, ref_values=values,
                           ref_model=model.state_dict(), ref_adam=optimizer.state_dict(),
                           ref_metrics=metrics, episode=record))
    torch.save(payload, OUT / 'fixture.pt')
    write('fixture.json', dict(**record, samples=len(samples), inference_batch=len(inference),
                              intruder_counts=dict(Counter(len(s['intruders']) for s in samples)),
                              parameters=sum(p.numel() for p in model.parameters()), metrics=metrics))


def fixed(payload, cfg):
    model, optimizer = model_and_optimizer(cfg, payload['weights'])
    device = cfg['device']
    rows = payload['samples']
    with torch.no_grad():
        logits, values = model(*collate(payload['inference'], device=device))
    mask = torch.isfinite(payload['ref_logits'])
    logits_cpu, values_cpu = logits.cpu(), values.cpu()
    assert torch.equal(torch.isfinite(logits_cpu), mask)
    # Explicit numerical checks, distinct from exact trace agreement.
    torch.testing.assert_close(logits_cpu[mask], payload['ref_logits'][mask], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(values_cpu, payload['ref_values'], rtol=2e-5, atol=2e-6)
    forward_error = float((logits_cpu[mask] - payload['ref_logits'][mask]).abs().max())
    value_error = float((values_cpu - payload['ref_values']).abs().max())
    gen = torch.Generator().manual_seed(SAMPLE_SEED)
    obs = {str(i): row for i, row in enumerate(payload['inference'])}
    for _ in range(5):
        select(model, obs, gen)
    sync(device)
    t = time.perf_counter()
    for _ in range(30):
        select(model, obs, gen)
    sync(device)
    inference_seconds = (time.perf_counter() - t) / 30
    update(model, optimizer, rows[:64], cfg, generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    times, errors = [], []
    for _ in range(3):
        model, optimizer = model_and_optimizer(cfg, payload['weights'])
        generator = torch.Generator().manual_seed(SHUFFLE_SEED)
        sync(device)
        t = time.perf_counter()
        metrics = update(model, optimizer, rows, cfg, generator=generator)
        sync(device)
        times.append(time.perf_counter() - t)
        errors.append(dict(model=diff_tree(cpu_tree(model.state_dict()), payload['ref_model']),
                           adam=diff_tree(cpu_tree(optimizer.state_dict()), payload['ref_adam'])))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v.cpu(), payload['ref_model'][k], rtol=3e-4, atol=3e-5)
        assert metrics['samples'] == len(rows)
    return dict(config=cfg, samples=len(rows), inference_batch=len(obs),
                selection_seconds_per_batch=inference_seconds, update_seconds=times,
                median_update_seconds=statistics.median(times), numerical_errors=errors,
                max_logit_error=forward_error, max_value_error=value_error, final_metrics=metrics)


def full_episode(payload, cfg):
    model, optimizer = model_and_optimizer(cfg, payload['weights'])
    samples, result = episode(model, cfg)
    sync(cfg['device'])
    t = time.perf_counter()
    metrics = update(model, optimizer, samples, cfg, generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    sync(cfg['device'])
    result['update_seconds'] = time.perf_counter() - t
    result['collection_update_seconds'] = result['summary']['collection_wall_seconds'] + result['update_seconds']
    result['samples_per_second'] = len(samples) / result['collection_update_seconds']
    result['same_trace_as_fixture'] = result['action_reward_trace_sha256'] == payload['episode']['action_reward_trace_sha256']
    result['metrics'] = metrics
    return result


def pool_worker(index, seeds, weights, ready, results, start):
    try:
        os.sched_setaffinity(0, {CORES[index]})
        torch.set_num_interop_threads(1)
        t = time.perf_counter()
        cfg = config()
        model, _ = model_and_optimizer(cfg, weights)
        ecfg, parts = load_environment_config(ENV_CONFIG)
        env = PaperEnvironment(ecfg, parts)
        ready.put(dict(index=index, init_seconds=time.perf_counter() - t))
        if not start.wait(90):
            raise RuntimeError('pool start timeout')
        t = time.perf_counter()
        records = []
        for seed in seeds:
            _, record = episode(model, cfg, seed, env)
            records.append(record)
        results.put(dict(index=index, records=records, wall_seconds=time.perf_counter() - t,
                         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    except Exception as error:
        results.put(dict(index=index, error=repr(error)))
        raise


def pool(payload, workers):
    ctx = mp.get_context('spawn')
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    seeds = list(range(SEED, SEED + 4))
    processes = [ctx.Process(target=pool_worker,
                            args=(i, seeds[i::workers], payload['weights'], ready, results, start))
                 for i in range(workers)]
    t = time.perf_counter()
    try:
        for p in processes:
            p.start()
        initialized = [ready.get(timeout=90) for _ in processes]
        startup = time.perf_counter() - t
        t = time.perf_counter()
        start.set()
        records = [results.get(timeout=240) for _ in processes]
        collection_wall = time.perf_counter() - t
        for p in processes:
            p.join(timeout=10)
            assert p.exitcode == 0
        if any('error' in r for r in records):
            raise RuntimeError(records)
        samples = sum(e['summary']['rollout_samples'] for r in records for e in r['records'])
        return dict(workers=workers, seeds=seeds, startup_seconds=startup,
                    collection_wall_seconds=collection_wall, samples=samples,
                    samples_per_second=samples / collection_wall, initialized=initialized,
                    records=sorted(records, key=lambda r: r['index']), frozen_policy_only=True)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # only the Process objects created above
            if p.pid is not None:
                p.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fixture', 'bench', 'pool'])
    parser.add_argument('--fixture')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, nargs='+', choices=[1, 2, 4], default=[1])
    parser.add_argument('--workers', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--episode', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and args.mode != 'bench':
        parser.error('CUDA is restricted to the single-process benchmark')
    torch.set_num_interop_threads(1)
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    guard = GPUWatch() if args.device == 'cuda' else None
    try:
        write('runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
              versions={n: importlib.metadata.version(n) for n in ('numpy', 'bluesky-simulator', 'openap')},
              affinity=sorted(os.sched_getaffinity(0)), interop=torch.get_num_interop_threads(),
              fp32_precision='ieee', adam_foreach=False, adam_fused=False,
              argv=vars(args), allocator_cap_mib=256 if guard else None))
        if args.mode == 'fixture':
            fixture()
        else:
            payload = torch.load(args.fixture, map_location='cpu', weights_only=True)
            if args.mode == 'pool':
                write('pool.json', pool(payload, args.workers))
            else:
                for threads in args.threads:
                    os.sched_setaffinity(0, set({1: [14], 2: [14, 15], 4: CORES}[threads]))
                    cfg = config(args.device, threads)
                    result = fixed(payload, cfg)
                    result['affinity'] = sorted(os.sched_getaffinity(0))
                    write(f'fixed-{threads}.json', result)
                    if args.episode:
                        result['episode'] = full_episode(payload, cfg)
                        write(f'bench-{threads}.json', result)
                    print(json.dumps({'threads': threads, 'update_seconds': result['update_seconds'],
                                      'episode_seconds': result.get('episode', {}).get('collection_update_seconds')}), flush=True)
        if guard:
            sync('cuda')
            write('gpu-memory.json', dict(max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  final_free_total_bytes=torch.cuda.mem_get_info()))
    finally:
        if guard:
            guard.close()


if __name__ == '__main__':
    main()

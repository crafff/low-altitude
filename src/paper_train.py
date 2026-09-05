"""Bounded CPU PPO with serial or synchronous episode collection, via lab."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch

from shared_ppo import SharedActorCritic, collate, compute_gae, update
from rollout_errors import CollectionCutoff


SCOPE = "literal-environment learning diagnostic; corridor containment unresolved"
EXECUTION_SCOPE = "execution-semantics learning diagnostic; corridor containment unresolved"
CHECKPOINT_SCHEMA = "bluesky.paper-like.training-checkpoint.v1"
PARALLEL_CHECKPOINT_SCHEMA = "bluesky.paper-like.training-checkpoint.parallel.v1"


def batch_size(cfg):
    settings = cfg.get("parallel_rollout")
    if settings is None:
        return 1
    if (not isinstance(settings, dict)
            or settings.get("mode") != "synchronous_complete_episodes"
            or settings.get("sampling_seed_rule") != "base_plus_episode_index"
            or type(settings.get("workers")) is not int
            or settings["workers"] != 4
            or type(settings.get("episodes_per_update")) is not int
            or settings["episodes_per_update"] != 4):
        raise ValueError("parallel rollout requires four synchronous complete episodes")
    cores = settings.get("cpu_ids")
    if (not isinstance(cores, list) or len(cores) != 4 or len(set(cores)) != 4
            or any(type(c) is not int or c < 0 for c in cores)):
        raise ValueError("parallel rollout needs four distinct CPU IDs")
    return 4


def checkpoint_schema(cfg):
    return PARALLEL_CHECKPOINT_SCHEMA if batch_size(cfg) > 1 else CHECKPOINT_SCHEMA


def validated_scope(cfg):
    """Accept only the two declared diagnostic scopes, without efficacy claims."""
    scope = cfg.get("scope")
    if not isinstance(scope, str) or scope not in (SCOPE, EXECUTION_SCOPE):
        raise ValueError("scope must be an explicitly supported learning diagnostic")
    return scope


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise CollectionCutoff("inner wall-clock cutoff reached between decisions")


def select_actions(model, observations, generator, *, deterministic=False):
    """Sorted persistent IDs and a caller-owned CPU random stream."""
    ids = sorted(observations)
    if not ids:
        return {}, {}, {}
    with torch.no_grad():
        logits, values = model(*collate([observations[acid] for acid in ids]))
        log_probs = torch.log_softmax(logits, dim=-1)
        actions = (logits.argmax(dim=-1) if deterministic else
                   torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1))
        chosen_log_probs = log_probs.gather(1, actions[:, None]).squeeze(-1)
    return ({acid: int(actions[i]) for i, acid in enumerate(ids)},
            {acid: float(chosen_log_probs[i]) for i, acid in enumerate(ids)},
            {acid: float(values[i]) for i, acid in enumerate(ids)})


def collect_episode(env, model, sampling_generator, ppo_cfg, *, observations=None,
                    deadline=None):
    """Collect a reset environment to termination; return (PPO samples, summary).

    All rows retain aircraft_id, reward, value and terminated for diagnostics,
    but only the compact summary is written to disk. On CollectionCutoff the
    caller must restore pre-episode RNG state and discard this episode.
    """
    started = time.perf_counter()
    observations = env.observations() if observations is None else observations
    trajectories, finished, histogram = {}, set(), Counter()
    decision = 0
    previous_mode = model.training
    model.eval()
    try:
        while not env.done:
            _check_deadline(deadline)
            if finished.intersection(observations):
                raise RuntimeError("an aircraft ID was reused after termination")
            snapshots = {acid: {key: np.array(obs[key], copy=True)
                                for key in ("own", "intruders", "action_mask")}
                         for acid, obs in observations.items()}
            actions, log_probs, values = select_actions(model, snapshots, sampling_generator)
            following, rewards, terminated, info = env.step(actions)
            controlled = set(snapshots)
            if set(rewards) != controlled or set(terminated) != controlled:
                raise RuntimeError("post-step reward/termination IDs differ from sampled aircraft")
            if bool(info["done"]) != bool(env.done):
                raise RuntimeError("environment done flags disagree")
            for acid in sorted(controlled):
                terminal = terminated[acid]
                if not isinstance(terminal, (bool, np.bool_)):
                    raise RuntimeError("termination flags must be boolean")
                if bool(terminal) == (acid in following):
                    raise RuntimeError("aircraft disappearance disagrees with termination")
                reward = float(rewards[acid])
                if not math.isfinite(reward):
                    raise RuntimeError("environment reward must be finite")
                row = dict(snapshots[acid], aircraft_id=acid, decision_index=decision,
                           action=actions[acid], old_log_prob=log_probs[acid],
                           value=values[acid], reward=reward, terminated=bool(terminal))
                trajectories.setdefault(acid, []).append(row)
                histogram[actions[acid]] += 1
                if terminal:
                    finished.add(acid)
            if finished.intersection(following):
                raise RuntimeError("a terminated aircraft reappeared in the next observation")
            observations = following
            decision += 1
        if observations:
            raise RuntimeError("a completed episode still has active observations")
        samples = []
        for acid in sorted(trajectories):
            rows = trajectories[acid]
            if not rows[-1]["terminated"]:
                raise RuntimeError("completed episode contains an unterminated aircraft trajectory")
            advantages, returns = compute_gae(
                [row["reward"] for row in rows], [row["value"] for row in rows],
                [row["terminated"] for row in rows], 0.,
                ppo_cfg["gamma"], ppo_cfg["gae_lambda"])
            for row, advantage, target in zip(rows, advantages, returns):
                samples.append(dict(row, advantage=float(advantage), **{"return": float(target)}))
        if not samples:
            raise RuntimeError("completed training episode produced no policy transitions")
        summary = env.summary(include_flights=False)
        if not summary["completed_population"]:
            raise RuntimeError("environment summary reports an unfinished population")
        summary = dict(summary, rollout_samples=len(samples), rollout_aircraft=len(trajectories),
                       collected_reward_sum=sum(row["reward"] for row in samples),
                       action_histogram=[histogram[i] for i in range(60)],
                       collection_wall_seconds=time.perf_counter()-started)
        return samples, summary
    finally:
        model.train(previous_mode)


def capture_rng(sampling_generator, shuffle_generator):
    numpy_state = np.random.get_state()
    return {"sampling": sampling_generator.get_state().clone(),
            "shuffle": shuffle_generator.get_state().clone(),
            "torch": torch.random.get_rng_state().clone(),
            "python": random.getstate(),
            "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])}}


def restore_rng(state, sampling_generator, shuffle_generator):
    sampling_generator.set_state(state["sampling"])
    shuffle_generator.set_state(state["shuffle"])
    torch.random.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))


def make_checkpoint(model, optimizer, *, completed_episodes, next_seed_index,
                    sampling_generator, shuffle_generator, configs, versions,
                    evaluation=None, best_checkpoint=None):
    """Snapshot tensors rather than retaining references changed by future steps."""
    scope = validated_scope(configs["training"])
    if evaluation is not None and evaluation.get("scope") != scope:
        raise ValueError("checkpoint evaluation scope differs from training scope")
    size = batch_size(configs["training"])
    if (type(completed_episodes) is not int or completed_episodes < 0
            or type(next_seed_index) is not int or next_seed_index != completed_episodes
            or completed_episodes % size):
        raise ValueError("checkpoint requires complete rollout batches and fresh scenario seeds")
    if best_checkpoint is not None and best_checkpoint.get("best_checkpoint") is not None:
        raise ValueError("best checkpoint nesting must be at most one level")
    if best_checkpoint is not None:
        _validate_checkpoint(best_checkpoint, configs, versions)
    return copy.deepcopy({"schema": checkpoint_schema(configs["training"]), "scope": scope,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "completed_episodes": completed_episodes, "next_seed_index": next_seed_index,
        "completed_update_batches": completed_episodes // size,
        "rng": capture_rng(sampling_generator, shuffle_generator),
        "configs": configs, "versions": versions, "evaluation": evaluation,
        "best_checkpoint": best_checkpoint})


def save_checkpoint(path, payload):
    """Atomic replacement using a temporary file owned by this function."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_checkpoint(payload, configs, versions):
    scope = validated_scope(configs["training"])
    size = batch_size(configs["training"])
    if payload.get("schema") != checkpoint_schema(configs["training"]):
        raise ValueError("unsupported training checkpoint")
    if payload.get("scope") != scope:
        raise ValueError("checkpoint scope differs from configured training scope")
    if payload["configs"] != configs:
        raise ValueError("checkpoint configurations/development cases are incompatible")
    if payload["versions"] != versions:
        raise ValueError("checkpoint software/source versions are incompatible")
    evaluation = payload.get("evaluation")
    if evaluation is not None and evaluation.get("scope") != scope:
        raise ValueError("checkpoint evaluation scope differs from training scope")
    episodes, index = payload["completed_episodes"], payload["next_seed_index"]
    if (type(episodes) is not int or episodes < 0 or type(index) is not int
            or index != episodes or episodes % size
            or (size > 1 and (type(payload.get("completed_update_batches")) is not int
                             or payload["completed_update_batches"] != episodes // size))):
        raise ValueError("invalid completed episode/next seed counters")
    for tensor in payload["model"].values():
        if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint model must contain finite CPU tensors")
    for state in payload["optimizer"]["state"].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and (value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
                raise ValueError("checkpoint Adam state must contain finite CPU tensors")
    best = payload.get("best_checkpoint")
    if best is not None:
        if best.get("best_checkpoint") is not None or best["completed_episodes"] > episodes:
            raise ValueError("invalid embedded best checkpoint")
        _validate_checkpoint(best, configs, versions)


def load_checkpoint(path, model, optimizer, *, sampling_generator, shuffle_generator,
                    configs, versions):
    payload = torch.load(Path(path), weights_only=True, map_location="cpu")
    _validate_checkpoint(payload, configs, versions)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], sampling_generator, shuffle_generator)
    return payload


def aggregate_summaries(summaries):
    """Ratio of exposure sums to flight-hour sums; never mean per-case rates."""
    planned = sum(row["planned"] for row in summaries)
    hours = sum(row["flight_hours"] for row in summaries)
    if planned <= 0 or hours <= 0:
        raise ValueError("development aggregation requires population and positive exposure")
    additive = ("completed", "failed_timeout", "path_length_m", "outside_corridor_aircraft_seconds",
                "outside_corridor_flights", "outside_altitude_aircraft_seconds", "changed_instructions",
                "policy_decisions", "return_sum")
    result = {key: sum(row[key] for row in summaries) for key in additive}
    for key in ('failed_route_exhausted', 'outside_exit_crossings'):
        result[key] = sum(row.get(key, 0) for row in summaries)
    result.update(planned=planned, flight_hours=hours, completed_fraction=result["completed"]/planned,
                  max_centerline_distance_m=max(row["max_centerline_distance_m"] for row in summaries))
    result["risk"] = {}
    for level in ("lowc", "nmac"):
        values = {key: sum(row["risk"][level][key] for row in summaries)
                  for key in ("unordered_pair_seconds", "directed_pair_seconds")}
        values["unordered_seconds_per_flight_hour"] = values["unordered_pair_seconds"]/hours
        values["directed_seconds_per_flight_hour"] = values["directed_pair_seconds"]/hours
        result["risk"][level] = values
    return result


def selection_key(aggregate, minimum_completed_fraction=.95):
    fraction = aggregate["completed_fraction"]
    nmac = aggregate["risk"]["nmac"]["unordered_seconds_per_flight_hour"]
    lowc = aggregate["risk"]["lowc"]["unordered_seconds_per_flight_hour"]
    if not all(math.isfinite(v) for v in (fraction, nmac, lowc)):
        raise ValueError("selection metrics must be finite")
    return (0, nmac, lowc, 0.) if fraction >= minimum_completed_fraction else (1, -fraction, nmac, lowc)


def _evaluate_case(env, model, scenario, policy, action_seed, deadline):
    observations = env.reset(scenario["seed"], scenario=copy.deepcopy(scenario))
    generator = torch.Generator(device="cpu").manual_seed(action_seed)
    histogram = Counter()
    while not env.done:
        _check_deadline(deadline)
        actions = None
        if policy != "nr":
            actions, _, _ = select_actions(model, observations, generator, deterministic=policy=="argmax")
            histogram.update(actions.values())
        observations, _, _, _ = env.step(actions)
    summary = env.summary(include_flights=False)
    if not summary["completed_population"]:
        raise RuntimeError("unfinished development population")
    return dict(summary, policy=policy, action_seed=None if policy=="nr" else action_seed,
                action_histogram=[histogram[i] for i in range(60)])


def evaluate_development(env, model, cases, scenarios, cfg, nr_cache, *, deadline=None):
    """Evaluate one frozen model on every selected case using separate RNG streams."""
    scope = validated_scope(cfg)
    if not cases or len(cases) != len(scenarios):
        raise ValueError("each declared development case needs its exact scenario")
    previous_mode = model.training
    model.eval()
    results = []
    policies = ["sample"] + (["argmax"] if cfg["report_argmax"] else [])
    try:
        for case, scenario in zip(cases, scenarios):
            key = (case["seed"], case["corridor_count"])
            action_seed = cfg["dev_action_seed_base"] + case["seed"]
            if key not in nr_cache:
                nr_cache[key] = _evaluate_case(env, model, scenario, "nr", action_seed, deadline)
            result = dict(case, action_seed=action_seed, nr=nr_cache[key])
            for policy in policies:
                result[policy] = _evaluate_case(env, model, scenario, policy, action_seed, deadline)
            results.append(result)
    finally:
        model.train(previous_mode)
    aggregates = {policy: aggregate_summaries([row[policy] for row in results])
                  for policy in ["nr", *policies]}
    key = selection_key(aggregates["sample"], cfg["selection_min_completed_fraction"])
    return {"scope": scope, "evaluation_scope": "development selection, not held-out evidence",
            "cases": results, "aggregate": aggregates, "selection_key": list(key),
            "primary_policy": "sample", "case_count": len(cases)}


class WallBudget:
    def __init__(self, started, seconds, cfg):
        self.deadline = started + seconds
        self.grace = cfg["deadline_grace_seconds"]
        self.estimate = cfg["initial_episode_seconds"]
        self.factor = cfg["timing_safety_factor"]
        self.batch_estimate = cfg.get("initial_batch_seconds", self.estimate)

    @property
    def collection_deadline(self):
        return self.deadline-self.grace

    def can_start(self, episode_units=1):
        return time.perf_counter()+episode_units*self.estimate*self.factor < self.collection_deadline

    def observe(self, duration):
        self.estimate = max(self.estimate, duration)

    def can_start_batch(self, evaluation_units, startup_seconds=0.):
        reserve = self.factor * (evaluation_units*self.estimate + self.batch_estimate)
        return time.perf_counter()+reserve+startup_seconds < self.collection_deadline

    def observe_batch(self, duration):
        self.batch_estimate = max(self.batch_estimate, duration)


def _versions():
    directory = Path(__file__).resolve().parent
    names = ("paper_train.py", "shared_ppo.py", "paper_environment.py", "paper_actions.py",
             "paper_observation.py", "paper_scenarios.py", "paper_performance.py",
             "nr_pilot.py", "bluesky_diagnostic.py", "route_completion.py", "navigation_refresh.py",
             "nominal_turn_speed.py", "parallel_rollout.py", "rollout_errors.py")
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "bluesky": importlib.metadata.version("bluesky-simulator"),
            "source_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}}


def _append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n")
        handle.flush()


def _validate_train_config(cfg, episodes, wall_seconds, eval_limit):
    if cfg["device"] != "cpu" or cfg["torch_num_threads"] != 1 or cfg["torch_num_interop_threads"] != 1:
        raise ValueError("this training block requires one CPU thread")
    validated_scope(cfg)
    size = batch_size(cfg)
    if cfg["dev_policy"] != "sample":
        raise ValueError("sampled primary development policy must be explicit")
    for key in ("training_seed", "training_scenario_seed_start", "sampling_seed", "shuffle_seed", "dev_action_seed_base"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(episodes) is not int or episodes < 0 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("episode target must be nonnegative and wall budget positive")
    for key in ("evaluate_every", "checkpoint_every"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if size > 1:
        if any(value % size for value in (episodes, cfg["evaluate_every"], cfg["checkpoint_every"])):
            raise ValueError("parallel episode target/evaluation/save intervals must align with complete batches")
        if not cfg.get("development_scenarios_path"):
            raise ValueError("parallel pilot requires the preserved development scenarios")
        for key in ("initial_batch_seconds", "parallel_startup_seconds"):
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be positive")
    if not isinstance(cfg["report_argmax"], bool):
        raise ValueError("report_argmax must be boolean")
    for key in ("deadline_grace_seconds", "initial_episode_seconds", "timing_safety_factor"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["timing_safety_factor"] < 1 or not 0 < cfg["selection_min_completed_fraction"] <= 1:
        raise ValueError("invalid timing or checkpoint-selection threshold")
    cases = cfg["development_cases"]
    if not cases or len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("development cases must have distinct seeds")
    if any(case["corridor_count"] not in (3, 4, 5) for case in cases):
        raise ValueError("development corridor count must be 3, 4 or 5")
    if any(isinstance(case["seed"], bool) or not isinstance(case["seed"], int) or case["seed"] < 0 for case in cases):
        raise ValueError("development seeds must be nonnegative integers")
    start = cfg["training_scenario_seed_start"]
    if any(start <= case["seed"] < start+episodes for case in cases):
        raise ValueError("training and development seeds overlap")
    if eval_limit is not None and not 1 <= eval_limit <= len(cases):
        raise ValueError("eval-limit must select a nonempty prefix of declared development cases")


def fixed_development_scenarios(cfg, cases):
    """Load the preserved scenarios, validating the full declaration, then select."""
    path = cfg.get("development_scenarios_path")
    if path is None:
        return None
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != cfg["development_scenarios_sha256"]:
        raise ValueError("preserved development scenario file hash differs")
    worlds = json.loads(raw)
    if (not isinstance(worlds, list) or len(worlds) != len(cfg["development_cases"])
            or len({w['seed'] for w in worlds}) != len(worlds)):
        raise ValueError("preserved development scenarios must be unique and complete")
    by_seed = {w['seed']: w for w in worlds}
    for case in cfg['development_cases']:
        world = by_seed.get(case['seed'])
        if world is None or len(world['corridors']) != case['corridor_count']:
            raise ValueError("preserved scenario seed/corridor declaration differs")
    return [by_seed[case['seed']] for case in cases]


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, help="Target TOTAL completed episodes, including resume")
    parser.add_argument("--resume", help="Checkpoint included by the launcher's read-only --input")
    parser.add_argument("--wall-seconds", type=float, help="Inner budget; leave margin in the outer launcher")
    parser.add_argument("--eval-limit", type=int, help="Explicit pilot prefix of the declared development cases")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    episodes = cfg["episodes"] if args.episodes is None else args.episodes
    wall_seconds = cfg["wall_seconds"] if args.wall_seconds is None else args.wall_seconds
    _validate_train_config(cfg, episodes, wall_seconds, args.eval_limit)
    scope = cfg["scope"]
    output = Path(os.environ["LAB_RUN_DIR"])
    if not output.is_dir():
        raise ValueError("LAB_RUN_DIR must be the launcher's existing writable output directory")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(cfg["training_seed"])
    np.random.seed(cfg["training_seed"])
    torch.manual_seed(cfg["training_seed"])
    from paper_environment import PaperEnvironment, load_environment_config
    from paper_scenarios import generate_scenario

    environment_cfg, parts = load_environment_config(cfg["environment_config"])
    ppo_cfg = json.loads(Path(cfg["ppo_config"]).read_text())
    cases = copy.deepcopy(cfg["development_cases"][:args.eval_limit])
    scenarios = fixed_development_scenarios(cfg, cases)
    configs = {"training": cfg, "environment": environment_cfg, "environment_parts": parts,
               "aircraft_types": json.loads(Path(environment_cfg["types_config"]).read_text()),
               "ppo": ppo_cfg, "effective_development_cases": cases}
    if scenarios is not None:
        configs['effective_development_scenarios'] = scenarios
    versions = _versions()
    model = SharedActorCritic(ppo_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg["learning_rate"],
                                 betas=tuple(ppo_cfg["adam_betas"]), eps=ppo_cfg["adam_eps"],
                                 weight_decay=ppo_cfg["weight_decay"])
    sampling = torch.Generator(device="cpu").manual_seed(cfg["sampling_seed"])
    shuffle = torch.Generator(device="cpu").manual_seed(cfg["shuffle_seed"])
    completed = next_index = 0
    best_checkpoint = last_evaluation = None
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, sampling_generator=sampling,
                                  shuffle_generator=shuffle, configs=configs, versions=versions)
        completed, next_index = payload["completed_episodes"], payload["next_seed_index"]
        last_evaluation, best_checkpoint = payload["evaluation"], payload["best_checkpoint"]
        if best_checkpoint is None and last_evaluation is not None:
            best_checkpoint = copy.deepcopy(payload)
    if episodes < completed:
        raise ValueError("episode target precedes the resumed completed episode count")
    budget = WallBudget(started, wall_seconds, cfg)

    def checkpoint(*, include_best=True):
        return make_checkpoint(model, optimizer, completed_episodes=completed, next_seed_index=next_index,
            sampling_generator=sampling, shuffle_generator=shuffle, configs=configs, versions=versions,
            evaluation=last_evaluation, best_checkpoint=best_checkpoint if include_best else None)

    save_checkpoint(output/"latest.pt", checkpoint())
    if best_checkpoint is not None:
        save_checkpoint(output/"best.pt", best_checkpoint)
    startup_rng = capture_rng(sampling, shuffle)
    env = PaperEnvironment(environment_cfg, parts)
    restore_rng(startup_rng, sampling, shuffle)
    if scenarios is None:
        scenarios = [generate_scenario(dict(parts["scenario"], corridor_counts=[case["corridor_count"]]),
                                       case["seed"], list(env.types)) for case in cases]
    nr_cache, evaluations = {}, []
    evaluation_units = len(cases)*(1+int(cfg["report_argmax"]))

    def evaluate(phase):
        nonlocal last_evaluation, best_checkpoint
        required = evaluation_units + len(cases)-len(nr_cache)
        if not budget.can_start(required):
            return False
        evaluation_started = time.perf_counter()
        before_evaluation_rng = capture_rng(sampling, shuffle)
        try:
            result = evaluate_development(env, model, cases, scenarios, cfg, nr_cache,
                                          deadline=budget.collection_deadline)
        except CollectionCutoff:
            return False
        finally:
            restore_rng(before_evaluation_rng, sampling, shuffle)
        result.update(completed_episodes=completed, phase=phase,
                      wall_seconds=time.perf_counter()-evaluation_started)
        durations = [row[policy]["wall_seconds"] for row in result["cases"]
                     for policy in ("nr", "sample", *(["argmax"] if cfg["report_argmax"] else []))]
        for duration in durations:
            budget.observe(duration)
        last_evaluation = result
        evaluations.append({key: result[key] for key in ("scope", "completed_episodes", "phase", "aggregate", "selection_key", "wall_seconds")})
        _append_json(output/"development.jsonl", result)
        if best_checkpoint is None or tuple(result["selection_key"]) < tuple(best_checkpoint["evaluation"]["selection_key"]):
            best_checkpoint = checkpoint(include_best=False)
            save_checkpoint(output/"best.pt", best_checkpoint)
        save_checkpoint(output/"latest.pt", checkpoint())
        print(json.dumps({"scope": scope, "event": "development", "completed_episodes": completed,
                          "phase": phase, "selection_key": result["selection_key"],
                          "sample": result["aggregate"]["sample"], "nr": result["aggregate"]["nr"]}, allow_nan=False), flush=True)
        return True

    stop_reason = "episode_target_reached"
    initial_completed = completed
    initial_evaluated = evaluate("initial_untrained" if completed == 0 else "initial_resumed")
    if not initial_evaluated:
        stop_reason = "insufficient_budget_for_initial_development_evaluation"
    size = batch_size(cfg)
    pool = None
    try:
        while initial_evaluated and completed < episodes:
            startup = cfg.get("parallel_startup_seconds", 0.) if size > 1 and pool is None else 0.
            can_start = (budget.can_start_batch(evaluation_units, startup) if size > 1
                         else budget.can_start(evaluation_units+1))
            if not can_start:
                stop_reason = "stopped_between_batches_to_reserve_final_evaluation"
                break
            before_rng = capture_rng(sampling, shuffle)
            seed = cfg["training_scenario_seed_start"]+next_index
            batch = None
            try:
                if size > 1 and pool is None:
                    from parallel_rollout import ParallelRolloutPool
                    settings = cfg['parallel_rollout']
                    pool_started = time.perf_counter()
                    pool = ParallelRolloutPool(cfg['environment_config'], ppo_cfg,
                        workers=settings['workers'], cpu_ids=settings['cpu_ids'],
                        scenario_seed_start=cfg['training_scenario_seed_start'],
                        sampling_seed=cfg['sampling_seed'], training_seed=cfg['training_seed'],
                        startup_deadline=min(budget.collection_deadline, pool_started+startup))
                    _append_json(output/'collection.jsonl', dict(event='pool_started',
                        wall_seconds=time.perf_counter()-pool_started))
                episode_started = time.perf_counter()
                if size > 1:
                    batch = pool.collect(copy.deepcopy(model.state_dict()),
                        list(range(next_index, next_index+size)),
                        policy_version=completed//size, deadline=budget.collection_deadline)
                    samples = batch['samples']
                    summary = aggregate_summaries([e['summary'] for e in batch['episodes']])
                else:
                    observations = env.reset(seed)
                    samples, summary = collect_episode(env, model, sampling, ppo_cfg,
                        observations=observations, deadline=budget.collection_deadline)
                _check_deadline(budget.collection_deadline)
            except CollectionCutoff:
                restore_rng(before_rng, sampling, shuffle)
                stop_reason = "partial_batch_discarded_at_resource_cutoff"
                break
            except Exception:
                restore_rng(before_rng, sampling, shuffle)
                save_checkpoint(output/"latest.pt", checkpoint())
                raise
            updates = update(model, optimizer, samples, ppo_cfg, shuffle)
            completed += size
            next_index += size
            last_evaluation = None
            duration = time.perf_counter()-episode_started
            if size > 1:
                budget.observe_batch(duration)
                row = {"scope": scope, "completed_episodes": completed,
                       "completed_update_batches": completed//size,
                       "scenario_seeds": [e['scenario_seed'] for e in batch['episodes']],
                       "episodes": batch['episodes'], "collection_timing": batch['timing'],
                       "policy_version": completed//size-1,
                       "environment": summary, "ppo": updates, "wall_seconds": duration,
                       "elapsed_seconds": time.perf_counter()-started}
            else:
                budget.observe(duration)
                row = {"scope": scope, "completed_episode": completed, "scenario_seed": seed,
                       "environment": summary, "ppo": updates, "wall_seconds": duration,
                       "elapsed_seconds": time.perf_counter()-started}
            _append_json(output/"training.jsonl", row)
            print(json.dumps({"scope": scope, "event": "training", "completed_episodes": completed,
                              "completed_update_batches": completed//size, "first_seed": seed,
                              "completed": summary["completed"], "planned": summary["planned"],
                              "return_sum": summary["return_sum"], "ppo": updates,
                              "wall_seconds": duration}, allow_nan=False), flush=True)
            if completed % cfg["checkpoint_every"] == 0:
                save_checkpoint(output/"latest.pt", checkpoint())
            if completed % cfg["evaluate_every"] == 0 and not evaluate("periodic"):
                stop_reason = "insufficient_budget_for_periodic_development_evaluation"
                break
    finally:
        if pool is not None:
            pool.close()
    final_evaluated = last_evaluation is not None and last_evaluation["completed_episodes"] == completed
    if initial_evaluated and not final_evaluated:
        final_evaluated = evaluate("final")
    save_checkpoint(output/"latest.pt", checkpoint())
    result = {"scope": scope, "configs": configs, "versions": versions,
              "initial_completed_episodes": initial_completed, "completed_episodes": completed,
              "episodes_completed_this_job": completed-initial_completed, "next_seed_index": next_index,
              "completed_update_batches": completed//size, "episodes_per_update": size,
              "target_completed_episodes": episodes, "stop_reason": stop_reason,
              "initial_evaluated": initial_evaluated, "final_evaluated": final_evaluated,
              "best_completed_episodes": None if best_checkpoint is None else best_checkpoint["completed_episodes"],
              "best_selection_key": None if best_checkpoint is None else best_checkpoint["evaluation"]["selection_key"],
              "evaluations": evaluations, "wall_seconds": time.perf_counter()-started,
              "inner_wall_budget_seconds": wall_seconds, "resumed_from": args.resume}
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

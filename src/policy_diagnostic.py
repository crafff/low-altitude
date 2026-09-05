"""Observe one frozen native policy replay without changing its training path."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from paper_train import (_check_deadline, _validate_checkpoint, _validate_train_config,
                         _versions, aggregate_summaries, select_actions)
from shared_ppo import SharedActorCritic


COMPONENTS = ('speed', 'altitude', 'lane')
SHAPE = (4, 5, 3)
STATUSES = ('arrived', 'flight_timeout', 'route_exhausted_without_arrival')
REWARD_KEYS = ('safety', 'efficiency', 'arrival', 'total')
OMITTED_RUNTIME_FIELDS = frozenset(('wall_seconds', 'process_peak_rss_mib'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(canonical([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def scientific(value):
    """Only elapsed runtime and RSS are excluded, at any summary nesting level."""
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k not in OMITTED_RUNTIME_FIELDS}
    if isinstance(value, (list, tuple)):
        return [scientific(v) for v in value]
    return value


def difference_paths(left, right, path='root'):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = [f'{path}.{key}: missing key' for key in sorted(left.keys() ^ right.keys())]
        for key in sorted(left.keys() & right.keys()):
            paths.extend(difference_paths(left[key], right[key], f'{path}.{key}'))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path+': length']
        return [p for i, (a, b) in enumerate(zip(left, right))
                for p in difference_paths(a, b, f'{path}[{i}]')]
    # Canonical scalar comparison preserves bool/int/float and exact values.
    return [] if canonical(left) == canonical(right) else [path]


def component_indices(action):
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 60:
        raise ValueError('Action must be an integer in [0, 60)')
    return int(action)//15, (int(action)//3)%5, int(action)%3


def entropy(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values[values > 0.]
    return -float(np.sum(positive*np.log(positive)))


def component_marginals(probabilities):
    joint = np.asarray(probabilities, dtype=np.float64).reshape(SHAPE)
    return [joint.sum(axis=(1, 2)), joint.sum(axis=(0, 2)), joint.sum(axis=(0, 1))]


def distribution_metrics(logits, mask):
    logits, mask = np.asarray(logits, dtype=np.float64), np.asarray(mask)
    if logits.shape != (60,) or mask.shape != (60,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Expected logits[60] and a nonempty bool action mask')
    if not np.isfinite(logits[mask]).all() or not np.isneginf(logits[~mask]).all():
        raise ValueError('Hook must observe finite legal logits and negative-infinity masked logits')
    probabilities = np.zeros(60, dtype=np.float64)
    weights = np.exp(logits[mask]-logits[mask].max())
    probabilities[mask] = weights/weights.sum()
    count = int(mask.sum())
    conditional = entropy(probabilities)
    normalized = conditional/math.log(count) if count > 1 else None
    if normalized is not None and not -1e-12 <= normalized <= 1.+1e-12:
        raise RuntimeError('Entropy exceeds its legal-action support')
    return dict(probabilities=probabilities, marginals=component_marginals(probabilities),
                valid_count=count, conditional_entropy=conditional,
                normalized_entropy=normalized)


class DecisionStats:
    """Keep conditional and pooled entropies separate while streaming rows."""
    def __init__(self):
        self.count = self.normalized_count = 0
        self.entropy_sum = self.normalized_sum = 0.
        self.probabilities = np.zeros(60, dtype=np.float64)
        self.actions = np.zeros(60, dtype=np.int64)
        self.valid_counts = Counter()

    def add(self, values, action):
        component_indices(action)
        if values['probabilities'][action] <= 0.:
            raise ValueError('Selected action is outside positive legal probability support')
        self.count += 1
        self.entropy_sum += values['conditional_entropy']
        if values['normalized_entropy'] is not None:
            self.normalized_count += 1
            self.normalized_sum += values['normalized_entropy']
        self.probabilities += values['probabilities']
        self.actions[action] += 1
        self.valid_counts[values['valid_count']] += 1

    def merge(self, other):
        self.count += other.count
        self.normalized_count += other.normalized_count
        self.entropy_sum += other.entropy_sum
        self.normalized_sum += other.normalized_sum
        self.probabilities += other.probabilities
        self.actions += other.actions
        self.valid_counts.update(other.valid_counts)

    def summary(self):
        if not self.count:
            return dict(aircraft_decisions=0)
        mixture = self.probabilities/self.count
        selected = self.actions/self.count
        return dict(aircraft_decisions=self.count,
                    mean_conditional_entropy=self.entropy_sum/self.count,
                    mean_entropy_over_log_valid=self.normalized_sum/self.normalized_count if self.normalized_count else None,
                    normalized_entropy_decisions=self.normalized_count,
                    probability_mixture_entropy=entropy(mixture),
                    empirical_selected_action_entropy=entropy(selected),
                    valid_count_histogram={str(k): v for k, v in sorted(self.valid_counts.items())},
                    mean_component_probabilities={k: v.tolist() for k, v in zip(COMPONENTS, component_marginals(mixture))},
                    selected_component_frequencies={k: v.tolist() for k, v in zip(COMPONENTS, component_marginals(selected))},
                    selected_action_histogram=self.actions.tolist(), nominal_selected_fraction=float(selected[37]))


class ForwardCapture:
    """A passive hook: never substitutes output, forwards again, or samples."""
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.pending = None
        self.handle = None

    def __enter__(self):
        self.handle = self.model.register_forward_hook(self._capture)
        return self

    def _capture(self, module, arguments, output):
        if self.pending is not None:
            raise RuntimeError('An unexpected second policy forward occurred')
        self.calls += 1
        self.pending = (output[0].detach().cpu().numpy().copy(),
                        arguments[3].detach().cpu().numpy().copy())
        # Returning None leaves the original output object unchanged.

    def take(self, expected):
        result, self.pending = self.pending, None
        if not expected:
            if result is not None:
                raise RuntimeError('Empty observations unexpectedly triggered a forward')
            return None
        if result is None or result[0].shape != (expected, 60) or result[1].shape != (expected, 60):
            raise RuntimeError('Expected exactly one policy forward for the sampled ID batch')
        return result

    def __exit__(self, *args):
        self.handle.remove()
        self.pending = None


class BoundedCSV:
    def __init__(self, path, fields, max_rows, max_bytes):
        self.path, self.fields = Path(path), fields
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.rows = self.bytes = 0
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open('x', encoding='utf-8', newline='')
        try:
            self._write(self.fields)
        except Exception:
            self.handle.close()
            raise
        return self

    def _write(self, values):
        buffer = io.StringIO(newline='')
        csv.writer(buffer, lineterminator='\n').writerow('null' if value is None else value for value in values)
        line = buffer.getvalue()
        size = len(line.encode('utf-8'))
        if self.bytes+size > self.max_bytes:
            raise RuntimeError('Declared diagnostic CSV byte budget exceeded')
        self.handle.write(line)
        self.bytes += size

    def write(self, values):
        if self.rows >= self.max_rows:
            raise RuntimeError('Declared diagnostic CSV row budget exceeded')
        if values.keys() != set(self.fields):
            raise ValueError('CSV row fields differ from the declared schema')
        self._write([values[key] for key in self.fields])
        self.rows += 1

    def __exit__(self, *args):
        self.handle.close()


def reward_ledger():
    return dict(decisions=0, terminal_rows=0, **dict.fromkeys(REWARD_KEYS, 0.))


def record_reward(ledger, reward, components, terminal, scale):
    if set(components) != set(REWARD_KEYS) or not all(math.isfinite(components[k]) for k in REWARD_KEYS):
        raise ValueError('Reward components must be the original finite four-part mapping')
    if not math.isclose(components['total'], reward, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Returned total differs from the sampled reward')
    reconstructed = components['safety']+scale*components['efficiency']+components['arrival']
    if not math.isclose(reconstructed, reward, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Reward components do not reconstruct the sampled reward')
    ledger['decisions'] += 1
    ledger['terminal_rows'] += int(terminal)
    for key in REWARD_KEYS:
        ledger[key] += components[key]


def return_checks(records, ledgers, scale, arrival_bonus):
    checks = dict(decision_counts=True, terminal_rows=True, per_flight_returns=True,
                  component_totals=True, arrival_bonus_only_on_arrival=True)
    for record in records:
        ledger = ledgers[record['id']]
        checks['decision_counts'] &= ledger['decisions'] == record['policy_decisions']
        checks['terminal_rows'] &= ledger['terminal_rows'] == int(ledger['decisions'] > 0)
        checks['per_flight_returns'] &= math.isclose(ledger['total'], record['return_sum'], rel_tol=0., abs_tol=1e-9)
        checks['component_totals'] &= math.isclose(ledger['safety']+scale*ledger['efficiency']+ledger['arrival'],
                                                  ledger['total'], rel_tol=0., abs_tol=1e-8)
        expected = arrival_bonus if record['status']=='arrived' and ledger['decisions'] else 0.
        checks['arrival_bonus_only_on_arrival'] &= ledger['arrival'] == expected
    return checks


CSV_FIELDS = [
    'scenario_seed', 'completed_episodes', 'decision_index', 'sim_time_s', 'global_step_end_s',
    'reward_time_s', 'aircraft', 'type', 'corridor_id', 'age_before_s', 'age_at_reward_s',
    'lat_deg', 'lon_deg', 'alt_m', 'ground_speed_mps', 'nominal_speed_mps',
    'previous_target_speed_mps', 'previous_target_alt_m', 'previous_target_lane_m',
    'altitude_active', 'lane_active', 'neighbor_count', 'valid_count', 'action_mask_little_endian_hex',
    'conditional_entropy', 'entropy_over_log_valid', 'selected_action', 'speed_index', 'altitude_index',
    'lane_index', 'selected_log_probability', 'value_prediction',
    *[f'{name}_probability_{i}' for name, width in zip(COMPONENTS, SHAPE) for i in range(width)],
    'expected_speed_ratio', 'accepted_target_speed_mps', 'accepted_speed_ratio',
    'accepted_target_alt_m', 'accepted_target_lane_m', 'reward_safety', 'reward_efficiency',
    'reward_scaled_efficiency', 'reward_arrival', 'reward_total', 'terminated', 'status_after_step',
]


def reference_evaluation(path, episode):
    text = Path(path).read_text()
    if Path(path).suffix == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
    matches = [row for row in rows if row.get('completed_episodes') == episode and 'cases' in row]
    if len(matches) != 1:
        raise ValueError(f'Reference must contain exactly one full evaluation for episode {episode}; found {len(matches)}')
    return matches[0]


def strict_inputs(cfg, checkpoint_path, reference_path):
    from paper_environment import load_environment_config

    training = json.loads(Path(cfg['training_config']).read_text())
    _validate_train_config(training, training['episodes'], training['wall_seconds'], None)
    expected = [dict(seed=53001+i, corridor_count=3+i%3) for i in range(12)]
    if cfg['development_cases'] != expected or training['development_cases'] != expected:
        raise ValueError('Keep all 12 original development cases in their declared order')
    environment, parts = load_environment_config(training['environment_config'])
    configs = dict(training=training, environment=environment, environment_parts=parts,
                   aircraft_types=json.loads(Path(environment['types_config']).read_text()),
                   ppo=json.loads(Path(training['ppo_config']).read_text()),
                   effective_development_cases=copy.deepcopy(expected))
    versions = _versions()
    # main verifies LAB_RUN_DIR before calling this function. No module-level load.
    payload = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    _validate_checkpoint(payload, configs, versions)
    reference = reference_evaluation(reference_path, payload['completed_episodes'])
    checkpoint_evaluation = payload['evaluation']
    if checkpoint_evaluation is None or checkpoint_evaluation.get('completed_episodes') != payload['completed_episodes']:
        raise ValueError('The checkpoint itself has no evaluation at its completed episode')
    keys = ('completed_episodes', 'scope', 'evaluation_scope', 'primary_policy', 'case_count',
            'cases', 'aggregate', 'selection_key')
    mismatches = difference_paths(scientific({k: checkpoint_evaluation[k] for k in keys}),
                                 scientific({k: reference[k] for k in keys}))
    if mismatches:
        raise ValueError('Explicit reference differs from checkpoint evaluation: '+', '.join(mismatches[:8]))
    if reference['case_count'] != 12 or len(reference['cases']) != 12 or reference['primary_policy'] != 'sample':
        raise ValueError('Reference must contain 12 sampled-policy cases')
    for case, expected_case in zip(reference['cases'], expected):
        if {k: case[k] for k in expected_case} != expected_case:
            raise ValueError('Reference case order/geometry differs from the original development set')
        seed = training['dev_action_seed_base']+case['seed']
        if case['action_seed'] != seed or case['sample']['action_seed'] != seed or case['sample']['policy'] != 'sample':
            raise ValueError('Reference sampled-policy RNG seed is incompatible')
        if case['nr']['policy'] != 'nr' or not case['nr']['completed_population']:
            raise ValueError('Reference must explicitly contain a completed NR summary')
    if sum(case['sample']['policy_decisions'] for case in reference['cases']) > cfg['csv_max_rows']:
        raise ValueError('Reference decision count already exceeds the declared CSV row limit')
    return payload, reference, configs, versions


def outcome_groups(flights, flight_stats, ledgers):
    groups = defaultdict(list)
    for row in flights:
        key = (row['scenario_seed'], row['id'])
        for group in ('all', 'type/'+row['type'], 'status/'+row['status'],
                      'type_status/'+row['type']+'/'+row['status']):
            groups[group].append((row, flight_stats[key], ledgers[key]))
    result = {}
    for name, members in sorted(groups.items()):
        stats = DecisionStats()
        for _, decisions, _ in members:
            stats.merge(decisions)
        count = len(members)
        rewards = {k: math.fsum(ledger[k] for _, _, ledger in members) for k in REWARD_KEYS}
        result[name] = dict(flights=count,
            statuses={s: sum(row['status']==s for row, _, _ in members) for s in STATUSES},
            flight_seconds=math.fsum(row['flight_seconds'] for row, _, _ in members),
            mean_terminal_age_s=math.fsum(row['terminal_time_s']-row['actual_entry_s'] for row, _, _ in members)/count,
            mean_return_per_flight=rewards['total']/count,
            mean_return_per_decision=rewards['total']/stats.count if stats.count else None,
            reward_components=rewards,
            outside_corridor_seconds=math.fsum(row['outside_corridor_seconds'] for row, _, _ in members),
            path_length_m=math.fsum(row['path_length_m'] for row, _, _ in members),
            decisions=stats.summary())
    return result


def run_case(env, model, capture, case, scenario, episode, cfg, csv_output, deadline, output):
    output.mkdir()
    write_json(output/'scenario.json', scenario)
    observations = env.reset(case['seed'], scenario=copy.deepcopy(scenario))
    generator = torch.Generator(device='cpu').manual_seed(case['action_seed'])
    initial_rng = tensor_digest(generator.get_state())
    initial_model = model_digest(model)
    starting_forwards = capture.calls
    histogram, action_digest = Counter(), hashlib.sha256()
    flight_stats = {acid: DecisionStats() for acid in env.records}
    ledgers = {acid: reward_ledger() for acid in env.records}
    lock_stats = defaultdict(DecisionStats)
    checks = dict(single_forward_per_nonempty_decision=True, sampling_uses_private_torch_rng=True,
                  recorded_mask_matches_original=True, all_recorded_masks_nonempty=True, selected_actions_legal=True,
                  reward_ids_match_sampled_ids=True, terminal_status_matches_flag=True)
    decision = nonempty_decisions = rows = 0
    scale = env.observation_cfg['efficiency_reward_scale']
    while not env.done:
        _check_deadline(deadline)
        now, ids = float(env.bs.sim.simt), sorted(observations)
        states = env.physical_states()
        if ids != sorted(states):
            raise RuntimeError('Diagnostic observations and live true IDs differ')
        before_rng = torch.random.get_rng_state().clone()
        before_calls = capture.calls
        actions, log_probs, values = select_actions(model, observations, generator)
        checks['sampling_uses_private_torch_rng'] &= torch.equal(before_rng, torch.random.get_rng_state())
        checks['single_forward_per_nonempty_decision'] &= capture.calls-before_calls == int(bool(ids))
        captured = capture.take(len(ids))
        nonempty_decisions += int(bool(ids))
        metrics = {}
        for i, acid in enumerate(ids):
            logits, mask = captured[0][i], captured[1][i]
            checks['recorded_mask_matches_original'] &= np.array_equal(mask, observations[acid]['action_mask'])
            checks['all_recorded_masks_nonempty'] &= bool(mask.any())
            checks['selected_actions_legal'] &= bool(mask[actions[acid]])
            metrics[acid] = distribution_metrics(logits, mask)
        histogram.update(actions.values())
        action_digest.update((canonical([now, actions])+'\n').encode())
        following, rewards, terminated, info = env.step(actions)
        expected_ids = set(ids)
        if not all(set(mapping) == expected_ids for mapping in (rewards, terminated, info['reward_components'])):
            raise RuntimeError('Returned reward/termination components differ from sampled IDs')
        end = float(env.bs.sim.simt)
        for acid in ids:
            state, record = states[acid], env.records[acid]
            action, metric = actions[acid], metrics[acid]
            terminal = bool(terminated[acid])
            checks['terminal_status_matches_flag'] &= (record['status'] in STATUSES) == terminal
            reward_time = record['terminal_time_s'] if terminal else end
            accepted = record['accepted_targets'] if terminal else env.actions.state_fields(acid)
            components = info['reward_components'][acid]
            record_reward(ledgers[acid], rewards[acid], components, terminal, scale)
            flight_stats[acid].add(metric, action)
            locks = f'altitude_{int(state.altitude_active)}_lane_{int(state.lane_active)}'
            lock_stats[locks].add(metric, action)
            lock_stats[record['type']+'/'+locks].add(metric, action)
            indexes = component_indices(action)
            row = dict(scenario_seed=case['seed'], completed_episodes=episode, decision_index=decision,
                sim_time_s=now, global_step_end_s=end, reward_time_s=reward_time,
                aircraft=acid, type=record['type'], corridor_id=record['corridor_id'],
                age_before_s=now-record['actual_entry_s'], age_at_reward_s=reward_time-record['actual_entry_s'],
                lat_deg=state.lat_deg, lon_deg=state.lon_deg, alt_m=state.alt_m,
                ground_speed_mps=state.ground_speed_mps, nominal_speed_mps=state.nominal_speed_mps,
                previous_target_speed_mps=state.target_speed_mps, previous_target_alt_m=state.target_alt_m,
                previous_target_lane_m=state.target_lane_m, altitude_active=state.altitude_active,
                lane_active=state.lane_active, neighbor_count=len(observations[acid]['intruders']),
                valid_count=metric['valid_count'], action_mask_little_endian_hex=np.packbits(
                    observations[acid]['action_mask'], bitorder='little').tobytes().hex(),
                conditional_entropy=metric['conditional_entropy'], entropy_over_log_valid=metric['normalized_entropy'],
                selected_action=action, **{k+'_index': v for k, v in zip(COMPONENTS, indexes)},
                selected_log_probability=log_probs[acid], value_prediction=values[acid],
                **{f'{name}_probability_{i}': float(value) for name, marginal in zip(COMPONENTS, metric['marginals'])
                   for i, value in enumerate(marginal)},
                expected_speed_ratio=float(np.dot(metric['marginals'][0], env.action_cfg['speed_ratios'])),
                accepted_target_speed_mps=accepted['target_speed_mps'],
                accepted_speed_ratio=accepted['target_speed_mps']/state.nominal_speed_mps,
                accepted_target_alt_m=accepted['target_alt_m'], accepted_target_lane_m=accepted['target_lane_m'],
                reward_safety=components['safety'], reward_efficiency=components['efficiency'],
                reward_scaled_efficiency=scale*components['efficiency'], reward_arrival=components['arrival'],
                reward_total=rewards[acid], terminated=terminal, status_after_step=record['status'])
            csv_output.write(row)
            rows += 1
        observations = following
        decision += 1
    full = env.summary(include_flights=True)
    flights = [dict(record, scenario_seed=case['seed'], diagnostic_rewards=ledgers[record['id']],
                    diagnostic_decisions=flight_stats[record['id']].summary()) for record in full.pop('flights')]
    sample = dict(full, policy='sample', action_seed=case['action_seed'],
                  action_histogram=[histogram[i] for i in range(60)])
    mismatches = difference_paths(scientific(case['sample']), scientific(sample))
    checks.update(return_checks(flights, ledgers, scale, env.observation_cfg['arrival_reward']))
    checks.update(reference_sample_exact=not mismatches,
                  original_environment_decision_steps=decision == sample['decision_steps'],
                  forward_count_reconciles=capture.calls-starting_forwards == nonempty_decisions,
                  sampled_aircraft_decisions_reconcile=rows == sample['policy_decisions'] == sum(histogram.values()),
                  population_completed=sample['completed_population'] and len(flights)==30,
                  model_state_unchanged=model_digest(model)==initial_model,
                  no_model_gradients=all(p.grad is None for p in model.parameters()),
                  model_remains_evaluation=not model.training)
    keyed_stats = {(case['seed'], acid): values for acid, values in flight_stats.items()}
    keyed_ledgers = {(case['seed'], acid): values for acid, values in ledgers.items()}
    result = dict(seed=case['seed'], scope=cfg['scope'], completed_episodes=episode,
        action_seed=case['action_seed'], sample=sample, nr=case['nr'], nr_execution='reused_reference_not_rerun',
        checks={k: bool(v) for k, v in checks.items()}, all_checks_passed=all(checks.values()),
        reference_difference_count=len(mismatches), reference_difference_paths=mismatches[:100],
        actual_forward_calls=capture.calls-starting_forwards, nonempty_decision_batches=nonempty_decisions,
        global_decision_steps=decision, sampled_aircraft_decisions=rows,
        private_sampling_rng_initial_sha256=initial_rng, private_sampling_rng_final_sha256=tensor_digest(generator.get_state()),
        action_stream_sha256=action_digest.hexdigest(), model_sha256=initial_model,
        flights=flights, groups=outcome_groups(flights, keyed_stats, keyed_ledgers),
        predecision_lock_groups={k: v.summary() for k, v in sorted(lock_stats.items())})
    write_json(output/'result.json', result)
    return result, flights, keyed_stats, keyed_ledgers, lock_stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Explicit checkpoint included by lab --input')
    parser.add_argument('--reference', required=True, help='Explicit full development JSON/JSONL included by lab --input')
    args = parser.parse_args(argv)
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be the existing launcher artifact directory')
    started = time.perf_counter()
    result = dict(all_checks_passed=False, cases=[])
    try:
        cfg = json.loads(Path(args.config).read_text())
        if cfg['schema'] != 'bluesky.policy-diagnostic.v1':
            raise ValueError('Unknown policy diagnostic schema')
        if not 0 < cfg['wall_seconds'] <= 225. or not 1 <= cfg['csv_max_rows'] <= 60000 or not 1 <= cfg['csv_max_bytes'] <= 128*1024**2:
            raise ValueError('Keep the bounded wall/CSV diagnostic budget')
        write_json(output/'input.json', dict(diagnostic_config=cfg, checkpoint=args.checkpoint, reference=args.reference,
                   checkpoint_sha256=file_digest(args.checkpoint), reference_sha256=file_digest(args.reference)))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        payload, reference, configs, versions = strict_inputs(cfg, args.checkpoint, args.reference)
        training = configs['training']
        write_json(output/'effective_config.json', dict(diagnostic=cfg, checkpoint_configs=configs, versions=versions,
                   diagnostic_source_sha256=file_digest(__file__), reference_phase=reference.get('phase')))
        random.seed(training['training_seed'])
        np.random.seed(training['training_seed'])
        torch.manual_seed(training['training_seed'])
        model = SharedActorCritic(configs['ppo'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        initial_model = model_digest(model)
        from paper_environment import PaperEnvironment
        from paper_scenarios import generate_scenario

        env = PaperEnvironment(configs['environment'], configs['environment_parts'])
        scenarios = [generate_scenario(dict(configs['environment_parts']['scenario'],
                     corridor_counts=[case['corridor_count']]), case['seed'], list(env.types))
                     for case in reference['cases']]
        all_flights, all_stats, all_ledgers = [], {}, {}
        locks = defaultdict(DecisionStats)
        csv_output = BoundedCSV(output/'decisions.csv', CSV_FIELDS, cfg['csv_max_rows'], cfg['csv_max_bytes'])
        before_hooks = tuple(model._forward_hooks)
        with torch.no_grad(), csv_output, ForwardCapture(model) as capture:
            for case, scenario in zip(reference['cases'], scenarios):
                values = run_case(env, model, capture, case, scenario, payload['completed_episodes'], cfg,
                                  csv_output, started+cfg['wall_seconds'], output/f"seed-{case['seed']}")
                case_result, flights, stats, ledgers, case_locks = values
                result['cases'].append(case_result)
                all_flights.extend(flights)
                all_stats.update(stats)
                all_ledgers.update(ledgers)
                for key, value in case_locks.items():
                    locks[key].merge(value)
                print(canonical(dict(seed=case['seed'], all_checks_passed=case_result['all_checks_passed'],
                                     aircraft_decisions=case_result['sampled_aircraft_decisions'])), flush=True)
        aggregate = dict(nr=reference['aggregate']['nr'],
                         sample=aggregate_summaries([case['sample'] for case in result['cases']]))
        differences = difference_paths(scientific(reference['aggregate']), scientific(aggregate))
        checks = dict(strict_checkpoint_source_and_config_validated=True,
                      reference_matches_checkpoint_completed_evaluation=True,
                      all_cases_match_reference=all(case['all_checks_passed'] for case in result['cases']),
                      all_twelve_cases_completed=len(result['cases'])==12,
                      aggregate_matches_reference=not differences,
                      csv_rows_reconcile=csv_output.rows==aggregate['sample']['policy_decisions'],
                      csv_bytes_reconcile=csv_output.bytes==(output/'decisions.csv').stat().st_size,
                      hooks_restored=tuple(model._forward_hooks)==before_hooks,
                      model_state_unchanged=model_digest(model)==initial_model,
                      no_model_gradients=all(p.grad is None for p in model.parameters()),
                      forwards_reconcile=capture.calls==sum(case['nonempty_decision_batches'] for case in result['cases']))
        result.update(scope=cfg['scope'], completed_episodes=payload['completed_episodes'],
            checks=checks, all_checks_passed=all(checks.values()), aggregate=aggregate,
            aggregate_difference_paths=differences[:100], actual_forward_calls=capture.calls,
            sampled_aircraft_decisions=csv_output.rows, csv_bytes=csv_output.bytes,
            model_sha256=initial_model, groups=outcome_groups(all_flights, all_stats, all_ledgers),
            predecision_lock_groups={k: v.summary() for k, v in sorted(locks.items())},
            nr_execution='reused_from_explicit_reference; no NR replay',
            inference_scope='One real select_actions forward per nonempty batch; hook-only float64 diagnostics, no extra draws.',
            interpretation='Observed associations across different flights and masks; no causal early-termination exploit claim.')
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), all_checks_passed=False)
    result['wall_seconds'] = time.perf_counter()-started
    write_json(output/'result.json', result)
    print(canonical(dict(all_checks_passed=result['all_checks_passed'], error=result.get('error'))), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Native perception-layer invariance/risk diagnostics, never PPO training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from observation_perturbation import ObservationPerturbation, load_config as load_perturbations
from paper_environment import PaperEnvironment, load_environment_config
from paper_rollout import constructed_scenario
from paper_train import select_actions
from shared_ppo import SharedActorCritic

CASE_SPECS = (
    ('nr', 'nr', 0., 0.), ('plain', 'plain', 0., 0.),
    ('zero', 'perception', 0., 0.), ('position', 'perception', 1., 0.),
    ('communication', 'perception', 0., 1.),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def update_digest(digest, value):
    digest.update((canonical(value)+'\n').encode())


def observations_packet(observations):
    """Preserve float32/mask bits and encoding metadata, not rounded decimals."""
    result = {}
    for acid in sorted(observations):
        obs = observations[acid]
        result[acid] = {
            key: dict(dtype=np.asarray(obs[key]).dtype.str, shape=list(np.asarray(obs[key]).shape),
                      bytes=np.ascontiguousarray(obs[key]).tobytes().hex())
            for key in ('own', 'intruders', 'action_mask')}
        result[acid].update(intruder_ids=list(obs['intruder_ids']), clipping_counts=obs['clipping_counts'])
    return result


def physical_packet(env):
    traf = env.bs.traf
    return dict(sim_time_s=float(env.bs.sim.simt), states={acid: {
        name: float(getattr(traf, field)[i]) for name, field in (
            ('lat_deg', 'lat'), ('lon_deg', 'lon'), ('alt_m', 'alt'), ('tas_mps', 'tas'),
            ('ground_speed_mps', 'gs'), ('vertical_speed_mps', 'vs'),
            ('heading_deg', 'hdg'), ('track_deg', 'trk'))}
        for acid, i in sorted((acid, traf.id2idx(acid)) for acid in traf.id)})


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        update_digest(digest, [name, array.dtype.str, list(array.shape)])
        digest.update(array.tobytes())
    return digest.hexdigest()


def scientific_summary(summary):
    return {k: v for k, v in summary.items() if k not in ('wall_seconds', 'process_peak_rss_mib')}


class PhysicalTrace:
    def __init__(self, env, layer, output, stack):
        self.layer = layer
        self.until = {}
        self.digest = hashlib.sha256()
        self.rows = self.steps = 0
        self.unavailable = dict.fromkeys(env.records, 0.)
        self.alive_seconds = 0.
        self.last_time = float(env.bs.sim.simt)
        self.presence = {}
        self.query_checks = True
        self.writer = csv.DictWriter(stack.enter_context((output/'physics.csv').open('x', newline='')),
            fieldnames=['sim_time_s', 'phase', 'aircraft', 'lat_deg', 'lon_deg', 'alt_m', 'tas_mps',
                        'ground_speed_mps', 'vertical_speed_mps', 'heading_deg', 'track_deg',
                        'interval_start_s', 'alive_interval_seconds', 'unavailable_interval_seconds',
                        'available_at_interval_start', 'available_at_sample', 'known_until_s'])
        self.writer.writeheader()
        self.record(env, initial=True)

    def record(self, env, initial=False):
        packet = physical_packet(env)
        now = packet['sim_time_s']
        if not initial and not math.isclose(now-self.last_time, env.dt, abs_tol=1e-8):
            raise RuntimeError('Physical callback omitted or repeated a native interval')
        update_digest(self.digest, packet)
        for acid, state in packet['states'].items():
            start = now if initial else max(self.last_time, env.records[acid]['actual_entry_s'])
            duration = now-start
            unavailable = 0.
            available_start = available_end = None
            if not initial:
                available_start = self.layer is None or self.layer.availability_at(acid, start)
                available_end = self.layer is None or self.layer.availability_at(acid, now)
                if not available_start:
                    unavailable = max(0., min(now, self.until.get(acid, start))-start)
                self.query_checks &= 0. <= unavailable <= duration
                self.unavailable[acid] += unavailable
                self.alive_seconds += duration
                self.presence[acid] = self.presence.get(acid, 0)+1
            self.writer.writerow(dict(sim_time_s=now, phase='initial_before_perception' if initial else 'post_physics_before_terminal',
                                      aircraft=acid, **state, interval_start_s=start,
                                      alive_interval_seconds=duration, unavailable_interval_seconds=unavailable,
                                      available_at_interval_start=available_start, available_at_sample=available_end,
                                      known_until_s=self.until.get(acid)))
            self.rows += 1
        if not initial:
            self.steps += 1
        self.last_time = now


def run_case(env, model, cfg, case, scenario, perturbation_base, output):
    output.mkdir()
    perception_cfg = copy.deepcopy(perturbation_base)
    perception_cfg.update(position_probability=case['position_probability'],
                          communication_probability=case['communication_probability'], position_sigma_m=cfg['position_sigma_m'])
    layer = ObservationPerturbation(perception_cfg, seed=cfg['perturbation_seed']) if case['mode']=='perception' else None
    initial_layer_rng = None if layer is None else copy.deepcopy(layer.state_dict()['rng_state'])
    write_json(output/'effective_config.json', dict(case=case, environment=env.cfg,
        parts=dict(scenario=env.scenario_cfg, action=env.action_cfg, observation=env.observation_cfg),
        perturbation=perception_cfg if layer else None, policy=model.cfg,
        model_seed=cfg['model_seed'], sampling_seed=cfg['sampling_seed'], perturbation_seed=cfg['perturbation_seed']))
    write_json(output/'scenario.json', scenario)
    observations = env.reset(scenario['seed'], scenario=copy.deepcopy(scenario))
    sampling = torch.Generator(device='cpu').manual_seed(cfg['sampling_seed'])
    initial_sampling = sampling.get_state().clone()
    observation_digest, action_digest = hashlib.sha256(), hashlib.sha256()
    inferred = held = inference_batches = decisions = 0
    checks = dict(snapshot_preserves_truth=True, snapshot_cache_identity=True,
                  zero_observations_bitwise_equal=True, zero_state_objects_preserved=True,
                  visible_ids_match_encoded_ids=True, masks_are_current_truth_masks=True,
                  invisible_ids_absent_from_neighbors=True, position_visibility_complete=True,
                  communication_has_no_encoded_ownships=True, full_action_ids_supplied=True,
                  holds_use_last_accepted_action=True, inference_ids_match_visible_ids=True,
                  snapshot_clock_complete=True)
    per_aircraft = {acid: dict(inference=0, holds=0) for acid in env.records}
    event_count = 0
    with ExitStack() as stack:
        trace = PhysicalTrace(env, layer, output, stack)
        env.on_physics_step = trace.record
        events_file = stack.enter_context((output/'perception_events.jsonl').open('x'))
        decisions_file = stack.enter_context((output/'decisions.jsonl').open('x'))
        observations_file = stack.enter_context((output/'observations.jsonl').open('x'))
        while not env.done:
            now = float(env.bs.sim.simt)
            truth = env.physical_states()
            original_truth = canonical({acid: asdict(state) for acid, state in truth.items()})
            native_before = canonical(physical_packet(env))
            encoded = observations
            snapshot = None
            if layer is not None:
                checks['snapshot_clock_complete'] &= now == decisions*5.
                snapshot = layer.snapshot(truth, now)
                checks['snapshot_cache_identity'] &= layer.snapshot(truth, now) is snapshot
                encoded = snapshot.encode(env.observation_cfg)
                for acid in encoded:
                    encoded[acid]['action_mask'] = env.actions.action_mask(acid)
                trace.until = dict(snapshot.until)
                checks['visible_ids_match_encoded_ids'] &= sorted(encoded) == list(snapshot.available_ids)
                checks['invisible_ids_absent_from_neighbors'] &= all(
                    set(obs['intruder_ids']) <= set(snapshot.available_ids) for obs in encoded.values())
                for event in snapshot.events:
                    events_file.write(canonical(dict(event))+'\n')
                    event_count += 1
                if case['id']=='zero':
                    checks['zero_observations_bitwise_equal'] &= canonical(observations_packet(encoded)) == canonical(observations_packet(observations))
                    checks['zero_state_objects_preserved'] &= all(snapshot.sensed_states[acid] is state for acid, state in truth.items())
                if case['id']=='position':
                    checks['position_visibility_complete'] &= set(encoded) == set(truth)
                if case['id']=='communication':
                    checks['communication_has_no_encoded_ownships'] &= not encoded
            checks['snapshot_preserves_truth'] &= (
                native_before == canonical(physical_packet(env))
                and original_truth == canonical({acid: asdict(state) for acid, state in truth.items()})
                and original_truth == canonical({acid: asdict(state) for acid, state in env.physical_states().items()}))
            checks['masks_are_current_truth_masks'] &= all(
                np.array_equal(obs['action_mask'], env.actions.action_mask(acid)) for acid, obs in encoded.items())
            packet = observations_packet(encoded)
            update_digest(observation_digest, [now, packet])
            observations_file.write(canonical(dict(sim_time_s=now, input_bits=packet,
                normalized_inputs={acid: dict(own=obs['own'].tolist(), intruders=obs['intruders'].tolist(),
                                             action_mask=obs['action_mask'].tolist(), intruder_ids=list(obs['intruder_ids']))
                                   for acid, obs in encoded.items()}))+'\n')
            inferred_actions, holds = {}, {}
            if case['mode']=='nr':
                actions = None
            else:
                if encoded:
                    inferred_actions, _, _ = select_actions(model, encoded, sampling)
                    inference_batches += 1
                checks['inference_ids_match_visible_ids'] &= set(inferred_actions) == set(encoded)
                inferred += len(inferred_actions)
                for acid in inferred_actions:
                    per_aircraft[acid]['inference'] += 1
                for acid in sorted(set(truth)-set(encoded)):
                    holds[acid] = env.actions.state_fields(acid)['accepted_action_index']
                    per_aircraft[acid]['holds'] += 1
                held += len(holds)
                actions = {**inferred_actions, **holds}
                checks['full_action_ids_supplied'] &= set(actions) == set(truth)
                checks['holds_use_last_accepted_action'] &= all(
                    action == env.actions.state_fields(acid)['accepted_action_index'] for acid, action in holds.items())
            update_digest(action_digest, [now, actions])
            decisions_file.write(canonical(dict(sim_time_s=now, active_ids=sorted(truth), visible_ids=sorted(encoded),
                inferred_actions=inferred_actions, held_actions=holds, environment_actions=actions,
                perturbation_until={} if snapshot is None else dict(snapshot.until),
                perturbation_statistics={} if snapshot is None else dict(snapshot.statistics)))+'\n')
            observations, _, _, _ = env.step(actions)
            decisions += 1
        env_summary = env.summary(include_flights=True)
        final_layer = None if layer is None else layer.state_dict()
        independent_unavailable = {}
        for record in env_summary['flights']:
            acid = record['id']
            intervals = [] if final_layer is None else final_layer['blackout_intervals'].get(acid, [])
            independent_unavailable[acid] = sum(max(0., min(end, record['terminal_time_s'])-max(start, record['actual_entry_s']))
                                                 for start, end in intervals)
        checks.update(
            complete_population=env_summary['completed_population'] and env_summary['planned']==2,
            physical_samples_cover_lifetimes=all(trace.presence.get(r['id'], 0)==round(r['flight_seconds']/env.dt)
                                                 for r in env_summary['flights']),
            physical_time_reconciles=math.isclose(trace.alive_seconds, env.flight_seconds, abs_tol=1e-9),
            unavailable_intervals_bounded=trace.query_checks,
            unavailable_lifetimes_reconcile=all(math.isclose(trace.unavailable[a], independent_unavailable[a], abs_tol=1e-9)
                                                 for a in trace.unavailable),
            no_model_gradients=all(p.grad is None for p in model.parameters()),
            inference_hold_accounting_reconciles=all(
                per_aircraft[r['id']]['inference']+per_aircraft[r['id']]['holds'] == r['policy_decisions']
                for r in env_summary['flights']),
        )
        if case['id']=='zero':
            checks['disabled_layer_rng_unchanged'] = final_layer['rng_state'] == initial_layer_rng
            checks['disabled_layer_no_events'] = event_count == 0
        if case['id']=='position':
            checks['position_events_present'] = final_layer['statistics']['position_triggers'] > 0
            checks['position_has_no_unavailable_time'] = sum(trace.unavailable.values()) == 0.
        if case['id']=='communication':
            checks.update(communication_zero_inference=inferred==inference_batches==0,
                          communication_sampling_rng_unused=torch.equal(sampling.get_state(), initial_sampling),
                          communication_full_lifetime_unavailable=math.isclose(sum(trace.unavailable.values()), env.flight_seconds, abs_tol=1e-9),
                          communication_truth_lowc_preserved=env_summary['risk']['lowc']['unordered_pair_seconds'] > 0.,
                          communication_truth_nmac_preserved=env_summary['risk']['nmac']['unordered_pair_seconds'] > 0.,
                          communication_no_changed_commands=all(
                              all(r['action_execution']['command_counts'][key]==0 for key in ('speed_commands','altitude_commands','lane_commands'))
                              for r in env_summary['flights']))
        result = dict(case=case, scope=cfg['scope'], checks={k: bool(v) for k, v in checks.items()},
            all_checks_passed=all(checks.values()), environment_summary=env_summary,
            observation_stream_sha256=observation_digest.hexdigest(), action_stream_sha256=action_digest.hexdigest(),
            physical_trajectory_sha256=trace.digest.hexdigest(), physical_rows=trace.rows, physical_steps=trace.steps,
            inference_aircraft_decisions=inferred, inference_batches=inference_batches, held_aircraft_decisions=held,
            per_aircraft_decisions=per_aircraft, unavailable_aircraft_seconds=sum(trace.unavailable.values()),
            unavailable_by_aircraft=trace.unavailable, independently_clipped_blackout_seconds=independent_unavailable,
            perception_events=event_count, perception_statistics={} if final_layer is None else final_layer['statistics'],
            perception_until={} if final_layer is None else final_layer['until'],
            model_sha256_after_case=model_digest(model),
            unavailable_time_definition='Sum over real pre-deletion native intervals; independently intersect merged blackouts with entry/terminal times.',
            inference_hold_scope='Standalone inference/hold diagnostics only; no PPO samples, missing-decision log-probabilities or GAE.')
        write_json(output/'result.json', result)
        write_json(output/'truth_risk_events.json', env.tracker.events)
        if final_layer is not None:
            write_json(output/'perception_state.json', final_layer)
        return result


def compare_cases(results, initial_model):
    by_id = {r['case']['id']: r for r in results}
    needed = {'nr', 'plain', 'zero', 'position', 'communication'}
    if set(by_id) != needed or any('error' in r for r in results):
        return dict(all_cases_completed_without_exception=False)
    plain, zero = by_id['plain'], by_id['zero']
    nr, communication = by_id['nr'], by_id['communication']
    checks = dict(
        zero_plain_observation_stream_bitwise_equal=plain['observation_stream_sha256']==zero['observation_stream_sha256'],
        zero_plain_action_stream_equal=plain['action_stream_sha256']==zero['action_stream_sha256'],
        zero_plain_physical_trajectory_equal=plain['physical_trajectory_sha256']==zero['physical_trajectory_sha256'],
        zero_plain_full_scientific_summary_equal=canonical(scientific_summary(plain['environment_summary']))==canonical(scientific_summary(zero['environment_summary'])),
        communication_nr_physical_trajectory_equal=communication['physical_trajectory_sha256']==nr['physical_trajectory_sha256'],
        communication_nr_true_risk_equal=canonical(communication['environment_summary']['risk'])==canonical(nr['environment_summary']['risk']),
        model_unchanged_all_cases=all(r['model_sha256_after_case']==initial_model for r in results),
    )
    physical_keys = ('completed','failed_timeout','failed_route_exhausted','flight_hours','path_length_m',
                     'outside_corridor_aircraft_seconds','outside_corridor_flights','max_centerline_distance_m',
                     'outside_altitude_aircraft_seconds','sim_seconds','physics_steps','max_active')
    checks['communication_nr_physical_outcomes_equal'] = all(
        communication['environment_summary'][key]==nr['environment_summary'][key] for key in physical_keys)
    checks['nr_truth_conflicts_nonzero'] = all(nr['environment_summary']['risk'][level]['unordered_pair_seconds']>0
                                              for level in ('lowc','nmac'))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    actual = tuple((c['id'], c['mode'], c['position_probability'], c['communication_probability']) for c in cfg['cases'])
    if cfg['schema'] != 'bluesky.perturbation-probe.v1' or actual != CASE_SPECS:
        raise ValueError('Keep the five predeclared perception integration cases')
    if (cfg['model_seed'], cfg['sampling_seed'], cfg['perturbation_seed'], cfg['position_sigma_m']) != (930001, 940001, 950001, 9.63):
        raise ValueError('Preserve the declared model/sampling/perception seeds and sigma')
    if version('bluesky-simulator') != cfg['bluesky_version'] or cfg['bluesky_version'] != '1.1.1':
        raise ValueError('Native diagnostic requires BlueSky1.1.1')
    env_cfg, parts = load_environment_config(cfg['environment_config'])
    if cfg['environment_config'] != 'configs/paper_environment_execution.json' or env_cfg['decision_seconds'] != 5. or parts['scenario']['dt_seconds'] != .25:
        raise ValueError('Preserve the execution environment and5s/0.25s timing')
    policy_cfg = json.loads(Path(cfg['policy_config']).read_text())
    perturbation_cfg = load_perturbations(cfg['perturbation_config'])
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be an existing artifact directory')
    write_json(output/'input.json', cfg)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg['model_seed'])
    model = SharedActorCritic(policy_cfg).eval().requires_grad_(False)
    initial_model = model_digest(model)
    scenario = constructed_scenario('crossing', 'M100', cfg['scenario_seed'])
    env = PaperEnvironment(env_cfg, parts)
    results = []
    with torch.no_grad():
        for case in cfg['cases']:
            try:
                result = run_case(env, model, cfg, case, scenario, perturbation_cfg, output/case['id'])
            except Exception as exc:
                result = dict(case=case, scope=cfg['scope'], error=f'{type(exc).__name__}: {exc}',
                              traceback=traceback.format_exc(), all_checks_passed=False)
                write_json(output/case['id']/'failure.json', result)
            results.append(result)
            print(canonical(dict(case=case['id'], all_checks_passed=result['all_checks_passed'],
                                 error=result.get('error'))), flush=True)
    comparisons = compare_cases(results, initial_model)
    summary = dict(scope=cfg['scope'], cases=results, comparison_checks=comparisons,
        all_checks_passed=all(r['all_checks_passed'] for r in results) and all(comparisons.values()),
        model_initializations=1, model_initial_sha256=initial_model,
        versions={name: version(name) for name in ('bluesky-simulator','numpy','torch')},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name('observation_perturbation.py'),
            Path(__file__).with_name('paper_environment.py'), Path(__file__).with_name('paper_train.py'),
            Path(__file__).with_name('shared_ppo.py'))})
    write_json(output/'result.json', summary)
    return 0 if summary['all_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

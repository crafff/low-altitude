"""Independent frozen sample/argmax modes with native sample-reference checks."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from paper_train import (_evaluate_case, _validate_checkpoint, _validate_train_config,
                         _versions, aggregate_summaries)
from policy_diagnostic import (canonical, difference_paths, file_digest, model_digest,
                               scientific, write_json)
from shared_ppo import SharedActorCritic


def mode_specs(episode):
    if type(episode) is not int or episode <= 0:
        raise ValueError('The trained checkpoint episode must be a positive integer')
    return ((f'episode{episode}_sample', f'episode{episode}', 'sample', episode),
            (f'episode{episode}_argmax', f'episode{episode}', 'argmax', episode),
            ('initial0_sample', 'initial0', 'sample', 0),
            ('initial0_argmax', 'initial0', 'argmax', 0))


MODE_SPECS = mode_specs(400)  # Public v1 contract retained.
ORIGINAL_CASES = tuple((53001+i, 3+i%3) for i in range(12))


def validate_config(cfg):
    if cfg['schema'] not in ('bluesky.checkpoint-policy-modes.v1', 'bluesky.checkpoint-policy-modes.v2'):
        raise ValueError('Unknown checkpoint policy modes configuration')
    expected_modes = mode_specs(cfg['checkpoint_completed_episodes'])
    v1 = cfg['schema'] == 'bluesky.checkpoint-policy-modes.v1'
    if v1 and cfg['checkpoint_completed_episodes'] != 400:
        raise ValueError('The v1 diagnostic is explicitly episode 400')
    if v1 or 'modes' in cfg:
        modes = tuple((m['id'], m['model'], m['policy'], m['completed_episodes']) for m in cfg['modes'])
        if modes != expected_modes or any(type(m['completed_episodes']) is not int for m in cfg['modes']):
            raise ValueError('Preserve all four derived modes and their order')
    if cfg['initialization_seed'] != 61001:
        raise ValueError('Preserve original initialization seed 61001')
    if cfg['primary_policy'] != 'sample':
        raise ValueError('Sample remains the primary policy; argmax cannot select or promote a model')
    if cfg['device'] != 'cpu' or cfg['torch_num_threads'] != 1 or cfg['torch_num_interop_threads'] != 1:
        raise ValueError('The diagnostic requires single-thread CPU execution')
    if not math.isfinite(cfg['wall_seconds']) or not 0 < cfg['wall_seconds'] <= 570.:
        raise ValueError('The inner diagnostic deadline is at most 570 seconds')
    if tuple((c['seed'], c['corridor_count']) for c in cfg['development_cases']) != ORIGINAL_CASES:
        raise ValueError('Preserve all 12 original development cases')
    if v1:
        return cfg
    return dict(cfg, modes=[dict(zip(('id', 'model', 'policy', 'completed_episodes'), spec))
                           for spec in expected_modes])


def resolve_trained_reference(episode, reference_trained=None, reference_400=None):
    """One explicit reference; the historical spelling retains its 400 meaning."""
    if (reference_trained is None) == (reference_400 is None):
        raise ValueError('Supply exactly one of --reference-trained or --reference-400')
    if reference_400 is not None and episode != 400:
        raise ValueError('--reference-400 is only valid for episode 400; use --reference-trained')
    path = reference_trained if reference_trained is not None else reference_400
    if not path:
        raise ValueError('The trained reference path must not be empty')
    return path


def select_reference(rows, episode):
    matches = [r for r in rows if type(r.get('completed_episodes')) is int
               and r['completed_episodes'] == episode and 'cases' in r]
    if len(matches) != 1:
        raise ValueError(f'Require exactly one full reference evaluation for episode {episode}, found {len(matches)}')
    return matches[0]


def load_reference(path, episode):
    text = Path(path).read_text()
    if Path(path).suffix == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    return select_reference(rows, episode)


def summary_checks(summary, policy, seed, action_seed, forward_rows=None):
    histogram = summary['action_histogram']
    terminal_count = summary['completed']+summary['failed_timeout']+summary['failed_route_exhausted']
    checks = dict(policy_label=summary['policy']==policy,
                  scenario_seed=summary['seed']==seed,
                  action_seed=summary['action_seed']==action_seed,
                  completed_population=bool(summary['completed_population']),
                  terminal_population_accounting=terminal_count==summary['planned'],
                  sixty_joint_action_classes=len(histogram)==60,
                  nonnegative_integer_action_counts=all(type(n) is int and n >= 0 for n in histogram),
                  action_histogram_reconciles=sum(histogram)==summary['policy_decisions'])
    if forward_rows is not None:
        checks['actual_forward_rows_reconcile'] = forward_rows==summary['policy_decisions']
    return checks


def validate_reference(reference, episode, training, scope):
    if (type(reference['completed_episodes']) is not int or reference['completed_episodes'] != episode or reference['scope'] != scope
            or reference['primary_policy'] != 'sample' or reference['case_count'] != 12
            or len(reference['cases']) != 12):
        raise ValueError('Reference episode/scope/policy/population differs from the declared evaluation')
    if tuple((c['seed'], c['corridor_count']) for c in reference['cases']) != ORIGINAL_CASES:
        raise ValueError('Reference case order or geometry differs from the original development set')
    for case in reference['cases']:
        action_seed = training['dev_action_seed_base']+case['seed']
        if case['sample']['planned'] != 30 or case['nr']['planned'] != 30:
            raise ValueError('Each original reference case must retain the full 30-aircraft population')
        if case['action_seed'] != action_seed or not all(summary_checks(case['sample'], 'sample', case['seed'], action_seed).values()):
            raise ValueError('Reference sample metadata or action accounting is inconsistent')
        if not all(summary_checks(case['nr'], 'nr', case['seed'], None).values()):
            raise ValueError('Reference NR metadata or action accounting is inconsistent')
    for policy in ('sample', 'nr'):
        actual = aggregate_summaries([case[policy] for case in reference['cases']])
        if difference_paths(scientific(reference['aggregate'][policy]), scientific(actual)):
            raise ValueError(f'Reference {policy} aggregate does not reconcile with its 12 cases')


def frozen_checks(model, initial_digest):
    return dict(model_tensors_unchanged=model_digest(model)==initial_digest,
                no_gradients=all(p.grad is None for p in model.parameters()),
                evaluation_mode=not model.training,
                gradients_disabled=all(not p.requires_grad for p in model.parameters()))


class ForwardAudit:
    """Passive counts only; the original helper owns every forward and action."""
    def __init__(self, model):
        self.model = model
        self.calls = self.rows = 0
        self.valid_masks = True
        self.handle = None
        self.restored = False

    def __enter__(self):
        self.before_hooks = tuple(self.model._forward_hooks)
        self.handle = self.model.register_forward_hook(self.capture)
        return self

    def capture(self, module, inputs, output):
        self.calls += 1
        self.rows += int(output[0].shape[0])
        self.valid_masks &= bool(inputs[3].any(dim=-1).all()) and output[0].shape[1]==60
        # None preserves the actual output and never draws from an RNG.

    def __exit__(self, *args):
        self.handle.remove()
        self.restored = tuple(self.model._forward_hooks)==self.before_hooks


def run_mode(env, model, mode, scenarios, reference, output, deadline):
    output.mkdir()
    initial = model_digest(model)
    records = []
    audit = ForwardAudit(model)
    with audit:
        for case, scenario in zip(reference['cases'], scenarios):
            record = dict(seed=case['seed'], corridor_count=case['corridor_count'],
                          mode=mode['id'], executed=False, completed=False, all_checks_passed=False)
            if time.perf_counter() >= deadline:
                record.update(status='not_run_budget_exhausted')
            else:
                before_calls, before_rows = audit.calls, audit.rows
                record['executed'] = True
                try:
                    summary = _evaluate_case(env, model, scenario, mode['policy'], case['action_seed'], deadline)
                    checks = summary_checks(summary, mode['policy'], case['seed'], case['action_seed'], audit.rows-before_rows)
                    checks.update(frozen_checks(model, initial))
                    differences = (difference_paths(scientific(case['sample']), scientific(summary))
                                   if mode['policy']=='sample' else None)
                    if differences is not None:
                        checks['sample_reference_exact'] = not differences
                    record.update(status='completed_actual_native_evaluation', completed=True, summary=summary,
                        checks=checks, all_checks_passed=all(checks.values()),
                        sample_reference_comparison='exact_except_wall_and_rss' if differences is not None else 'not_applicable_to_argmax',
                        sample_reference_difference_count=None if differences is None else len(differences),
                        sample_reference_difference_paths=None if differences is None else differences[:100])
                except Exception as exc:
                    record.update(status='incomplete_native_evaluation_error', error=f'{type(exc).__name__}: {exc}',
                                  traceback=traceback.format_exc())
                record.update(actual_policy_forwards=audit.calls-before_calls,
                              actual_policy_aircraft_rows=audit.rows-before_rows)
            records.append(record)
            write_json(output/f"seed-{case['seed']}.json", record)
            print(canonical(dict(mode=mode['id'], seed=case['seed'], status=record['status'],
                                 all_checks_passed=record['all_checks_passed'])), flush=True)
    complete = len(records)==12 and all(row['completed'] for row in records)
    aggregate = aggregate_summaries([row['summary'] for row in records]) if complete else None
    differences = None
    checks = dict(all_twelve_native_evaluations_complete=complete,
                  all_case_checks=all(row['all_checks_passed'] for row in records),
                  all_masks_have_legal_joint_actions=audit.valid_masks,
                  forward_hook_restored=audit.restored, **frozen_checks(model, initial))
    if complete and mode['policy']=='sample':
        differences = difference_paths(scientific(reference['aggregate']['sample']), scientific(aggregate))
        checks['sample_aggregate_reference_exact'] = not differences
    result = dict(mode=mode, primary_policy='sample', cases=records, aggregate=aggregate,
        all_checks_passed=all(checks.values()), checks=checks,
        actual_native_episodes_started=sum(row['executed'] for row in records),
        actual_native_episodes_completed=sum(row['completed'] for row in records),
        actual_policy_forwards=audit.calls, actual_policy_aircraft_rows=audit.rows,
        model_before_sha256=initial, model_after_sha256=model_digest(model),
        sample_aggregate_reference_difference_paths=None if differences is None else differences[:100],
        action_histogram=None if not complete else [sum(row['summary']['action_histogram'][i] for row in records) for i in range(60)])
    write_json(output/'result.json', result)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Explicit configured-episode checkpoint included by lab --input')
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument('--reference-trained', help='Explicit full development JSON/JSONL for the configured trained episode')
    references.add_argument('--reference-400', help='Historical spelling for an explicit full episode-400 development JSON/JSONL')
    parser.add_argument('--reference-0', required=True, help='Explicit original episode-0 development JSON/JSONL')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be the existing launcher output directory')
    started = time.perf_counter()
    result = dict(all_checks_passed=False, modes=[], primary_policy='sample')
    try:
        cfg = validate_config(json.loads(Path(args.config).read_text()))
        episode = cfg['checkpoint_completed_episodes']
        trained_key = f'episode{episode}'
        trained_reference = resolve_trained_reference(episode, args.reference_trained, args.reference_400)
        reference_key = 'reference400' if cfg['schema'] == 'bluesky.checkpoint-policy-modes.v1' else 'reference_trained'
        input_paths = dict(checkpoint=args.checkpoint, **{reference_key: trained_reference}, reference0=args.reference_0)
        input_identity = {key: dict(path=path, sha256=file_digest(path)) for key, path in input_paths.items()}
        write_json(output/'input.json', dict(config=cfg, explicit_inputs=input_identity))
        training = json.loads(Path(cfg['training_config']).read_text())
        _validate_train_config(training, episode, training['wall_seconds'], None)
        if training['training_seed'] != cfg['initialization_seed'] or training['development_cases'] != cfg['development_cases']:
            raise ValueError('Current training configuration differs from original initialization or development cases')

        # The original main seeds these three streams before importing the native
        # adapter/loading its JSON configurations and constructing the first model.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        random.seed(training['training_seed'])
        np.random.seed(training['training_seed'])
        torch.manual_seed(training['training_seed'])
        from paper_environment import PaperEnvironment, load_environment_config
        from paper_scenarios import generate_scenario

        env_cfg, parts = load_environment_config(training['environment_config'])
        ppo_cfg = json.loads(Path(training['ppo_config']).read_text())
        configs = dict(training=training, environment=env_cfg, environment_parts=parts,
            aircraft_types=json.loads(Path(env_cfg['types_config']).read_text()), ppo=ppo_cfg,
            effective_development_cases=copy.deepcopy(cfg['development_cases']))
        versions = _versions()
        initial_model = SharedActorCritic(ppo_cfg)
        initial_model.eval().requires_grad_(False)
        # No module-level deserialization; this is reached only inside the CLI.
        payload = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        _validate_checkpoint(payload, configs, versions)
        if payload['completed_episodes'] != episode:
            raise ValueError(f'The supplied checkpoint is not completed episode {episode}')
        refs = {episode: load_reference(trained_reference, episode), 0: load_reference(args.reference_0, 0)}
        for reference_episode, reference in refs.items():
            validate_reference(reference, reference_episode, training, payload['scope'])
        stored = payload['evaluation']
        if stored is None or type(stored.get('completed_episodes')) is not int or stored['completed_episodes'] != episode:
            raise ValueError(f'Checkpoint lacks its own completed-{episode} evaluation')
        differences = difference_paths(scientific(stored), scientific(refs[episode]))
        if differences:
            raise ValueError(f'{episode} reference differs from checkpoint evaluation: '+', '.join(differences[:8]))
        nr_differences = difference_paths(scientific([case['nr'] for case in refs[0]['cases']]),
                                         scientific([case['nr'] for case in refs[episode]['cases']]))
        nr_differences += difference_paths(scientific(refs[0]['aggregate']['nr']), scientific(refs[episode]['aggregate']['nr']))
        if nr_differences:
            raise ValueError(f'Original and {episode} NR references differ: '+', '.join(nr_differences[:8]))
        checkpoint_model = copy.deepcopy(initial_model)
        checkpoint_model.load_state_dict(payload['model'], strict=True)
        models = {trained_key: checkpoint_model, 'initial0': initial_model}
        identities = {key: model_digest(model) for key, model in models.items()}
        independent_storage = not ({p.data_ptr() for p in initial_model.parameters()} &
                                   {p.data_ptr() for p in checkpoint_model.parameters()})
        if not independent_storage or identities[trained_key]==identities['initial0']:
            raise ValueError('Trained and original model states are unexpectedly aliased or identical')
        write_json(output/'effective_config.json', dict(diagnostic=cfg, checkpoint_configs=configs,
            scientific_versions=versions, model_initial_sha256=identities,
            diagnostic_source_sha256=file_digest(__file__), identity_helper_source_sha256=file_digest(Path(__file__).with_name('policy_diagnostic.py')),
            initialization=f'Original random.seed/np.random.seed/torch.manual_seed then SharedActorCritic; seed 61001. Model{episode} is an independent copy loaded strictly from supplied tensors. No optimizer is constructed.',
            source_fact='Paper p.6 specifies sampling during training; its p.17 frozen-policy/Monte-Carlo text does not settle sample versus argmax deployment.'))
        write_json(output/'reused_nr.json', dict(source='both explicit references, verified exact except wall/RSS',
            nr_native_episodes_executed=0, cases=[dict(seed=c['seed'], nr=c['nr']) for c in refs[episode]['cases']],
            aggregate=refs[episode]['aggregate']['nr']))
        # Match the original main's restoration of global RNGs across BlueSky init.
        startup_torch = torch.random.get_rng_state().clone()
        startup_numpy, startup_python = np.random.get_state(), random.getstate()
        env = PaperEnvironment(env_cfg, parts)
        torch.random.set_rng_state(startup_torch)
        np.random.set_state(startup_numpy)
        random.setstate(startup_python)
        scenarios = [generate_scenario(dict(parts['scenario'], corridor_counts=[case['corridor_count']]),
                                      case['seed'], list(env.types)) for case in cfg['development_cases']]
        write_json(output/'scenarios.json', scenarios)
        with torch.no_grad():
            for mode in cfg['modes']:
                record = run_mode(env, models[mode['model']], mode, scenarios, refs[mode['completed_episodes']],
                                  output/mode['id'], started+cfg['wall_seconds'])
                result['modes'].append(record)
        checks = dict(strict_checkpoint_source_config_version_validated=True,
            both_reference_nr_results_agree=True, four_mode_records_preserved=len(result['modes'])==4,
            all_mode_checks=all(mode['all_checks_passed'] for mode in result['modes']),
            forty_eight_native_cases_completed=sum(mode['actual_native_episodes_completed'] for mode in result['modes'])==48,
            independent_model_parameter_storage=independent_storage,
            both_models_unchanged=all(all(frozen_checks(model, identities[key]).values()) for key, model in models.items()),
            explicit_input_files_unchanged=all(file_digest(item['path'])==item['sha256'] for item in input_identity.values()))
        result.update(scope=cfg['scope'], checks=checks, all_checks_passed=all(checks.values()),
            model_initial_sha256=identities, model_final_sha256={key: model_digest(model) for key, model in models.items()},
            actual_native_episodes_completed=sum(mode['actual_native_episodes_completed'] for mode in result['modes']),
            actual_nr_native_episodes=0, model_selection_performed=False, checkpoint_writes=0,
            interpretation='Four independently executed deployment modes. Sample remains primary; argmax is a legal joint 60-class decision, never independently maximized component marginals. No model promotion or efficacy claim.')
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), all_checks_passed=False)
    result['wall_seconds'] = time.perf_counter()-started
    write_json(output/'result.json', result)
    print(canonical(dict(all_checks_passed=result['all_checks_passed'], error=result.get('error'))), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

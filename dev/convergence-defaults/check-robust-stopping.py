#!/usr/bin/env python3
"""Verify fixed robust LM stopping ablations without selecting defaults."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).with_name('check-least-squares.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

POLICIES = {
    'robust_gradient_default': set(),
    'robust_relative_probe': {'robust_orthogonality', 'model_reduction', 'trial_step', 'trust_radius'},
    'robust_model_probe': {'model_reduction'},
    'robust_step_probe': {'trial_step'},
    'robust_model_step_probe': {'model_reduction', 'trial_step'},
    'robust_gradient_probe': {'robust_orthogonality'},
    'robust_radius_probe': {'trust_radius'},
}
ROUTES = ('lm_normal_nielsen', 'lm_normal_trust', 'lm_qr_nielsen', 'lm_qr_trust')


def quality(run, publications):
    # Every ablation receives the same independent unit-scale quality threshold.
    return pilot.robust_quality(dict(run, policy='robust_default'), publications)


def compare_prefix(probe, control):
    pilot.require(all(all(a[k] == b[k] for k in ('kind', 'point', 'outcome'))
                      for a, b in zip(probe, control)), 'stopping ablation changed the common callback prefix')
    return min(len(probe), len(control))


def terminal_trial(observations, checks):
    last = observations[-1]
    if last['trial'] != 'true' or not last['trial_cost']:
        return None
    passing = {c['name'] for c in checks if c['sequence'] == last['sequence'] and c['passed'] == 'true'}
    both = {'model_reduction', 'trial_step'} <= passing
    accepted = last['accepted'] == 'true'
    return dict(sequence=last['sequence'], base_cost=float(last['base_cost']),
                trial_cost=float(last['trial_cost']), actual=float(last['actual']),
                predicted=float(last['predicted']), gain_ratio=float(last['ratio']),
                damping=float(last['damping']), step=pilot.numbers(last['step']),
                base_gradient=pilot.numbers(last['gradient']), diagonal=pilot.numbers(last['diagonal']),
                accepted=accepted, published=last['published'] == 'true', passing=sorted(passing),
                model_and_step=both, accepted_model_and_step=accepted and both,
                ratio_above_quarter_model_and_step=both and float(last['ratio']) > .25)


def verify(directory):
    runs, grouped, statistics = pilot.verify(directory)
    expected = {(dataset, precision, route, policy) for dataset, case in pilot.ROBUST_CASES.items()
                if case[4] is None for precision in ('f32', 'f64') for route in ROUTES for policy in POLICIES}
    pilot.require(len(runs) == len(expected) and
                  {(r['dataset'], r['precision'], r['route'], r['policy']) for r in runs} == expected,
                  'robust stopping coverage')
    pilot.require(all(r['cap'] == '4000' and r['start'] == '1' for r in runs), 'robust stopping caps and starts')
    defaults = {(r['dataset'], r['precision'], r['route']): r for r in runs
                if r['policy'] == 'robust_gradient_default'}
    summary = {name: Counter() for name in POLICIES}
    records = []
    for run in runs:
        identifier = run['id']
        checks = grouped['checks'][identifier]
        observations = grouped['native'][identifier]
        enabled = POLICIES[run['policy']]
        tol = pilot.native('1e-4' if run['precision'] == 'f32' else '1e-8', run['precision'])
        keyed = {o['sequence']: o for o in observations}
        for check in checks:
            name = check['name']
            pilot.require(name in {'absolute_gradient', 'robust_orthogonality', 'model_reduction', 'trial_step', 'trust_radius'},
                          'unknown stopping comparison')
            if name == 'absolute_gradient':
                wanted = pilot.native('1e-8', run['precision']) if run['policy'] == 'robust_gradient_default' else 0.
            else:
                wanted = tol if name in enabled and not (name == 'trust_radius' and run['route'].endswith('nielsen')) else None
                if name == 'trust_radius':
                    observation = keyed[check['sequence']]
                    finite_trial = all(observation[k] and math.isfinite(float(observation[k]))
                                       for k in ('base_cost', 'trial_cost', 'actual', 'predicted', 'ratio'))
                    if not finite_trial:
                        wanted = None
            pilot.require((float(check['tolerance']) if check['tolerance'] else None) == wanted,
                          f'{identifier}: frozen stopping configuration')
            if name == 'model_reduction' and check['evidence']:
                observation = keyed[check['sequence']]
                pilot.require(all(check[a] == observation[b] for a, b in
                                  [('actual', 'actual'), ('predicted', 'predicted'), ('reference_cost', 'base_cost'), ('ratio', 'ratio')]),
                              'model reduction evidence differs from its trial')
        pubs = grouped['publications'][identifier]
        point_quality = quality(run, pubs)
        control = defaults[run['dataset'], run['precision'], run['route']]
        control_quality = quality(control, grouped['publications'][control['id']])
        prefix = compare_prefix(grouped['leaves'][identifier], grouped['leaves'][control['id']])
        premature = (point_quality is not None and not point_quality['quality_passed'] and
                     control_quality is not None and control_quality['quality_passed'] and
                     int(run['work']) < int(control['work']) and run['outcome'] == 'converged')
        trial = terminal_trial(observations, checks)
        record = dict({k: run[k] for k in ('id', 'dataset', 'precision', 'route', 'policy', 'outcome', 'criteria', 'work', 'denied', 'returned')},
                      quality=point_quality, terminal_trial=trial, confirmed_premature=premature,
                      paired_default=dict(id=control['id'], work=control['work'], outcome=control['outcome'],
                                          common_prefix_work=prefix, quality=control_quality,
                                          available_improvement=point_quality['cost'] - control_quality['cost']
                                          if point_quality is not None and control_quality is not None else None))
        counts = summary[run['policy']]
        counts['runs'] += 1
        counts[run['outcome']] += 1
        counts['quality_passed'] += bool(point_quality and point_quality['quality_passed'])
        counts['confirmed_premature'] += premature
        counts['work'] += int(run['work'])
        records.append(record)
    return dict(schema=1, purpose='robust-lm-stopping-ablation', status='passed', policies_selected=False,
                holdout_candidate_outcomes='sealed', statistics=statistics, summary=summary, runs=records,
                quality='common unit-scale parameter and gradient limits: 1e-6 f64, 1e-3 f32; not CDP-1 eligibility certificates',
                composition='conjunction and acceptance/ratio filters evaluated only at measured terminal trials; no new solver policy implemented',
                solve_timing='instrumented, unoptimized; no speed comparison', verification_outside_solve_ledger=True)


def verify_baseline(directory, baseline):
    runs, grouped, _ = pilot.verify(directory)
    previous, old_grouped, _ = pilot.verify(baseline)
    pilot.require(len(previous) == 352, 'baseline route coverage')
    previous = {r['id']: r for r in previous}
    matched = 0
    for run in runs:
        if run['policy'] not in ('robust_gradient_default', 'robust_relative_probe'):
            continue
        old_id = run['id'].replace('robust_gradient_default', 'robust_default')
        pilot.require(old_id in previous, 'missing baseline control')
        before = previous[old_id]
        pilot.require(all(run[k] == before[k] for k in
                          ('outcome', 'criteria', 'work', 'denied', 'returned', 'cost_evals', 'residual_evals', 'jacobian_evals')),
                      'baseline control outcome changed')
        for name in grouped:
            after_rows, before_rows = grouped[name][run['id']], old_grouped[name][old_id]
            pilot.require(len(after_rows) == len(before_rows) and
                          all(all(a[k] == b[k] for k in a if k != 'id') for a, b in zip(after_rows, before_rows)),
                          f'baseline control {name} changed')
        matched += 1
    pilot.require(matched == 96, 'baseline control count')
    return dict(runs=352, matching_ablation_controls=matched, all_control_callbacks_publications_and_native_records_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    if args.baseline:
        report['baseline_reproduction'] = verify_baseline(args.directory, args.baseline)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(dict(statistics=report['statistics'], summary=report['summary'])))


if __name__ == '__main__':
    main()

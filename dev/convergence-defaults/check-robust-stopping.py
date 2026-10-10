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
EXTENDED_STARTS = {name: ([-1.] if name == 'robust_arctan' else [-.5, -.5, .5, .5] if '_rank4' in name else [-1., 1., -.5, .5])
                   for name in pilot.ROBUST_EXTENDED_CASES}


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


def verify(directory, extended=False):
    runs, grouped, statistics = pilot.verify(directory)
    cases = pilot.ROBUST_EXTENDED_CASES if extended else pilot.ROBUST_CASES
    expected = {(dataset, precision, route, policy, str(cap)) for dataset, case in cases.items()
                if case[4] is None for precision in ('f32', 'f64') for route in ROUTES for policy in POLICIES
                for cap in ((0, 1, 2, 4000) if extended and policy == 'robust_gradient_default' else (4000,))}
    pilot.require(len(runs) == len(expected) and
                  {(r['dataset'], r['precision'], r['route'], r['policy'], r['cap']) for r in runs} == expected,
                  'robust stopping coverage')
    pilot.require(all(r['start'] == '1' for r in runs), 'robust stopping starts')
    defaults = {(r['dataset'], r['precision'], r['route']): r for r in runs
                if r['policy'] == 'robust_gradient_default' and r['cap'] == '4000'}
    summary = {name: Counter() for name in POLICIES}
    budget_summary = Counter()
    records = []
    for run in runs:
        identifier = run['id']
        if extended and run['cap'] != '0':
            leaves = grouped['leaves'][identifier]
            pilot.require(leaves and pilot.numbers(leaves[0]['point']) == EXTENDED_STARTS[run['dataset']] and
                          leaves[0]['kind'] == 'ResidualJacobian' and leaves[0]['outcome'] == 'Completed',
                          'extended fixture initialization changed')
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
        trial = terminal_trial(observations, checks) if observations else None
        record = dict({k: run[k] for k in ('id', 'dataset', 'precision', 'route', 'policy', 'cap', 'outcome', 'stage', 'criteria', 'work', 'denied', 'returned', 'cost_evals', 'residual_evals', 'jacobian_evals')},
                      completed_iterations=int(pubs[-1]['iteration']) if pubs and run['returned'] == 'true' else None,
                      last_published_iteration=int(pubs[-1]['iteration']) if pubs else None,
                      physical_calls_by_kind=Counter(leaf['kind'] for leaf in grouped['leaves'][identifier]),
                      native_model_solves=sum(int(o['model_solves']) for o in observations),
                      quality=point_quality, last_publication_quality=pilot.robust_quality(dict(run, policy='robust_default'), pubs, last_publication=True),
                      terminal_trial=trial, confirmed_premature=premature,
                      paired_default=dict(id=control['id'], work=control['work'], outcome=control['outcome'],
                                          common_prefix_work=prefix, quality=control_quality,
                                          available_improvement=point_quality['cost'] - control_quality['cost']
                                          if point_quality is not None and control_quality is not None else None))
        counts = summary[run['policy']] if run['cap'] == '4000' else budget_summary
        counts['runs'] += 1
        counts[run['outcome']] += 1
        counts['quality_passed'] += bool(point_quality and point_quality['quality_passed'])
        counts['confirmed_premature'] += premature
        counts['work'] += int(run['work'])
        records.append(record)
    return dict(schema=1, purpose='robust-lm-extended-controls' if extended else 'robust-lm-stopping-ablation', status='passed', policies_selected=False,
                holdout_candidate_outcomes='sealed', statistics=statistics, summary=summary, budget_summary=budget_summary, runs=records,
                quality='common unit-scale parameter and gradient limits: 1e-6 f64, 1e-3 f32; not CDP-1 eligibility certificates',
                composition='conjunction and acceptance/ratio filters evaluated only at measured terminal trials; no new solver policy implemented',
                solve_timing='instrumented, unoptimized; no speed comparison', verification_outside_solve_ledger=True)


def verify_reproduction(directory, previous):
    matched = {}
    for name in ('runs', 'leaves', 'publications', 'native', 'checks'):
        after = pilot.load(directory, name)
        before = pilot.load(previous, name)
        # The instrumented clock is not expected to reproduce bit for bit.
        pilot.require(len(after) == len(before) and all(
            {k: v for k, v in a.items() if k != 'elapsed_seconds'} ==
            {k: v for k, v in b.items() if k != 'elapsed_seconds'} for a, b in zip(after, before)),
            f'original ablation {name} changed')
        matched[name] = len(after)
    pilot.require(matched['runs'] == 336, 'original ablation reproduction coverage')
    return dict(runs=336, records=matched, all_original_records_equal_except_elapsed_time=True)


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


def recovery_transition(before, after, precision):
    differences = [j for j, (a, b) in enumerate(zip(before, after)) if a != b]
    pilot.require(bool(differences), 'missing damping recovery transition')
    index = differences[0]
    old, new = before[index], after[index]
    pilot.require(index >= 2 and old['trial'] == new['trial'] == 'true' and
                  old['diagonal'] == new['diagonal'] and old['gradient'] == new['gradient'] and
                  old['base_cost'] == new['base_cost'] and before[index - 2]['accepted'] == 'true',
                  'recovery changed before the accepted curvature transition')
    previous = pilot.numbers(before[index - 2]['diagonal'])
    current = pilot.numbers(old['diagonal'])
    pilot.require(len(previous) == len(current) and all(x > 0 for x in previous) and
                  all(y >= x for x, y in zip(previous, current)), 'invalid recovery diagonal growth')
    ratio = max(x / y for x, y in zip(previous, current))
    expected = float(old['damping']) * ratio
    unit = 2 ** (-24 if precision == 'f32' else -53)
    pilot.require(0 < ratio < 1e-5 and
                  abs(float(new['damping']) - expected) <= 32 * unit * expected,
                  'recovery damping does not compensate for diagonal growth')
    return dict(sequence=old['sequence'], diagonal_ratio=ratio,
                previous_diagonal=previous, new_diagonal=current,
                damping_before=float(old['damping']), damping_after=float(new['damping']),
                step_before=pilot.numbers(old['step']), step_after=pilot.numbers(new['step']))


def verify_recovery(directory, previous, report):
    old_report = verify(previous, extended=True)
    old = {r['id']: r for r in old_report['runs']}
    pilot.require(set(old) == {r['id'] for r in report['runs']}, 'recovery comparison coverage')
    tables = ('runs', 'publications', 'leaves', 'native', 'checks')
    unchanged = {r['id'] for r in report['runs'] if r['route'].endswith('trust') or r['cap'] != '4000'}
    matched = {}
    for table in tables:
        before = [{k: v for k, v in row.items() if k != 'elapsed_seconds'}
                  for row in pilot.load(previous, table) if row['id'] in unchanged]
        after = [{k: v for k, v in row.items() if k != 'elapsed_seconds'}
                 for row in pilot.load(directory, table) if row['id'] in unchanged]
        pilot.require(before == after, f'unaffected recovery control {table} changed')
        matched[table] = len(after)
    pilot.require(len(unchanged) == 468, 'unaffected recovery control coverage')
    expected = {(loss, route) for loss in ('robust_huber_rank4', 'robust_arctan_rank4')
                for route in ('lm_normal_nielsen', 'lm_qr_nielsen')}
    failures = [r for r in old_report['runs'] if r['policy'] == 'robust_gradient_default' and
                r['cap'] == '4000' and not r['quality']['quality_passed']]
    pilot.require({(r['dataset'], r['route']) for r in failures} == expected and len(failures) == 4 and
                  all(r['precision'] == 'f32' and r['outcome'] == 'stalled' for r in failures),
                  'previous four stalls changed')
    native_before = pilot.load(previous, 'native')
    native_after = pilot.load(directory, 'native')
    recovered = []
    for run in report['runs']:
        before = old[run['id']]
        if before not in failures:
            continue
        pilot.require(run['quality']['quality_passed'], 'rank-deficient stall was not recovered')
        transition = recovery_transition([r for r in native_before if r['id'] == run['id']],
                                         [r for r in native_after if r['id'] == run['id']], run['precision'])
        recovered.append(dict(id=run['id'], before={k: before[k] for k in ('outcome', 'work', 'quality')},
                              after={k: run[k] for k in ('outcome', 'work', 'quality')}, transition=transition))
    for policy in ('robust_gradient_default', 'robust_gradient_probe', 'robust_radius_probe'):
        pilot.require(report['summary'][policy]['quality_passed'] == 72, 'gradient control recovery quality')
    return dict(previous_summary=old_report['summary'], recovered=recovered,
                unchanged_runs=len(unchanged), unchanged_records=matched,
                unaffected_records_equal_except_elapsed_time=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--previous', type=Path, help='compare all original ablation CSV records, excluding elapsed time')
    parser.add_argument('--extended', action='store_true', help='verify the larger, rank-deficient, and arctangent controls')
    parser.add_argument('--recovery-from', type=Path, help='verify recovery against the frozen extended controls')
    args = parser.parse_args()
    pilot.require(not (args.extended and args.baseline), 'original baseline does not contain the extended fixtures')
    pilot.require(not (args.extended and args.previous), 'original ablation does not contain the extended fixtures')
    report = verify(args.directory, args.extended)
    if args.recovery_from:
        pilot.require(args.extended, 'recovery comparison requires extended controls')
        report['recovery'] = verify_recovery(args.directory, args.recovery_from, report)
    if args.baseline:
        report['baseline_reproduction'] = verify_baseline(args.directory, args.baseline)
    if args.previous:
        report['original_reproduction'] = verify_reproduction(args.directory, args.previous)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(dict(statistics=report['statistics'], summary=report['summary'])))


if __name__ == '__main__':
    main()

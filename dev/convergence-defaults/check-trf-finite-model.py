#!/usr/bin/env python3
"""Compare the frozen robust pilot with its legacy-TRF safeguard recheck."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(directory, name):
    with (directory / f'{name}.csv').open(newline='') as source:
        return list(csv.DictReader(source))


def grouped(directory, name):
    result = defaultdict(list)
    for row in rows(directory, name):
        result[row['id']].append(row)
    return result


def compare(old, new, report):
    old_runs = {r['id']: r for r in rows(old, 'runs')}
    new_runs = {r['id']: r for r in rows(new, 'runs')}
    require(len(old_runs) == len(new_runs) == 352 and old_runs.keys() == new_runs.keys(), 'case coverage changed')
    old_leaves, new_leaves = grouped(old, 'leaves'), grouped(new, 'leaves')
    old_pubs, new_pubs = grouped(old, 'publications'), grouped(new, 'publications')
    quality = {r['id']: r['quality'] for r in report['runs']}
    changed = []
    for identifier, run in new_runs.items():
        before, after = old_leaves[identifier], new_leaves[identifier]
        require(len(after) <= len(before), f'{identifier}: extra callbacks')
        require(all(all(a[k] == b[k] for k in ('kind', 'point', 'outcome')) for a, b in zip(before, after)), f'{identifier}: callback prefix changed')
        previous = old_runs[identifier]
        if any(run[k] != previous[k] for k in ('outcome', 'work', 'denied', 'returned')):
            require(previous['outcome'] == 'callback_error' and run['outcome'] == 'failed' and run['returned'] == 'true' and quality[identifier]['quality_passed'], f'{identifier}: unexpected outcome change')
            changed.append(dict(id=identifier, old_work=int(previous['work']), new_work=int(run['work']), point=quality[identifier]['point'], quality=quality[identifier]))
        if old_pubs[identifier]:
            require(new_pubs[identifier] and all(old_pubs[identifier][-1][k] == new_pubs[identifier][-1][k] for k in ('point', 'cost')), f'{identifier}: final publication changed')
        else:
            require(not new_pubs[identifier], f'{identifier}: unexpected publication')
    require(len(changed) == 4 and all('f32-1-trf_legacy-robust_default-4000' in c['id'] for c in changed), 'affected case set changed')
    require(not report['statistics'].get('nonfinite_model_predictions', 0), 'non-finite trial prediction survived safeguard')
    return dict(purpose='legacy-trf-finite-model-regression', runs=352, unchanged_outcomes=348,
                changed=changed, all_callback_prefixes_equal=True, all_final_published_points_and_costs_equal=True,
                old_physical_calls=sum(int(r['work']) for r in old_runs.values()),
                new_physical_calls=sum(int(r['work']) for r in new_runs.values()), statistics=report['statistics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('recheck', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    require(report['status'] == 'passed' and report['statistics']['runs'] == 352, 'independent verification did not pass')
    result = compare(args.baseline, args.recheck, report)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(changed_cases=len(result['changed']), old_work=result['old_physical_calls'], new_work=result['new_physical_calls'])))


if __name__ == '__main__':
    main()

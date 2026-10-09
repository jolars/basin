"""Verify retained witness records and exercise preflight rejection paths."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference-tools'))
from preflight import failed_clauses
from decimal import Decimal as D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    full = json.loads((args.directory / 'full-report.json').read_text())
    if len(report['references']) != 15 or len(report['cases']) != 60:
        raise ValueError('incomplete retained report')
    if report['references'] != full['references']:
        raise ValueError('retained reference certificates changed')
    for case, original in zip(report['cases'], full['cases']):
        for field in case.keys() - {'witnesses'}:
            if case[field] != original[field]:
                raise ValueError(f'retained field changed: {field}')
        for label, witness in case['witnesses'].items():
            if witness != original['witnesses'][label]:
                raise ValueError('retained witness changed')
        for target in case['targets']:
            if target['status'] == 'eligible':
                if failed_clauses(case['witnesses'][target['witness']], D(target['q'])):
                    raise ValueError('eligible witness fails a quality or uncertainty clause')
            elif target['status'] != 'reference-pending':
                raise ValueError('unknown eligibility status')
    tests = []
    reference_bytes = (args.directory / 'references.json').read_bytes()
    points_bytes = (args.directory / 'points.csv').read_bytes()
    native_rows = list(csv.DictReader((args.directory / 'native.csv').open(newline='')))
    fields = list(native_rows[0])
    with tempfile.TemporaryDirectory(prefix='nist-reference-check-') as temporary:
        directory = Path(temporary)

        def reject(name, expected, mutate):
            reference = json.loads(reference_bytes)
            rows = [row.copy() for row in native_rows]
            mutate(reference, rows)
            (directory / 'references.json').write_text(json.dumps(reference))
            (directory / 'points.csv').write_bytes(points_bytes)
            with (directory / 'native.csv').open('w', newline='') as out:
                writer = csv.DictWriter(out, fields)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run([sys.executable, str(ROOT / 'reference-tools/preflight.py'),
                                     'check', str(directory), '--output', str(directory / 'summary.json')],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0 or expected not in result.stderr:
                raise ValueError(f'{name}: expected rejection {expected}: {result.stderr}')
            tests.append(name)

        reject('truncated references', 'reference coverage mismatch', lambda r, rows: r['references'].pop())
        reject('changed source hash', 'source changed',
               lambda r, rows: r['source_sha256'].__setitem__(next(iter(r['source_sha256'])), '0' * 64))
        reject('missing native point', 'native witness coverage mismatch', lambda r, rows: rows.pop())
        reject('duplicate native point', 'native witness coverage mismatch', lambda r, rows: rows.append(rows[0].copy()))
        reject('altered native coordinate', 'native coordinates changed', lambda r, rows: rows[0].__setitem__('x0', '0'))

        def corrupt_cost(reference, rows):
            rows[0]['cost'] = rows[0]['cost_abs_terms'] = '1000000'

        reject('independent cost integrity', 'native cost integrity check failed', corrupt_cost)

        (directory / 'holdout.csv').write_text('dataset,precision,point,x0,x1,x2,x3,x4,x5\nChwirut1,f64,reference,1,1,1,,,\n')
        result = subprocess.run([str(args.probe.resolve()), '--points', str(directory / 'holdout.csv'),
                                 '--output', str(directory / 'holdout-native.csv')],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0 or 'development only' not in result.stderr:
            raise ValueError('native probe accepted a holdout dataset')
        tests.append('holdout native probe')
        result = subprocess.run([str(args.probe.resolve()), '--points', str(args.directory / 'points.csv'),
                                 '--output', str(args.directory / 'native.csv')],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            raise ValueError('native probe overwrote prior output')
        tests.append('output overwrite refusal')
    eligible = sum(t['status'] == 'eligible' for c in report['cases'] for t in c['targets'])
    total = sum(len(c['targets']) for c in report['cases'])
    print(json.dumps(dict(status='passed', references=15, precision_start_cases=60,
                          native_points=len(native_rows), target_combinations=total,
                          eligible=eligible, reference_pending=total - eligible,
                          negative_controls=tests), indent=2))


if __name__ == '__main__':
    main()

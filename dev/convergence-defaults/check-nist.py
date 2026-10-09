"""Independently check NIST model CSVs using 100-digit standard-library Decimal.

This validates formulas and printed-value reproduction, not solver outcomes or
rigorous precision certificates. It uses no Rust formulas or cached objectives.
"""

import csv
import hashlib
import json
import math
import re
import struct
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 100
PI = D('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148')
ONE = D(1)


def sin_cos(x):
    x = x % (2 * PI)
    if x > PI:
        x -= 2 * PI
    s, c = x, ONE
    st, ct = x, ONE
    for k in range(1, 180):
        st *= -x * x / D((2 * k) * (2 * k + 1))
        ct *= -x * x / D((2 * k - 1) * (2 * k))
        s += st
        c += ct
        if max(abs(st), abs(ct)) < D('1e-102'):
            return s, c
    raise AssertionError('trigonometric series did not converge')


def atan(x):
    if x < 0:
        return -atan(-x)
    if x > 1:
        return PI / 2 - atan(1 / x)
    multiplier = 1
    while x > D('.2'):
        x = x / (1 + (1 + x * x).sqrt())
        multiplier *= 2
    term, result = x, x
    for k in range(1, 100):
        term *= -x * x
        increment = term / (2 * k + 1)
        result += increment
        if abs(increment) < D('1e-102'):
            return multiplier * result
    raise AssertionError('arctangent series did not converge')


def power(x, p):
    return (p * x.ln()).exp()


def response(name, b, x):
    t = x[0]
    if name in {'Misra1a', 'BoxBOD'}:
        return b[0] * (1 - (-b[1] * t).exp())
    if name == 'Misra1b':
        return b[0] * (1 - 1 / (1 + b[1] * t / 2) ** 2)
    if name == 'Misra1c':
        return b[0] * (1 - 1 / (1 + 2 * b[1] * t).sqrt())
    if name == 'Misra1d':
        return b[0] * b[1] * t / (1 + b[1] * t)
    if name.startswith('Chwirut'):
        return (-b[0] * t).exp() / (b[1] + b[2] * t)
    if name == 'DanWood':
        return b[0] * power(t, b[1])
    if name == 'Bennett5':
        return b[0] * power(b[1] + t, -1 / b[2])
    if name.startswith('Lanczos'):
        return sum(b[i] * (-b[i+1] * t).exp() for i in (0, 2, 4))
    if name == 'MGH17':
        return b[0] + b[1] * (-t * b[3]).exp() + b[2] * (-t * b[4]).exp()
    if name == 'MGH10':
        return b[0] * (b[1] / (t + b[2])).exp()
    if name.startswith('Gauss'):
        return b[0] * (-b[1] * t).exp() + sum(b[i] * (-((t-b[i+1])/b[i+2])**2).exp() for i in (2, 5))
    if name == 'Eckerle4':
        return b[0] / b[1] * (-((t-b[2])/b[1])**2 / 2).exp()
    if name in {'Kirby2', 'Hahn1', 'Thurber'}:
        n = (len(b) - 1) // 2
        return sum(b[i] * t**i for i in range(n+1)) / (1 + sum(b[n+i] * t**i for i in range(1, n+1)))
    if name == 'MGH09':
        return b[0] * (t*t + t*b[1]) / (t*t + t*b[2] + b[3])
    if name == 'Nelson':
        return b[0] - b[1] * t * (-b[2] * x[1]).exp()
    if name == 'Roszman1':
        return b[0] - b[1] * t - atan(b[2] / (t-b[3])) / PI
    if name == 'ENSO':
        s, c = sin_cos(2 * PI * t / 12)
        y = b[0] + b[1] * c + b[2] * s
        for i in (3, 6):
            s, c = sin_cos(2 * PI * t / b[i])
            y += b[i+1] * c + b[i+2] * s
        return y
    if name in {'Rat42', 'Rat43'}:
        q = 1 + (b[1] - b[2] * t).exp()
        return b[0] / (q if name == 'Rat42' else power(q, 1 / b[3]))
    raise AssertionError(name)


def derivative(name, b, x, j):
    h = max(abs(b[j]), D('1e-12')) * D('1e-22')
    plus, minus = b.copy(), b.copy()
    plus[j] += h
    minus[j] -= h
    return (response(name, plus, x) - response(name, minus, x)) / (2*h)


def native(value, precision):
    value = float(value)
    if precision == 'f32':
        value = struct.unpack('f', struct.pack('f', value))[0]
    return D.from_float(value)


def half_width(value):
    return D(5).scaleb(D(value).as_tuple().exponent - 1)


def main():
    assert len(sys.argv) == 2, 'usage: check-nist.py <model-csv>'
    path = Path(sys.argv[1])
    root = Path(__file__).resolve().parent / 'nist'
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = list(csv.DictReader(path.open(newline='')))
    grouped = {}
    for row in rows:
        assert None not in row and None not in row.values()
        key = (row['dataset'], row['precision'], row['point'])
        grouped.setdefault(key, []).append(row)
    expected_keys = {(d['id'], p, point) for d in manifest['datasets'] for p in ('f32', 'f64') for point in ('start1', 'start2', 'reference')}
    assert set(grouped) == expected_keys
    summaries = []
    total_derivatives = 0
    for dataset in manifest['datasets']:
        name = dataset['id']
        snapshot = (root / dataset['path']).read_bytes()
        assert hashlib.sha256(snapshot).hexdigest() == dataset['snapshot_sha256']
        text = snapshot.decode('ascii')
        observations = [line.split() for line in re.split(r'(?m)^Data:\s+y[^\n]*\n', text)[1].splitlines() if line.strip()]
        parameter_rows = re.findall(r'(?m)^\s*b\d+\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+\S+\s*$', text)
        assert [r[0] for r in parameter_rows] == dataset['start1']
        assert [r[1] for r in parameter_rows] == dataset['start2']
        assert [r[2] for r in parameter_rows] == dataset['certified_parameters']
        assert re.search(r'Residual Sum of Squares:\s*(\S+)', text)[1] == dataset['certified_rss']
        assert len(observations) == dataset['observations']
        assert all(len(row) == dataset['predictors'] + 1 for row in observations)
        reference_residuals, rounding_deltas = [], []
        b = list(map(D, dataset['certified_parameters']))
        for row in observations:
            obs, x = D(row[0]), list(map(D, row[1:]))
            y = response(name, b, x)
            reference_residuals.append(y - (obs.ln() if name == 'Nelson' else obs))
            rounding_deltas.append(sum(abs(derivative(name, b, x, j)) * half_width(dataset['certified_parameters'][j]) for j in range(len(b))))
        exact_rss = sum(r*r for r in reference_residuals)
        published = D(dataset['certified_rss'])
        # This local sensitivity screen is not an interval proof or a target certificate.
        allowance = half_width(dataset['certified_rss']) + sum(8*abs(r)*d + 16*d*d for r, d in zip(reference_residuals, rounding_deltas))
        assert abs(exact_rss - published) <= allowance, (name, exact_rss, published, allowance)
        summaries.append({'dataset': name, 'published_rss': str(published), 'rss_at_printed_parameters': str(exact_rss), 'rounding_screen_allowance': str(allowance)})
        for precision in ('f64', 'f32'):
            eps = D(2) ** (-52 if precision == 'f64' else -23)
            for point in ('start1', 'start2', 'reference'):
                values = dataset['certified_parameters' if point == 'reference' else point]
                b = [native(v, precision) for v in values]
                emitted = grouped[(name, precision, point)]
                assert len(emitted) == len(observations)
                actual_residuals = []
                for i, (row, output) in enumerate(zip(observations, emitted)):
                    assert output['schema'] == '1' and output['family'] == dataset['family']
                    assert output['partition'] == dataset['partition'] and int(output['row']) == i
                    obs, x = native(row[0], precision), [native(v, precision) for v in row[1:]]
                    y = response(name, b, x)
                    target = obs.ln() if name == 'Nelson' else obs
                    g = [derivative(name, b, x, j) for j in range(len(b))]
                    condition = max(abs(y), abs(target), sum(abs(v*w) for v, w in zip(b, g)), D('1e-30'))
                    bound = 128 * eps * condition
                    assert abs(D(output['response']) - y) <= bound, (name, precision, point, i, 'response')
                    residual = D(output['residual'])
                    assert abs(residual - (y-target)) <= bound, (name, precision, point, i, 'residual')
                    for j, expected in enumerate(g):
                        # Trigonometric basis zeros have absolute phase-rounding error.
                        floor = ONE if name == 'ENSO' and j in (1, 2, 4, 5, 7, 8) else D('1e-30')
                        if name == 'ENSO' and j in (3, 6):
                            floor = abs(2*PI*x[0]/(b[j]*b[j])) * (abs(b[j+1]) + abs(b[j+2]))
                        if name in {'Kirby2', 'Hahn1', 'Thurber'}:
                            degree = (len(b)-1)//2
                            terms = [b[k]*x[0]**k for k in range(degree+1)]
                            denominator_terms = [ONE] + [b[degree+k]*x[0]**k for k in range(1, degree+1)]
                            amplification = sum(map(abs, terms))/max(abs(sum(terms)), D('1e-90')) + 2*sum(map(abs, denominator_terms))/abs(sum(denominator_terms))
                            floor = max(floor, abs(expected)*amplification)
                        assert abs(D(output[f'j{j}']) - expected) <= 512*eps*max(abs(expected), floor), (name, precision, point, i, j)
                    assert all(output[f'j{j}'] == '' for j in range(len(b), 9))
                    total_derivatives += len(g)
                    actual_residuals.append(residual)
                rss = D(emitted[0]['rss'])
                assert all(D(r['rss']) == rss for r in emitted)
                sum_rss = sum(r*r for r in actual_residuals)
                # Sequential native accumulation gets a dimension-dependent bound.
                assert abs(rss - sum_rss) <= 4*len(observations)*eps*max(sum_rss, D('1e-60')), (name, precision, point, 'rss')
                assert math.isfinite(float(rss))
    repository = root.parent.parent.parent
    source_paths = [Path(__file__).resolve(), root / 'manifest.json', repository / 'Cargo.lock',
                    repository / 'crates/competitor-bench/src/bin/verify_nist.rs',
                    repository / 'crates/competitor-bench/src/convergence/nist.rs',
                    repository / 'crates/competitor-bench/tests/nist_models.rs']
    source_paths += sorted((repository / 'crates/competitor-bench/src/convergence/nist').glob('*.rs'))
    source_hashes = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    print(json.dumps({'schema': 1, 'source_sha256': source_hashes, 'status': 'passed', 'datasets': 27, 'primary_starts': 54, 'observations': 2176, 'emitted_rows': len(rows), 'checked_derivatives': total_derivatives, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'precision_certificates': 'pending; model validation only', 'references': summaries}, indent=2))


if __name__ == '__main__':
    main()

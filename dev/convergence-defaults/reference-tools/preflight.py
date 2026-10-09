"""Refine development NIST references and certify analytic native witnesses.

Run prepare first, evaluate points with verify_nist_witness, then run check.
No Basin solver or candidate stopping policy participates in reference work.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import struct
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

sys.dont_write_bytecode = True

from interval import Interval as I, PRECISION
from models import ROOT, Jet, certify, coordinate_scales, datasets, evaluate, response, solve

getcontext().prec = PRECISION
spec = importlib.util.spec_from_file_location('independent_nist', ROOT / 'check-nist.py')
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


def write_json(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')


def source_hashes():
    repository = ROOT.parents[1]
    paths = [ROOT / 'check-nist.py', ROOT / 'nist/manifest.json', ROOT / 'protocol.md',
             repository / 'Cargo.lock', repository / 'crates/competitor-bench/src/bin/verify_nist_witness.rs',
             repository / 'crates/competitor-bench/src/convergence/nist.rs']
    paths += sorted((Path(__file__).parent).glob('*.py'))
    paths += sorted((repository / 'crates/competitor-bench/src/convergence/nist').glob('*.rs'))
    return {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def refine(case, method):
    point = list(map(D, case['certified_parameters']))
    scales = coordinate_scales(case)
    for iteration in range(300):
        if method == 'newton-ad':
            result = evaluate(case, point)
            matrix, gradient = result.h, result.g
        else:
            # The second implementation uses the existing independent formulas
            # and numerical response derivatives, not the new AD expressions.
            n = len(point)
            matrix = [[D(0) for _ in range(n)] for _ in range(n)]
            gradient = [D(0)] * n
            for row in case['data']:
                target = row[0].ln() if case['id'] == 'Nelson' else row[0]
                residual = independent.response(case['id'], point, row[1:]) - target
                derivatives = [independent.derivative(case['id'], point, row[1:], j) for j in range(n)]
                for i in range(n):
                    gradient[i] += residual * derivatives[i]
                    for j in range(n):
                        matrix[i][j] += derivatives[i] * derivatives[j]
        step = solve(matrix, gradient)
        point = [v - d for v, d in zip(point, step)]
        if max(abs(d / s) for d, s in zip(step, scales)) < D('1e-70'):
            return point, iteration + 1
    raise ValueError(f"{case['id']}: {method} did not refine")


def native(value, precision):
    value = float(value)
    return struct.unpack('!f', struct.pack('!f', value))[0] if precision == 'f32' else value


def neighbor(value, precision, direction):
    if precision == 'f64':
        return math.nextafter(value, math.inf if direction > 0 else -math.inf)
    if value == 0:
        return direction * struct.unpack('!f', struct.pack('!I', 1))[0]
    bits = struct.unpack('!I', struct.pack('!f', value))[0]
    bits += direction if value > 0 else -direction
    return struct.unpack('!f', struct.pack('!I', bits))[0]


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    references = []
    points = []
    for case in datasets():
        print(f"refining {case['id']}", file=sys.stderr, flush=True)
        point, newton_iterations = refine(case, 'newton-ad')
        comparison, gn_iterations = refine(case, 'gauss-newton-central')
        scales = coordinate_scales(case)
        # The independent finite-difference implementation has O(1e-44) bias.
        agreement = max(abs((a - b) / s) for a, b, s in zip(point, comparison, scales))
        if agreement > D('1e-35'):
            raise ValueError(f"independent refinement disagreement: {case['id']}: {agreement}")
        certificate = certify(case, point)
        published_rss = I(case['certified_rss']) + I(-independent.half_width(case['certified_rss']),
                                                  independent.half_width(case['certified_rss']))
        refined_rss = 2 * I(*certificate['objective_interval'])
        if refined_rss.hi < published_rss.lo or published_rss.hi < refined_rss.lo:
            raise ValueError(f"{case['id']}: refined RSS disagrees with printed RSS rounding interval")
        certificate.update(dataset=case['id'], family=case['family'], partition=case['partition'],
                           snapshot_sha256=case['snapshot_sha256'], objective_convention='half-RSS',
                           newton_iterations=newton_iterations, gauss_newton_iterations=gn_iterations,
                           independent_point=list(map(str, comparison)), scaled_method_difference=str(agreement),
                           published_rss=case['certified_rss'],
                           refined_rss_overlaps_published_rounding_interval=True,
                           published_parameter_half_widths=[str(independent.half_width(v)) for v in case['certified_parameters']],
                           printed_parameter_rss=str(2 * evaluate(case, list(map(D, case['certified_parameters']))).v))
        references.append(certificate)
        for precision in ('f64', 'f32'):
            rounded = [native(x, precision) for x in point]
            candidates = [('rounded', rounded)]
            for j in range(len(point)):
                for direction in (-1, 1):
                    trial = rounded.copy()
                    trial[j] = neighbor(trial[j], precision, direction)
                    candidates.append((f'neighbor{j}{"p" if direction > 0 else "m"}', trial))
            candidates += [(label, [native(v, precision) for v in case[label]]) for label in ('start1', 'start2')]
            for label, values in candidates:
                points.append([case['id'], precision, label] + [repr(v) for v in values]
                              + [''] * (6 - len(values)))
    with (directory / 'points.csv').open('x', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['dataset', 'precision', 'point'] + [f'x{j}' for j in range(6)])
        writer.writerows(points)
    write_json(directory / 'references.json', dict(schema=1, purpose='reference-preflight',
               decimal_digits=PRECISION, source_sha256=source_hashes(), references=references,
               points_sha256=hashlib.sha256((directory / 'points.csv').read_bytes()).hexdigest()))


def interval(pair):
    return I(*pair)


def exact_float(value, precision):
    number = float(value)
    if not math.isfinite(number) or native(number, precision) != number:
        raise ValueError('CSV value is not finite native arithmetic')
    return D.from_float(number)


def quality(case, certificate, point, emitted, precision, scale):
    n = len(point)
    exact = evaluate(case, list(map(I, point)))
    data = [[D.from_float(native(v, precision)) for v in row] for row in case['data']]
    converted = evaluate(case, list(map(I, point)), data)
    cost = I(exact_float(emitted['cost'], precision))
    gradient = [I(exact_float(emitted[f'g{j}'], precision)) for j in range(n)]
    ref = interval(certificate['objective_interval'])
    if exact.v.hi < ref.lo:
        raise ValueError('witness objective invalidates the reference interval')
    coordinate = list(map(D, certificate['coordinate_scales']))
    gap = max(D(0), ((exact.v - I(ref.lo)) / scale).hi)
    stationarity = max((g * s / scale).magnitude() for g, s in zip(exact.g, coordinate))
    parameter = max(((I(v) - interval(box)) / s).magnitude()
                    for v, box, s in zip(point, certificate['parameter_box'], coordinate))
    u = I(D(2) ** (-53 if precision == 'f64' else -24))
    # Separate source-to-native data conversion from actual arithmetic error.
    conversion_f = (exact.v - converted.v).magnitude()
    arithmetic_f = (cost - converted.v).magnitude()
    abs_terms_f = I(exact_float(emitted['cost_abs_terms'], precision))
    if abs_terms_f.lo < 0 or abs_terms_f.lo != cost.lo:
        raise ValueError('native absolute cost terms are inconsistent')
    screen_f = max((8 * u * abs_terms_f).hi, (8 * u * I(converted.v.magnitude())).hi)
    conversion_g = [(a - b).magnitude() for a, b in zip(exact.g, converted.g)]
    arithmetic_g = [(a - b).magnitude() for a, b in zip(gradient, converted.g)]
    screens_g = [(8 * u * I(exact_float(emitted[f'ga{j}'], precision))).hi for j in range(n)]
    if any(exact_float(emitted[f'ga{j}'], precision) < gradient[j].magnitude() for j in range(n)):
        raise ValueError('native absolute gradient terms are inconsistent')
    # The observed native values must agree with a broad independent sanity
    # bound; this is an integrity check, not an interval arithmetic certificate.
    bounds = [I(0) for _ in range(n)]
    absolute_gradient_terms = [I(0) for _ in range(n)]
    response_terms = I(0)
    b = [Jet.variable(I(v), n, j) for j, v in enumerate(point)]
    for row in data:
        target = I(row[0]).ln() if case['id'] == 'Nelson' else I(row[0])
        y = response(case['id'], b, list(map(I, row[1:])))
        response_terms += (I(y.v.magnitude()) + I(target.magnitude())) ** 2 / 2
        for j in range(n):
            bounds[j] += (I(y.v.magnitude()) + I(target.magnitude())) * I(y.g[j].magnitude())
            absolute_gradient_terms[j] += I((y.v - target).magnitude()) * I(y.g[j].magnitude())
    screens_g = [max(screen, (8 * u * terms).hi) for screen, terms in zip(screens_g, absolute_gradient_terms)]
    if arithmetic_f > (4096 * u * response_terms).hi:
        raise ValueError('native cost integrity check failed')
    if any(e > (4096 * u * bound).hi for e, bound in zip(arithmetic_g, bounds)):
        raise ValueError('native gradient integrity check failed')
    objective_uncertainty = ((I(ref.width()) + conversion_f + max(arithmetic_f, screen_f)) / scale).hi
    gradient_uncertainty = max(((I(a) + max(b, c) + g.width()) * s / scale).hi
                               for a, b, c, g, s in zip(conversion_g, arithmetic_g, screens_g, exact.g, coordinate))
    parameter_uncertainty = max((I(interval(box).width()) / s).hi
                                for box, s in zip(certificate['parameter_box'], coordinate))
    return dict(point=list(map(str, point)), native_cost=str(cost.lo),
                objective_upper=str(gap), stationarity_upper=str(stationarity), parameter_upper=str(parameter),
                objective_uncertainty=str(objective_uncertainty), stationarity_uncertainty=str(gradient_uncertainty),
                parameter_uncertainty=str(parameter_uncertainty),
                data_conversion_cost_bound=str(conversion_f), native_cost_error_bound=str(arithmetic_f),
                cost_rounding_screen=str(screen_f),
                data_conversion_gradient_bounds=list(map(str, conversion_g)),
                native_gradient_error_bounds=list(map(str, arithmetic_g)),
                gradient_rounding_screens=list(map(str, screens_g)))


def failed_clauses(measure, q):
    root = I(q).sqrt().lo
    tests = [('objective', D(measure['objective_upper']) <= q),
             ('stationarity', D(measure['stationarity_upper']) <= root),
             ('parameter', D(measure['parameter_upper']) <= root),
             ('objective uncertainty margin', q > (10 * I(measure['objective_uncertainty'])).hi),
             ('stationarity uncertainty margin', root > (10 * I(measure['stationarity_uncertainty'])).hi),
             ('parameter uncertainty margin', root > (10 * I(measure['parameter_uncertainty'])).hi)]
    return [name for name, passed in tests if not passed]


def passes(measure, q):
    return not failed_clauses(measure, q)


def check(directory, output):
    saved = json.loads((directory / 'references.json').read_text())
    if saved['schema'] != 1 or len(saved['references']) != len(datasets()):
        raise ValueError('reference coverage mismatch')
    if saved['source_sha256'] != source_hashes():
        raise ValueError('source changed since reference preparation')
    if saved['points_sha256'] != hashlib.sha256((directory / 'points.csv').read_bytes()).hexdigest():
        raise ValueError('witness points changed')
    planned = list(csv.DictReader((directory / 'points.csv').open(newline='')))
    emitted = list(csv.DictReader((directory / 'native.csv').open(newline='')))
    def key(row):
        return row['dataset'], row['precision'], row['point']
    keyed = {key(row): row for row in emitted}
    if len(keyed) != len(emitted) or set(keyed) != {key(row) for row in planned}:
        raise ValueError('native witness coverage mismatch')
    results = []
    for case, certificate in zip(datasets(), saved['references']):
        if case['id'] != certificate['dataset']:
            raise ValueError('reference coverage mismatch')
        point = list(map(D, certificate['point']))
        verified = certify(case, point)
        for field in verified:
            if verified[field] != certificate[field]:
                raise ValueError(f'certificate changed: {field}')
        modeled = [I(row[0]).ln() if case['id'] == 'Nelson' else I(row[0]) for row in case['data']]
        mean = sum(modeled) / len(modeled)
        absolute = sum((y - mean) ** 2 for y in modeled) / 2
        if absolute.lo <= 0:
            absolute = sum(y ** 2 for y in modeled) / 2
        if absolute.lo <= 0:
            absolute = I(1)
        for precision in ('f64', 'f32'):
            subset = [row for row in planned if row['dataset'] == case['id'] and row['precision'] == precision]
            native_rows = {}
            for row in subset:
                result = keyed[key(row)]
                if result['schema'] != '1' or result['partition'] != 'development' or result['family'] != case['family']:
                    raise ValueError('native metadata mismatch')
                values = [exact_float(row[f'x{j}'], precision) for j in range(len(point))]
                if any(exact_float(result[f'x{j}'], precision) != v for j, v in enumerate(values)):
                    raise ValueError('native coordinates changed')
                native_rows[row['point']] = values, result
            grid = ['1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '1e-12'] if precision == 'f64' else ['1e-2', '1e-3', '1e-4', '1e-5', '1e-6']
            raw_measures = {label: quality(case, certificate, values, result, precision, I(1))
                            for label, (values, result) in native_rows.items() if label not in ('start1', 'start2')}
            for start in ('start1', 'start2'):
                initial = evaluate(case, list(map(I, native_rows[start][0]))).v
                reference = interval(certificate['objective_interval'])
                initial_gap = initial - reference
                # Freeze a conservative positive normalization, independently
                # of candidate outcomes and with its interval retained.
                relative = I(max(D(0), initial_gap.lo, -initial_gap.hi), initial_gap.magnitude())
                scale = I(max(absolute.lo, relative.lo), max(absolute.hi, relative.hi))
                measures = {}
                for label, unscaled in raw_measures.items():
                    measure = unscaled.copy()
                    for field in ('objective_upper', 'stationarity_upper', 'objective_uncertainty', 'stationarity_uncertainty'):
                        measure[field] = str((I(unscaled[field]) / I(scale.lo)).hi)
                    measures[label] = measure
                eligibility = []
                for q in grid:
                    witnesses = [label for label, m in measures.items() if passes(m, D(q))]
                    label = min(witnesses, key=lambda name: (D(measures[name]['objective_upper']), name)) if witnesses else None
                    eligibility.append(dict(q=q, status='eligible' if label else 'reference-pending', witness=label,
                                            reason='certified representable witness and tenfold uncertainty margin' if label
                                            else 'tested witnesses do not certify the joint target; no impossibility claim',
                                            rounded_witness_failed_clauses=failed_clauses(measures['rounded'], D(q))))
                results.append(dict(dataset=case['id'], family=case['family'], partition='development',
                                    precision=precision, derivative_mode='analytic', start=start,
                                    absolute_scale_interval=absolute.json(), initial_gap_interval=relative.json(),
                                    objective_scale_interval=scale.json(), frozen_objective_scale=str(scale.lo),
                                    targets=eligibility, witnesses=measures,
                                    finite_difference_eligibility='pending; derivative bias not certified'))
        print(f"checked {case['id']}", file=sys.stderr, flush=True)
    report = dict(schema=1, purpose='development-reference-and-analytic-witness-preflight',
                  status='passed', decimal_digits=PRECISION, source_sha256=source_hashes(),
                  references=saved['references'], cases=results,
                  native_csv_sha256=hashlib.sha256((directory / 'native.csv').read_bytes()).hexdigest(),
                  points_sha256=saved['points_sha256'], holdout_candidate_outcomes='sealed',
                  limitations=['local minimum certificates, not global or start-basin proofs',
                               'witness arithmetic errors bound only the tested points',
                               'finite-difference derivative eligibility pending',
                               'native nalgebra 0.34 witness adapter; full backend gate remains open'])
    # Retain every target and its chosen witness. Full neighbor measurements
    # remain reproducible in the raw report, rather than inflating Git records.
    write_json(directory / 'full-report.json', report)
    for result in report['cases']:
        chosen = {t['witness'] for t in result['targets'] if t['witness']}
        if not chosen:
            chosen = {'rounded'}
        result['witnesses'] = {label: result['witnesses'][label] for label in sorted(chosen)}
    write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('directory', type=Path)
    check_parser = sub.add_parser('check')
    check_parser.add_argument('directory', type=Path)
    check_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.directory)
    else:
        check(args.directory, args.output)


if __name__ == '__main__':
    main()

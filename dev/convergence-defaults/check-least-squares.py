#!/usr/bin/env python3
"""Check native decisions and verify published points outside the solve ledger.

This pilot accepts development NIST families only. It does not select defaults.
Native identity checks use a stated roundoff allowance; quality uses independent
100-digit intervals and the frozen reference/precision eligibility register.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal as D, getcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference-tools'))
from interval import Interval as I, Jet, PRECISION
from models import datasets, response
from preflight import exact_float, native, failed_clauses
getcontext().prec = PRECISION


class First(Jet):
    """First-order forward AD; no full Hessian is needed for point quality."""
    def __init__(self, value, gradient):
        self.v, self.g = value, gradient

    @classmethod
    def constant(cls, value, n):
        return cls(value, [value * 0] * n)

    def of(self, other):
        return other if isinstance(other, First) else self.constant(I.of(other) if isinstance(self.v, I) else D(other), len(self.g))

    def __add__(self, other):
        other = self.of(other)
        return First(self.v + other.v, [a + b for a, b in zip(self.g, other.g)])

    __radd__ = __add__

    def __neg__(self):
        return First(-self.v, [-g for g in self.g])

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        other = self.of(other)
        return First(self.v * other.v, [a * other.v + self.v * b for a, b in zip(self.g, other.g)])

    __rmul__ = __mul__

    def reciprocal(self):
        v = 1 / self.v
        return First(v, [-g * v ** 2 for g in self.g])

    def __truediv__(self, other):
        return self * self.of(other).reciprocal()

    def __rtruediv__(self, other):
        return self.of(other) / self

    def __pow__(self, n):
        return First(self.v ** n, [n * self.v ** (n - 1) * g for g in self.g]) if n else self.constant(I(1), len(self.g))

    def exp(self):
        v = self.v.exp()
        return First(v, [v * g for g in self.g])

    def ln(self):
        return First(self.v.ln(), [g / self.v for g in self.g])

    def sqrt(self):
        v = self.v.sqrt()
        return First(v, [g / (2 * v) for g in self.g])


def evaluate(case, point, data=None):
    b = [First.variable(I(x), len(point), j) for j, x in enumerate(point)]
    objective = First.constant(I(0), len(point))
    absolute = [I(0) for _ in point]
    rows = []
    for row in case['data'] if data is None else data:
        row = list(map(I, row))
        target = row[0].ln() if case['id'] == 'Nelson' else row[0]
        y = response(case['id'], b, row[1:])
        r = y - target
        objective += r ** 2 / 2
        rows.append((r.v, y.g))
        for j in range(len(point)):
            absolute[j] += I(r.v.magnitude()) * I(y.g[j].magnitude())
    return objective, absolute, rows


def branches(certificate):
    boxes = [I(*pair) for pair in certificate['parameter_box']]
    if certificate['dataset'].startswith('Lanczos'):
        return [[boxes[j] for pair in p for j in (2 * pair, 2 * pair + 1)]
                for p in itertools.permutations(range(3))]
    if certificate['dataset'] == 'MGH17':
        return [boxes, [boxes[j] for j in (0, 2, 1, 4, 3)]]
    return [boxes]


def parameter_error(certificate, point):
    coordinate = list(map(D, certificate['coordinate_scales']))
    return min(max(((I(v) - box) / scale).magnitude() for v, box, scale in zip(point, boxes, coordinate))
               for boxes in branches(certificate))


def quality(case, certificate, eligibility, point, emitted):
    precision = eligibility['precision']
    scale = I(eligibility['frozen_objective_scale'])
    exact, _, _ = evaluate(case, point)
    data = [[D.from_float(native(v, precision)) for v in row] for row in case['data']]
    converted, terms, _ = evaluate(case, point, data)
    cost = I(exact_float(emitted['cost'], precision))
    gradient = [I(exact_float(emitted[f'g{j}'], precision)) for j in range(len(point))]
    reference = I(*certificate['objective_interval'])
    coordinate = list(map(D, certificate['coordinate_scales']))
    unit = I(D(2) ** (-53 if precision == 'f64' else -24))
    conversion_f = (exact.v - converted.v).magnitude()
    arithmetic_f = (cost - converted.v).magnitude()
    screen_f = (8 * unit * I(converted.v.magnitude())).hi
    conversion_g = [(a - b).magnitude() for a, b in zip(exact.g, converted.g)]
    arithmetic_g = [(a - b).magnitude() for a, b in zip(gradient, converted.g)]
    screens_g = [max((8 * unit * term).hi, (8 * unit * I(exact_float(emitted[f'ga{j}'], precision))).hi)
                 for j, term in enumerate(terms)]
    measure = dict(objective_upper=str(max(D(0), ((exact.v - I(reference.lo)) / scale).hi)),
                   stationarity_upper=str(max((g * s / scale).magnitude() for g, s in zip(exact.g, coordinate))),
                   parameter_upper=str(parameter_error(certificate, point)),
                   objective_uncertainty=str(((I(reference.width()) + conversion_f + max(arithmetic_f, screen_f)) / scale).hi),
                   stationarity_uncertainty=str(max(((I(a) + max(b, c) + g.width()) * s / scale).hi
                       for a, b, c, g, s in zip(conversion_g, arithmetic_g, screens_g, exact.g, coordinate))),
                   parameter_uncertainty=str(max((I(box.width()) / s).hi
                       for branch in branches(certificate) for box, s in zip(branch, coordinate))),
                   independent_cost_interval=exact.v.json(), native_verification_cost=emitted['cost'])
    if exact.v.hi < reference.lo:
        # A local reference need not be the best minimum reached by this start.
        measure['below_local_reference'] = True
    return measure


def load(directory, name):
    with (directory / f'{name}.csv').open(newline='') as file:
        return list(csv.DictReader(file))


def numbers(value):
    return [float(x) for x in value.split(';')] if value else []


def close(a, b, unit, scale=None):
    if not math.isfinite(a) or not math.isfinite(b):
        return math.isnan(a) and math.isnan(b) or a == b
    return abs(a - b) <= 4096 * unit * max(abs(a), abs(b), scale or 0, 1e-300)


def require(test, message):
    if not test:
        raise ValueError(message)


ROBUST_CASES = {
    'robust_huber_outlier': ('huber', 'outlier', D(1), D('.5'), None),
    'robust_huber_scaled': ('huber', 'outlier', D('.5'), D('.25'), None),
    'robust_soft_l1': ('soft_l1', 'symmetric', D(1), D(0), None),
    'robust_cauchy': ('cauchy', 'linear', D(1), D(1), None),
    'robust_huber_kink': ('huber', 'linear', D('.5'), D(1), None),
    'robust_nonfinite': ('soft_l1', 'nonfinite', D(1), D(1), None),
    'robust_huber_bound': ('huber', 'outlier', D('.5'), D('.125'), (D(0), D('.125'))),
}


def robust_values(run, point):
    loss, model, scale, _, _ = ROBUST_CASES[run['dataset']]
    x = D.from_float(point[0])
    if model == 'outlier':
        residuals, jacobian = [x, x, x - 8], [D(1)] * 3
    elif model == 'symmetric':
        residuals, jacobian = [x - 2, x + 2], [D(1)] * 2
    elif model == 'nonfinite':
        if x < 0 or x > D.from_float(native('1.1', run['precision'])):
            return math.inf, math.nan, []
        residuals, jacobian = [1 + (x - 1) ** 2], [2 * (x - 1)]
    else:
        residuals, jacobian = [x - 1], [D(1)]
    costs, derivatives, curvatures = [], [], []
    for r in residuals:
        if loss == 'huber':
            costs.append(r * r / 2 if abs(r) <= scale else scale * (abs(r) - scale / 2))
            derivatives.append(r if abs(r) <= scale else scale.copy_sign(r))
            curvatures.append(D(1) if abs(r) <= scale else D(0))
        elif loss == 'soft_l1':
            root = (1 + (r / scale) ** 2).sqrt()
            costs.append(r * r / (root + 1))
            derivatives.append(r / root)
            curvatures.append(1 / root ** 3)
        else:
            z = (r / scale) ** 2
            costs.append(scale ** 2 * (1 + z).ln() / 2)
            derivatives.append(r / (1 + z))
            curvatures.append((1 - z) / (1 + z) ** 2)
    epsilon = D(2) ** (-23 if run['precision'] == 'f32' else -52)
    rows = [[float(max(c, epsilon).sqrt() * j)] for c, j in zip(curvatures, jacobian)]
    return float(sum(costs)), float(sum(d * j for d, j in zip(derivatives, jacobian))), rows


def robust_quality(run, pubs):
    _, _, _, reference, bounds = ROBUST_CASES[run['dataset']]
    unit = 2 ** (-24 if run['precision'] == 'f32' else -53)
    for p in pubs:
        x = numbers(p['point'])
        require(len(x) == 1, 'robust point dimension')
        if bounds:
            require(float(bounds[0]) <= x[0] <= float(bounds[1]), 'robust feasibility')
        cost, _, _ = robust_values(run, x)
        require(close(float(p['cost']), cost, unit), 'robust independent published objective')
    if run['returned'] != 'true':
        return None
    x = numbers(pubs[-1]['point'])
    cost, gradient, _ = robust_values(run, x)
    reference_cost, _, _ = robust_values(run, [float(reference)])
    kkt = abs(gradient) if bounds is None else abs(x[0] - min(float(bounds[1]), max(float(bounds[0]), x[0] - gradient)))
    error = abs(x[0] - float(reference))
    limit = 1e-3 if run['precision'] == 'f32' else 1e-6
    if run['policy'] == 'robust_relative_probe':
        limit = 2e-2 if run['precision'] == 'f32' else 1e-4
    return dict(point=x, reference=float(reference), cost=cost, reference_cost=reference_cost,
                objective_gap=cost-reference_cost, gradient=gradient, stationarity=kkt,
                parameter_error=error, quality_limit=limit, quality_passed=error <= limit and kkt <= limit)


BOX_CASES = {
    'box_active': ([0., -.5], [.5, 0.]),
    'box_mixed_fixed': ([.25, -2.], [.25, 0.]),
    'box_stationary': ([-1., -2.], [1., 0.]),
    'box_all_fixed': ([.25, -.25], [.25, -.25]),
}


def box_quality(run, pubs):
    lower, upper = BOX_CASES[run['dataset']]
    target = [min(hi, max(lo, t)) for lo, hi, t in zip(lower, upper, [1., -1.])]
    for p in pubs:
        x = numbers(p['point'])
        require(len(x) == 2 and all(lo <= v <= hi for v, lo, hi in zip(x, lower, upper)), 'box publication feasibility')
        residual = [D.from_float(x[0]) - 1, D.from_float(x[1]) + 1]
        if run['dataset'] == 'box_stationary':
            residual[0] = 2 - D.from_float(x[0]) ** 2
        cost = sum(v * v for v in residual) / 2
        unit = 2 ** (-24 if run['precision'] == 'f32' else -53)
        require(close(float(p['cost']), float(cost), unit), 'box independent objective')
    if run['returned'] != 'true':
        require(run['outcome'] in ('initialization_error', 'callback_error'), 'box budget outcome')
        return None
    x = numbers(pubs[-1]['point'])
    if run['dataset'] == 'box_stationary':
        require(x == [0., -1.] and run['criteria'] == 'scaled_gradient' and run['outcome'] == 'converged', 'stationary negative control')
        return dict(point=x, references=[[-1., -1.], [1., -1.]], parameter_error=1., objective_gap=1.5, projected_kkt=0., quality_passed=False, classification='stationary-nonminimum')
    error = max(abs(v - t) for v, t in zip(x, target))
    # Projection with a unit step is an independent box KKT residual.
    kkt = max(abs(v - min(hi, max(lo, t))) for v, t, lo, hi in zip(x, [1., -1.], lower, upper))
    limit = 1e-3 if run['precision'] == 'f32' else 1e-6
    require(error <= limit and kkt <= limit, 'box returned quality')
    return dict(point=x, reference=target, parameter_error=error, projected_kkt=kkt, quality_limit=limit, quality_passed=True)


def verify(directory):
    runs = load(directory, 'runs')
    grouped = {}
    for name in ('publications', 'leaves', 'native', 'checks'):
        grouped[name] = defaultdict(list)
        for r in load(directory, name):
            grouped[name][r['id']].append(r)
    require(len({r['id'] for r in runs}) == len(runs), 'duplicate run')
    run_ids = {r['id'] for r in runs}
    require(all(set(rows) <= run_ids for rows in grouped.values()), 'orphan observation')
    statistics = Counter()
    cases = {c['id']: c for c in datasets()}
    jacobian_cache = {}
    def independent_jacobian(run, point):
        if run['dataset'] in ROBUST_CASES:
            return robust_values(run, point)[2]
        if run['dataset'] == 'box_stationary':
            return [[-2 * point[0], 0.], [0., 1.]]
        if run['dataset'] in BOX_CASES:
            return [[1., 0.], [0., 1.]]
        if run['dataset'] not in cases:
            return [[1.0 if run['dataset'].startswith('linear') else 2 * (point[0] - 1)]]
        key = run['dataset'], run['precision'], tuple(point)
        if key not in jacobian_cache:
            case = cases[run['dataset']]
            b = [First.variable(D.from_float(x), len(point), j) for j, x in enumerate(point)]
            jacobian_cache[key] = [[float(g) for g in response(case['id'], b, [D.from_float(native(v, run['precision'])) for v in row[1:]]).g] for row in case['data']]
        return jacobian_cache[key]
    for run in runs:
        identifier = run['id']
        pubs, leaves, observations, checks = [grouped[k][identifier] for k in grouped]
        unit = 2 ** (-24 if run['precision'] == 'f32' else -53)
        require([int(c['work']) for c in leaves] == list(range(1, len(leaves) + 1)), f'{identifier}: work sequence')
        require(int(run['work']) == len(leaves) <= int(run['cap']), f'{identifier}: physical budget')
        require(all(c['kind'] != 'Cost' for c in leaves), f'{identifier}: synthesized cost charged physically')
        counts = Counter(c['kind'] for c in leaves)
        if run['residual_evals']:
            difference_r = int(run['residual_evals']) - counts['Residual'] - counts['ResidualJacobian']
            difference_j = int(run['jacobian_evals']) - counts['Jacobian'] - counts['ResidualJacobian']
            require(difference_r >= 0 and difference_j >= 0 and difference_r + difference_j == int(run['denied']), f'{identifier}: logical/physical counts')
            require(int(run['cost_evals']) == 0, f'{identifier}: synthesized cost is not a CostFunction request')
        else:
            require(run['outcome'] == 'initialization_error' and not pubs, f'{identifier}: missing logical counts')
        require((run['returned'] == 'true') == (run['outcome'] in ('converged', 'failed', 'stalled', 'limit', 'other')), f'{identifier}: returned-point ownership')
        require([int(o['sequence']) for o in observations] == sorted({int(o['sequence']) for o in observations}), f'{identifier}: duplicate sequence')
        keyed = {o['sequence']: o for o in observations}
        leaf_keyed = {c['work']: c for c in leaves}
        robust = run['dataset'] in ROBUST_CASES
        if robust:
            robust_quality(run, pubs)
        for o in observations:
            if robust and o['gradient']:
                base_point = next((numbers(p['point']) for p in reversed(pubs) if int(p['work']) <= int(o['work_before'])), None)
                require(base_point is not None, f'{identifier}: robust base publication')
                base_cost, gradient, rows = robust_values(run, base_point)
                require(close(float(o['base_cost']), base_cost, unit), f'{identifier}: robust base objective')
                expected = gradient
                if run['route'] == 'trf_full' and o['trial'] == 'true':
                    expected *= numbers(o['coordinate_scale'])[0]
                require(len(numbers(o['gradient'])) == 1 and close(numbers(o['gradient'])[0], expected, unit, max(1., abs(expected))), f'{identifier}: independent robust gradient')
                if run['route'].startswith('lm_') and o['trial'] != 'true':
                    require(close(numbers(o['diagonal'])[0], sum(row[0] ** 2 for row in rows), unit), f'{identifier}: safeguarded robust model diagonal')
            if o['trial'] != 'true':
                statistics['model_observations'] += 1
                continue
            statistics['attempted_trials'] += 1
            if not o['residual_work']:
                require(not o['trial_cost'] and run['outcome'] == 'callback_error', f'{identifier}: pending trial')
                statistics['budget_denied_trials'] += 1
                continue
            leaf = leaf_keyed[o['residual_work']]
            require(leaf['kind'] == 'Residual' and int(o['work_before']) + 1 == int(leaf['work']), f'{identifier}: trial leaf matching')
            if not o['trial_cost']:
                require(leaf['outcome'].startswith('Failed'), f'{identifier}: incomplete residual')
                statistics['failed_trial_callbacks'] += 1
                continue
            base = float(o['base_cost']); trial = float(o['trial_cost'])
            if robust:
                expected_cost, _, _ = robust_values(run, numbers(leaf['point']))
                require(close(trial, expected_cost, unit), f'{identifier}: robust trial objective')
            h, g, diagonal, curvature = [numbers(o[k]) for k in ('step', 'gradient', 'diagonal', 'curvature')]
            actual = base - trial - (sum(c * v * v for c, v in zip(curvature, h)) / 2 if run['route'] == 'trf_legacy' else 0)
            if o['actual']:
                require(close(float(o['actual']), actual, unit, abs(base) + abs(trial)), f'{identifier}: actual decrease')
            if run['route'] != 'trf_full':
                predicted = (float(o['damping']) * sum(d * v * v for d, v in zip(diagonal, h)) - sum(v * w for v, w in zip(h, g))) / 2
                # Native squared-step products can underflow before damping rescales them.
                subnormal = 2 ** (-149 if run['precision'] == 'f32' else -1074)
                lost_products = abs(float(o['damping'])) * subnormal * len(h) * max([1] + list(map(abs, diagonal)))
                prediction_scale = (abs(float(o['damping'])) * sum(abs(d * v * v) for d, v in zip(diagonal, h)) + sum(abs(v * w) for v, w in zip(h, g))) / 2
                if robust:
                    rows = robust_values(run, base_point)[2]
                    independent = -sum(v * w for v, w in zip(g, h)) - sum((row[0] * h[0]) ** 2 for row in rows) / 2 - sum(c * v * v for c, v in zip(curvature, h)) / 2
                    independent_scale = sum(abs(v * w) for v, w in zip(g, h)) + sum((row[0] * h[0]) ** 2 for row in rows) / 2 + sum(abs(c * v * v) / 2 for c, v in zip(curvature, h))
                    require(close(float(o['predicted']), independent, unit, independent_scale), f'{identifier}: independent robust model decrease')
                    if run['dataset'] == 'robust_huber_bound':
                        predicted = independent
                        prediction_scale = independent_scale
                require(close(float(o['predicted']), predicted, unit, prediction_scale) or abs(float(o['predicted']) - predicted) <= lost_products, f'{identifier}: predicted decrease')
                expected_ratio = actual / float(o['predicted']) if float(o['predicted']) > 0 else 0.0
                require(close(float(o['ratio']), expected_ratio, unit, (abs(base) + abs(trial)) / max(abs(float(o['predicted'])), 1e-300)), f'{identifier}: safeguarded ratio')
                accepted = float(o['ratio']) > 0
            else:
                base_point = next((numbers(p['point']) for p in reversed(pubs) if int(p['work']) <= int(o['work_before'])), None)
                require(base_point is not None, f'{identifier}: trial base publication')
                rows = independent_jacobian(run, base_point)
                scale = numbers(o['coordinate_scale']); free = [int(v) for v in o['free'].split(';')]
                jh = [sum(j[k] * d * v for k, d, v in zip(free, scale, h)) for j in rows]
                terms = [v * w for v, w in zip(g, h)] + [v * v / 2 for v in jh] + [c * v * v / 2 for c, v in zip(curvature, h)]
                predicted = -sum(terms)
                prediction_scale = sum(abs(v * w) for v, w in zip(g, h)) + sum(sum(abs(j[k] * d * v) for k, d, v in zip(free, scale, h)) ** 2 / 2 for j in rows) + sum(abs(c * v * v) / 2 for c, v in zip(curvature, h))
                require(close(float(o['predicted']), predicted, unit, prediction_scale), f'{identifier}: independent TRF model decrease')
                accepted = math.isfinite(trial) and actual > 0
                if math.isfinite(trial):
                    require(close(float(o['ratio']), actual / float(o['predicted']), unit, (abs(base) + abs(trial)) / float(o['predicted'])), f'{identifier}: TRF ratio')
                hnorm = math.hypot(*h)
                ratio = float(o['ratio']) if o['ratio'] else -math.inf
                before = float(o['radius_before'])
                after = hnorm / 4 if ratio < .25 else 2 * before if ratio > .75 and hnorm > .95 * before else before
                require(close(float(o['radius_after']), after, unit), f'{identifier}: TRF radius')
            require((o['accepted'] == 'true') == accepted, f'{identifier}: acceptance')
            if o['published'] == 'true':
                require(accepted and any(p['point'] == leaf['point'] and int(p['work']) >= int(leaf['work']) for p in pubs), f'{identifier}: accepted publication')
            statistics['accepted_trials' if accepted else 'rejected_trials'] += 1
            statistics['published_trials'] += o['published'] == 'true'
            statistics['accepted_unpublished_trials'] += accepted and o['published'] != 'true'
            statistics['nonfinite_trials'] += not math.isfinite(trial)
        for c in checks:
            require(c['sequence'] in keyed, f'{identifier}: orphan check')
            if c['name'] == 'no_free_parameters':
                require(c['evidence'] == 'no_free_parameters' and c['passed'] == 'true' and not c['tolerance'], f'{identifier}: structural evidence')
                require(not keyed[c['sequence']]['free'], f'{identifier}: structural free coordinates')
                statistics['structural_checks'] += 1
                continue
            if not c['tolerance']:
                require(not c['passed'] and not c['evidence'], f'{identifier}: disabled comparison')
                statistics['disabled_checks'] += 1
                continue
            statistics['passing_checks' if c['passed'] == 'true' else 'failing_checks'] += 1
            if c['evidence'] in ('upper_bound', 'factored_upper_bound'):
                value = D(c['value']) * D(2) ** int(c['value_exponent'] or 0)
                bound = D(c['bound']) * D(2) ** int(c['bound_exponent'] or 0)
                passed = value <= bound
                require(passed == (c['passed'] == 'true') or abs(value - bound) <= D(str(4096 * unit)) * max(abs(value), abs(bound)), f'{identifier}: {c["name"]} comparison')
                o = keyed[c['sequence']]
                if c['name'] == 'absolute_gradient':
                    require(close(float(value), max(map(abs, numbers(o['gradient']))), unit), f'{identifier}: gradient evidence')
                if c['name'] in ('orthogonality', 'robust_orthogonality'):
                    o = keyed[c['sequence']]
                    g = numbers(o['gradient']); diag = numbers(o['diagonal']); rnorm = math.sqrt(2 * float(o['base_cost']))
                    quotient = max((abs(v) / math.sqrt(d) / rnorm if v and d and rnorm else 0 for v, d in zip(g, diag)), default=0)
                    require(close(float(value), quotient, unit), f'{identifier}: orthogonality operands')
                if c['name'] == 'scaled_gradient':
                    g = numbers(o['gradient']); diag = numbers(o['diagonal'])
                    metric = max((abs(v) / d for v, d in zip(g, diag)), default=0) if diag else max(map(abs, g), default=0)
                    if run['dataset'] in BOX_CASES and run['route'] == 'trf_full':
                        x = next(numbers(p['point']) for p in reversed(pubs) if int(p['work']) <= int(o['work_before']))
                        lower, upper = BOX_CASES[run['dataset']]
                        free = [i for i in range(2) if lower[i] != upper[i]]
                        require([int(i) for i in o['free'].split(';') if i] == free, f'{identifier}: free-coordinate map')
                        raw = [x[0] - 1, x[1] + 1]
                        if run['dataset'] == 'box_stationary':
                            raw[0] = -2 * x[0] * (2 - x[0] ** 2)
                        require(all(close(v, raw[i], unit) for v, i in zip(g, free)) and len(g) == len(free), f'{identifier}: independent free gradient')
                        metric = max((abs(raw[i]) * (upper[i] - x[i] if raw[i] < 0 else x[i] - lower[i]) for i in free), default=0)
                    if robust and run['route'] == 'trf_full':
                        x = next(numbers(p['point']) for p in reversed(pubs) if int(p['work']) <= int(o['work_before']))
                        _, raw, _ = robust_values(run, x)
                        bounds = ROBUST_CASES[run['dataset']][4]
                        v = 1. if bounds is None or raw == 0 else float(bounds[1]) - x[0] if raw < 0 else x[0] - float(bounds[0])
                        metric = abs(raw) * v
                    require(close(float(value), metric, unit, 1. if robust else None), f'{identifier}: scaled-gradient evidence')
                if c['name'] in ('trial_step', 'trust_radius'):
                    if c['name'] == 'trial_step':
                        require(close(float(value), math.hypot(*numbers(o['step'])), unit), f'{identifier}: step evidence')
                    else:
                        require(close(float(value), float(o['radius_after']), unit), f'{identifier}: radius evidence')
                    trial_leaf = leaf_keyed[o['residual_work']]
                    x = numbers(trial_leaf['point']) if o['accepted'] == 'true' else next(numbers(p['point']) for p in reversed(pubs) if int(p['work']) <= int(o['work_before']))
                    expected_ref = math.hypot(*x) if c['name'] == 'trial_step' else math.sqrt(sum(d * v * v for d, v in zip(numbers(o['diagonal']), x)))
                    ref = float(D(c['reference']) * D(2) ** int(c['reference_exponent'] or 0))
                    require(close(ref, expected_ref, unit), f'{identifier}: {c["name"]} reference iterate')
                    require(close(float(bound), float(c['tolerance']) * ref, unit), f'{identifier}: factored bound')
            elif c['evidence'] == 'model_reduction':
                tol = float(c['tolerance']); actual = float(c['actual']); predicted = float(c['predicted']); ref = float(c['reference_cost']); ratio = float(c['ratio'])
                passed = abs(actual) <= tol * ref and predicted <= tol * ref and ratio <= 2
                require(passed == (c['passed'] == 'true'), f'{identifier}: model-reduction conjunction')
            else:
                require(c['passed'] == 'false' and not c['evidence'], f'{identifier}: unexpected evidence')
        if run['outcome'] == 'converged':
            last = observations[-1]['sequence']
            passing = [c['name'] for c in checks if c['sequence'] == last and c['passed'] == 'true']
            require(run['criteria'].split(';') == passing, f'{identifier}: all passing termination criteria')
        statistics['runs'] += 1
        statistics[f'outcome_{run["outcome"]}'] += 1
    return runs, grouped, dict(statistics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite report')
    runs, grouped, statistics = verify(args.directory)
    report = dict(schema=1, purpose='least-squares-measurement-pilot', status='passed', statistics=statistics,
                  native_integrity_allowance='4096 native unit roundoffs plus a bound for underflow before damping rescales squared steps; identities, not accuracy certificates',
                  solve_timing='instrumented, unoptimized; no speed comparison', policies_selected=False,
                  verification_outside_solve_ledger=True, holdout_candidate_outcomes='sealed', runs=[])
    nist_cases = {c['id']: c for c in datasets()}
    is_nist = all(r['dataset'] in nist_cases for r in runs)
    is_box = all(r['dataset'] in BOX_CASES for r in runs)
    is_robust = all(r['dataset'] in ROBUST_CASES for r in runs)
    require(is_nist or is_box or is_robust or all(r['dataset'] in ('linear', 'linear_budget', 'nonzero', 'nonfinite') for r in runs), 'mixed or unsupported phase')
    if is_robust:
        lm_routes = ('lm_normal_nielsen', 'lm_normal_trust', 'lm_qr_nielsen', 'lm_qr_trust')
        expected = set()
        for d, (_, _, _, _, bounds) in ROBUST_CASES.items():
            for p in ('f32', 'f64'):
                for cap in (0, 1, 2, 4000):
                    for route in (('trf_legacy', 'trf_full') if bounds else lm_routes + ('trf_legacy', 'trf_full')):
                        expected.add((d, p, cap, route, 'robust_default'))
                if bounds is None:
                    for route in lm_routes:
                        expected.add((d, p, 4000, route, 'robust_relative_probe'))
        require(len(runs) == 352 and {(r['dataset'], r['precision'], int(r['cap']), r['route'], r['policy']) for r in runs} == expected, 'robust route coverage')
        for run in runs:
            report['runs'].append(dict({k: run[k] for k in ('id', 'outcome', 'criteria', 'work', 'denied', 'returned')}, quality=robust_quality(run, grouped['publications'][run['id']])))
    elif is_box:
        require(len(runs) == 32 and {(r['dataset'], r['precision'], int(r['cap']), r['route'], r['policy']) for r in runs} ==
                {(d, p, c, 'trf_full', 'bounded_default') for d in BOX_CASES for p in ('f32', 'f64') for c in (0, 1, 2, 4000)}, 'bounded coverage')
        for run in runs:
            pubs = grouped['publications'][run['id']]
            quality = box_quality(run, pubs)
            if run['dataset'] == 'box_all_fixed' and int(run['cap']) > 0:
                require(run['criteria'] == 'no_free_parameters' and int(run['work']) == 1 and int(run['jacobian_evals']) == 0, 'all-fixed callback contract')
            report['runs'].append(dict({k: run[k] for k in ('id', 'outcome', 'criteria', 'work', 'denied', 'returned')}, quality=quality))
    elif not is_nist:
        require(len(runs) == 80, 'analytic route coverage')
        for run in runs:
            pubs = grouped['publications'][run['id']]
            if run['dataset'] == 'linear_budget':
                require(run['outcome'] in ('initialization_error', 'callback_error'), 'budget fixture incorrectly converged')
            elif run['returned'] == 'true':
                x = numbers(pubs[-1]['point'])[0]
                require(abs(x - 1) < (2e-2 if run['policy'] == 'relative_probe' and run['precision'] == 'f32' else 1e-3 if run['precision'] == 'f32' else 1e-5), 'analytic returned point')
            report['runs'].append({k: run[k] for k in ('id', 'outcome', 'criteria', 'work', 'denied', 'returned')})
    else:
        require(len(runs) == 360 and args.probe, 'NIST route coverage or missing independent native probe')
        saved = json.loads((ROOT / 'nist-reference-eligibility.json').read_text())
        for name, digest in saved['source_sha256'].items():
            require(hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest() == digest, f'frozen reference source changed: {name}')
        certificates = {c['dataset']: c for c in saved['references']}
        eligible = {(c['dataset'], c['precision'], c['start']): c for c in saved['cases']}
        require({(r['dataset'], r['precision'], int(r['start']), r['route']) for r in runs} ==
                set(itertools.product(nist_cases, ('f64', 'f32'), (1, 2), ('lm_normal_nielsen', 'lm_normal_trust', 'lm_qr_nielsen', 'lm_qr_trust', 'trf_legacy', 'trf_full'))), 'NIST variants missing')
        points = {}; point_id = {}
        def add(run, text, force=False):
            point = numbers(text)
            if not all(map(math.isfinite, point)):
                return None
            certificate = certificates[run['dataset']]
            if not force and parameter_error(certificate, list(map(D.from_float, point))) > D('.1'):
                return None
            key = run['dataset'], run['precision'], text
            if key not in point_id:
                label = f'p{len(points)}'; point_id[key] = label
                points[label] = key
            return point_id[key]
        for run in runs:
            pubs = grouped['publications'][run['id']]
            for i, p in enumerate(pubs):
                add(run, p['point'], force=i == len(pubs) - 1)
            for leaf in grouped['leaves'][run['id']]:
                if leaf['outcome'] == 'Completed':
                    add(run, leaf['point'])
        with (args.directory / 'verification-points.csv').open('x', newline='') as out:
            writer = csv.writer(out); writer.writerow(['dataset', 'precision', 'point'] + [f'x{j}' for j in range(6)])
            for label, (dataset, precision, text) in points.items():
                x = text.split(';'); writer.writerow([dataset, precision, label] + x + [''] * (6 - len(x)))
        subprocess.run([str(args.probe), '--points', str(args.directory / 'verification-points.csv'), '--output', str(args.directory / 'verification-native.csv')], check=True)
        emitted = {r['point']: r for r in load(args.directory, 'verification-native')}
        require(set(emitted) == set(points), 'verification point coverage')
        for label, (dataset, precision, text) in points.items():
            row = emitted[label]; point = numbers(text)
            require(row['schema'] == '1' and row['dataset'] == dataset and row['precision'] == precision and row['partition'] == 'development', 'native verification metadata')
            require([float(row[f'x{j}']) for j in range(len(point))] == point, 'native verification coordinates')
            require(row['cost_abs_terms'] == row['cost'], 'native verification cost terms')
            require(all(float(row[f'ga{j}']) >= abs(float(row[f'g{j}'])) for j in range(len(point))), 'native verification gradient terms')
        measures = {}
        for run in runs:
            case = nist_cases[run['dataset']]; certificate = certificates[run['dataset']]
            eligibility = eligible[run['dataset'], run['precision'], f'start{run["start"]}']
            def measure_point(text):
                label = point_id.get((run['dataset'], run['precision'], text))
                if label is None:
                    return None
                cache = label, eligibility['frozen_objective_scale']
                if cache not in measures:
                    measures[cache] = quality(case, certificate, eligibility, list(map(D.from_float, numbers(text))), emitted[label])
                return measures[cache]
            pubs = grouped['publications'][run['id']]; leaves = grouped['leaves'][run['id']]
            parameters = {}
            def possible(text, q):
                if not all(map(math.isfinite, numbers(text))):
                    return False
                if text not in parameters:
                    parameters[text] = parameter_error(certificate, list(map(D.from_float, numbers(text))))
                return parameters[text] <= I(q).sqrt().lo
            last_quality = measure_point(pubs[-1]['point']) if pubs else None
            targets = []
            for target in eligibility['targets']:
                q = D(target['q']); eligible_target = target['status'] == 'eligible'
                def passed(m):
                    return eligible_target and m is not None and not failed_clauses(m, q)
                first = next((p['work'] for p in pubs if eligible_target and possible(p['point'], q) and passed(measure_point(p['point']))), None)
                sampled = next((c['work'] for c in leaves if eligible_target and c['outcome'] == 'Completed' and possible(c['point'], q) and passed(measure_point(c['point']))), None)
                returned = run['returned'] == 'true' and passed(last_quality)
                targets.append(dict(q=target['q'], eligibility=target['status'], first_publication_work=first,
                                    first_sample_work=sampled, returned_pass=returned))
            report['runs'].append(dict(id=run['id'], dataset=run['dataset'], precision=run['precision'], start=run['start'], route=run['route'],
                outcome=run['outcome'], work=int(run['work']), returned=run['returned'] == 'true',
                last_published_quality=last_quality,
                returned_quality=last_quality if run['returned'] == 'true' else None, targets=targets))
            print(f'checked {run["id"]}', file=sys.stderr, flush=True)
        report['verification_sha256'] = {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest() for name in ('verification-points.csv', 'verification-native.csv')}
        report['verification_points'] = len(points)
        report['quality_evaluations'] = len(measures)
        report['reference_register_sha256'] = hashlib.sha256((ROOT / 'nist-reference-eligibility.json').read_bytes()).hexdigest()
        report['quality_limitations'] = ['local references do not prove global optimality or either-start basin membership',
            'targets marked reference-pending are withheld', 'native probe arithmetic screens cover these tested points only',
            'parameter gates exclude points that cannot pass the joint target being tested; last publications always receive full quality verification',
            'finite differences, robust losses, transformed cases, and other backends remain outside this run']
    report['input_sha256'] = {name: hashlib.sha256((args.directory / f'{name}.csv').read_bytes()).hexdigest()
                            for name in ('runs', 'publications', 'leaves', 'native', 'checks')}
    with args.output.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(statistics, sort_keys=True))


if __name__ == '__main__':
    main()

"""Development-only source formulas for independent reference arithmetic."""

import hashlib
import json
import re
from decimal import Decimal as D
from pathlib import Path

from interval import Interval, Jet, UP

ROOT = Path(__file__).resolve().parents[1]


def datasets():
    manifest = json.loads((ROOT / 'nist/manifest.json').read_text())
    cases = []
    for entry in manifest['datasets']:
        if entry['partition'] != 'development':
            continue
        snapshot = (ROOT / 'nist' / entry['path']).read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != entry['snapshot_sha256']:
            raise ValueError(f"changed snapshot: {entry['id']}")
        lines = re.split(r'(?m)^Data:\s+y[^\n]*\n', snapshot.decode('ascii'))[1].splitlines()
        rows = [[D(v) for v in line.split()] for line in lines if line.strip()]
        if len(rows) != entry['observations']:
            raise ValueError('observation dimension mismatch')
        cases.append(dict(entry, data=rows))
    return cases


def response(name, b, x):
    t = x[0]
    if name in ('Misra1a', 'BoxBOD'):
        return b[0] * (1 - (-b[1] * t).exp())
    if name == 'Misra1b':
        return b[0] * (1 - (1 + b[1] * t / 2) ** -2)
    if name == 'Misra1c':
        return b[0] * (1 - 1 / (1 + 2 * b[1] * t).sqrt())
    if name == 'Misra1d':
        return b[0] * b[1] * t / (1 + b[1] * t)
    if name.startswith('Lanczos'):
        return sum(b[j] * (-b[j + 1] * t).exp() for j in (0, 2, 4))
    if name == 'MGH17':
        return b[0] + b[1] * (-b[3] * t).exp() + b[2] * (-b[4] * t).exp()
    if name == 'MGH10':
        return b[0] * (b[1] / (t + b[2])).exp()
    if name == 'DanWood':
        return b[0] * (b[1] * t.ln()).exp()
    if name == 'Bennett5':
        return b[0] * (-(b[1] + t).ln() / b[2]).exp()
    if name == 'Nelson':
        return b[0] - b[1] * t * (-b[2] * x[1]).exp()
    if name in ('Rat42', 'Rat43'):
        q = 1 + (b[1] - b[2] * t).exp()
        return b[0] / (q if name == 'Rat42' else (q.ln() / b[3]).exp())
    raise ValueError(f'not a development model: {name}')


def evaluate(case, point, data=None):
    n = len(point)
    b = [Jet.variable(value, n, j) for j, value in enumerate(point)]
    result = Jet.constant(point[0] * 0, n)
    for row in case['data'] if data is None else data:
        cast = Interval.of if isinstance(point[0], Interval) else D
        row = [cast(v) for v in row]
        target = row[0].ln() if case['id'] == 'Nelson' else row[0]
        residual = response(case['id'], b, row[1:]) - target
        result += residual ** 2 / 2
    return result


def solve(a, b):
    """Pivoted Decimal elimination is an approximation, never a certificate."""
    n = len(b)
    rows = [list(row) + [value] for row, value in zip(a, b)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(rows[i][j]))
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        if divisor == 0:
            raise ValueError('singular approximation')
        rows[j] = [v / divisor for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [v - factor * w for v, w in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def coordinate_scales(case):
    return [max(abs(D(a)), abs(D(b)), abs(D(c))) or D(1)
            for a, b, c in zip(case['certified_parameters'], case['start1'], case['start2'])]


def certify(case, point, radius=D('1e-50')):
    """Krawczyk inclusion and interval LDL certify a unique strict local minimum."""
    n = len(point)
    scales = coordinate_scales(case)
    center = evaluate(case, list(map(Interval, point)))
    delta = [Interval(-radius, radius) * scale for scale in scales]
    box = [Interval(x) + d for x, d in zip(point, delta)]
    full = evaluate(case, box)
    columns = [solve([[v.midpoint() for v in row] for row in center.h],
                     [D(i == j) for i in range(n)]) for j in range(n)]
    inverse = [[Interval(columns[j][i]) for j in range(n)] for i in range(n)]
    defect = [[Interval(int(i == j)) - sum(inverse[i][k] * full.h[k][j] for k in range(n))
               for j in range(n)] for i in range(n)]
    krawczyk = [Interval(point[i]) - sum(inverse[i][j] * center.g[j] for j in range(n))
                + sum(defect[i][j] * delta[j] for j in range(n)) for i in range(n)]
    contraction = D(0)
    for i in range(n):
        row_bound = D(0)
        for j in range(n):
            row_bound = UP.add(row_bound, (defect[i][j] * scales[j] / scales[i]).magnitude())
        contraction = max(contraction, row_bound)
    if contraction >= 1 or not all(x.lo < k.lo <= k.hi < x.hi for x, k in zip(box, krawczyk)):
        raise ValueError('Krawczyk inclusion failed')
    # Positive interval LDL pivots prove positive definiteness throughout the box.
    lower = [[Interval(0) for _ in range(n)] for _ in range(n)]
    pivots = []
    for i in range(n):
        lower[i][i] = Interval(1)
        for j in range(i):
            lower[i][j] = (full.h[i][j] * scales[i] * scales[j]
                           - sum(lower[i][k] * pivots[k] * lower[j][k] for k in range(j))) / pivots[j]
        pivot = full.h[i][i] * scales[i] ** 2 - sum(lower[i][k] ** 2 * pivots[k] for k in range(i))
        if pivot.lo <= 0:
            raise ValueError('positive definiteness not established')
        pivots.append(pivot)
    return {'classification': 'validated-local', 'point': list(map(str, point)),
            'parameter_box': [v.json() for v in box], 'objective_interval': full.v.json(),
            'gradient_interval': [v.json() for v in full.g],
            'krawczyk_image': [v.json() for v in krawczyk], 'scaled_contraction_upper': str(contraction),
            'scaled_ldl_pivots': [v.json() for v in pivots],
            'coordinate_scales': list(map(str, scales)),
            'local_identifiability': 'positive definite objective Hessian; local parameter branch',
            'parameter_symmetry': ('all permutations of the three amplitude/rate pairs'
                                   if case.get('id', '').startswith('Lanczos') else
                                   'exchange the two amplitude/rate pairs, retaining the intercept'
                                   if case.get('id') == 'MGH17' else 'none known for this development model'),
            'global_claim': False, 'start_basin_membership': ['pending', 'pending']}

#!/usr/bin/env python3
"""Independent AD and mutation tests for the least-squares pilot checker."""
import csv
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).with_name('check-least-squares.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from models import evaluate as second_order


class Arithmetic(unittest.TestCase):
    def test_first_order_matches_full_reference_ad(self):
        for case in pilot.datasets():
            point = [pilot.D(v) for v in case['certified_parameters']]
            fast, _, _ = pilot.evaluate(case, point)
            full = second_order(case, list(map(pilot.I, point)))
            for a, b in zip([fast.v] + fast.g, [full.v] + full.g):
                self.assertLessEqual(max(a.lo, b.lo), min(a.hi, b.hi), case['id'])
                self.assertLess((a - b).magnitude(), pilot.D('1e-70') * max(pilot.D(1), a.magnitude()))

    def test_symmetry_preserves_parameter_quality(self):
        saved = pilot.json.loads((pilot.ROOT / 'nist-reference-eligibility.json').read_text())
        for certificate in saved['references']:
            for branch in pilot.branches(certificate):
                point = [b.midpoint() for b in branch]
                self.assertLess(pilot.parameter_error(certificate, point), pilot.D('1e-49'))


class Roundoff(unittest.TestCase):
    def test_cancellation_uses_absolute_operation_terms(self):
        unit = 2 ** -53
        terms = [2 ** 53, 1.0, -(2 ** 53)]
        rounded = sum(terms)
        exact = float(sum(map(pilot.D.from_float, terms)))
        self.assertFalse(pilot.close(rounded, exact, unit))
        self.assertTrue(pilot.close(rounded, exact, unit, sum(map(abs, terms))))


class Evidence(unittest.TestCase):
    directory = None

    def test_analytic_records_and_negative_controls(self):
        if self.directory is None:
            self.skipTest('provide an analytic output directory')
        pilot.verify(self.directory)
        changes = [
            ('runs', lambda rows: rows[0].update(work='99999')),
            ('runs', lambda rows: rows[0].update(residual_evals='99999')),
            ('runs', lambda rows: rows[0].update(criteria='orthogonality')),
            ('native', lambda rows: next(r for r in rows if r['accepted'] == 'true').update(accepted='false')),
            ('native', lambda rows: next(r for r in rows if r['published'] == 'true').update(residual_work='1')),
            ('checks', lambda rows: next(r for r in rows if r['passed'] == 'true').update(value='99999')),
            ('leaves', lambda rows: rows[0].update(kind='Cost')),
        ]
        for name, change in changes:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for file in ('runs', 'publications', 'native', 'checks', 'leaves'):
                    shutil.copyfile(self.directory / f'{file}.csv', directory / f'{file}.csv')
                rows = pilot.load(directory, name)
                change(rows)
                with (directory / f'{name}.csv').open('w', newline='') as out:
                    writer = csv.DictWriter(out, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
                with self.assertRaises(ValueError):
                    pilot.verify(directory)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        Evidence.directory = Path(sys.argv.pop())
    unittest.main()

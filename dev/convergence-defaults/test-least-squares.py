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


class BoxQuality(unittest.TestCase):
    def test_box_rejects_infeasible_and_inaccurate_returns(self):
        run = dict(dataset='box_active', precision='f64', returned='true', outcome='converged')
        good = dict(point='0.5;-0.5', cost='0.25')
        pilot.box_quality(run, [good])
        for outcome in ('failed', 'stalled'):
            self.assertTrue(pilot.box_quality(dict(run, outcome=outcome), [good])['quality_passed'])
        for bad in (dict(point='0.6;-0.5', cost='0.205'),
                    dict(point='0.1;-0.1', cost='0.81'),
                    dict(point='0.5;-0.5', cost='0.5')):
            with self.assertRaises(ValueError):
                pilot.box_quality(run, [bad])

    def test_stationary_stop_fails_minimum_quality(self):
        run = dict(dataset='box_stationary', precision='f64', returned='true', outcome='converged', criteria='scaled_gradient')
        result = pilot.box_quality(run, [dict(point='0;-1', cost='2')])
        self.assertFalse(result['quality_passed'])
        self.assertEqual(result['objective_gap'], 1.5)


class RobustReference(unittest.TestCase):
    def test_qr_prediction_allowance_includes_unsummed_gradient_terms(self):
        unit = 2**-53
        h = 1.3314095455969066e-7
        gradient = -2.663114961887203e-7
        independent = -gradient*h - h*h
        observed = 1.773045301903516e-14
        summed = abs(gradient*h) + h*h
        self.assertFalse(pilot.close(observed, independent, unit, summed))
        self.assertTrue(pilot.close(observed, independent, unit, summed + abs(h)*2))

    def test_subnormal_product_rounding_allowance(self):
        observed, independent = 0., 1.5468596943914052e-45
        self.assertFalse(pilot.close(observed, independent, 2**-24, 5.1609820902990446e-42))
        self.assertLess(abs(observed-independent), 4 * 2**-149)

    def test_known_minima_and_bound_kkt(self):
        for name, (_, _, _, reference, bounds) in pilot.ROBUST_CASES.items():
            run = dict(dataset=name, precision='f64')
            cost, gradient, _ = pilot.robust_values(run, [float(reference)])
            if bounds:
                self.assertLess(gradient, 0)
            else:
                self.assertEqual(gradient, 0)
            for delta in (-.01, .01):
                x = float(reference) + delta
                if bounds and not float(bounds[0]) <= x <= float(bounds[1]):
                    continue
                trial_cost, trial_gradient, _ = pilot.robust_values(run, [x])
                self.assertGreater(trial_cost, cost, name)
                if bounds is None:
                    self.assertGreater(trial_gradient * delta, 0, name)

    def test_gradient_matches_independent_objective_difference(self):
        for name in pilot.ROBUST_CASES:
            run = dict(dataset=name, precision='f64')
            x = .1
            h = 1e-5
            _, gradient, _ = pilot.robust_values(run, [x])
            plus = pilot.robust_values(run, [x + h])[0]
            minus = pilot.robust_values(run, [x - h])[0]
            self.assertAlmostEqual(gradient, (plus-minus)/(2*h), delta=1e-8)

    def test_curvature_safeguards_are_precision_specific(self):
        for precision, epsilon in [('f64', 2**-52), ('f32', 2**-23)]:
            _, _, rows = pilot.robust_values(dict(dataset='robust_cauchy', precision=precision), [-1.])
            self.assertAlmostEqual(rows[0][0]**2, epsilon, delta=epsilon*1e-14)
            _, _, rows = pilot.robust_values(dict(dataset='robust_huber_outlier', precision=precision), [.5])
            self.assertAlmostEqual(rows[2][0]**2, epsilon, delta=epsilon*1e-14)

    def test_quality_keeps_poor_and_failed_returns(self):
        run = dict(dataset='robust_huber_outlier', precision='f64', returned='true', policy='robust_default', outcome='failed')
        result = pilot.robust_quality(run, [dict(point='.5', cost='7.25')])
        self.assertTrue(result['quality_passed'])
        result = pilot.robust_quality(dict(run, outcome='converged'), [dict(point='0', cost='7.5')])
        self.assertFalse(result['quality_passed'])
        error_run = dict(run, returned='false', outcome='callback_error')
        self.assertIsNone(pilot.robust_quality(error_run, [dict(point='.5', cost='7.25')]))
        self.assertTrue(pilot.robust_quality(error_run, [dict(point='.5', cost='7.25')], last_publication=True)['quality_passed'])
        with self.assertRaises(ValueError):
            pilot.robust_quality(run, [dict(point='.5', cost='28.375')])


class Evidence(unittest.TestCase):
    directory = None

    def test_analytic_records_and_negative_controls(self):
        if self.directory is None:
            self.skipTest('provide an analytic output directory')
        pilot.verify(self.directory)
        changes = [
            ('runs', lambda rows: rows[0].update(work='99999')),
            ('runs', lambda rows: rows[0].update(residual_evals='99999')),
            ('runs', lambda rows: next(r for r in rows if r['outcome'] == 'converged').update(criteria='orthogonality')),
            ('native', lambda rows: next(r for r in rows if r['accepted'] == 'true').update(accepted='false')),
            ('native', lambda rows: next(r for r in rows if r['published'] == 'true').update(residual_work='1')),
            ('checks', lambda rows: next(r for r in rows if r['passed'] == 'true').update(value='99999')),
            ('leaves', lambda rows: rows[0].update(kind='Cost')),
        ]
        if pilot.load(self.directory, 'runs')[0]['dataset'] in pilot.ROBUST_CASES:
            changes.extend([
                ('publications', lambda rows: next(r for r in rows if r['cost']).update(cost='99999')),
                ('native', lambda rows: next(r for r in rows if r['gradient']).update(gradient='99999')),
                ('native', lambda rows: next(r for r in rows if r['trial_cost']).update(trial_cost='99999')),
                ('native', lambda rows: next(r for r in rows if r['trial_cost'] and r['predicted'] and pilot.math.isfinite(float(r['predicted']))).update(predicted='99999')),
                ('native', lambda rows: next(r for r in rows if r['predicted'] and not pilot.math.isfinite(float(r['predicted']))).update(accepted='true')),
            ])
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

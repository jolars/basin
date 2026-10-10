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
    def test_decimal_arctangent(self):
        for value in ('0', '.001', '.1', '1', '4', '1e20'):
            self.assertAlmostEqual(float(pilot.decimal_atan(pilot.D(value))), pilot.math.atan(float(value)), delta=3e-16)
        quarter_pi = pilot.D('0.785398163397448309615660845819875721049292349843776455243736148076954101571552249657008706335529266995537')
        self.assertLess(abs(pilot.decimal_atan(pilot.D(1)) - quarter_pi), pilot.D('1e-98'))

    def test_extended_minima_and_identifiable_directions(self):
        for name, (loss, model, _, reference, _) in pilot.ROBUST_EXTENDED_CASES.items():
            run = dict(dataset=name, precision='f64')
            point = list(map(float, reference))
            cost, gradient, _ = pilot.robust_values(run, point)
            self.assertEqual(gradient, [0.] * len(point), name)
            if model == 'rank4':
                expected = {'huber': 3., 'soft_l1': 2 * (5 ** .5 - 1), 'cauchy': pilot.math.log(5), 'arctan': pilot.math.atan(4)}[loss]
                self.assertAlmostEqual(cost, expected, delta=3e-16)
                flat = [2., -1., 3., -4.]
                shifted_cost, shifted_gradient, _ = pilot.robust_values(run, flat)
                self.assertEqual(shifted_cost, cost)
                self.assertEqual(shifted_gradient, gradient)
                result = pilot.robust_quality(dict(run, returned='true', policy='robust_default'),
                                              [dict(point='2;-1;3;-4', cost=str(cost))])
                self.assertEqual(result['parameter_error'], 0.)
                self.assertTrue(result['quality_passed'])
            else:
                self.assertEqual(cost, 0.)
            for k in range(len(point)):
                for delta in (-.01, .01):
                    perturbed = point.copy()
                    perturbed[k] += delta
                    trial_cost, trial_gradient, _ = pilot.robust_values(run, perturbed)
                    self.assertGreater(trial_cost, cost, name)
                    self.assertGreater(trial_gradient[k] * delta, 0, name)

    def test_extended_gradient_matches_objective_difference(self):
        for name in pilot.ROBUST_EXTENDED_CASES:
            run = dict(dataset=name, precision='f64')
            point = [-.75] if name == 'robust_arctan' else [-.75, .25, .75, -.25]
            _, gradient, rows = pilot.robust_values(run, point)
            self.assertTrue(all(len(row) == len(point) for row in rows))
            for k, derivative in enumerate(gradient):
                h = 1e-5
                plus, minus = point.copy(), point.copy()
                plus[k] += h
                minus[k] -= h
                difference = (pilot.robust_values(run, plus)[0] - pilot.robust_values(run, minus)[0]) / (2 * h)
                self.assertAlmostEqual(derivative, difference, delta=2e-8, msg=name)

    def test_arctangent_negative_curvature_and_rank_quality(self):
        for precision, epsilon in [('f64', 2**-52), ('f32', 2**-23)]:
            _, _, rows = pilot.robust_values(dict(dataset='robust_arctan', precision=precision), [-1.])
            self.assertAlmostEqual(rows[0][0]**2, epsilon, delta=epsilon*1e-14)
        run = dict(dataset='robust_cauchy_rank4', precision='f64', returned='true', policy='robust_default')
        point = [2., -1., 3., -3.75]
        cost = pilot.robust_values(run, point)[0]
        result = pilot.robust_quality(run, [dict(point='2;-1;3;-3.75', cost=str(cost))])
        self.assertEqual(result['parameter_error'], .125)
        self.assertFalse(result['quality_passed'])
        with self.assertRaises(ValueError):
            pilot.robust_quality(run, [dict(point='0;0', cost=str(cost))])

    def test_qr_projection_roundoff_uses_clipped_rhs(self):
        run = dict(dataset='robust_huber_linear4', precision='f64')
        point = [1.6494017565635213, -1.6494017565635208, -.4999999999999982, .4999999999999982]
        step = [-4.35064215424544e-8, 4.3506421617627035e-8, 5.2754183593327415e-8, -5.275418338529763e-8]
        observed, independent = 7.090717309744541e-7, 7.090717312665235e-7
        unit = 2 ** -53
        self.assertFalse(pilot.close(observed, independent, unit, 1.4181434905879226e-6))
        self.assertTrue(pilot.close(observed, independent, unit, pilot.qr_projection_scale(run, point, step)))
        self.assertFalse(pilot.close(99999., independent, unit, pilot.qr_projection_scale(run, point, step)))

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
                self.assertLess(gradient[0], 0)
            else:
                self.assertEqual(gradient, [0])
            for delta in (-.01, .01):
                x = float(reference) + delta
                if bounds and not float(bounds[0]) <= x <= float(bounds[1]):
                    continue
                trial_cost, trial_gradient, _ = pilot.robust_values(run, [x])
                self.assertGreater(trial_cost, cost, name)
                if bounds is None:
                    self.assertGreater(trial_gradient[0] * delta, 0, name)

    def test_gradient_matches_independent_objective_difference(self):
        for name in pilot.ROBUST_CASES:
            run = dict(dataset=name, precision='f64')
            x = .1
            h = 1e-5
            _, gradient, _ = pilot.robust_values(run, [x])
            plus = pilot.robust_values(run, [x + h])[0]
            minus = pilot.robust_values(run, [x - h])[0]
            self.assertAlmostEqual(gradient[0], (plus-minus)/(2*h), delta=1e-8)

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
        if pilot.load(self.directory, 'runs')[0]['dataset'] in pilot.ALL_ROBUST_CASES:
            changes.extend([
                ('publications', lambda rows: next(r for r in rows if r['cost']).update(cost='99999')),
                ('native', lambda rows: next(r for r in rows if r['gradient']).update(gradient='99999')),
                ('native', lambda rows: next(r for r in rows if r['trial_cost']).update(trial_cost='99999')),
                ('native', lambda rows: next(r for r in rows if r['trial_cost'] and r['predicted'] and pilot.math.isfinite(float(r['predicted']))).update(predicted='99999')),
            ])
            if any(r['predicted'] and not pilot.math.isfinite(float(r['predicted'])) for r in pilot.load(self.directory, 'native')):
                changes.append(('native', lambda rows: next(r for r in rows if r['predicted'] and not pilot.math.isfinite(float(r['predicted']))).update(accepted='true')))
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

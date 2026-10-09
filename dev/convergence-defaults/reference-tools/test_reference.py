"""Analytic checks and negative controls for the certificate machinery."""

import unittest
from decimal import Decimal as D, getcontext
from fractions import Fraction
from unittest.mock import patch

from interval import Interval as I, Jet, PRECISION
from models import certify, datasets, evaluate
from preflight import failed_clauses, neighbor

getcontext().prec = PRECISION


class Certificates(unittest.TestCase):
    def test_native_neighbors_and_tenfold_uncertainty_gate(self):
        self.assertLess(neighbor(1.0, 'f32', -1), 1)
        self.assertGreater(neighbor(1.0, 'f32', 1), 1)
        self.assertLess(neighbor(-1.0, 'f32', -1), -1)
        self.assertGreater(neighbor(-1.0, 'f32', 1), -1)
        measure = dict(objective_upper='0', stationarity_upper='0', parameter_upper='0',
                       objective_uncertainty='1e-5', stationarity_uncertainty='0', parameter_uncertainty='0')
        self.assertEqual(failed_clauses(measure, D('1e-3')), [])
        self.assertEqual(failed_clauses(measure, D('1e-4')), ['objective uncertainty margin'])

    def test_directed_arithmetic_encloses_exact_rationals(self):
        for a in ('-1.23456789', '0', '0.000001', '987.654321'):
            for b in ('-9', '0.0003', '7'):
                left, right = I(a), I(b)
                for actual, exact in ((left + right, Fraction(a) + Fraction(b)),
                                      (left - right, Fraction(a) - Fraction(b)),
                                      (left * right, Fraction(a) * Fraction(b)),
                                      (left / right, Fraction(a) / Fraction(b))):
                    self.assertLessEqual(Fraction(actual.lo), exact)
                    self.assertGreaterEqual(Fraction(actual.hi), exact)
        self.assertEqual((I(-2, 3) ** 2).lo, 0)
        self.assertEqual((I(-2, 3) ** 2).hi, 9)
        with self.assertRaises(ValueError):
            I(1) / I(-1, 1)

    def test_elementary_enclosures(self):
        self.assertLess((I(0).exp() - 1).lo, 0)
        self.assertGreater((I(0).exp() - 1).hi, 0)
        self.assertLessEqual((I(2).ln().exp() - 2).lo, 0)
        self.assertGreaterEqual((I(2).ln().exp() - 2).hi, 0)
        self.assertLessEqual((I(2).sqrt() ** 2).lo, 2)
        self.assertGreaterEqual((I(2).sqrt() ** 2).hi, 2)

    def test_jet_chain_rule_analytic_hessian(self):
        x, y = [Jet.variable(D(v), 2, i) for i, v in enumerate(('2', '3'))]
        result = x ** 2 * y + (x * y).exp()
        exp = D(6).exp()
        self.assertLess(abs(result.g[0] - (12 + 3 * exp)), D('1e-94'))
        self.assertLess(abs(result.h[0][1] - (4 + 7 * exp)), D('1e-94'))
        self.assertLess(abs(result.h[1][1] - 4 * exp), D('1e-94'))

    def test_krawczyk_and_ldl_distinguish_minimum_and_maximum(self):
        case = dict(certified_parameters=['1', '2'], start1=['0', '0'], start2=['3', '4'])

        def quadratic(case, point):
            x, y = [Jet.variable(value, 2, j) for j, value in enumerate(point)]
            return (x - 1) ** 2 + 2 * (y - 2) ** 2

        with patch('models.evaluate', quadratic):
            certificate = certify(case, [D(1), D(2)])
            self.assertEqual(certificate['classification'], 'validated-local')
            with self.assertRaisesRegex(ValueError, 'Krawczyk'):
                certify(case, [D('1.01'), D(2)])
        with patch('models.evaluate', lambda c, p: -quadratic(c, p)):
            with self.assertRaisesRegex(ValueError, 'positive definiteness'):
                certify(case, [D(1), D(2)])

    def test_development_partition_and_hessian_numerical_check(self):
        cases = datasets()
        self.assertEqual(len(cases), 15)
        self.assertTrue(all(c['partition'] == 'development' for c in cases))
        for case in cases:
            point = list(map(D, case['certified_parameters']))
            exact = evaluate(case, point)
            for j in range(len(point)):
                h = max(abs(point[j]), D('1e-12')) * D('1e-25')
                plus, minus = point.copy(), point.copy()
                plus[j] += h
                minus[j] -= h
                pg, mg = evaluate(case, plus).g, evaluate(case, minus).g
                for i in range(len(point)):
                    difference = (pg[i] - mg[i]) / (2 * h)
                    self.assertLess(abs(difference - exact.h[i][j]),
                                    D('1e-40') * max(D(1), abs(exact.h[i][j])),
                                    (case['id'], i, j))


if __name__ == '__main__':
    unittest.main()

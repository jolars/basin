#!/usr/bin/env python3
"""Negative controls for paired stopping evidence and fixed configurations."""
import csv
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('stopping', Path(__file__).with_name('check-robust-stopping.py'))
stopping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stopping)


class Composition(unittest.TestCase):
    def test_good_agreement_and_acceptance_do_not_require_stationarity(self):
        trial = dict(sequence='2', trial='true', base_cost='.5', trial_cost='.4999999999',
                     actual='1e-10', predicted='1e-10', ratio='1', damping='1e10',
                     step='1e-10', gradient='1', diagonal='1', accepted='true', published='true')
        checks = [dict(sequence='2', name=name, passed='true') for name in ('model_reduction', 'trial_step')]
        result = stopping.terminal_trial([trial], checks)
        self.assertTrue(result['model_and_step'])
        self.assertTrue(result['accepted_model_and_step'])
        self.assertTrue(result['ratio_above_quarter_model_and_step'])
        self.assertEqual(result['base_gradient'], [1.])
        checks[-1]['sequence'] = '1'
        self.assertFalse(stopping.terminal_trial([trial], checks)['model_and_step'])

    def test_prefix_rejects_changed_trajectory(self):
        leaf = dict(kind='Residual', point='0', outcome='Completed')
        self.assertEqual(stopping.compare_prefix([leaf], [leaf, leaf]), 1)
        for key in leaf:
            with self.subTest(key=key), self.assertRaises(ValueError):
                stopping.compare_prefix([dict(leaf, **{key: 'altered'})], [leaf])

    def test_quality_is_common_to_every_policy(self):
        for policy in stopping.POLICIES:
            run = dict(dataset='robust_cauchy', precision='f64', returned='true', outcome='converged', policy=policy)
            x = 1.00005
            cost = stopping.pilot.robust_values(run, [x])[0]
            result = stopping.quality(run, [dict(point=str(x), cost=str(cost))])
            self.assertEqual(result['quality_limit'], 1e-6)
            self.assertFalse(result['quality_passed'])


class Evidence(unittest.TestCase):
    directory = None

    def test_records_and_mutation_controls(self):
        if self.directory is None:
            self.skipTest('provide a robust stopping output directory')
        stopping.verify(self.directory)
        mutations = [
            ('runs', lambda rows: rows.pop()),
            ('runs', lambda rows: rows[0].update(cap='4001')),
            ('runs', lambda rows: rows[0].update(policy='robust_model_probe')),
            ('checks', lambda rows: next(r for r in rows if not r['tolerance']).update(tolerance='0', passed='false')),
            ('checks', lambda rows: next(r for r in rows if r['name'] == 'absolute_gradient').update(tolerance='1', bound='1')),
            ('checks', lambda rows: next(r for r in rows if r['evidence'] == 'model_reduction' and r['passed'] == 'true').update(actual='0')),
        ]
        for name, mutation in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for file in ('runs', 'publications', 'leaves', 'native', 'checks'):
                    shutil.copyfile(self.directory / f'{file}.csv', directory / f'{file}.csv')
                rows = stopping.pilot.load(directory, name)
                mutation(rows)
                with (directory / f'{name}.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    stopping.verify(directory)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        Evidence.directory = Path(sys.argv.pop())
    unittest.main()

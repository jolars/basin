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

    def test_recovery_checks_the_first_changed_damping(self):
        accepted = dict(trial='true', accepted='true', diagonal='1e-7;1e-7')
        model = dict(trial='false', accepted='', diagonal='1;1')
        trial = dict(sequence='4', trial='true', accepted='false', diagonal='1;1',
                     gradient='1;1', base_cost='2', damping='1000000', step='1e-7;1e-7')
        after = dict(trial, damping='0.1', step='0.1;0.1')
        before = [accepted, model, trial]
        self.assertAlmostEqual(stopping.recovery_transition(before, [accepted, model, after], 'f32')['diagonal_ratio'], 1e-7)
        for change in (dict(damping='1'), dict(gradient='0;0'), dict(diagonal='2;2')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                stopping.recovery_transition(before, [accepted, model, dict(after, **change)], 'f32')
        with self.assertRaises(ValueError):
            stopping.recovery_transition(before, before, 'f32')
        accepted = dict(accepted, diagonal='1e-7;2e-7')
        before = [accepted, model, trial]
        after = dict(after, damping='0.2')
        self.assertAlmostEqual(stopping.recovery_transition(before, [accepted, model, after], 'f32')['diagonal_ratio'], 2e-7)
        with self.assertRaises(ValueError):
            stopping.recovery_transition(before, [accepted, model, dict(after, damping='0.1')], 'f32')

    def test_reproduction_checks_every_csv_except_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ('before', 'after')]
            for path in paths:
                path.mkdir()
                for name in ('runs', 'leaves', 'publications', 'native', 'checks'):
                    with (path / f'{name}.csv').open('w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=['id', 'point', 'elapsed_seconds'])
                        writer.writeheader()
                        writer.writerows(dict(id=str(k), point='1', elapsed_seconds=str(paths.index(path)))
                                         for k in range(336 if name == 'runs' else 1))
            self.assertEqual(stopping.verify_reproduction(paths[1], paths[0])['runs'], 336)
            with (paths[1] / 'leaves.csv').open('w') as file:
                file.write('id,point,elapsed_seconds\n0,2,1\n')
            with self.assertRaises(ValueError):
                stopping.verify_reproduction(paths[1], paths[0])


class Evidence(unittest.TestCase):
    directory = None
    extended = False

    def test_records_and_mutation_controls(self):
        if self.directory is None:
            self.skipTest('provide a robust stopping output directory')
        stopping.verify(self.directory, self.extended)
        mutations = [
            ('runs', lambda rows: rows.pop()),
            ('runs', lambda rows: rows[0].update(cap='4001')),
            ('runs', lambda rows: rows[0].update(policy='robust_model_probe')),
            ('checks', lambda rows: next(r for r in rows if not r['tolerance']).update(tolerance='0', passed='false')),
            ('checks', lambda rows: next(r for r in rows if r['name'] == 'absolute_gradient').update(tolerance='1', bound='1')),
            ('checks', lambda rows: next(r for r in rows if r['evidence'] == 'model_reduction' and r['passed'] == 'true').update(actual='0')),
        ]
        if self.extended:
            mutations.extend([
                ('publications', lambda rows: next(r for r in rows if '_rank4-' in r['id']).update(point='0;0;0;0')),
                ('native', lambda rows: next(r for r in rows if '_linear4-' in r['id'] and r['gradient']).update(gradient='0')),
                ('native', lambda rows: next(r for r in rows if '_rank4-' in r['id'] and r['diagonal']).update(diagonal='0;0;0;0')),
            ])
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
                    stopping.verify(directory, self.extended)


if __name__ == '__main__':
    if '--extended' in sys.argv:
        sys.argv.remove('--extended')
        Evidence.extended = True
    if len(sys.argv) == 2:
        Evidence.directory = Path(sys.argv.pop())
    unittest.main()

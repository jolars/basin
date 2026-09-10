"""Regression checks for the LP comparison's pre-timing numerical gate."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import reproduce_cobyla_gap as gap


class LpVerificationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "inputs": 1,
            "mismatches": 0,
            "max_relative_step_error": 1e-16,
            "max_scaled_metric_error": 1e-16,
            "basin_step_bits": [[0, 1]],
            "reference_step_bits": [[0, 2]],
        }

    def test_reference_roundoff_does_not_require_identical_bits(self):
        gap.check_lp_result(self.baseline, self.baseline)

    def test_changed_basin_step_is_rejected_even_if_reference_tolerance_passes(self):
        candidate = dict(self.baseline, basin_step_bits=[[0, 2]])
        with self.assertRaisesRegex(ValueError, "Basin LP steps changed"):
            gap.check_lp_result(candidate, self.baseline)

    def test_invalid_reference_metrics_and_missing_steps_are_rejected(self):
        for changes in [
            {"mismatches": 1},
            {"max_relative_step_error": 2e-6},
            {"max_scaled_metric_error": 2e-7},
            {"max_scaled_metric_error": float("nan")},
            {"basin_step_bits": []},
            {"reference_step_bits": []},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gap.check_lp_result(dict(self.baseline, **changes), self.baseline)

    def run_comparison(self, changed=False):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            builds = {key: root / key for key in ["before", "after"]}
            trace = builds["before"] / "traces" / "camel.jsonl"
            trace.parent.mkdir(parents=True)
            trace.write_text("baseline inputs\n")
            args = SimpleNamespace(
                command="compare-lp",
                cases=["camel"],
                cpu=2,
                rounds=2,
                output=root / "samples.csv",
            )

            def probe(binary, mode, path, repeats=1, cpu=None):
                self.assertEqual(path, trace)
                result = dict(self.baseline, ns=10.0, mode=mode)
                if changed and binary == builds["after"] / "probe":
                    result["basin_step_bits"] = [[0, 2]]
                return result

            with (
                patch.object(gap, "comparison_builds", return_value=(builds, {})),
                patch.object(gap.prima, "probe", side_effect=probe) as calls,
            ):
                if changed:
                    with self.assertRaisesRegex(ValueError, "Basin LP steps changed"):
                        gap.compare_lp(args)
                    self.assertFalse(args.output.exists())
                    self.assertTrue(
                        all(len(call.args) < 4 for call in calls.call_args_list)
                    )
                else:
                    gap.compare_lp(args)
                    self.assertEqual(len(args.output.read_text().splitlines()), 9)
                    self.assertTrue(args.output.with_suffix(".metadata.json").exists())
                    timed = [
                        call for call in calls.call_args_list if len(call.args) >= 4
                    ]
                    for offset in [0, 4]:
                        basin = [
                            i
                            for i, call in enumerate(timed[offset : offset + 4])
                            if call.args[1] == "lp-basin"
                        ]
                        self.assertEqual(max(basin) - min(basin), len(basin) - 1)

    def test_every_contestant_replays_the_baseline_trace(self):
        self.run_comparison()

    def test_mismatch_prevents_timed_batches(self):
        self.run_comparison(changed=True)


if __name__ == "__main__":
    unittest.main()

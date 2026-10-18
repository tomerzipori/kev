"""Edge tests for the draft Phase 5 exact group-bound helpers."""

import json
import math
import runpy
import unittest
from pathlib import Path

import torch


_MODULE = runpy.run_path(
    str(Path(__file__).with_name("phase5-readiness-statistics.py"))
)
order_statistic_rank = _MODULE["order_statistic_rank"]
threshold_from_group_maxima = _MODULE["threshold_from_group_maxima"]
exact_binomial_upper = _MODULE["exact_binomial_upper"]
conservative_row_error_bounds = _MODULE["conservative_row_error_bounds"]
required_zero_event_groups = _MODULE["required_zero_event_groups"]
canonical_probability_decision = _MODULE["canonical_probability_decision"]


class ExactGroupBoundTests(unittest.TestCase):
    def test_order_statistic_minimum_and_four_model_multiplicity(self):
        rank, bound = order_statistic_rank(
            299,
            row_fpr_target=0.01,
            max_rows_per_group=1,
            familywise_delta=0.05,
            candidate_models=1,
        )
        self.assertEqual(rank, 299)
        self.assertLessEqual(bound, 0.05)
        with self.assertRaisesRegex(ValueError, "too few"):
            order_statistic_rank(
                298,
                row_fpr_target=0.01,
                max_rows_per_group=1,
                familywise_delta=0.05,
                candidate_models=1,
            )

        rank, bound = order_statistic_rank(
            437,
            row_fpr_target=0.01,
            max_rows_per_group=1,
            familywise_delta=0.05,
            candidate_models=4,
        )
        self.assertEqual(rank, 437)
        self.assertLessEqual(bound, 0.05 / 4)

    def test_cluster_cap_requires_more_independent_groups(self):
        self.assertEqual(
            required_zero_event_groups(
                row_error_target=0.01,
                max_rows_per_group=5,
                delta=0.05,
                candidate_models=4,
            ),
            2189,
        )
        rank, bound = order_statistic_rank(
            2189,
            row_fpr_target=0.01,
            max_rows_per_group=5,
            familywise_delta=0.05,
            candidate_models=4,
        )
        self.assertEqual(rank, 2189)
        self.assertLessEqual(bound, 0.05 / 4)

    def test_ge_runtime_excludes_cutoff_ties(self):
        result = threshold_from_group_maxima(
            [0.2, 0.2, 0.7],
            ["a", "b", "c"],
            row_fpr_target=0.9,
            max_rows_per_group=1,
            familywise_delta=0.9,
            candidate_models=1,
            runtime_uses_greater_equal=True,
            design_approved=True,
        )
        threshold = result["effective_threshold"]
        self.assertGreater(threshold, result["cutoff"])
        self.assertFalse(0.2 >= threshold)

    def test_canonical_comparator_preserves_torch_item_values_through_json(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                probability = float(torch.tensor(0.1, dtype=dtype, device="cpu").item())
                tied_threshold = json.loads(json.dumps(probability, allow_nan=False))
                next_threshold = json.loads(
                    json.dumps(math.nextafter(probability, math.inf), allow_nan=False)
                )
                self.assertEqual(tied_threshold, probability)
                self.assertTrue(canonical_probability_decision(probability, tied_threshold))
                self.assertFalse(canonical_probability_decision(probability, next_threshold))
        self.assertNotEqual(float(torch.tensor(0.1, dtype=torch.float32).item()), 0.1)

    def test_zero_event_bound_is_positive_and_row_bound_uses_cap(self):
        group_upper = exact_binomial_upper(0, 400, 0.05)
        self.assertGreater(group_upper, 0.0)
        self.assertAlmostEqual(exact_binomial_upper(1, 10, 0.05), 0.3941633024, places=9)
        result = conservative_row_error_bounds(
            group_ids=[f"g{i}" for i in range(400)],
            groups_with_any_error=0,
            max_rows_per_group=5,
            delta=0.05,
            design_approved=True,
        )
        self.assertAlmostEqual(result["row_error_upper"], min(1.0, 5 * group_upper))
        self.assertGreater(result["row_error_upper"], 0.0)
        self.assertAlmostEqual(
            result["row_recall_lower_if_error_means_miss"],
            1 - result["row_error_upper"],
        )

    def test_cp_inversion_is_stable_for_large_n_and_boundary_counts(self):
        # Reference values from scipy.stats.beta.ppf(0.95, x + 1, n - x).
        self.assertAlmostEqual(
            exact_binomial_upper(1, 2000, 0.05),
            0.002369713367656844,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            exact_binomial_upper(1999, 2000, 0.05),
            0.9999743536816786,
            delta=1e-10,
        )
        self.assertEqual(exact_binomial_upper(2000, 2000, 0.05), 1.0)

    def test_unapproved_or_over_cap_design_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not approved"):
            threshold_from_group_maxima(
                [0.1], ["g"], row_fpr_target=0.01, max_rows_per_group=1,
                familywise_delta=0.05, candidate_models=1,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            conservative_row_error_bounds(
                group_ids=["g", "g"], groups_with_any_error=0,
                max_rows_per_group=1, delta=0.05, design_approved=True,
            )
        with self.assertRaisesRegex(ValueError, "too few"):
            order_statistic_rank(
                400, row_fpr_target=0.01, max_rows_per_group=5,
                familywise_delta=0.05, candidate_models=4,
            )


if __name__ == "__main__":
    unittest.main()

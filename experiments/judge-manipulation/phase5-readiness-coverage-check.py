"""Seeded CPU stress check for clustered rare-FPR threshold selection.

This is a design diagnostic, not a test suite or a proof of nominal coverage.
It uses synthetic logistic-normal group scores and a quantile-reduced threshold
grid to compare pointwise and max-deviation percentile bootstrap selection.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import expit


TARGET = 0.01
REPLICATES = 10000
SEED = 20261018
SCENARIOS = (
    {"name": "iid_equal_200x5", "groups": 200, "size": "equal5", "sigma": 0.0},
    {"name": "correlated_equal_200x5", "groups": 200, "size": "equal5", "sigma": 1.0},
    {"name": "correlated_unequal_cap5_400g", "groups": 400, "size": "uniform1to5", "sigma": 1.0},
    {"name": "correlated_unequal_cap15_200g", "groups": 200, "size": "zipf_cap15", "sigma": 1.5},
    {"name": "correlated_few_unequal_60g", "groups": 60, "size": "zipf_cap30", "sigma": 1.5},
)


def group_sizes(rng: np.random.Generator, groups: int, kind: str) -> np.ndarray:
    if kind == "equal5":
        return np.full(groups, 5, dtype=np.int64)
    if kind == "uniform1to5":
        return rng.integers(1, 6, size=groups, dtype=np.int64)
    cap = 15 if kind == "zipf_cap15" else 30
    return np.clip(rng.zipf(1.7, size=groups), 1, cap).astype(np.int64)


def marginal_tail(score_threshold: float, sigma: float) -> float:
    if sigma == 0:
        return float(expit(-score_threshold))
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    latent = math.sqrt(2.0) * sigma * nodes
    return float(np.dot(weights, expit(latent - score_threshold)) / math.sqrt(math.pi))


def cutoff_for_rate(rate: float, sigma: float) -> float:
    return brentq(lambda cutoff: marginal_tail(cutoff, sigma) - rate, -20.0, 20.0)


def wilson(successes: int, trials: int) -> list[float] | None:
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def simulate_scenario(config: dict, seed: int, outer: int, bootstrap: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = config["groups"]
    sigma = config["sigma"]
    quantiles = np.linspace(0.90, 0.999, 20)
    method_stats = {
        "pointwise": {"selected": 0, "false_selected": 0, "unsupported": 0},
        "max_deviation": {"selected": 0, "false_selected": 0, "unsupported": 0},
    }
    all_grid_covered = 0
    near_one_pct_covered = 0
    selected_rates = {name: [] for name in method_stats}
    row_counts = []
    cluster_cv = []
    observed_min_size = None
    observed_max_size = 0

    for _ in range(outer):
        sizes = group_sizes(rng, groups, config["size"])
        row_counts.append(int(sizes.sum()))
        cluster_cv.append(float(sizes.std() / sizes.mean()) if sizes.mean() else 0.0)
        observed_min_size = int(sizes.min()) if observed_min_size is None else min(observed_min_size, int(sizes.min()))
        observed_max_size = max(observed_max_size, int(sizes.max()))
        latent = rng.normal(0.0, sigma, size=groups)
        group_scores = [latent[g] + rng.logistic(size=int(sizes[g])) for g in range(groups)]
        all_scores = np.concatenate(group_scores)
        thresholds = np.unique(np.quantile(all_scores, quantiles, method="linear"))
        group_fp = np.asarray(
            [[np.count_nonzero(scores >= threshold) for threshold in thresholds] for scores in group_scores],
            dtype=np.int64,
        )
        observed = group_fp.sum(axis=0) / sizes.sum()
        true = np.asarray([marginal_tail(float(t), sigma) for t in thresholds])
        multinomial = rng.multinomial(groups, np.full(groups, 1 / groups), size=bootstrap)
        boot_den = multinomial @ sizes
        boot_fp = multinomial @ group_fp
        boot_rates = boot_fp / boot_den[:, None]
        observed_minus_bootstrap = observed[None, :] - boot_rates
        simultaneous_offset = float(np.quantile(
            observed_minus_bootstrap.max(axis=1), 0.95, method="higher"
        ))
        upper_pointwise = np.quantile(boot_rates, 0.95, axis=0, method="higher")
        upper_simultaneous = observed + simultaneous_offset
        all_grid_covered += int(np.all(upper_simultaneous >= true))
        near_index = int(np.argmin(np.abs(true - TARGET)))
        near_one_pct_covered += int(upper_pointwise[near_index] >= true[near_index])

        for name, upper in (("pointwise", upper_pointwise), ("max_deviation", upper_simultaneous)):
            # Boundary samples with no false positives contain no bootstrap
            # information about the unseen tail and cannot qualify a threshold.
            unsupported = (observed == 0) & np.all(boot_rates == 0, axis=0)
            method_stats[name]["unsupported"] += int(np.any(unsupported))
            eligible = np.flatnonzero((upper <= TARGET) & ~unsupported)
            if not len(eligible):
                continue
            selected = int(eligible[0])
            method_stats[name]["selected"] += 1
            selected_rates[name].append(float(true[selected]))
            false = bool(true[selected] > TARGET)
            method_stats[name]["false_selected"] += int(false)

    summary = {}
    for name, values in method_stats.items():
        selected = values["selected"]
        summary[name] = {
            "selection_rate": selected / outer,
            "false_selection_rate_unconditional": values["false_selected"] / outer,
            "false_selection_rate_conditional_on_selection": (
                values["false_selected"] / selected if selected else None
            ),
            "false_selection_wilson_95": wilson(values["false_selected"], outer),
            "selection_count": selected,
            "false_selection_count": values["false_selected"],
            "boundary_unsupported_count": values["unsupported"],
            "mean_true_fpr_when_selected": (
                float(np.mean(selected_rates[name])) if selected_rates[name] else None
            ),
        }
    near_grid_count = near_one_pct_covered
    band_grid_count = all_grid_covered
    return {
        "scenario": config["name"],
        "independent_groups": groups,
        "cluster_size_rule": config["size"],
        "rows_mean": float(np.mean(row_counts)),
        "cluster_size_min": observed_min_size,
        "cluster_size_max": observed_max_size,
        "cluster_size_cv_mean": float(np.mean(cluster_cv)),
        "random_intercept_sd": sigma,
        "bootstrap": {"replicates": bootstrap, "seed": seed},
        "outer_datasets": outer,
        "candidate_thresholds_per_dataset": len(thresholds),
        "adaptive_nearest_1pct_pointwise_upper_coverage": near_grid_count / outer,
        "adaptive_nearest_1pct_pointwise_coverage_mc_wilson_95": wilson(near_grid_count, outer),
        "max_deviation_simultaneous_grid_coverage": all_grid_covered / outer,
        "max_deviation_grid_coverage_mc_wilson_95": wilson(band_grid_count, outer),
        "selection": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--outer", type=int, default=150)
    parser.add_argument("--bootstrap", type=int, default=999)
    parser.add_argument("--out", type=Path, default=Path("experiments/judge-manipulation/phase5-readiness-coverage-results.json"))
    args = parser.parse_args()
    if args.outer < 1 or args.bootstrap < 99:
        parser.error("use at least one outer dataset and 99 bootstrap replicates")
    results = {
        "schema_version": 1,
        "purpose": "CPU coverage stress diagnostic only; not a proof of 95% coverage or an approval of the interval method.",
        "seed": args.seed,
        "outer_datasets_per_scenario": args.outer,
        "bootstrap_replicates_per_dataset": args.bootstrap,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "data_generating_process": {
            "scores": "group Normal(0, sigma) random intercept plus within-group standard logistic noise",
            "true_row_fpr": "computed from the logistic-normal marginal tail integral",
            "threshold_search": "20 thresholds chosen from observed development-score quantiles 0.90 through 0.999; a reduced adaptive grid, not every distinct score",
            "bootstrap": "resample independent groups jointly with replacement; all rows in each group move together; row-weighted FPR is recomputed",
            "methods": "pointwise 95th percentile cluster-bootstrap upper limits versus a 95th percentile max-deviation band over the reduced grid",
            "zero_event": "a threshold with zero observed and zero bootstrap false positives is unsupported, never eligible",
        },
        "scenarios": [
            simulate_scenario(config, args.seed + index, args.outer, args.bootstrap)
            for index, config in enumerate(SCENARIOS)
        ],
        "interpretation_limit": "Monte Carlo rates are conditional on these synthetic score and cluster-size models. The stress check does not establish coverage for the real corpus, arbitrary group dependence, the full observed-score threshold grid, or a 10,000-replicate final analysis.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

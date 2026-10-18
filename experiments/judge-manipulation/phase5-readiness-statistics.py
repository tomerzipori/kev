"""Draft exact, conservative group-bound helpers for Phase 5 readiness.

This module is not an approved Phase 5 gate. It assumes independent groups from
the target group mixture and a reviewed hard maximum number of rows per group.
It keeps all rows and bounds row-weighted error rates by group-any-error rates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence


def _validate_design(
    *, group_ids: Sequence[str], max_rows_per_group: int, design_approved: bool
) -> None:
    if not design_approved:
        raise ValueError("group independence and size-cap design are not approved")
    if not isinstance(max_rows_per_group, int) or max_rows_per_group < 1:
        raise ValueError("max_rows_per_group must be a positive integer")
    if not group_ids:
        raise ValueError("at least one group is required")
    if any(not isinstance(group_id, str) or not group_id.strip() for group_id in group_ids):
        raise ValueError("every row needs a nonempty string group_id")
    sizes: dict[str, int] = defaultdict(int)
    for group_id in group_ids:
        sizes[group_id] += 1
    if max(sizes.values()) > max_rows_per_group:
        raise ValueError("observed group exceeds the reviewed hard size cap")


def canonical_probability_decision(probability_value: float, persisted_threshold: float) -> bool:
    """Compare preserved Python scalar values using the registered >= rule.

    Callers must pass ``tensor.item()`` for a model probability and the numeric
    threshold loaded from its JSON registration. Do not cast either value back
    into the model tensor dtype before comparing.
    """
    probability = float(probability_value)
    threshold = float(persisted_threshold)
    if not math.isfinite(probability) or not math.isfinite(threshold):
        raise ValueError("probability and threshold must be finite")
    return probability >= threshold


def _binomial_upper_tail(n: int, probability: float, first_success: int) -> float:
    """P(Binomial(n, probability) >= first_success), evaluated stably."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0, 1]")
    if first_success <= 0:
        return 1.0
    if first_success > n:
        return 0.0
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    log_terms = [_binomial_log_pmf(n, probability, k) for k in range(first_success, n + 1)]
    return min(1.0, _logsumexp(log_terms))


def _binomial_cdf(n: int, probability: float, last_success: int) -> float:
    """P(Binomial(n, probability) <= last_success), without 1-tail cancellation."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0, 1]")
    if last_success < 0:
        return 0.0
    if last_success >= n:
        return 1.0
    if probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0
    log_terms = [_binomial_log_pmf(n, probability, k) for k in range(last_success + 1)]
    return min(1.0, _logsumexp(log_terms))


def _binomial_log_pmf(n: int, probability: float, successes: int) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(n - successes + 1)
        + successes * math.log(probability)
        + (n - successes) * math.log1p(-probability)
    )


def _logsumexp(log_values: Sequence[float]) -> float:
    peak = max(log_values)
    if peak == -math.inf:
        return 0.0
    return math.exp(peak) * math.fsum(math.exp(value - peak) for value in log_values)


def order_statistic_rank(
    independent_groups: int,
    *,
    row_fpr_target: float,
    max_rows_per_group: int,
    familywise_delta: float,
    candidate_models: int,
) -> tuple[int, float]:
    """Return the smallest valid ascending rank k and its binomial violation bound.

    The group-any-FP target is row_fpr_target / max_rows_per_group. The returned
    rank is one-based and predicts positive only for scores strictly greater
    than the k-th ascending order statistic. Bonferroni divides delta across
    the preregistered candidate_models.
    """
    if independent_groups < 1:
        raise ValueError("independent_groups must be positive")
    if not (0.0 < row_fpr_target < 1.0):
        raise ValueError("row_fpr_target must be in (0, 1)")
    if max_rows_per_group < 1 or candidate_models < 1:
        raise ValueError("group cap and candidate_models must be positive")
    if not (0.0 < familywise_delta < 1.0):
        raise ValueError("familywise_delta must be in (0, 1)")

    per_model_delta = familywise_delta / candidate_models
    any_fp_target = row_fpr_target / max_rows_per_group
    order_success_probability = 1.0 - any_fp_target
    if _binomial_upper_tail(
        independent_groups, order_success_probability, independent_groups
    ) > per_model_delta:
        raise ValueError("too few independent groups for the requested FPR and multiplicity")

    low, high = 1, independent_groups
    while low < high:
        middle = (low + high) // 2
        bound = _binomial_upper_tail(
            independent_groups, order_success_probability, middle
        )
        if bound <= per_model_delta:
            high = middle
        else:
            low = middle + 1
    return low, _binomial_upper_tail(
        independent_groups, order_success_probability, low
    )


def threshold_from_group_maxima(
    negative_scores: Sequence[float],
    group_ids: Sequence[str],
    *,
    row_fpr_target: float,
    max_rows_per_group: int,
    familywise_delta: float,
    candidate_models: int,
    runtime_uses_greater_equal: bool = False,
    design_approved: bool = False,
) -> dict[str, float | int | bool]:
    """Draft NP threshold using group maximum scores; never drops a row."""
    if len(negative_scores) != len(group_ids):
        raise ValueError("scores and group_ids must have equal length")
    _validate_design(
        group_ids=group_ids,
        max_rows_per_group=max_rows_per_group,
        design_approved=design_approved,
    )
    if any(not math.isfinite(float(score)) for score in negative_scores):
        raise ValueError("all scores must be finite")

    maxima: dict[str, float] = {}
    for group_id, score in zip(group_ids, negative_scores, strict=True):
        score = float(score)
        maxima[group_id] = max(maxima.get(group_id, -math.inf), score)
    sorted_maxima = sorted(maxima.values())
    rank, violation_bound = order_statistic_rank(
        len(sorted_maxima),
        row_fpr_target=row_fpr_target,
        max_rows_per_group=max_rows_per_group,
        familywise_delta=familywise_delta,
        candidate_models=candidate_models,
    )
    cutoff = sorted_maxima[rank - 1]
    effective_cutoff = math.nextafter(cutoff, math.inf) if runtime_uses_greater_equal else cutoff
    return {
        "independent_groups": len(sorted_maxima),
        "max_rows_per_group": max_rows_per_group,
        "candidate_models": candidate_models,
        "rank_one_based": rank,
        "cutoff": cutoff,
        "effective_threshold": effective_cutoff,
        "runtime_uses_greater_equal": runtime_uses_greater_equal,
        "strictly_exceeds_cutoff": True,
        "violation_bound_per_model": violation_bound,
        "familywise_delta": familywise_delta,
    }


def exact_binomial_upper(groups_with_event: int, independent_groups: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound for a Bernoulli group event."""
    if independent_groups < 1 or not (0 <= groups_with_event <= independent_groups):
        raise ValueError("event count must be between zero and a positive group count")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if groups_with_event == independent_groups:
        return 1.0
    if groups_with_event == 0:
        return -math.expm1(math.log(delta) / independent_groups)

    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        # CP upper endpoint solves P(Bin(n, p) <= x) = delta.
        cdf = _binomial_cdf(independent_groups, middle, groups_with_event)
        if cdf > delta:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def conservative_row_error_bounds(
    *,
    group_ids: Sequence[str],
    groups_with_any_error: int,
    max_rows_per_group: int,
    delta: float,
    design_approved: bool = False,
) -> dict[str, float | int]:
    """Bound row FPR/FNR via group-any-error probability under the hard cap."""
    _validate_design(
        group_ids=group_ids,
        max_rows_per_group=max_rows_per_group,
        design_approved=design_approved,
    )
    groups = len(set(group_ids))
    group_upper = exact_binomial_upper(groups_with_any_error, groups, delta)
    row_error_upper = min(1.0, max_rows_per_group * group_upper)
    return {
        "independent_groups": groups,
        "groups_with_any_error": groups_with_any_error,
        "group_event_upper": group_upper,
        "row_error_upper": row_error_upper,
        "row_recall_lower_if_error_means_miss": 1.0 - row_error_upper,
        "delta": delta,
        "max_rows_per_group": max_rows_per_group,
    }


def required_zero_event_groups(
    *, row_error_target: float, max_rows_per_group: int, delta: float, candidate_models: int = 1
) -> int:
    """Minimum groups so zero group errors meet a Bonferroni-adjusted bound."""
    if not (0.0 < row_error_target < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("targets and delta must be in (0, 1)")
    if max_rows_per_group < 1 or candidate_models < 1:
        raise ValueError("group cap and candidate_models must be positive")
    per_model_delta = delta / candidate_models
    group_error_target = row_error_target / max_rows_per_group
    return math.ceil(math.log(per_model_delta) / math.log1p(-group_error_target))


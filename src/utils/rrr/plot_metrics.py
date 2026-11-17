"""Small, artifact-independent calculations for RRR result figures."""

from __future__ import annotations

import numpy as np


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PEAK_FLOOR = 0.005


def fisher_mean(values, axis=None):
    """Average correlations in Fisher-z space."""
    values = np.asarray(values, dtype=float)
    z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
    return np.tanh(np.nanmean(z, axis=axis))


def monotonic_normalized_curve(ranks, correlations):
    """Return zero-based relative rank and a monotonic curve in [0, 1]."""
    ranks = np.asarray(ranks, dtype=float)
    correlations = np.asarray(correlations, dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(correlations)
    ranks, correlations = ranks[valid], correlations[valid]
    if len(ranks) < 2:
        return np.array([]), np.array([])
    order = np.argsort(ranks)
    ranks, correlations = ranks[order], correlations[order]
    unique, inverse = np.unique(ranks, return_inverse=True)
    if len(unique) != len(ranks):
        correlations = np.bincount(inverse, weights=correlations) / np.bincount(inverse)
        ranks = unique
    if len(ranks) < 2 or ranks[-1] == ranks[0]:
        return np.array([]), np.array([])
    curve = np.maximum.accumulate(np.maximum(correlations, 0))
    peak = curve[-1]
    if peak <= 0:
        return np.array([]), np.array([])
    return (ranks - ranks[0]) / (ranks[-1] - ranks[0]), curve / peak


def normalized_log_rank_auc(ranks, correlations):
    """Integrate normalized monotonic performance over normalized log rank."""
    ranks = np.asarray(ranks, dtype=float)
    correlations = np.asarray(correlations, dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(correlations) & (ranks > 0)
    ranks, correlations = ranks[valid], correlations[valid]
    if len(ranks) < 2:
        return np.nan
    order = np.argsort(ranks)
    ranks, correlations = ranks[order], correlations[order]
    unique, inverse = np.unique(ranks, return_inverse=True)
    if len(unique) != len(ranks):
        correlations = np.bincount(inverse, weights=correlations) / np.bincount(inverse)
        ranks = unique
    if len(ranks) < 2 or not np.isclose(ranks[0], 1):
        raise ValueError(f"Log-rank AUC requires rank 1; got {ranks[0]:g}")
    curve = np.maximum.accumulate(np.maximum(correlations, 0))
    if curve[-1] <= 0:
        return np.nan
    return float(np.trapezoid(curve / curve[-1], x=np.log(ranks) / np.log(ranks[-1])))


def rank_quantiles(ranks, target_correlations, levels=QUANTILES):
    """Calculate quantiles across targets for each row of a rank-by-target array."""
    ranks = np.asarray(ranks, dtype=float)
    values = np.asarray(target_correlations, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(ranks):
        raise ValueError("Expected one rank-by-target row for each rank")
    return np.nanquantile(values, levels, axis=1).T


def normalized_quantile_curve(values, min_peak=PEAK_FLOOR):
    """Normalize one quantile's monotonic curve when its peak passes the floor."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    curve = np.maximum.accumulate(np.maximum(values[valid], 0))
    peak = float(curve[-1])
    if peak <= min_peak or peak <= 0:
        return np.full(values.shape, np.nan)
    result = np.full(values.shape, np.nan)
    result[valid] = curve / peak
    return result


def mean_saturation_rank(ranks, correlations, fraction=0.99):
    """Interpolate the first rank reaching a fraction of peak mean correlation."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ranks = np.asarray(ranks, dtype=float)
    correlations = np.asarray(correlations, dtype=float)
    if ranks.ndim != 1 or correlations.shape != ranks.shape:
        raise ValueError("Expected one mean correlation per rank")
    finite = np.isfinite(ranks) & np.isfinite(correlations)
    ranks, correlations = ranks[finite], correlations[finite]
    if not len(ranks):
        return np.nan
    order = np.argsort(ranks)
    ranks, correlations = ranks[order], correlations[order]
    threshold = fraction * correlations.max()
    reached = np.flatnonzero(correlations >= threshold)
    if not len(reached):
        return np.nan
    index = int(reached[0])
    if index == 0:
        return float(ranks[0])
    left, right = correlations[index - 1:index + 1]
    weight = (threshold - left) / (right - left)
    return float(ranks[index - 1] + weight * (ranks[index] - ranks[index - 1]))


def quantile_saturation_rank(ranks, quantiles, fraction=0.99, min_peak=PEAK_FLOOR):
    """Latest first crossing among eligible quantiles, interpolated in rank."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ranks = np.asarray(ranks, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[0] != len(ranks):
        raise ValueError("quantiles must have one row per rank")
    crossings = []
    for column in quantiles.T:
        valid = np.isfinite(ranks) & np.isfinite(column)
        q_ranks, values = ranks[valid], column[valid]
        if not len(values):
            continue
        normalized = normalized_quantile_curve(values, min_peak)
        if not np.isfinite(normalized).any():
            continue
        index = int(np.searchsorted(normalized, fraction, side="left"))
        if index >= len(q_ranks):
            continue
        if index == 0:
            crossings.append(float(q_ranks[0]))
        else:
            left, right = normalized[index - 1:index + 1]
            weight = (fraction - left) / (right - left) if right > left else 1.0
            crossings.append(float(q_ranks[index - 1] + weight * (q_ranks[index] - q_ranks[index - 1])))
    return max(crossings) if crossings else np.nan


def explained_variance_curve(correlations, axis=0, scale=None):
    """Mean squared correlation per rank from per-target correlations.

    `correlations` has ranks on the first axis and any number of target axes after
    it, already averaged across evaluation folds (Fisher space). Negative
    correlations are clipped to zero before squaring; non-finite targets are
    ignored. Returns one value per rank: the average variance explained (r^2) per
    target, which is the summed r^2 divided by the number of targets. Summing
    instead would not move the saturation rank.

    `scale` optionally divides every target's r^2 (for example by its squared noise ceiling,
    giving the fraction of explainable variance). It broadcasts against the target axes
    (`correlations.shape[1:]`); targets whose scale is not a positive finite number are ignored.
    """
    values = np.asarray(correlations, dtype=float)
    squared = np.maximum(values, 0) ** 2
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        squared = squared / np.where(scale > 0, scale, np.nan)
        values = np.where(np.isfinite(squared), values, np.nan)
    squared = np.where(np.isfinite(values), squared, np.nan)
    flat = squared.reshape(values.shape[0], -1)
    with np.errstate(all="ignore"):
        counts = np.isfinite(flat).sum(axis=1)
        return np.where(counts > 0, np.nansum(flat, axis=1) / np.maximum(counts, 1), np.nan)


def explained_variance_saturation_rank(ranks, curve, fraction=0.99):
    """First interpolated rank at which the explained-variance curve reaches
    `fraction` of its own maximum, treating the curve as monotonic.

    Uses the same monotonic envelope and interpolation as the quantile statistic,
    with no minimum-peak cutoff (explained variance is small by nature). Curves
    whose maximum is not positive give NaN.
    """
    curve = np.asarray(curve, dtype=float).reshape(-1, 1)
    return quantile_saturation_rank(ranks, curve, fraction, min_peak=0.0)


def average_normalized_quantiles(curves, grid=None, min_peak=PEAK_FLOOR):
    """Average eligible quantiles equally across model layers on relative rank."""
    if grid is None:
        grid = np.r_[0.0, np.geomspace(1e-4, 1.0, 199)]
    grid = np.asarray(grid, dtype=float)
    if not curves:
        return grid, np.empty((len(grid), 0)), np.array([], dtype=int)
    n_quantiles = curves[0][1].shape[1]
    sums = np.zeros((len(grid), n_quantiles))
    counts = np.zeros(n_quantiles, dtype=int)
    for ranks, matrix in curves:
        for index, column in enumerate(np.asarray(matrix).T):
            normalized = normalized_quantile_curve(column, min_peak)
            x, y = monotonic_normalized_curve(ranks, normalized)
            if len(x):
                sums[:, index] += np.interp(grid, x, y)
                counts[index] += 1
    result = np.full_like(sums, np.nan)
    present = counts > 0
    result[:, present] = sums[:, present] / counts[present]
    return grid, result, counts


EXACT_SIGN_FLIP_MAX_N = 14


def paired_sign_flip(first, second, *, alternative="less", permutations=10_000, seed=0):
    """One-sided paired sign-flip p-value.

    With at most `EXACT_SIGN_FLIP_MAX_N` pairs every one of the 2^n sign patterns is enumerated
    (an exact test whose smallest p-value is 1/2^n); with more pairs, `permutations` random sign
    patterns are drawn with the notebook's fixed seed.
    """
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return np.nan
    observed = float(np.mean(difference))
    n = len(difference)
    if n <= EXACT_SIGN_FLIP_MAX_N:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        shuffled = (signs * difference).mean(axis=1)
        tolerance = 1e-12 * max(1.0, abs(observed))
        exceed = np.count_nonzero(shuffled <= observed + tolerance if alternative == "less"
                                  else shuffled >= observed - tolerance)
        return exceed / 2 ** n
    rng = np.random.default_rng(seed)
    shuffled = np.asarray([
        np.mean(difference * rng.choice((-1.0, 1.0), size=len(difference)))
        for _ in range(permutations)
    ])
    exceed = np.count_nonzero(shuffled <= observed if alternative == "less"
                              else shuffled >= observed)
    return (exceed + 1) / (permutations + 1)


def mean_by_model(values, models):
    """Average `values` within each model: layers of one model are not independent.

    Returns `(model names sorted, per-model means)`; non-finite values are ignored.
    """
    values = np.asarray(values, dtype=float)
    models = np.asarray(models, dtype=object)
    names = sorted({str(model) for model, value in zip(models, values) if np.isfinite(value)})
    means = [float(np.mean([value for model, value in zip(models, values)
                            if str(model) == name and np.isfinite(value)]))
             for name in names]
    return names, np.asarray(means)


def model_sign_flip(first, second, models, *, alternative="less"):
    """Paired sign-flip test with one unit per model (layer values are averaged first).

    `first`, `second` and `models` are parallel over (model, layer) entries. Returns NaN with
    fewer than two models, since no sign pattern other than the observed one is informative.
    """
    names, first_means = mean_by_model(first, models)
    _, second_means = mean_by_model(second, models)
    if len(names) < 2:
        return np.nan
    return paired_sign_flip(first_means, second_means, alternative=alternative)

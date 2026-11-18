"""Saturation rank as a function of lag for the iEEG datasets.

For every model layer and method, each lag has its own explained-variance rank curve
(mean squared correlation over all electrodes at that lag, folds Fisher-averaged,
negatives clipped to zero). Its saturation rank is the first interpolated rank reaching
99% of the curve's own maximum, exactly as in the main figures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting.final_axes import MODEL_COLORS, MODEL_NAMES, finish_axis, paper_style
from src.utils.plotting.final_data import (
    LABELS, PAIRS, _cache_fingerprint, _read_cache, _write_cache, load_pair_products,
)
from src.utils.plotting.final_bootstrap import FRACTION, LEVEL, N_BOOT, bootstrap_counts
from src.utils.rrr.plot_metrics import (
    explained_variance_curve, explained_variance_saturation_rank, model_sign_flip,
)

IEEG_DATASETS = ("podcast", "cars-2")
METHODS = ("rrr", "pca")
METHOD_LABELS = {"rrr": "RRR", "pca": "PCA"}
CACHE_NAME = ".final_lag_saturation_cache.pkl"


def _lag_saturation_one(entry, fraction):
    """(lags, saturation rank per lag) of one method's products (None without per-lag sums)."""
    if entry is None or entry["lags"] is None:
        return None
    lags, sums, counts, _ = entry["lags"]
    with np.errstate(all="ignore"):  # pooled over all electrodes of every cluster at each lag
        curves = sums.sum(axis=0) / counts.sum(axis=0)  # [rank, lag]
    saturation = np.array([
        explained_variance_saturation_rank(entry["ranks"], curves[:, index], fraction)
        for index in range(len(lags))])
    return lags, saturation


def load_lag_saturation(results_root, fraction=0.99):
    """{(dataset, model, layer, method): (lags, saturation rank per lag)} for the iEEG datasets."""
    path = results_root / CACHE_NAME
    fingerprint = _cache_fingerprint(results_root, f"lag_saturation_{fraction}")
    cached = _read_cache(path, fingerprint)
    if cached is not None:
        return cached
    result = {}
    products = load_pair_products(results_root)
    for dataset in IEEG_DATASETS:
        for model, layer in PAIRS:
            internal = products.get((dataset, model, layer), {}).get("internal", {})
            for method in METHODS:
                found = _lag_saturation_one(internal.get(method), fraction)
                if found is not None:
                    result[(dataset, model, layer, method)] = found
    _write_cache(path, fingerprint, result)
    return result


def _log2_saturation(ranks, curves, fraction=FRACTION):
    """Vectorized `explained_variance_saturation_rank` of curves [draw, rank, lag], as log2."""
    envelope = np.maximum.accumulate(np.maximum(curves, 0), axis=1)
    peak = envelope[:, -1]
    reached = envelope >= fraction * peak[:, None]
    first = np.argmax(reached, axis=1)  # [draw, lag]
    previous = np.maximum(first - 1, 0)
    right = np.take_along_axis(envelope, first[:, None], axis=1)[:, 0]
    left = np.take_along_axis(envelope, previous[:, None], axis=1)[:, 0]
    with np.errstate(all="ignore"):
        weight = np.where(right > left, (fraction * peak - left) / (right - left), 1.0)
        saturation = np.where(first == 0, ranks[0],
                              ranks[previous] + weight * (ranks[first] - ranks[previous]))
        return np.where(peak > 0, np.log2(saturation), np.nan)


def load_lag_saturation_intervals(results_root, exclude_models=(), n_boot=N_BOOT):
    """{(dataset, method): (lags, low, high)} bootstrap interval of the mean log2 saturation rank.

    Clusters (Podcast: subjects; Cars 2: depth probes of a subject) are resampled with replacement,
    shared by all layers of a dataset. In each resample the per-lag pooled explained-variance curve
    of every model layer is rebuilt from the cluster sums, its log2 saturation rank found, and the
    log2 ranks averaged over layers; the interval is the percentile range over resamples. Values
    are returned as ranks (2 ** log2 rank).
    """
    path = results_root / CACHE_NAME.replace(".pkl", "_ci.pkl")
    fingerprint = _cache_fingerprint(
        results_root, f"lag_saturation_ci_{n_boot}_{sorted(exclude_models)}")
    cached = _read_cache(path, fingerprint)
    if cached is not None:
        return cached
    result = {}
    products = load_pair_products(results_root)
    for dataset in IEEG_DATASETS:
        multiplicity, expected = None, None
        totals = {method: None for method in METHODS}
        lags = None
        for model, layer in PAIRS:
            if model in exclude_models:
                continue
            internal = products.get((dataset, model, layer), {}).get("internal", {})
            for method in METHODS:
                entry = internal.get(method)
                if entry is None or entry["lags"] is None:
                    continue
                ranks = entry["ranks"]
                lags, sums, counts, names = entry["lags"]
                if multiplicity is None or names != expected:
                    expected, multiplicity = names, bootstrap_counts(len(names), n_boot)
                n_rank, n_lag = sums.shape[1:]
                with np.errstate(all="ignore"):
                    curves = ((multiplicity @ sums.reshape(len(names), -1))
                              / (multiplicity @ counts.reshape(len(names), -1))
                              ).reshape(n_boot, n_rank, n_lag)
                log_rank = _log2_saturation(ranks, curves)
                finite = np.isfinite(log_rank)
                if totals[method] is None:
                    totals[method] = [np.zeros_like(log_rank), np.zeros(log_rank.shape)]
                totals[method][0] += np.where(finite, log_rank, 0)
                totals[method][1] += finite
        for method, accumulated in totals.items():
            if accumulated is None:
                continue
            with np.errstate(all="ignore"):
                mean = accumulated[0] / np.where(accumulated[1] > 0, accumulated[1], np.nan)
            low, high = np.nanpercentile(mean, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2], axis=0)
            result[(dataset, method)] = (lags, 2 ** low, 2 ** high)
    _write_cache(path, fingerprint, result)
    return result


def benjamini_hochberg(p_values, alpha=0.05):
    """Boolean mask of discoveries and BH-adjusted q-values; NaN p-values are never discoveries."""
    p_values = np.asarray(p_values, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    q = np.full(p_values.shape, np.nan)
    if not len(valid):
        return np.zeros(p_values.shape, dtype=bool), q
    order = valid[np.argsort(p_values[valid])]
    adjusted = p_values[order] * len(valid) / np.arange(1, len(valid) + 1)
    q[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    return q <= alpha, q


def lag_saturation_significance(data, alpha=0.05):
    """{dataset: (lags, p, q, significant)} testing RRR < PCA saturation rank at every lag.

    At each lag the log2 saturation ranks of a model's layers are averaged (layers are not
    independent) and RRR is compared with PCA by the one-sided exact paired sign-flip test across
    models, as in the distribution plots. Benjamini-Hochberg false-discovery-rate control at
    `alpha` is applied across the lags of each dataset.
    """
    result = {}
    for dataset in IEEG_DATASETS:
        layers = sorted({key[1:3] for key in data if key[0] == dataset
                         and (dataset, *key[1:3], "rrr") in data and (dataset, *key[1:3], "pca") in data},
                        key=lambda item: (item[0], -1 if item[1] is None else item[1]))
        if not layers:
            continue
        lags = data[(dataset, *layers[0], "rrr")][0]
        with np.errstate(all="ignore"):
            logs = {method: np.stack([np.log2(np.where(data[(dataset, *layer, method)][1] > 0,
                                                       data[(dataset, *layer, method)][1], np.nan))
                                      for layer in layers]) for method in METHODS}
        models = [layer[0] for layer in layers]
        p = np.array([model_sign_flip(logs["rrr"][:, index], logs["pca"][:, index], models)
                      for index in range(len(lags))])
        significant, q = benjamini_hochberg(p, alpha)
        result[dataset] = (lags, p, q, significant)
    return result


def lag_saturation_figure(data, intervals=None, significance=None, *, ranks_label="Saturation rank"):
    """One panel per iEEG dataset: faint RRR line per model layer, black RRR mean, PCA mean overlaid.

    Means are taken in log2 rank. `intervals` (from `load_lag_saturation_intervals`) adds shaded
    bootstrap bands around each mean.
    """
    pca_color = "#dc2626"
    styles = {"rrr": ("black", "RRR mean over model layers"), "pca": (pca_color, "PCA mean over model layers")}
    with paper_style():
        figure, axes = plt.subplots(1, len(IEEG_DATASETS), figsize=(6.5 * len(IEEG_DATASETS), 4.6),
                                    sharex=True, sharey=True, squeeze=False)
        axes = axes[0]
        shown = set()
        for ax, dataset in zip(axes, IEEG_DATASETS):
            ax.set_title(LABELS[dataset])
            drawn = False
            for method in METHODS:
                members = {key: value for key, value in data.items()
                           if key[0] == dataset and key[3] == method}
                if not members:
                    continue
                drawn = True
                logs = []
                for (_, model, _, _), (lags, saturation) in members.items():
                    with np.errstate(all="ignore"):
                        logs.append(np.log2(np.where(saturation > 0, saturation, np.nan)))
                    if method == "rrr":
                        ax.plot(lags, saturation, color=MODEL_COLORS[model], alpha=.3, linewidth=1)
                        shown.add(model)
                with np.errstate(all="ignore"):
                    mean = 2 ** np.nanmean(np.stack(logs), axis=0)
                color, label = styles[method]
                if intervals is not None and (dataset, method) in intervals:
                    band_lags, low, high = intervals[(dataset, method)]
                    ax.fill_between(band_lags, low, high, color=color, alpha=.22, linewidth=0)
                ax.plot(lags, mean, color=color, linewidth=2.6, label=label)
            if not drawn:
                ax.text(.5, .5, "Not available", ha="center", va="center",
                        transform=ax.transAxes, color="#64748b")
                continue
            if significance is not None and dataset in significance:
                sig_lags, _, _, flagged = significance[dataset]
                ax.plot(sig_lags[flagged], np.full(int(flagged.sum()), .97), linestyle="", marker="*",
                        markersize=8, color="black", transform=ax.get_xaxis_transform(), clip_on=False,
                        label="RRR < PCA (BH FDR 0.05)")
            ax.axvline(0, color="#64748b", linewidth=1, linestyle=":")
            ax.set_yscale("log", base=2)
            ax.set_xlabel("Lag (s)")
            finish_axis(ax)
        axes[0].set_ylabel(ranks_label)
        for model in MODEL_COLORS:
            if model in shown:
                axes[0].plot([], [], color=MODEL_COLORS[model], linewidth=2.2, label=MODEL_NAMES[model])
        handles, labels = axes[0].get_legend_handles_labels()
        figure.tight_layout()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 0.0),
                      ncol=3, frameon=False, fontsize=12)
    plt.close(figure)
    return figure

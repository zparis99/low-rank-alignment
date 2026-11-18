"""Compact per-result products, computed while the result matrix is already in memory.

Several consumers need the same per-target r^2 of one (dataset, model, layer, method) result: the
figure curves, the bootstrap intervals (saturation, AUC, ROI), the per-ROI curves and the
saturation-rank-by-lag figure. Each used to re-read the result files. `result_products` derives
everything those consumers need from one read; the products are small (cluster x rank sums), so
they are cached on disk next to the main summaries (`final_data.load_pair_products`).

Resampling clusters are subjects (Podcast, fMRI) or, for Cars 2 (sEEG, only three subjects),
depth probes of a subject.
"""

from __future__ import annotations

import re

import numpy as np

from src.utils.rrr.plot_metrics import explained_variance_curve

VARIANTS = ("normalized", "raw_fmri")


def probe_of(electrode):
    """Depth probe of an sEEG contact: its name without the trailing contact number."""
    return re.sub(r"\d+$", "", str(electrode))


def cluster_labels(dataset, targets):
    """Resampling cluster of every target (row of the RRR target table)."""
    subjects = targets["subject"].astype(str)
    if dataset == "cars-2":
        return (subjects + "/" + targets["electrode"].map(probe_of)).to_numpy()
    return subjects.to_numpy()


def sums_by_cluster(squared, labels, names=None):
    """Summed r^2 and pair counts per cluster: `(sums [cluster, rank], counts, names)`.

    Summing every cluster and dividing gives exactly the plotted mean r^2 per (target, lag) pair.
    """
    names = sorted(set(labels)) if names is None else names
    sums = np.empty((len(names), squared.shape[0]))
    counts = np.empty_like(sums)
    for index, name in enumerate(names):
        block = squared[:, labels == name, :].reshape(squared.shape[0], -1)
        sums[index] = np.nansum(block, axis=1)
        counts[index] = np.isfinite(block).sum(axis=1)
    return sums, counts, names


def lag_cluster_sums(squared, labels):
    """Like `sums_by_cluster`, keeping the lag axis: `(sums [cluster, rank, lag], counts, names)`."""
    names = sorted(set(labels))
    sums = np.stack([np.nansum(squared[:, labels == name, :], axis=1) for name in names])
    counts = np.stack([np.isfinite(squared[:, labels == name, :]).sum(axis=1) for name in names])
    return sums, counts, names


def squared_variants(per_pair, scale):
    """`{variant: [rank, target, lag] r^2}` from the fold-averaged correlations `per_pair`.

    r^2 is the correlation clipped at zero and squared. "normalized" is what the main figures
    use: fMRI r^2 is divided by the voxel's squared noise ceiling (`scale`; voxels without one
    become NaN). fMRI also has "raw_fmri", the undivided r^2.
    """
    per_pair = np.asarray(per_pair, dtype=float)
    squared = np.where(np.isfinite(per_pair), np.maximum(per_pair, 0) ** 2, np.nan)
    if scale is None:
        return {"normalized": squared}
    normalized = squared / np.where(scale > 0, scale, np.nan)[None, :, None]
    return {"normalized": normalized, "raw_fmri": squared}


def result_products(dataset, ranks, per_pair, scale, targets, *, region_index=None, lags=None,
                    cluster_only=False):
    """Everything downstream consumers need from one result, from its fold-averaged correlations.

    `per_pair` is `[rank, target, lag]` (Fisher mean over folds), `scale` the fMRI per-voxel
    squared noise ceiling (None for iEEG). Returns `{"ranks", "clusters", "roi", "roi_curves",
    "lags"}`: per variant the per-cluster r^2 sums/counts/names (`clusters`), the same per
    ROI (`roi`, regions with targets) and the pooled per-ROI curves (`roi_curves`), plus for iEEG
    the per-lag cluster sums `(lags, sums, counts, names)`. `region_index` maps each region to
    the indices of its targets (omit for no ROI products), `cluster_only` keeps just `clusters`
    (directed transfer results).
    """
    per_pair = np.asarray(per_pair, dtype=float)
    labels = cluster_labels(dataset, targets)
    products = {"ranks": np.asarray(ranks, dtype=float), "clusters": {}, "roi": {},
                "roi_curves": {}, "roi_counts": {}, "lags": None}
    if region_index is not None and not cluster_only:
        products["roi_counts"] = {region: int(len(keep)) for region, keep in region_index.items()}
    for variant, squared in squared_variants(per_pair, scale).items():
        products["clusters"][variant] = sums_by_cluster(squared, labels)
        if cluster_only:
            continue
        if variant == "normalized" and lags is not None and dataset != "fmri":
            products["lags"] = (np.asarray(lags), *lag_cluster_sums(squared, labels))
        if region_index is None:
            continue
        roi, curves = {}, {}
        for region, keep in region_index.items():
            if not len(keep):
                continue
            roi[region] = sums_by_cluster(squared[:, keep, :], labels[keep])
            curves[region] = explained_variance_curve(
                per_pair[:, keep, :],
                scale=None if scale is None or variant == "raw_fmri" else scale[keep][:, None])
        products["roi"][variant], products["roi_curves"][variant] = roi, curves
    return products

"""Bootstrap confidence intervals for the saturation ranks shown in the distribution plots.

Internal runs: clusters of targets are resampled with replacement and the pooled explained-
variance curve is recomputed from the resampled cluster sums, so each cluster counts in
proportion to its number of (target, lag) pairs, exactly like the plotted curves. A cluster is
a subject (Podcast, fMRI) or, for Cars 2 (sEEG, only three subjects), a depth probe of a subject.

Leave-one-subject-out runs: the plotted value is the mean over held-out subjects of each
subject's saturation rank, so held-out subjects are resampled and re-averaged.

Every interval is for one model: in each resample the log2 saturation ranks of the model's layers
are averaged (the unit used by the significance tests) and the 2.5th and 97.5th percentiles over
resamples are returned. All models of a dataset share the same resamples, so intervals are
comparable across models and methods.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from src.utils.plotting.final_data import (
    DATASETS, _ev_scale, _fisher, _lagged, _load_matrix, _metric_name, _rrr,
)
from src.utils.rrr.plot_metrics import explained_variance_saturation_rank, normalized_log_rank_auc

N_BOOT = 2000
SEED = 0
FRACTION = .99
LEVEL = 95.0
METHODS = ("rrr", "pca")


def probe_of(electrode):
    """Depth probe of an sEEG contact: its name without the trailing contact number."""
    return re.sub(r"\d+$", "", str(electrode))


def cluster_labels(dataset, targets):
    """Resampling cluster of every target (row of the RRR target table)."""
    subjects = targets["subject"].astype(str)
    if dataset == "cars-2":
        return (subjects + "/" + targets["electrode"].map(probe_of)).to_numpy()
    return subjects.to_numpy()


def squared_correlations(results_root, dataset, model, layer, method, fmri_noise_normalized=True,
                         source=None):
    """Per-target explained variance at every rank: `(ranks, [rank, target, lag] r^2, targets)`.

    r^2 is the fold-averaged correlation clipped at zero and squared (fMRI: divided by the voxel's
    squared noise ceiling, voxels without one are NaN). `source` selects the directed
    cross-dataset (transfer) results for target `dataset` instead of the internal run. Returns
    None when the method has no per-target metrics.
    """
    if source is None:
        artifact = _rrr(Path(results_root), "internal", dataset, model, layer
                        ).load_artifacts(["target_manifest"])
        matrix = _load_matrix(artifact, method, _metric_name(dataset, method))
    else:
        artifact = _rrr(Path(results_root), "cross_dataset", dataset, model, layer, source)
        if dataset == "fmri":
            candidates = ("corr_per_target",)
        elif method == "rrr":
            candidates = ("nr_orth_corr_per_target", "nr_corr_per_target")
        else:
            candidates = ("pca_corr_per_target",)
        matrix = next((m for m in (_load_matrix(artifact, method, name) for name in candidates)
                       if m is not None), None)
    if matrix is None:
        return None
    values = np.asarray(_fisher(_lagged(matrix), axis=1), dtype=float)  # [rank, target, lag]
    squared = np.where(np.isfinite(values), np.maximum(values, 0) ** 2, np.nan)
    scale = _ev_scale(results_root, dataset, matrix.targets, fmri_noise_normalized)
    if scale is not None:  # fMRI: fraction of explainable variance; no ceiling -> dropped
        squared = squared / np.where(scale > 0, scale, np.nan)[None, :, None]
    return np.asarray(matrix.ranks, dtype=float), squared, matrix.targets


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


def cluster_sums(results_root, dataset, model, layer, method, fmri_noise_normalized=True):
    """Per-cluster summed r^2 and pair counts at every rank for one internal RRR/PCA curve.

    Returns `(ranks, sums [cluster, rank], counts [cluster, rank], cluster names)`.
    """
    loaded = squared_correlations(results_root, dataset, model, layer, method,
                                  fmri_noise_normalized)
    if loaded is None:
        return None
    ranks, squared, targets = loaded
    sums, counts, names = sums_by_cluster(squared, cluster_labels(dataset, targets))
    return ranks, sums, counts, names


def bootstrap_counts(n_clusters, n_boot=N_BOOT, seed=SEED):
    """[n_boot, n_clusters] multiplicities of each cluster in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot)


def resampled_log2_saturation(ranks, sums, counts, multiplicity, fraction=FRACTION):
    """log2 saturation rank of the pooled curve in every bootstrap resample."""
    total = multiplicity @ counts
    with np.errstate(all="ignore"):
        curves = (multiplicity @ sums) / total
    return np.log2([explained_variance_saturation_rank(ranks, curve, fraction)
                    for curve in curves])


def _interval(draws):
    """Percentile interval of per-model draws [layers, resamples], averaged over layers."""
    with np.errstate(all="ignore"):
        per_model = np.nanmean(draws, axis=0)
    low, high = np.nanpercentile(per_model, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return float(low), float(high)


def internal_intervals(results_root, model_layers, datasets=tuple(DATASETS), log=None,
                       fmri_noise_normalized=True):
    """`{(dataset, model, method): (low, high)}` in log2 saturation rank for internal runs."""
    result = {}
    for dataset in datasets:
        multiplicity, expected = None, None
        for model, layers in model_layers:
            draws = {method: [] for method in METHODS}
            for layer in layers:
                for method in METHODS:
                    try:
                        sums = cluster_sums(results_root, dataset, model, layer, method,
                                            fmri_noise_normalized)
                    except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError):
                        sums = None
                    if sums is None:
                        continue
                    ranks, cluster_sum, cluster_count, names = sums
                    if multiplicity is None or names != expected:
                        # Targets are shared by every model layer, so the resamples are too.
                        expected = names
                        multiplicity = bootstrap_counts(len(names))
                    draws[method].append(resampled_log2_saturation(
                        ranks, cluster_sum, cluster_count, multiplicity))
            for method, rows in draws.items():
                if rows:
                    result[(dataset, model, method)] = _interval(np.asarray(rows))
            if log:
                log(f"{dataset} {model}: {len(layers)} layers")
    return result


def loso_intervals(loso_curves, saturation_rank, min_subjects=5):
    """`{(dataset, model, method): (low, high)}` for LOSO runs, resampling held-out subjects.

    `loso_curves` maps `(dataset, model, layer, subject, method)` to `(ranks, curve)` (the
    explained-variance LOSO curves). Datasets with fewer than `min_subjects` held-out subjects
    are skipped: three subjects give only ten distinct resamples.
    """
    by_layer = {}
    for (dataset, model, layer, subject, method), (ranks, curve) in loso_curves.items():
        rank = saturation_rank(ranks, curve)
        if np.isfinite(rank):
            by_layer.setdefault((dataset, model, method, layer), {})[str(subject)] = rank
    result, groups = {}, {}
    for (dataset, model, method, layer), by_subject in by_layer.items():
        groups.setdefault((dataset, model, method), []).append(by_subject)
    for (dataset, model, method), layers in groups.items():
        subjects = sorted({s for layer in layers for s in layer})
        if len(subjects) < min_subjects:
            continue
        multiplicity = bootstrap_counts(len(subjects))
        draws = []
        for by_subject in layers:
            ranks = np.array([by_subject.get(s, np.nan) for s in subjects])
            present = np.isfinite(ranks)
            weights = multiplicity * present
            with np.errstate(all="ignore"):
                draws.append(np.log2((weights @ np.where(present, ranks, 0.0)) / weights.sum(1)))
        result[(dataset, model, method)] = _interval(np.asarray(draws))
    return result


def saturation_intervals(results_root, loso_curves, model_layers, log=None,
                         fmri_noise_normalized=True):
    """All bootstrap intervals: `{"internal": {...}, "loso": {...}}` keyed `(dataset, model, method)`.

    `loso_curves` is the explained-variance LOSO curve table of `load_final_data` (loaded with the
    same `fmri_noise_normalized` setting); `model_layers` is `[(model, [layers])]`.
    """
    from src.utils.plotting.final_figures import _saturation_rank

    def loso_rank(ranks, curve):
        return _saturation_rank("explained_variance", ranks, curve, FRACTION)

    return {"internal": internal_intervals(results_root, model_layers, log=log,
                                          fmri_noise_normalized=fmri_noise_normalized),
            "loso": loso_intervals(loso_curves, loso_rank)}


MIN_CLUSTERS = 5  # regions with fewer subjects / probes than this get no interval


def roi_intervals(results_root, data_roots, regions, model_layers, fmri_noise_normalized=True,
                  min_clusters=MIN_CLUSTERS, log=None):
    """`{(dataset, region, model, method): (low, high)}` log2 saturation-rank intervals per ROI.

    Within each bihemispheric ROI the clusters (subjects; Cars 2 depth probes) of its targets are
    resampled. Regions with fewer than `min_clusters` clusters are skipped. All models share the
    resamples of a region.
    """
    from scripts.run_rrr import _dataset_adapter_name
    from src.core.registry import get_dataset_adapter
    from src.utils.atlas.core import bihemispheric_region_groups

    groups = bihemispheric_region_groups()
    result = {}
    for dataset in DATASETS:
        adapter = get_dataset_adapter(_dataset_adapter_name(DATASETS[dataset]))
        region_info = None
        multiplicity = {}
        for model, layers in model_layers:
            draws = {}
            for layer in layers:
                for method in METHODS:
                    try:
                        loaded = squared_correlations(results_root, dataset, model, layer, method,
                                                      fmri_noise_normalized)
                    except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError):
                        loaded = None
                    if loaded is None:
                        continue
                    ranks, squared, targets = loaded
                    if region_info is None:
                        located = targets.copy()
                        located[["x", "y", "z"]] = adapter.get_electrode_coordinates(
                            located, str(data_roots[dataset]))
                        atlas = adapter.get_target_atlas_labels(located, str(data_roots[dataset]))
                        clusters = cluster_labels(dataset, targets)
                        region_info = {
                            region: np.flatnonzero(np.isin(atlas, list(groups[region])))
                            for region in regions}
                    for region in regions:
                        keep = region_info[region]
                        if not len(keep):
                            continue
                        names = sorted(set(clusters[keep]))
                        if len(names) < min_clusters:
                            continue
                        sums, counts, _ = sums_by_cluster(
                            squared[:, keep, :], clusters[keep], names)
                        if region not in multiplicity:
                            multiplicity[region] = bootstrap_counts(len(names))
                        draws.setdefault((region, method), []).append(
                            resampled_log2_saturation(ranks, sums, counts, multiplicity[region]))
            for (region, method), rows in draws.items():
                result[(dataset, region, model, method)] = _interval(np.asarray(rows))
            if log:
                log(f"ROI {dataset} {model}")
    return result


AUC_COMPARISONS = (
    ("Within RRR \u2212 Transfer RRR", ("within", "rrr"), ("transfer", "rrr")),
    ("Transfer RRR \u2212 Transfer PCA", ("transfer", "rrr"), ("transfer", "pca")),
    ("Within PCA \u2212 Transfer PCA", ("within", "pca"), ("transfer", "pca")),
)


def auc_intervals(results_root, model_layers, fmri_noise_normalized=True, log=None):
    """`{(target, source, model, comparison title): (low, high)}` for normalized-AUC differences.

    Target clusters (subjects; Cars 2 probes) are resampled once per draw and the within-dataset and
    transfer curves are rebuilt from the same draw, so each draw's difference is paired. The
    intervals are per model: the differences of its layers are averaged within every draw.
    """
    result = {}
    for target in DATASETS:
        for source in DATASETS:
            if source == target:
                continue
            multiplicity, expected = None, None
            for model, layers in model_layers:
                draws = {title: [] for title, _, _ in AUC_COMPARISONS}
                for layer in layers:
                    curves = {}
                    for kind in ("within", "transfer"):
                        for method in METHODS:
                            try:
                                loaded = squared_correlations(
                                    results_root, target, model, layer, method,
                                    fmri_noise_normalized, None if kind == "within" else source)
                            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError):
                                loaded = None
                            if loaded is not None:
                                ranks, squared, targets = loaded
                                sums, counts, names = sums_by_cluster(
                                    squared, cluster_labels(target, targets))
                                curves[(kind, method)] = (ranks, sums, counts, names)
                    if len(curves) < 4:
                        continue
                    names = curves[("within", "rrr")][3]
                    if any(value[3] != names for value in curves.values()):
                        continue  # the target clusters differ between runs: cannot pair draws
                    if multiplicity is None or names != expected:
                        expected, multiplicity = names, bootstrap_counts(len(names))
                    aucs = {}
                    for key, (ranks, sums, counts, _) in curves.items():
                        with np.errstate(all="ignore"):
                            resampled = (multiplicity @ sums) / (multiplicity @ counts)
                        aucs[key] = np.array([normalized_log_rank_auc(ranks, curve)
                                              for curve in resampled])
                    for title, first, second in AUC_COMPARISONS:
                        draws[title].append(aucs[first] - aucs[second])
                for title, rows in draws.items():
                    if rows:
                        result[(target, source, model, title)] = _interval(np.asarray(rows))
                if log:
                    log(f"AUC {source}->{target} {model}")
    return result


def interval_tables(results_root, loso_curves, model_layers, data_roots, regions,
                    fmri_noise_normalized=True, log=None):
    """All bootstrap intervals, each piece cached separately (see `final_subspace_data.cached`).

    Returns `{"internal", "loso", "auc", "roi"}`: saturation-rank intervals keyed
    `(dataset, model, method)` (log2), `(dataset, region, model, method)` for ROIs (log2), and
    AUC-difference intervals keyed `(target, source, model, comparison)`.
    """
    from src.utils.plotting.final_subspace_data import cached

    tag = "" if fmri_noise_normalized else "_raw_fmri"
    saturation = cached(results_root, "saturation_cis" + tag, lambda: saturation_intervals(
        results_root, loso_curves, model_layers, log=log,
        fmri_noise_normalized=fmri_noise_normalized))
    return {
        **saturation,
        "auc": cached(results_root, "auc_cis" + tag, lambda: auc_intervals(
            results_root, model_layers, fmri_noise_normalized, log)),
        "roi": cached(results_root, "roi_cis" + tag, lambda: roi_intervals(
            results_root, data_roots, regions, model_layers, fmri_noise_normalized, log=log)),
    }

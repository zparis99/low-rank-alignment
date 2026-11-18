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

import numpy as np

from src.utils.plotting.final_data import DATASETS
from src.utils.plotting.final_products import (  # noqa: F401  (re-exported)
    VARIANTS, cluster_labels, probe_of, sums_by_cluster,
)
from src.utils.rrr.plot_metrics import explained_variance_saturation_rank, normalized_log_rank_auc

N_BOOT = 2000
SEED = 0
FRACTION = .99
LEVEL = 95.0
METHODS = ("rrr", "pca")

# Every interval is computed from `final_data.load_pair_products` (cluster sums of the already-read
# results, no file reads here) for the main variant (fMRI r^2 as a fraction of explainable
# variance) and, for fMRI targets only, the raw-r^2 variant.


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


def internal_intervals(products, model_layers, datasets=tuple(DATASETS), log=None):
    """`{variant: {(dataset, model, method): (low, high)}}` in log2 saturation rank (internal runs)."""
    result = {variant: {} for variant in VARIANTS}
    for dataset in datasets:
        multiplicity, expected = None, None
        for model, layers in model_layers:
            draws = {variant: {method: [] for method in METHODS} for variant in VARIANTS}
            for layer in layers:
                internal = products.get((dataset, model, layer), {}).get("internal", {})
                for method in METHODS:
                    entry = internal.get(method)
                    if entry is None:
                        continue
                    for variant, (cluster_sum, cluster_count, names) in entry["clusters"].items():
                        if multiplicity is None or names != expected:
                            # Targets are shared by every model layer, so the resamples are too.
                            expected = names
                            multiplicity = bootstrap_counts(len(names))
                        draws[variant][method].append(resampled_log2_saturation(
                            entry["ranks"], cluster_sum, cluster_count, multiplicity))
            for variant, by_method in draws.items():
                for method, rows in by_method.items():
                    if rows:
                        result[variant][(dataset, model, method)] = _interval(np.asarray(rows))
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


def saturation_intervals(products, loso_curves, model_layers, raw_loso_curves=None, log=None):
    """`{variant: {"internal": {...}, "loso": {...}}}` keyed `(dataset, model, method)`.

    `loso_curves` / `raw_loso_curves` are the explained-variance LOSO curve tables of
    `load_final_data` (main and `raw_fmri`); `model_layers` is `[(model, [layers])]`.
    """
    from src.utils.plotting.final_figures import _saturation_rank

    def loso_rank(ranks, curve):
        return _saturation_rank("explained_variance", ranks, curve, FRACTION)

    internal = internal_intervals(products, model_layers, log=log)
    return {"normalized": {"internal": internal["normalized"],
                           "loso": loso_intervals(loso_curves, loso_rank)},
            "raw_fmri": {"internal": internal["raw_fmri"],
                         "loso": loso_intervals(raw_loso_curves or {}, loso_rank)}}


MIN_CLUSTERS = 5  # regions with fewer subjects / probes than this get no interval


def roi_intervals(products, regions, model_layers, min_clusters=MIN_CLUSTERS, log=None):
    """`{variant: {(dataset, region, model, method): (low, high)}}` log2 saturation-rank intervals.

    Within each bihemispheric ROI the clusters (subjects; Cars 2 depth probes) of its targets are
    resampled. Regions with fewer than `min_clusters` clusters are skipped. All models share the
    resamples of a region.
    """
    result = {variant: {} for variant in VARIANTS}
    for dataset in DATASETS:
        multiplicity = {}
        for model, layers in model_layers:
            draws = {variant: {} for variant in VARIANTS}
            for layer in layers:
                internal = products.get((dataset, model, layer), {}).get("internal", {})
                for method in METHODS:
                    entry = internal.get(method)
                    if entry is None:
                        continue
                    for variant, by_region in entry["roi"].items():
                        for region in regions:
                            if region not in by_region:
                                continue
                            sums, counts, names = by_region[region]
                            if len(names) < min_clusters:
                                continue
                            if region not in multiplicity:
                                multiplicity[region] = bootstrap_counts(len(names))
                            draws[variant].setdefault((region, method), []).append(
                                resampled_log2_saturation(
                                    entry["ranks"], sums, counts, multiplicity[region]))
            for variant, by_key in draws.items():
                for (region, method), rows in by_key.items():
                    result[variant][(dataset, region, model, method)] = _interval(np.asarray(rows))
            if log:
                log(f"ROI {dataset} {model}")
    return result


AUC_COMPARISONS = (
    ("Within RRR \u2212 Transfer RRR", ("within", "rrr"), ("transfer", "rrr")),
    ("Transfer RRR \u2212 Transfer PCA", ("transfer", "rrr"), ("transfer", "pca")),
    ("Within PCA \u2212 Transfer PCA", ("within", "pca"), ("transfer", "pca")),
)


def auc_intervals(products, model_layers, log=None):
    """`{variant: {(target, source, model, comparison title): (low, high)}}` for AUC differences.

    Target clusters (subjects; Cars 2 probes) are resampled once per draw and the within-dataset and
    transfer curves are rebuilt from the same draw, so each draw's difference is paired. The
    intervals are per model: the differences of its layers are averaged within every draw.
    """
    result = {variant: {} for variant in VARIANTS}
    for target in DATASETS:
        for source in DATASETS:
            if source == target:
                continue
            multiplicity, expected = None, None
            for model, layers in model_layers:
                draws = {variant: {title: [] for title, _, _ in AUC_COMPARISONS}
                         for variant in VARIANTS}
                for layer in layers:
                    pair = products.get((target, model, layer), {})
                    curves = {variant: {} for variant in VARIANTS}
                    for kind, found in (("within", pair.get("internal", {})),
                                        ("transfer", pair.get("cross", {}).get(source, {}))):
                        for method in METHODS:
                            entry = found.get(method)
                            for variant, (sums, counts, names) in (
                                    entry["clusters"].items() if entry else ()):
                                curves[variant][(kind, method)] = (
                                    entry["ranks"], sums, counts, names)
                    for variant, variant_curves in curves.items():
                        if len(variant_curves) < 4:
                            continue
                        names = variant_curves[("within", "rrr")][3]
                        if any(value[3] != names for value in variant_curves.values()):
                            continue  # the target clusters differ between runs: cannot pair draws
                        if multiplicity is None or names != expected:
                            expected, multiplicity = names, bootstrap_counts(len(names))
                        aucs = {}
                        for key, (ranks, sums, counts, _) in variant_curves.items():
                            with np.errstate(all="ignore"):
                                resampled = (multiplicity @ sums) / (multiplicity @ counts)
                            aucs[key] = np.array([normalized_log_rank_auc(ranks, curve)
                                                  for curve in resampled])
                        for title, first, second in AUC_COMPARISONS:
                            draws[variant][title].append(aucs[first] - aucs[second])
                for variant, by_title in draws.items():
                    for title, rows in by_title.items():
                        if rows:
                            result[variant][(target, source, model, title)] = _interval(
                                np.asarray(rows))
                if log:
                    log(f"AUC {source}->{target} {model}")
    return result


def interval_tables(results_root, loso_curves, model_layers, raw_loso_curves=None, log=None):
    """All bootstrap intervals, each piece cached separately (see `final_subspace_data.cached`).

    Returns `{"internal", "loso", "auc", "roi", "raw_fmri": {the same four}}`: saturation-rank
    intervals keyed `(dataset, model, method)` (log2), `(dataset, region, model, method)` for
    ROIs (log2), and AUC-difference intervals keyed `(target, source, model, comparison)`. The
    `raw_fmri` entry is the same for raw fMRI r^2 and only has fMRI-target keys. Everything is
    computed from `load_pair_products`, so no result is read here.
    """
    from src.utils.atlas.core import bihemispheric_region_groups
    from src.utils.plotting.final_data import load_pair_products
    from src.utils.plotting.final_subspace_data import cached

    regions = list(bihemispheric_region_groups())
    products = lambda: load_pair_products(results_root)
    saturation = cached(results_root, "saturation_cis_v3", lambda: saturation_intervals(
        products(), loso_curves, model_layers, raw_loso_curves, log=log))
    auc = cached(results_root, "auc_cis_v3", lambda: auc_intervals(products(), model_layers, log))
    roi = cached(results_root, "roi_cis_v3", lambda: roi_intervals(
        products(), regions, model_layers, log=log))
    only_fmri = lambda table: {key: value for key, value in table.items() if key[0] == "fmri"}
    return {
        **saturation["normalized"], "auc": auc["normalized"], "roi": roi["normalized"],
        "raw_fmri": {
            "internal": only_fmri(saturation["raw_fmri"]["internal"]),
            "loso": only_fmri(saturation["raw_fmri"]["loso"]),
            "auc": only_fmri(auc["raw_fmri"]),
            "roi": only_fmri(roi["raw_fmri"]),
        },
    }

"""Per-ROI explained-variance saturation: curve loading and the two ROI figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_rrr import _dataset_adapter_name
from src.core.registry import get_dataset_adapter
from src.utils.atlas.core import bihemispheric_region_groups
from src.utils.plotting.final_axes import (
    add_bracket_headroom, finish_axis, normalized_curve_axis, paper_style, with_model_legend,
)
from src.utils.plotting.final_data import (
    DATASETS, LABELS, PAIRS, _ev_scale, _fisher, _lagged, _load_matrix, _rrr,
)
from src.utils.plotting.final_figures import (
    _saturation_comparison_axis, _saturation_values,
)
from src.utils.rrr.plot_metrics import explained_variance_curve

METHODS = ("rrr", "pca")


def load_roi_curves(results_root, data_roots, regions=None, pairs=PAIRS,
                    fmri_noise_normalized=True):
    """Mean-r^2-per-target rank curves for every dataset, ROI, model layer and method.

    Same definition as the final-plots explained-variance curves (folds Fisher-averaged,
    negatives clipped, squared, averaged over all targets / electrode-lag pairs; fMRI voxels
    divided by their squared noise ceiling unless `fmri_noise_normalized=False`), but
    over the targets inside one bihemispheric ROI. `data_roots` maps each dataset key
    to its raw data root. Returns `(curves, counts, gaps)`: `curves[(dataset, region,
    model, layer, method)] = (ranks, curve[:, None])`, `counts[(dataset, region)]` the
    number of targets (electrodes/voxels) in the ROI, and a list of missing results.
    """
    groups = bihemispheric_region_groups()
    regions = list(groups) if regions is None else list(regions)
    curves, counts, gaps = {}, {}, []
    for dataset in DATASETS:
        adapter = get_dataset_adapter(_dataset_adapter_name(DATASETS[dataset]))
        for model, layer in pairs:
            try:
                artifact = _rrr(results_root, "internal", dataset, model, layer
                                ).load_artifacts(["target_manifest"])
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"{(dataset, model, layer)}: {error}")
                continue
            for method in METHODS:
                metric = ("corr_per_target" if dataset == "fmri" else
                          "nr_orth_corr_per_target" if method == "rrr" else "pca_corr_per_target")
                matrix = _load_matrix(artifact, method, metric)
                if matrix is None:
                    continue
                targets = matrix.targets.copy()
                targets[["x", "y", "z"]] = adapter.get_electrode_coordinates(
                    targets, str(data_roots[dataset]))
                labels = adapter.get_target_atlas_labels(targets, str(data_roots[dataset]))
                values = _lagged(matrix)
                scale = _ev_scale(results_root, dataset, matrix.targets, fmri_noise_normalized)
                for region in regions:
                    keep = np.isin(labels, list(groups[region]))
                    counts[(dataset, region)] = int(keep.sum())
                    if not keep.any():
                        continue
                    curve = explained_variance_curve(
                        _fisher(values[:, :, keep], axis=1),
                        scale=None if scale is None else scale[keep][:, None])
                    curves[(dataset, region, model, layer, method)] = (
                        np.asarray(matrix.ranks, dtype=float), curve[:, None])
    return curves, counts, gaps


def _panel_title(dataset, count):
    """Dataset name with the number of targets in the region (voxels for fMRI, else electrodes)."""
    unit = "voxels" if dataset == "fmri" else "electrodes"
    return f"{LABELS[dataset]}, n = {count:,} {unit}"


def _grid(regions, title):
    figure, axes = plt.subplots(len(regions), len(DATASETS),
                                figsize=(4.6 * len(DATASETS), 3.1 * len(regions)),
                                squeeze=False)
    figure.suptitle(title)
    return figure, axes


def roi_normalized_curves(curves, counts, regions):
    """ROI rows x dataset columns: each model layer's normalized curve plus the mean."""
    with paper_style():
        figure, axes = _grid(regions, "Normalized mean explained variance by ROI")
        for row, region in enumerate(regions):
            for column, dataset in enumerate(DATASETS):
                ax = axes[row, column]
                members = {method: [value for key, value in curves.items()
                                    if key[:2] == (dataset, region) and key[-1] == method]
                           for method in METHODS}
                if not any(members.values()):
                    ax.text(.5, .5, "no targets", ha="center", va="center",
                            transform=ax.transAxes, color="#64748b")
                    finish_axis(ax)
                else:
                    normalized_curve_axis(
                        ax, {m: [(r, c[:, 0]) for r, c in v] for m, v in members.items()})
                ax.set_title(_panel_title(dataset, counts.get((dataset, region), 0)), fontsize=14)
                if column == 0:
                    ax.set_ylabel(f"{region}\nNormalized performance")
                else:
                    ax.set_ylabel("")
                if row != len(regions) - 1:
                    ax.set_xlabel("")
        handles, labels = next((a.get_legend_handles_labels() for a in axes.flat
                                if a.get_legend_handles_labels()[0]), ([], []))
        figure.legend(handles, labels, frameon=False, loc="upper right", ncol=2)
        figure.tight_layout(rect=(0, 0, 1, .97))
        return figure


@with_model_legend
def roi_saturation_distributions(curves, regions, counts=None, *, fraction=.99,
                                 separate_model=None, intervals=None):
    """ROI rows x dataset columns: paired RRR vs PCA saturation ranks over model layers.

    `counts` (`(dataset, region)` -> number of targets) puts each region's electrode / voxel
    count in the panel titles. `separate_model` draws that model's layers as their own pair of violins.
    `intervals` maps `(dataset, region, model, method)` to a log2 saturation-rank interval, drawn as
    a whisker through each model-mean square (missing for regions with too few subjects / probes).
    """
    with paper_style():
        figure, axes = _grid(
            regions, f"Saturation rank by ROI ({fraction * 100:g}% of maximum explained variance)")
        for row, region in enumerate(regions):
            for column, dataset in enumerate(DATASETS):
                ax = axes[row, column]
                ranks = _saturation_values(
                    {(dataset, *key[2:]): value for key, value in curves.items()
                     if key[:2] == (dataset, region)},
                    dataset, METHODS, fraction, saturation_mode="explained_variance")
                panel_intervals = None
                if intervals:
                    panel_intervals = {
                        label: {model: interval for (d, r, model, m), interval in intervals.items()
                                if (d, r, m) == (dataset, region, method)}
                        for method, label in zip(METHODS, ["RRR", "PCA"])}
                _saturation_comparison_axis(
                    ax, ranks, list(METHODS), ["RRR", "PCA"], [("RRR", "PCA")],
                    show_mean_value=separate_model is None, separate_model=separate_model,
                    intervals=panel_intervals)
                if separate_model is not None:
                    add_bracket_headroom(ax, 1)
                ax.set_title(LABELS[dataset] if counts is None else
                             _panel_title(dataset, counts.get((dataset, region), 0)), fontsize=14)
                if column == 0:
                    ax.set_ylabel(f"{region}\nSaturation rank")
                else:
                    ax.set_ylabel("")
        figure.tight_layout(rect=(0, 0, 1, .97))
        return figure

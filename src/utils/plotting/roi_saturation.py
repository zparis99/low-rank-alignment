"""Per-ROI explained-variance saturation: curve loading and the two ROI figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.utils.atlas.core import bihemispheric_region_groups
from src.utils.plotting.final_axes import (
    add_bracket_headroom, legend_anchor_y, thin_log2_rank_ticks, finish_axis, normalized_curve_axis, paper_style, with_model_legend,
)
from src.utils.plotting.final_data import DATASETS, LABELS, PAIRS, load_pair_products
from src.utils.plotting.final_figures import (
    _saturation_comparison_axis, _saturation_values,
)

METHODS = ("rrr", "pca")


def roi_curve_tables(products, regions=None, pairs=PAIRS):
    """Mean-r^2-per-target rank curves for every dataset, ROI, model layer and method.

    Same definition as the final-plots explained-variance curves (folds Fisher-averaged,
    negatives clipped, squared, averaged over all targets / electrode-lag pairs; fMRI voxels
    divided by their squared noise ceiling), but over the targets inside one bihemispheric ROI.
    `products` is `final_data.load_pair_products(results_root)` (the curves are derived while
    the results are read for the main figures). Returns `(curves, raw_fmri_curves, counts, gaps)`:
    `curves[(dataset, region, model, layer, method)] = (ranks, curve[:, None])`,
    `raw_fmri_curves` the same for fMRI with raw r^2 (no noise-ceiling division),
    `counts[(dataset, region)]` the number of targets (electrodes/voxels) in the ROI, and a list
    of missing results.
    """
    regions = list(bihemispheric_region_groups()) if regions is None else list(regions)
    curves, raw_curves, counts, gaps = {}, {}, {}, []
    for dataset in DATASETS:
        for model, layer in pairs:
            internal = products.get((dataset, model, layer), {}).get("internal")
            if not internal:
                gaps.append(f"{(dataset, model, layer)}: no internal results")
                continue
            for method in METHODS:
                entry = internal.get(method)
                if entry is None:
                    continue
                ranks = entry["ranks"]
                for variant, target in (("normalized", curves), ("raw_fmri", raw_curves)):
                    for region, curve in entry["roi_curves"].get(variant, {}).items():
                        if region in regions:
                            target[(dataset, region, model, layer, method)] = (ranks, curve[:, None])
                counts.update({(dataset, region): count
                               for region, count in entry["roi_counts"].items()
                               if region in regions})
    return curves, raw_curves, counts, gaps


def load_roi_curves(results_root, data_roots=None, regions=None, pairs=PAIRS):
    """`(curves, counts, gaps)` of `roi_curve_tables` for the results under `results_root`.

    `data_roots` is accepted for compatibility and unused: the ROI products are built from the
    standard `data/` layout next to `results/` (see `final_data.load_pair_products`).
    """
    curves, _, counts, gaps = roi_curve_tables(load_pair_products(results_root), regions, pairs)
    return curves, counts, gaps


def _panel_title(dataset, count):
    """Dataset name with the number of targets in the region (voxels for fMRI, else electrodes)."""
    unit = "voxels" if dataset == "fmri" else "electrodes"
    return f"{LABELS[dataset]}, n = {count:,} {unit}"


def _grid(regions, title, datasets=tuple(DATASETS)):
    figure, axes = plt.subplots(len(regions), len(datasets),
                                figsize=(4.6 * len(datasets), 3.1 * len(regions)),
                                sharey=True, squeeze=False)  # one y scale to compare all panels
    figure.suptitle(title)
    return figure, axes


def roi_normalized_curves(curves, counts, regions, datasets=tuple(DATASETS)):
    """ROI rows x dataset columns: each model layer's normalized curve plus the mean."""
    with paper_style():
        figure, axes = _grid(regions, "Normalized mean explained variance by ROI", datasets)
        for row, region in enumerate(regions):
            for column, dataset in enumerate(datasets):
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
                ax.set_xticks([0, .01, .1, 1], ["0", "1%", "10%", "100%"])  # 0.1% crowds the 0 tick
                ax.set_title(_panel_title(dataset, counts.get((dataset, region), 0)), fontsize=14)
                ax.set_ylabel(region if column == 0 else "")
                if row != len(regions) - 1:
                    ax.set_xlabel("")
        handles, labels = next((a.get_legend_handles_labels() for a in axes.flat
                                if a.get_legend_handles_labels()[0]), ([], []))
        figure.supylabel("Normalized performance", fontsize=17)
        figure.tight_layout(rect=(0, 0, 1, .97))
        figure.legend(handles, labels, frameon=False, fontsize=13, ncol=len(handles),
                      loc="upper center", bbox_to_anchor=(.5, legend_anchor_y(figure)), handlelength=1.6,
                      columnspacing=1.0, handletextpad=.4, borderaxespad=0)
        return figure


@with_model_legend
def roi_saturation_distributions(curves, regions, counts=None, *, fraction=.99,
                                 separate_model=None, intervals=None, datasets=tuple(DATASETS)):
    """ROI rows x dataset columns: paired RRR vs PCA saturation ranks over model layers.

    `counts` (`(dataset, region)` -> number of targets) puts each region's electrode / voxel
    count in the panel titles. `separate_model` draws that model's layers as their own pair of violins.
    `intervals` maps `(dataset, region, model, method)` to a log2 saturation-rank interval, drawn as
    a whisker through each model-mean square (missing for regions with too few subjects / probes).
    """
    with paper_style():
        figure, axes = _grid(
            regions, f"Saturation rank by ROI ({fraction * 100:g}% of maximum explained variance)",
            datasets)
        for row, region in enumerate(regions):
            for column, dataset in enumerate(datasets):
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
                ax.set_title(LABELS[dataset] if counts is None else
                             _panel_title(dataset, counts.get((dataset, region), 0)), fontsize=14)
                if column == 0:
                    ax.set_ylabel(f"{region}\nSaturation rank")
                else:
                    ax.set_ylabel("")
        if separate_model is not None:
            add_bracket_headroom(axes[0, 0], 1)  # once: the y axes are shared
        thin_log2_rank_ticks(axes[0, 0])  # short shared panels: avoid overlapping tick labels
        figure.tight_layout(rect=(0, 0, 1, .97))
        return figure

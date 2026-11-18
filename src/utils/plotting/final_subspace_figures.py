"""Paper figures for noise-ceiling distributions, PCA dimensionality and subspace consistency.

Every function takes the plain data from `final_subspace_data` and returns a Matplotlib
figure drawn in the final-plot style (`paper_style`, `finish_axis`, shared colors and
dataset labels). Datasets without results get a labelled empty panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from src.utils.plotting.final_axes import (
    COLORS, MODEL_COLORS, MODEL_NAMES, finish_axis, format_p_value, legend_anchor_y, paper_style,
)
from src.utils.plotting.final_data import DATASETS, LABELS
from src.utils.plotting.final_subspace_data import PCA_THRESHOLDS
from src.utils.rrr.plot_metrics import mean_by_model, paired_sign_flip

THRESHOLD_COLORS = ("#E4A02B", "#dc2626", "#7B4B94")
MEDIAN_COLOR = "#dc2626"
CURVE_COLOR = "#4b5563"
CHANCE = {"color": "black", "linestyle": (0, (1, 2)), "linewidth": 1.4}
SATURATION = {"color": "#dc2626", "linestyle": "--", "linewidth": 1.6}
TICKS, SMALL = 15, 13  # type sizes inside the dense per-layer grids


def _panel_title(model, layer):
    name = MODEL_NAMES.get(model, model)
    return name if layer is None else f"{name}, layer {layer}"


def _missing(ax, text="Not available"):
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes,
            color="#64748b", fontsize=SMALL + 1)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)


def _layer_grid(model_layers, *, sharex=False):
    columns = max(len(layers) for _, layers in model_layers)
    figure, axes = plt.subplots(
        len(model_layers), columns, figsize=(3.4 * columns, 2.7 * len(model_layers)),
        sharex=sharex, sharey=True, squeeze=False)
    for row, (_, layers) in zip(axes, model_layers):
        for ax in row[len(layers):]:
            ax.axis("off")
    return figure, axes


def noise_ceiling_histograms(fractions, model_layers, *, x_max=1.5):
    """Per-voxel fraction of explainable variance captured, one histogram per model layer."""
    bins = np.linspace(0.0, x_max, 61)
    with paper_style():
        figure, axes = _layer_grid(model_layers, sharex=True)
        for row, (model, layers) in zip(axes, model_layers):
            for ax, layer in zip(row, layers):
                values = fractions[(model, layer)]
                ax.hist(np.clip(values, None, x_max), bins=bins,
                        weights=np.full(len(values), 1.0 / len(values)),
                        color=COLORS["rrr"], edgecolor="white", linewidth=.3)
                ax.axvline(np.median(values), color=MEDIAN_COLOR, linewidth=1.6)
                ax.set_title(_panel_title(model, layer), fontsize=14, fontweight="bold")
                ax.text(.97, .95, f"Median: {np.median(values):.2f}\n$>$ 1: {np.mean(values > 1):.1%}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=SMALL)
                ax.tick_params(labelsize=TICKS, labelbottom=True)
                finish_axis(ax)
        figure.supxlabel(r"Fraction of explainable variance captured, $r^2 / CC_{max}^2$"
                         " (values above 1.5 in last bin)", fontsize=16)
        figure.supylabel("Fraction of voxels", fontsize=16)
        figure.tight_layout()
    return figure


def _threshold_curve(ax, cumulative, needed, *, line_width=2):
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=CURVE_COLOR, linewidth=line_width)
    for (threshold, count), color in zip(needed.items(), THRESHOLD_COLORS):
        ax.axvline(count, color=color, linestyle="--", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, 1.02)
    finish_axis(ax)


def neural_pca_figure(summaries):
    """Cumulative variance of the neural targets' PCA, one panel per dataset.

    `summaries` maps dataset key to a PCA summary (`cumulative`, `needed`) or None.
    """
    with paper_style():
        figure, axes = plt.subplots(1, len(DATASETS), figsize=(5.6 * len(DATASETS), 4.6),
                                    sharey=True, squeeze=False)
        for ax, dataset in zip(axes[0], DATASETS):
            ax.set_title(LABELS[dataset])
            summary = summaries.get(dataset)
            if summary is None:
                _missing(ax)
                continue
            _threshold_curve(ax, summary["cumulative"], summary["needed"])
            for (threshold, count), color in zip(summary["needed"].items(), THRESHOLD_COLORS):
                ax.plot([], [], color=color, linestyle="--",
                        label=f"{threshold:.0%} of variance: {count}")
            ax.legend(frameon=False, loc="upper left", fontsize=14)
            ax.set_xlabel("Number of principal components")
        axes[0, 0].set_ylabel("Cumulative fraction\nof variance explained")
        figure.tight_layout()
    return figure


def embedding_pca_figure(result, model_layers, dataset):
    """Cumulative variance of every model layer's embedding PCA over one dataset's words."""
    with paper_style():
        figure, axes = _layer_grid(model_layers)
        for row, (model, layers) in zip(axes, model_layers):
            for ax, layer in zip(row, layers):
                summary = result["summaries"][(model, layer)]
                ax.set_title(_panel_title(model, layer), fontsize=14, fontweight="bold")
                if summary is None:
                    _missing(ax)
                    continue
                _threshold_curve(ax, summary["cumulative"], summary["needed"], line_width=1.8)
                # Upper left: the curve is low there, and it keeps the text off the dashed lines.
                ax.text(.04, .96, "\n".join(f"{t:.0%}: {n}" for t, n in summary["needed"].items()),
                        transform=ax.transAxes, ha="left", va="top", fontsize=SMALL)
                ax.set_title(_panel_title(model, layer), fontsize=14, fontweight="bold")
                ax.tick_params(labelsize=TICKS)
        thresholds = ", ".join(f"{t:.0%}" for t in PCA_THRESHOLDS)
        figure.suptitle(f"{LABELS[dataset]} words (n = {result['n_words']:,})", fontsize=18)
        figure.supxlabel(f"Number of principal components (dashed lines: {thresholds} of variance)",
                         fontsize=16)
        figure.supylabel("Cumulative fraction of variance explained", fontsize=16)
        figure.tight_layout()
    return figure


# --- subspace consistency -------------------------------------------------------------------


def _rank_axis(ax):
    ax.set_xscale("log", base=2)
    finish_axis(ax)
    ax.grid(axis="x", alpha=.20)


def _legend_below(ax, ncol=1, y=-.22):
    ax.legend(frameon=False, fontsize=14, loc="upper center", bbox_to_anchor=(.5, y), ncol=ncol,
              columnspacing=1.0, handlelength=1.6, handletextpad=.5)


def subspace_consistency_figure(data, sample_model, sample_layer):
    """Overlap vs rank: LOSO (top row, one panel per dataset), cross-dataset (bottom row)."""
    datasets = list(DATASETS)
    ranks = np.array([2 ** p for p in range(int(np.log2(data["dim"])) + 1)])
    chance = ranks / data["dim"]
    with paper_style():
        figure, axes = plt.subplots(2, len(datasets), figsize=(6.2 * len(datasets), 10),
                                    sharey=True)
        for ax, dataset in zip(axes[0], datasets):
            ax.set_title(f"{LABELS[dataset]} LOSO")
            frame = data["loso"].get(dataset)
            if frame is None:
                _missing(ax)
                continue
            for method, group in frame.groupby("method"):
                per_rank = group.groupby("rank")["overlap"]
                mean, sem = per_rank.mean(), per_rank.sem()
                ax.plot(mean.index, mean.values, "o-", color=COLORS[method], linewidth=2,
                        markersize=5, label=method)
                ax.fill_between(mean.index, mean - sem, mean + sem, color=COLORS[method],
                                alpha=.2, linewidth=0)
            ax.plot(ranks, chance, label="Chance ($r/d$)", **CHANCE)
            saturation = data["saturation"].get(dataset)
            if saturation is not None:
                ax.axvline(saturation, label=f"Saturation rank: {saturation:.0f}", **SATURATION)
            _rank_axis(ax)
            _legend_below(ax, ncol=2, y=-.12)
        pairs = (("podcast", "cars-2"), ("podcast", "fmri"), ("cars-2", "fmri"))
        colors = {"PCA vs PCA": (COLORS["PCA"], "-"), "RRR vs RRR": (COLORS["RRR"], "-")}
        for ax, (first, second) in zip(axes[1], pairs):
            ax.set_title(f"{LABELS[first]} vs. {LABELS[second]}")
            frame = data["cross"]
            frame = None if frame is None else frame[frame["pair"] == f"{first} - {second}"]
            if frame is None or frame.empty:
                _missing(ax)
                continue
            styles = {**colors,
                      f"RRR ({first}) vs PCA ({second})": ("#E4A02B", "--"),
                      f"RRR ({second}) vs PCA ({first})": ("#7B4B94", ":")}
            for label, group in frame.groupby("comparison", sort=False):
                color, linestyle = styles[label]
                ax.plot(group["rank"], group["overlap"], marker="o", markersize=5, color=color,
                        linestyle=linestyle, linewidth=2,
                        label=label.replace(f"({first})", f"[{LABELS[first]}]").replace(f"({second})", f"[{LABELS[second]}]"))
            ax.plot(ranks, chance, label="Chance ($r/d$)", **CHANCE)
            _rank_axis(ax)
            ax.set_xlabel("Rank $k$")
            _legend_below(ax)
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean squared cosine\n(subspace overlap)")
        for ax in axes[:, 1:].flat:
            ax.set_ylabel("")
        ax_ylim = (0, 1)
        for ax in axes.flat:
            ax.set_ylim(ax_ylim)
        figure.suptitle(f"Subspace consistency: {MODEL_NAMES.get(sample_model, sample_model)}, "
                        f"layer {sample_layer}")
        figure.tight_layout(h_pad=2.5)
    return figure


def loso_principal_angle_figure(data, sample_model, sample_layer):
    """Squared cosine of every principal angle per held-out subject (smallest angle first).

    One row per dataset, one column per rank. The mean of each curve is the subspace overlap
    shown in the consistency figure; the dotted line is its chance level, $r/d$.
    """
    datasets, ranks = list(DATASETS), data["ranks"]
    with paper_style():
        figure, axes = plt.subplots(len(datasets), len(ranks), sharey=True, squeeze=False,
                                    figsize=(4.3 * len(ranks), 3.7 * len(datasets)))
        for row, dataset in zip(axes, datasets):
            vectors = data["cosines"].get(dataset)
            for ax, rank in zip(row, ranks):
                ax.set_title(f"{LABELS[dataset]}, rank {rank}", fontsize=16)
                if vectors is None:
                    _missing(ax)
                    continue
                x = np.arange(1, rank + 1)
                for method in ("PCA", "RRR"):
                    cosines = vectors[(method, rank)]  # [subjects, rank]
                    for subject_cosines in cosines:
                        ax.plot(x, subject_cosines, color=COLORS[method], alpha=.3, linewidth=.9)
                    ax.plot(x, np.median(cosines, axis=0), color=COLORS[method], linewidth=2.6,
                            label=f"{method} (median)")
                ax.axhline(rank / data["dim"], label="Chance ($r/d$)", **CHANCE)
                ax.set_ylim(0, 1.02)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                finish_axis(ax)
                ax.grid(axis="x", alpha=.20)
            row[0].set_ylabel("Squared cosine of\nprincipal angle")
        for ax in axes[-1]:
            ax.set_xlabel("Angle index (smallest first)")
        handles, labels = next((a.get_legend_handles_labels() for a in axes.flat
                                if a.get_legend_handles_labels()[0]), ([], []))
        figure.suptitle(f"LOSO principal angles: {MODEL_NAMES.get(sample_model, sample_model)}, "
                        f"layer {sample_layer} (thin lines: held-out subjects)")
        figure.tight_layout(rect=(0, 0, 1, .96))
        figure.legend(handles, labels, frameon=False, fontsize=14, ncol=len(handles),
                      loc="upper center", bbox_to_anchor=(.5, legend_anchor_y(figure)),
                      columnspacing=1.2, handlelength=1.6, handletextpad=.4)
    return figure


def _layer_summaries(summaries, dataset):
    return {key: value for key, value in summaries.items() if key[0] == dataset}


def _model_legend(ax, models, extra=(), loc=None):
    """One legend under the whole figure (call after the layout is final), so it covers no curve."""
    for model in MODEL_COLORS:
        if model in models:
            ax.plot([], [], color=MODEL_COLORS[model], linewidth=2.2, label=MODEL_NAMES[model])
    for kwargs in extra:
        ax.plot([], [], **kwargs)
    figure = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=14, ncol=len(handles),
                  loc="upper center", bbox_to_anchor=(.5, legend_anchor_y(figure)),
                  columnspacing=1.2, handlelength=1.6, handletextpad=.4)


def _mean_curve(stack):
    stack = np.array(stack)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0)
    return mean, np.isfinite(stack).sum(axis=0) >= max(3, len(stack) // 2)


def aligned_delta_figure(summaries, grid=np.arange(-5, 5.01, .25)):
    """RRR - PCA excess overlap vs log2(rank / saturation rank), one panel per dataset."""
    with paper_style():
        figure, axes = plt.subplots(1, len(DATASETS), figsize=(5.8 * len(DATASETS), 4.8), sharey=True)
        shown = set()
        for ax, dataset in zip(axes, DATASETS):
            members = _layer_summaries(summaries, dataset)
            ax.set_title(f"{LABELS[dataset]} LOSO" + (f" (n = {len({k[1] for k in members})} models)" if members else ""))
            if not members:
                _missing(ax)
                continue
            by_model = {}
            for (_, model, _), s in members.items():
                x = np.log2(s["ranks"] / s["saturation"])
                curve = np.interp(grid, x, s["rrr"] - s["pca"], left=np.nan, right=np.nan)
                by_model.setdefault(model, []).append(curve)
                ax.plot(grid, curve, color=MODEL_COLORS[model], alpha=.4, linewidth=1)
                shown.add(model)
            # Layers of one model are not independent: average within each model, then across.
            with np.errstate(all="ignore"):
                stack = [np.nanmean(curves, axis=0) for curves in by_model.values()]
            mean, enough = _mean_curve(stack)
            ax.plot(grid[enough], mean[enough], color="black", linewidth=2.8, label="Mean over models")
            ax.axhline(0, color="#64748b", linewidth=1)
            ax.axvline(0, label="Saturation rank", **SATURATION)
            ax.set_xlabel("log$_2$(rank / saturation rank)")
            finish_axis(ax)
        axes[0].set_ylabel("RRR $-$ PCA excess overlap\nover chance")
        figure.tight_layout()
        _model_legend(axes[0], shown)
    return figure


def saturation_split_figure(table, separate_model=None):
    """Mean RRR - PCA excess overlap below vs above the saturation rank, one line per layer.

    With `separate_model`, that model's layers get their own mean (dashed, offset) and tests.
    """
    with paper_style():
        figure, axes = plt.subplots(1, len(DATASETS), figsize=(5.8 * len(DATASETS), 5), sharey=True)
        shown = set()
        for ax, dataset in zip(axes, DATASETS):
            frame = table[table["dataset"] == dataset].dropna(subset=["below", "above"])
            ax.set_title(f"{LABELS[dataset]} LOSO" + (f" (n = {frame['model'].nunique()} models)"
                                                       if len(frame) else ""))
            if frame.empty:
                _missing(ax)
                continue
            for _, row in frame.iterrows():
                ax.plot([0, 1], [row["below"], row["above"]], color=MODEL_COLORS[row["model"]],
                        marker="o", markersize=4, alpha=.5, linewidth=1.1)
                shown.add(row["model"])
            ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
            is_separate = frame["model"] == separate_model
            for subset, color, linestyle, top in ((frame[~is_separate], "black", "-", 1.0),
                                                  (frame[is_separate],
                                                   MODEL_COLORS.get(separate_model), "--", .9)):
                if subset.empty:
                    continue
                # Layers of one model are not independent: one value per model (mean over its
                # layers) feeds the group mean and the sign-flip tests.
                names, below = mean_by_model(subset["below"], subset["model"])
                _, above = mean_by_model(subset["above"], subset["model"])
                for name, low, high in zip(names, below, above):
                    ax.plot([0, 1], [low, high], color=MODEL_COLORS[name], marker="s",
                            markersize=7, markeredgecolor="black", linewidth=2, zorder=4)
                ax.plot([0, 1], [below.mean(), above.mean()], color=color, linestyle=linestyle,
                        marker="D", linewidth=2.8, zorder=5)
                if len(names) < 2:
                    ax.text(.5, top, "no test (1 model)", color=color, ha="center", va="top",
                            transform=ax.get_xaxis_transform(), fontsize=14)
                    continue
                zeros = np.zeros(len(names))
                for x, values, direction, sign in ((0, below, "greater", ">"),
                                                   (1, above, "less", "<")):
                    p = paired_sign_flip(values, zeros, alternative=direction)
                    ax.text(x, top, f"{sign} 0: {format_p_value(p)}", color=color,
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=14)
            ax.set_xticks([0, 1], ["Below\nsaturation", "Above\nsaturation"])
            ax.set_xlim(-.4, 1.4)
            finish_axis(ax)
        axes[0].set_ylabel("Mean RRR $-$ PCA excess\noverlap over chance")
        extra = [{"color": "black", "marker": "D", "label": "Mean over models"}]
        if separate_model is not None:
            extra.append({"color": MODEL_COLORS[separate_model], "marker": "D", "linestyle": "--",
                          "label": f"{MODEL_NAMES[separate_model]} mean"})
        figure.tight_layout()
        _model_legend(axes[0], shown, extra)
    return figure

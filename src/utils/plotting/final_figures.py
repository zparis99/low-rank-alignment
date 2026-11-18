"""Compositions for the curated paper figures."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from src.utils.plotting.final_axes import (
    COLORS, MODEL_COLORS, MODEL_NAMES, add_decoding_gradient, correlation_scatter_axis, decoding_relative_axis,
    distribution_axis, finish_axis,
    format_p_value,
    normalized_curve_axis, pairwise_test_brackets, paper_style, quantile_axis, rank_curve_axis,
    add_bracket_headroom, separate_color, separate_group_captions, separate_label,
    target_bar_axis, thin_log2_rank_ticks, with_model_legend, legend_anchor_y,
)
from src.utils.plotting.final_data import DATASETS, LABELS, QUANTILES
from scripts.utilities.check_final_significance import MODEL_LAYERS
from src.utils.rrr.plot_metrics import (
    PEAK_FLOOR, average_normalized_quantiles, explained_variance_saturation_rank,
    fisher_mean, model_sign_flip, normalized_log_rank_auc, paired_sign_flip,
    quantile_saturation_rank,
)

# How each saturation mode reads its curve tables and measures saturation.
# "quantile": correlation curves, saturation from target-quantile curves.
# "explained_variance": mean r^2 per target curves; saturation is the rank reaching a
# fraction of the curve's own maximum.
MODES = {
    "quantile": {"column": "correlation", "aggregate": fisher_mean,
                 "ylabel": "Correlation"},
    "explained_variance": {"column": "explained_variance", "aggregate": "mean",
                           "ylabel": "Mean $r^2$\nper target"},
}


def _mode(saturation_mode):
    try:
        return MODES[saturation_mode]
    except KeyError:
        raise ValueError(f"saturation_mode must be one of {tuple(MODES)}, "
                         f"got {saturation_mode!r}") from None


def _saturation_rank(saturation_mode, ranks, matrix, fraction, min_peak=PEAK_FLOOR):
    """Saturation rank of one stored curve table entry under the chosen mode."""
    if saturation_mode == "explained_variance":
        return explained_variance_saturation_rank(ranks, np.asarray(matrix)[:, 0], fraction)
    return quantile_saturation_rank(ranks, matrix, fraction, min_peak)


def without_models(data, models, *, index=1):
    """`data` with every result of the given models removed.

    `data` may be a DataFrame with a `model` column, a dict keyed by tuples holding the model
    at position `index` (default 1: `(dataset, model, ...)`), a list of such tuples, or a
    NamedTuple of those (for example `ExplainedVarianceData`).
    """
    if data is None:
        return None
    models = set(models)
    if hasattr(data, "_fields"):
        return data._replace(**{name: without_models(getattr(data, name), models, index=index)
                                for name in data._fields})
    if isinstance(data, pd.DataFrame):
        return data[~data["model"].isin(models)]
    if isinstance(data, dict):
        return {key: value for key, value in data.items() if key[index] not in models}
    return [row for row in data if row[index] not in models]


def _split_model(ranks, model):
    """Split `{method: [((model, layer), rank), ...]}` into (others, the given model's)."""
    others = {m: [row for row in rows if row[0][0] != model] for m, rows in ranks.items()}
    chosen = {m: [row for row in rows if row[0][0] == model] for m, rows in ranks.items()}
    return others, chosen


def _layer(value):
    return None if pd.isna(value) else int(value)


def _pair_rows(frame, model, layer):
    selected = frame[frame["model"].eq(model)]
    return selected[selected["layer"].isna() if layer is None else selected["layer"].eq(layer)]


def _curves(frame, saturation_mode="quantile"):
    settings = _mode(saturation_mode)
    result = {}
    for method, group in frame.groupby("method"):
        grouped = group.groupby("rank")[settings["column"]].agg(settings["aggregate"]).sort_index()
        result[method] = (grouped.index.to_numpy(), grouped.to_numpy())
    return result


def _members(frame, saturation_mode="quantile"):
    settings = _mode(saturation_mode)
    members = defaultdict(list)
    for (_, _, method), group in frame.groupby(["model", "layer", "method"], dropna=False):
        if "subject" in group:
            group = group.groupby(["method", "rank"], as_index=False)[settings["column"]].agg(
                settings["aggregate"])
        members[method].append(_curves(group, saturation_mode)[method])
    return members


def _saturation_values(quantiles, dataset, methods, fraction, *, min_peak=PEAK_FLOOR,
                       saturation_mode="quantile"):
    values = defaultdict(list)
    for key, (ranks, matrix) in quantiles.items():
        if key[0] != dataset or key[-1] not in methods:
            continue
        rank = _saturation_rank(saturation_mode, ranks, matrix, fraction, min_peak)
        if np.isfinite(rank):
            values[key[-1]].append((key[1:3], rank))
    return values


def _saturation_comparison_axis(ax, ranks, methods, labels, tests, *,
                                show_mean_value=False, separate_model=None,
                                tick_labels=None, intervals=None):
    """Draw paired model-layer ranks with the aggregate comparison style.

    With `separate_model`, that model's layers are drawn as their own set of violins (labelled
    '... (random)') right of the others, with their own paired tests; every other model forms
    the main set. `tick_labels` optionally maps a label to the text shown on the x axis.
    `intervals` optionally maps a label to `{model: (low, high)}` bootstrap intervals (log2 rank).
    """
    separate_ranks = None
    if separate_model is not None:
        ranks, separate_ranks = _split_model(ranks, separate_model)

    def paired_rows(source, row_labels):
        by_pair = {method: dict(source.get(method, [])) for method in methods}
        common = set.intersection(*(set(values) for values in by_pair.values()))
        pairs = sorted(common, key=str)
        rows = [{label: by_pair[method][pair] for method, label in zip(methods, row_labels)}
                for pair in pairs]
        return pairs, rows

    pairs, paired = paired_rows(ranks, labels)
    groups = {label: [row[label] for row in paired] for label in labels}
    models = {label: [pair[0] for pair in pairs] for label in labels}
    separate_labels = [separate_label(label) for label in labels]
    all_rows, colors = list(paired), {}
    if separate_ranks is not None:
        separate_pairs, separate_paired = paired_rows(separate_ranks, separate_labels)
        for label in separate_labels:
            groups[label] = [row[label] for row in separate_paired]
            models[label] = [pair[0] for pair in separate_pairs]
        all_rows += separate_paired
        colors = {separate_label(label): separate_color(COLORS.get(label, "#6ab2e2"))
                  for label in labels}
    model_intervals = None
    if intervals:
        model_intervals = {label: intervals.get(label.removesuffix(separate_label("")), {})
                           for label in groups}
    distribution_axis(ax, groups, log2_rank=True, paired=all_rows,
                      show_mean_value=show_mean_value, point_models=models,
                      label_colors=colors, model_intervals=model_intervals)
    shown = [text.get_text() for text in ax.get_xticklabels()]
    if separate_ranks is not None:
        # Both groups share the same tick names; captions say which group is which.
        base = [text.removesuffix(separate_label("")) for text in shown]
        ax.set_xticklabels([(tick_labels or {}).get(text, text) for text in base])
        n_main = sum(not text.endswith(separate_label("")) for text in shown)
        separate_group_captions(ax, n_main, len(shown), offset=50 if tick_labels else 38)
    elif tick_labels:
        ax.set_xticklabels([tick_labels.get(text, text) for text in shown])
    if not paired:
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_xlim(.45, len(labels) + .55)
        return

    def brackets(rows, row_pairs, row_tests):
        # One unit per model: each model's layers are averaged before the sign-flip test.
        models = [pair[0] for pair in row_pairs]
        p_values = np.asarray([
            model_sign_flip([np.log2(row[first]) for row in rows],
                            [np.log2(row[second]) for row in rows], models)
            for first, second in row_tests
        ])
        if not np.isfinite(p_values).all():
            return  # fewer than two models: no test
        order = np.argsort(p_values)
        adjusted = np.empty(len(p_values))
        adjusted[order] = np.maximum.accumulate(np.minimum(
            1.0, (len(p_values) - np.arange(len(p_values))) * p_values[order]
        ))
        pairwise_test_brackets(ax, list(groups), [
            (first, second, p) for (first, second), p in zip(row_tests, adjusted)
        ])

    brackets(paired, pairs, tests)
    if separate_ranks is not None and separate_paired:
        brackets(separate_paired, separate_pairs,
                 [(separate_label(first), separate_label(second)) for first, second in tests])


def _loso_saturation_values(quantiles, dataset, fraction, saturation_mode="quantile"):
    by_pair = defaultdict(list)
    for key, (ranks, matrix) in quantiles.items():
        if key[0] != dataset:
            continue
        rank = _saturation_rank(saturation_mode, ranks, matrix, fraction)
        if np.isfinite(rank):
            by_pair[(key[1], key[2], key[4])].append(rank)
    values = defaultdict(list)
    for (model, layer, method), ranks in by_pair.items():
        values[method].append(((model, layer), np.mean(ranks)))
    return values


@with_model_legend
def rrr_aggregate(internal, loso, quantiles, loso_quantiles, full_rows,
                  selected_pair, *, mode="internal", fraction=.99,
                  saturation_mode="quantile", separate_model=None, distributions_only=False,
                  intervals=None, fmri_noise_normalized=True, datasets=tuple(DATASETS)):
    """Two stacked curve columns beside the full-height rank distributions (one per dataset).

    With `saturation_mode="explained_variance"`, pass the explained-variance tables:
    `internal`/`loso` frames, the `curves`/`loso_curves` dictionaries in place of the
    quantile dictionaries, and the explained-variance `full_rows`.

    `separate_model` draws that model's layers as their own set of saturation violins next to
    the others. Without it every model in the inputs is pooled (drop models beforehand with
    `without_models`). `distributions_only` draws just the saturation distributions (panel C).
    `intervals` maps `(dataset, model, method)` to a bootstrap interval in log2 saturation rank;
    it is drawn as a whisker through each model-mean square. `fmri_noise_normalized` only
    labels the fMRI sample-curve panel (pass False for the raw-r^2 appendix data). `datasets`
    restricts the figure to those datasets (one row of curves and one distribution each).
    """
    settings = _mode(saturation_mode)
    count = len(datasets)
    with paper_style():
        if distributions_only:
            width = 19 if separate_model is not None and mode == 'loso' else 14
            figure, distribution_axes = plt.subplots(
                1, count, sharey=True, squeeze=False,
                figsize=(width if count == len(DATASETS) else max(6.5, width * count / 3), 6.6))
            distribution_axes = distribution_axes[0]
        else:
            figure = plt.figure(figsize=(19, 7.4 if count == len(DATASETS) else 4.6))
            grid = figure.add_gridspec(
                count, 3, width_ratios=[1.4, 1.4, 3] if mode == "internal" else [1.0, 1.0, 4],
                hspace=.55, wspace=.38,
            )
            saturation_grid = grid[:, 2].subgridspec(1, count, wspace=.10)
        source = internal if mode == "internal" else loso
        methods = ["rrr", "pca"] if mode == "internal" else [
            "RRR baseline", "RRR transfer", "PCA baseline"]
        average = fisher_mean if saturation_mode == "quantile" else np.nanmean
        baseline_map = {(d, m, l): float(average(v)) for d, m, l, v in full_rows}
        first_saturation_ax = first_sample_ax = first_mean_ax = None
        for row, dataset in enumerate(datasets):
            frame = source[source["dataset"] == dataset]
            sample = _pair_rows(frame, *selected_pair)
            sample_curves = _curves(sample, saturation_mode)
            if mode == "internal":
                ranks = _saturation_values(quantiles, dataset, methods, fraction,
                                           saturation_mode=saturation_mode)
                markers = {
                    method: _saturation_rank(
                        saturation_mode, *quantiles[(dataset, *selected_pair, method)], fraction)
                    for method in methods if (dataset, *selected_pair, method) in quantiles
                }
            else:
                ranks = _loso_saturation_values(loso_quantiles, dataset, fraction,
                                                saturation_mode)
                markers = {}
                for method in methods:
                    subject_ranks = [_saturation_rank(saturation_mode, *value, fraction)
                                     for key, value in loso_quantiles.items()
                                     if key[:3] == (dataset, *selected_pair) and key[-1] == method]
                    finite = [value for value in subject_ranks if np.isfinite(value)]
                    if finite:
                        markers[method] = float(np.mean(finite))
            if not distributions_only:
                sample_ax = figure.add_subplot(grid[row, 0])
                first_sample_ax = first_sample_ax or sample_ax
                rank_curve_axis(sample_ax, sample_curves, saturation=markers,
                                baseline=baseline_map.get((dataset, *selected_pair)),
                                ylabel=settings["ylabel"])
                sample_ax.set_title(LABELS[dataset], fontsize=15)
                if row != count - 1:
                    sample_ax.set_xlabel("")
                if row != count // 2:
                    sample_ax.set_ylabel("")
                mean_ax = figure.add_subplot(grid[row, 1])
                first_mean_ax = first_mean_ax or mean_ax
                normalized_curve_axis(mean_ax, _members(frame, saturation_mode))
                mean_ax.set_title(LABELS[dataset])
                mean_ax.set_xticks([0, .01, .1, 1], ["0", "1%", "10%", "100%"])
                if row != count - 1:
                    mean_ax.set_xlabel("")
                if row != count // 2:
                    mean_ax.set_ylabel("")
            distribution_ax = (distribution_axes[row] if distributions_only else
                               figure.add_subplot(saturation_grid[0, row],
                                                  sharey=first_saturation_ax))
            if first_saturation_ax is None:
                first_saturation_ax = distribution_ax
            labels = [method.upper() if mode == "internal" else method for method in methods]
            tests = [("RRR", "PCA")] if mode == "internal" else [
                ("RRR transfer", "PCA baseline"),
                ("RRR baseline", "RRR transfer"),
            ]
            tick_labels = None
            if mode == "loso":
                tick_labels = {"RRR baseline": "RRR\nbaseline", "RRR transfer": "RRR\ntransfer",
                               "PCA baseline": "PCA\nbaseline"}
            panel_intervals = None
            if intervals:
                panel_intervals = {
                    label: {model: interval for (d, model, m), interval in intervals.items()
                            if d == dataset and m == method}
                    for method, label in zip(methods, labels)}
            _saturation_comparison_axis(distribution_ax, ranks, methods, labels, tests,
                                        show_mean_value=separate_model is None,
                                        separate_model=separate_model,
                                        tick_labels=tick_labels, intervals=panel_intervals)
            if mode == "loso":
                distribution_ax.tick_params(axis="x", labelsize=14)
            if row != 0:
                distribution_ax.set_ylabel("")
                distribution_ax.tick_params(axis="y", labelleft=False)
            distribution_ax.set_title(LABELS[dataset])
        if separate_model is not None:
            add_bracket_headroom(distribution_ax, len(tests))
        peak_text = ("Peak Performance" if saturation_mode == "quantile"
                     else "Maximum Explained Variance")
        title = f"Saturation Distributions ({fraction * 100:g}% {peak_text})"
        if distributions_only:
            figure.suptitle(title, fontsize=18, y=.995)
            figure.tight_layout(rect=(0, 0, 1, .975))
            return figure
        # The explained-variance y label is two lines and needs a wider left margin.
        figure.subplots_adjust(left=.055 if saturation_mode == "quantile" else .075,
                               right=.985, top=.89, bottom=.10)
        sample_box = sample_ax.get_position()
        mean_box = mean_ax.get_position()
        # One legend for panels A and B below them, so it never covers a curve.
        entries = {}
        for legend_ax in (first_sample_ax, first_mean_ax):
            entries.update(zip(*reversed(legend_ax.get_legend_handles_labels())))
        figure.legend(list(entries.values()), list(entries), frameon=False, fontsize=13,
                      ncol=len(entries), loc="upper center",
                      bbox_to_anchor=((sample_box.x0 + mean_box.x1) / 2, legend_anchor_y(figure)),
                      handlelength=1.6, columnspacing=1.0, handletextpad=.4, borderaxespad=0)
        saturation_start = first_saturation_ax.get_position()
        saturation_end = distribution_ax.get_position()
        figure.text((sample_box.x0 + sample_box.x1) / 2, .965,
                    "A. Sample Rank Curves", ha="center",
                    va="bottom", fontsize=16, weight="bold")
        figure.text((mean_box.x0 + mean_box.x1) / 2, .965,
                    "B. Mean Convergence", ha="center",
                    va="bottom", fontsize=16, weight="bold")
        figure.text((saturation_start.x0 + saturation_end.x1) / 2, .965,
                    f"C. {title}",
                    ha="center", va="bottom", fontsize=16, weight="bold")
        return figure


def _cross_auc(frame, saturation_mode="quantile"):
    column = _mode(saturation_mode)["column"]
    aucs = defaultdict(dict)
    for (model, layer, method), group in frame.groupby(
        ["model", "layer", "method"], dropna=False
    ):
        value = normalized_log_rank_auc(group["rank"], group[column])
        if np.isfinite(value):
            aucs[(model, _layer(layer))][method] = value
    return aucs


@with_model_legend
def cross_aggregate(cross, saturation_mode="quantile", separate_model=None,
                    distributions_only=False, intervals=None, targets=tuple(DATASETS)):
    """Six directed panels and the three current normalized-AUC comparisons.

    `targets` keeps only the directed pairs whose target dataset is listed (for example just
    "fmri": podcast → fmri and cars-2 → fmri).

    `separate_model` draws that model's layers as their own AUC-difference violins instead of
    pooling them. `distributions_only` draws just those three distribution panels. `intervals`
    maps `(target, source, model, comparison title)` to a bootstrap interval in AUC-difference
    units, drawn as a whisker through each model-mean square.

    For `saturation_mode="explained_variance"` pass the explained-variance cross frame.
    """
    directed = [(t, s) for t in targets for s in DATASETS if s != t]
    with paper_style():
        if distributions_only:
            figure = plt.figure(figsize=(12 if len(directed) > 2 else 7, 12))
            right = figure.add_gridspec(3, 1, hspace=.30)
        else:
            figure = plt.figure(figsize=(21, 9))
            grid = figure.add_gridspec(3, 6)
            left = grid[:, :3].subgridspec(2, -(-len(directed) // 2), hspace=.28, wspace=.10)
            right = grid[:, 3:].subgridspec(3, 1, hspace=.30)
        comparisons = [
            ("Within RRR − Transfer RRR", "Within RRR", "Transfer RRR"),
            ("Transfer RRR − Transfer PCA", "Transfer RRR", "Transfer PCA"),
            ("Within PCA − Transfer PCA", "Within PCA", "Transfer PCA"),
        ]
        groups = [defaultdict(list) for _ in comparisons]
        group_models = [defaultdict(list) for _ in comparisons]
        separate_groups = [defaultdict(list) for _ in comparisons]
        separate_models = [defaultdict(list) for _ in comparisons]
        first_directed_ax = None
        directions = {}  # direction label -> (target, source)
        for index, (target, source) in enumerate(directed):
            panel = cross[(cross["target_dataset"] == target)
                          & (cross["source_dataset"] == source)]
            if not distributions_only:
                ax = figure.add_subplot(
                    left[index % 2, index // 2], sharey=first_directed_ax,
                )
                if first_directed_ax is None:
                    first_directed_ax = ax
                normalized_curve_axis(ax, _members(panel, saturation_mode))
                ax.set_title(f"{LABELS[source]} →\n{LABELS[target]}", fontsize=14)
                ax.set_xlabel("")
                if index // 2 != 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
                ax.set_xticks([0, .01, .1, 1], ["0", "1%", "10%", "100%"])
                if index % 2 == 1 and index // 2 > 0:
                    ax.set_xticks([0, .01, .1, 1], ["", "1%", "10%", "100%"])
                ax.tick_params(axis="both", labelsize=14)
                ax.tick_params(axis="x", labelbottom=index % 2 == 1)
                if index == 0:
                    first_curve_ax = ax
            label = f"{LABELS[source].split(' (')[0]} → {LABELS[target].split(' (')[0]}"
            directions[label] = (target, source)
            for (model, _), values in _cross_auc(panel, saturation_mode).items():
                is_separate = model == separate_model
                targets = (separate_groups, separate_models) if is_separate else (groups, group_models)
                for output, models, (_, first, second) in zip(*targets, comparisons):
                    if first in values and second in values:
                        output[label].append(values[first] - values[second])
                        models[label].append(model)
        for row, (output, (title, _, _)) in enumerate(zip(groups, comparisons)):
            ax = figure.add_subplot(right[row, 0])
            point_models = group_models[row]
            colors = {}
            for label, values in separate_groups[row].items():
                output[separate_label(label)] = values
                point_models[separate_label(label)] = separate_models[row][label]
                colors[separate_label(label)] = separate_color(COLORS["rrr"])
            model_intervals = None
            if intervals:
                model_intervals = {}
                for group in output:
                    target, source = directions[group.removesuffix(separate_label(""))]
                    model_intervals[group] = {
                        model: interval for (t, src, model, name), interval in intervals.items()
                        if (t, src, name) == (target, source, title)}
            distribution_axis(ax, output, show_points=True, point_models=point_models,
                              label_colors=colors, model_intervals=model_intervals)
            ax.axhline(0, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Normalized AUC difference" if row == 1 else "")
            ax.tick_params(axis="x", labelbottom=row == 2, labelsize=14)
            if separate_groups[row]:
                n_main = len(groups[row]) - len(separate_groups[row])
                separate_group_captions(ax, n_main, len(groups[row]), offset=95,
                                        captions=row == 2)
            if row == 2:
                if separate_groups[row]:
                    ax.set_xticklabels([text.get_text().removesuffix(separate_label(""))
                                        for text in ax.get_xticklabels()])
                plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=14)
            for position, (group, values) in enumerate(output.items(), start=1):
                if values:
                    p = model_sign_flip(values, np.zeros(len(values)), point_models[group],
                                        alternative="greater")
                    mark = "*" if p < .05 else "n.s."
                    # Twelve narrow groups: stack the p-value vertically so labels never collide.
                    ax.text(position, .98,
                            "no test (1 model)" if np.isnan(p) else f"{mark} {format_p_value(p)}",
                            transform=ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=11 if separate_groups[row] else 12,
                            rotation=90 if separate_groups[row] else 0)
            if separate_groups[row]:
                add_bracket_headroom(ax, 1, extra=.18)
        if distributions_only:
            figure.suptitle("Normalized AUC Differences", fontsize=18, y=.995)
            figure.tight_layout(rect=(0, 0, 1, .975))
            return figure
        figure.subplots_adjust(left=.05, right=.995, top=.875, bottom=.17,
                               hspace=.45, wspace=.55)
        left_start, left_end = (grid[0, column].get_position(figure) for column in (0, 2))
        right_start, right_end = (grid[0, column].get_position(figure) for column in (3, 5))
        figure.text((left_start.x0 + left_end.x1) / 2, .985,
                    "A. Directed Transfer Curves", ha="center", va="top",
                    fontsize=16, weight="bold")
        figure.text((right_start.x0 + right_end.x1) / 2, .985,
                    "B. Normalized AUC Differences", ha="center", va="top",
                    fontsize=16, weight="bold")
        rank_label = figure.text((left_start.x0 + left_end.x1) / 2, .10,
                                 "Relative rank", ha="center", fontsize=14)
        # Curve legend under the A plots (not over the curves), level with the model legend.
        handles, labels = first_curve_ax.get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, fontsize=13, ncol=len(handles),
                      loc="upper center",
                      bbox_to_anchor=((left_start.x0 + left_end.x1) / 2, rank_label.get_window_extent(
                          figure.canvas.get_renderer()).transformed(figure.transFigure.inverted()).y0 - .012),
                      handlelength=1.6, columnspacing=1.0, handletextpad=.4, borderaxespad=0)
        return figure


def cross_gap_scatter(cross, saturation_mode="quantile"):
    """Correlate matched internal-minus-transfer PCA and RRR AUC gaps."""
    with paper_style():
        figure, ax = plt.subplots(figsize=(8.5, 9))
        all_points = []
        colors = plt.get_cmap("tab10").colors
        for index, (target, source) in enumerate(
            (t, s) for t in DATASETS for s in DATASETS if s != t
        ):
            panel = cross[(cross["target_dataset"] == target)
                          & (cross["source_dataset"] == source)]
            points = np.asarray([
                (item["Within PCA"] - item["Transfer PCA"],
                 item["Within RRR"] - item["Transfer RRR"])
                for item in _cross_auc(panel, saturation_mode).values()
                if {"Within PCA", "Transfer PCA", "Within RRR", "Transfer RRR"} <= item.keys()
            ]).reshape(-1, 2)
            if len(points):
                all_points.append(points)
                ax.scatter(points[:, 0], points[:, 1], s=35, alpha=.72,
                           color=colors[index], edgecolors="white", linewidths=.4,
                           label=f"{LABELS[source]} → {LABELS[target]}, n={len(points)}")
        if all_points:
            points = np.vstack(all_points)
            lo, hi = min(0, points.min()), max(0, points.max())
            pad = max((hi - lo) * .06, .01)
            limits = (lo - pad, hi + pad)
            ax.plot(limits, limits, color="#94a3b8", linestyle="--", label="Equal gaps")
            if len(points) >= 2 and np.unique(points[:, 0]).size > 1:
                slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
                ax.plot(limits, slope * np.asarray(limits) + intercept,
                        color="#111827", linewidth=2, label="Pooled linear fit")
            if len(points) >= 2 and np.unique(points[:, 0]).size > 1 and np.unique(points[:, 1]).size > 1:
                from scipy.stats import pearsonr

                correlation = pearsonr(points[:, 0], points[:, 1])
                r, p = correlation.statistic, correlation.pvalue
                p_text = format_p_value(p)
            else:
                r, p_text = np.nan, "p=NA"
            ax.set(xlim=limits, ylim=limits)
            ax.set_title(f"Internal − transfer AUC gaps\nPearson r = {r:.2f}, {p_text}, n = {len(points)}")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2,
                      columnspacing=1.0, handletextpad=.4)
        ax.axhline(0, color="#64748b", linewidth=.8)
        ax.axvline(0, color="#64748b", linewidth=.8)
        ax.set_xlabel("Within PCA − transfer PCA normalized AUC")
        ax.set_ylabel("Within RRR − transfer RRR normalized AUC")
        finish_axis(ax)
        figure.subplots_adjust(left=.14, right=.97, top=.90, bottom=.27)
        return figure


def _model_layout(items, order):
    positions, labels, colors, centers = [], [], [], {}
    # Same model colors as every distribution plot; unknown models fall back to a tab20 shade.
    color_map = {model: MODEL_COLORS.get(model, plt.get_cmap("tab20")(i / max(len(order) - 1, 1)))
                 for i, model in enumerate(order)}
    x = 0.0
    for model in order:
        layers = sorted([layer for name, layer in items if name == model],
                        key=lambda value: (value is not None, value or -1))
        if not layers:
            continue
        start = x
        for layer in layers:
            positions.append(x)
            labels.append("none" if layer is None else str(layer))
            colors.append(color_map[model])
            x += 1
        centers[model] = (start + x - 1) / 2
        x += .8
    return positions, labels, colors, centers, color_map


def model_order(full_rows):
    scores = defaultdict(list)
    for _, model, _, values in full_rows:
        finite = np.asarray(values)[np.isfinite(values)]
        if len(finite):
            scores[model].append(float(np.mean(finite)))
    return sorted(scores, key=lambda model: np.mean(scores[model]))


def _model_grid_style(ax, positions, labels, centers):
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    for model, center in centers.items():
        ax.text(center, -.30, MODEL_NAMES.get(model, model), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=14)
    finish_axis(ax)


def encoding_grid(full_rows, order):
    with paper_style():
        figure, axes = plt.subplots(3, 1, figsize=(20, 13), sharex=False)
        for ax, dataset in zip(axes, DATASETS):
            entries = {(model, layer): values for d, model, layer, values in full_rows
                       if d == dataset}
            positions, labels, colors, centers, _ = _model_layout(entries, order)
            target_bar_axis(ax, positions,
                            [entries[key] for key in entries_order(entries, order)],
                            colors, error="sd")
            _model_grid_style(ax, positions, labels, centers)
            ax.set_ylabel("Mean correlation ± SD")
            ax.set_title(LABELS[dataset])
        figure.suptitle("Full encoding-model performance", y=.995)
        figure.subplots_adjust(left=.06, right=.99, top=.925, bottom=.12, hspace=.75)
        return figure


def entries_order(entries, order):
    return [(model, layer) for model in order
            for layer in sorted((layer for name, layer in entries if name == model),
                                key=lambda value: (value is not None, value or -1))]


def significance_grid(significant, order):
    with paper_style():
        order = list(order) + sorted({model for _, model, _ in significant} - set(order))
        figure, axes = plt.subplots(3, 1, figsize=(20, 13))
        for ax, dataset in zip(axes, DATASETS):
            entries = {(model, layer): len(targets) for (d, model, layer), targets
                       in significant.items() if d == dataset}
            positions, labels, colors, centers, _ = _model_layout(entries, order)
            values = [entries[key] for key in entries_order(entries, order)]
            ax.bar(positions, values, color=colors, width=.8)
            _model_grid_style(ax, positions, labels, centers)
            ax.set(ylabel="Significant targets", title=LABELS[dataset])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        figure.suptitle("Significant electrodes or voxels by model layer", y=.995)
        figure.subplots_adjust(left=.06, right=.99, top=.925, bottom=.12, hspace=.75)
        return figure


def significance_latex(significant, order):
    """Return model rows with relative-depth columns and grouped datasets."""
    order = list(order) + sorted({model for _, model, _ in significant} - set(order))
    depths = (0, .25, .5, .75, 1)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Dataset & Model & 0\% & 25\% & 50\% & 75\% & 100\% \\",
             r"\midrule"]
    for dataset in DATASETS:
        entries = {(model, layer): targets for (d, model, layer), targets
                   in significant.items() if d == dataset}
        union = set().union(*entries.values()) if entries else set()
        dataset_label = LABELS[dataset]
        for model in order:
            model_entries = {layer: targets for (m, layer), targets in entries.items() if m == model}
            if not model_entries:
                continue
            name = model.replace("_", r"\_")
            if None in model_entries:
                cells = r"\multicolumn{5}{c}{" + f"{len(model_entries[None]):,} (layerless)" + "}"
            else:
                configured_layers = {name: layers for name, layers in MODEL_LAYERS.values()}
                max_layer = max(configured_layers.get(model, MODEL_LAYERS.get(model, (None, []))[1]))
                counts = {layer / max_layer: len(targets) for layer, targets in model_entries.items()}
                cells = " & ".join(f"{counts[depth]:,}" if depth in counts else "--" for depth in depths)
            lines.append(f"{dataset_label} & {name} & {cells} " + r"\\")
            dataset_label = ""
        lines.extend([f"{dataset_label} & Union & " + r"\multicolumn{5}{c}{" + f"{len(union):,}" + "} " + r"\\", r"\midrule"])
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def quantile_grid(quantiles, selected_pair, *, fraction=.99):
    with paper_style():
        figure, axes = plt.subplots(3, 2, figsize=(13, 10.5))
        for row, dataset in enumerate(DATASETS):
            for col, method in enumerate(("rrr", "pca")):
                ax = axes[row, col]
                value = quantiles.get((dataset, *selected_pair, method))
                if value is not None:
                    ranks, matrix = value
                    quantile_axis(ax, ranks, matrix, QUANTILES)
                    sat = quantile_saturation_rank(ranks, matrix, fraction)
                    if np.isfinite(sat):
                        ax.axvline(sat, color=COLORS[method], linestyle="--", linewidth=1.4)
                ax.set_title(f"{LABELS[dataset]} · {method.upper()}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, title="Target quantile", frameon=False,
                      loc="upper center", bbox_to_anchor=(.5, .935), ncol=len(labels))
        layer_text = "no layer" if selected_pair[1] is None else f"layer {selected_pair[1]}"
        figure.suptitle(f"Target correlation quantiles · "
                        f"{MODEL_NAMES.get(selected_pair[0], selected_pair[0])}, {layer_text}", y=.995)
        figure.tight_layout(rect=(0, 0, 1, .875))
        return figure


def average_quantile_grid(quantiles):
    with paper_style():
        figure, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True, sharey=True)
        palette = plt.get_cmap("viridis")(np.linspace(.08, .92, len(QUANTILES)))
        for row, dataset in enumerate(DATASETS):
            for col, method in enumerate(("rrr", "pca")):
                ax = axes[row, col]
                curves = [value for key, value in quantiles.items()
                          if key[0] == dataset and key[-1] == method]
                grid, means, counts = average_normalized_quantiles(curves)
                for index, (level, color) in enumerate(zip(QUANTILES, palette)):
                    if index < means.shape[1] and counts[index]:
                        ax.plot(grid, means[:, index], color=color, linewidth=2,
                                label=f"{level:.0%} (n={counts[index]})")
                ax.set_xscale("symlog", linthresh=.01)
                ax.set(xlim=(0, 1), ylim=(0, 1.05), title=f"{LABELS[dataset]} · {method.upper()}",
                       xlabel="Relative rank" if row == 2 else "",
                       ylabel="Mean normalized\nquantile performance" if col == 0 and row == 1 else "")
                finish_axis(ax)
        axes[0, 0].legend(frameon=False, loc="lower right")
        figure.suptitle("Mean quantile convergence across model layers · peak > 0.005", y=.995)
        figure.tight_layout(rect=(0, 0, 1, .975))
        return figure


def _quantile_robustness_values(quantiles, dataset, *, fraction=.99):
    """Use the latest crossing of the positive-peak 25/50/75% quantiles."""
    columns = [QUANTILES.index(level) for level in (.25, .50, .75)]
    values = defaultdict(list)
    for key, (ranks, matrix) in quantiles.items():
        if key[0] != dataset or key[-1] not in ("rrr", "pca"):
            continue
        rank = quantile_saturation_rank(
            ranks, np.asarray(matrix)[:, columns], fraction=fraction, min_peak=0,
        )
        if np.isfinite(rank):
            values[key[-1]].append((key[1:3], rank))
    return values


@with_model_legend
def quantile_saturation_robustness(quantiles, *, fraction=.99):
    """Separate internal RRR/PCA robustness comparison using target quantiles."""
    with paper_style():
        figure, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            ranks = _quantile_robustness_values(quantiles, dataset, fraction=fraction)
            _saturation_comparison_axis(
                ax, ranks, ("rrr", "pca"), ("RRR", "PCA"),
                (("RRR", "PCA"),), show_mean_value=True,
            )
            ax.set_title(LABELS[dataset])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{2 ** value:g}"))
        axes[0].set_ylabel("Saturation rank $k$")
        figure.suptitle(f"Internal saturation robustness · all eligible quantiles at {fraction * 100:g}%",
                        y=.995)
        figure.text(.5, .02, "25th, 50th, and 75th percentiles · positive peaks · monotonic curves",
                    ha="center", fontsize=14)
        figure.tight_layout(rect=(0, .06, 1, .96))
        return figure


@with_model_legend
def sensitivity_grid(results, values, *, mode="internal", variable="fraction",
                     saturation_mode="quantile"):
    """Compare paired saturation ranks across peak fractions or internal cutoffs.

    `results` is the quantile dictionary, or the explained-variance `curves` /
    `loso_curves` dictionary for `saturation_mode="explained_variance"` (which has no
    minimum-peak cutoff, so only `variable="fraction"` applies).
    """
    _mode(saturation_mode)
    if saturation_mode == "explained_variance" and variable == "floor":
        raise ValueError("explained-variance saturation has no minimum-peak cutoff")
    values = tuple(values)
    if not values:
        raise ValueError("values must contain at least one sensitivity setting")
    if mode not in {"internal", "loso"}:
        raise ValueError("mode must be 'internal' or 'loso'")
    if variable not in {"floor", "fraction"}:
        raise ValueError("variable must be 'floor' or 'fraction'")
    if variable == "floor" and mode != "internal":
        raise ValueError("Peak-cutoff sensitivity requires mode='internal'")
    methods = ("rrr", "pca") if mode == "internal" else (
        "RRR baseline", "RRR transfer", "PCA baseline"
    )
    labels = ("RRR", "PCA") if mode == "internal" else methods
    tests = (("RRR", "PCA"),) if mode == "internal" else (
        ("RRR transfer", "PCA baseline"), ("RRR baseline", "RRR transfer")
    )
    with paper_style():
        figure, axes = plt.subplots(
            len(values), len(DATASETS),
            figsize=(16, 3.15 * len(values) + 1.1), sharey=True,
            squeeze=False,
        )
        for row, value in enumerate(values):
            for col, dataset in enumerate(DATASETS):
                ranks = (_saturation_values(
                    results, dataset, methods, .99 if variable == "floor" else value,
                    min_peak=value if variable == "floor" else PEAK_FLOOR,
                    saturation_mode=saturation_mode,
                )
                         if mode == "internal" else
                         _loso_saturation_values(results, dataset, value, saturation_mode))
                ax = axes[row, col]
                _saturation_comparison_axis(
                    ax, ranks, methods, labels, tests, show_mean_value=True,
                )
                if variable == "floor":
                    ax.set_ylabel("")
                if mode == "loso":
                    ax.set_xticklabels(["RRR\nbaseline", "RRR\ntransfer", "PCA\nbaseline"])
                if row == 0:
                    ax.set_title(LABELS[dataset])
                if col == 0 and variable != "floor":
                    ax.set_ylabel(f"{value:g}" if variable == "floor" else f"{value * 100:g}%",
                                  labelpad=8)
                elif col != 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
        thin_log2_rank_ticks(axes[0, 0])  # shared y: label every other power of two
        title = "Internal" if mode == "internal" else "Leave-one-subject-out"
        setting = ("minimum quantile peak correlation" if variable == "floor"
                   else "required fraction of maximum explained variance"
                   if saturation_mode == "explained_variance" else "required peak fraction")
        basis = "quantile" if saturation_mode == "quantile" else "explained-variance"
        figure.suptitle(f"{title} RRR {basis} saturation sensitivity: {setting}")
        figure.text(.015, .5, "Saturation rank $k$", rotation=90,
                    va="center", ha="center", fontsize=15)
        figure.subplots_adjust(left=.18 if variable == "floor" else .13,
                               right=.99, top=.89, bottom=.06,
                               hspace=.54, wspace=.10)
        if variable == "floor":
            for row, value in enumerate(values):
                position = axes[row, 0].get_position()
                figure.text(.125, (position.y0 + position.y1) / 2,
                            f"Minimum peak\n{value:g}", ha="right", va="center",
                            fontsize=12)
        return figure


def _decoding_relative(decoding):
    required = {"space", "subspace", "k", "kind", "task", "primary_score"}
    if decoding.empty or not required <= set(decoding):
        return pd.DataFrame()
    frame = decoding[(decoding["generation"] == "within_dataset")
                     & (decoding["space"] == "rank")
                     & (decoding["subspace"].isin(["rrr", "pca"]))
                     & (decoding["label_offset"] == 0)].copy()
    volume_tasks = {"podcast": "intensity_rms", "cars-2": "native_intensity_rms",
                    "fmri": "intensity_rms"}
    for dataset, task in volume_tasks.items():
        frame.loc[frame["dataset"].eq(dataset) & frame["task"].eq(task), "task"] = "volume"
    frame["layer"] = frame["layer"].fillna(-1).astype(int)
    keys = ["dataset", "model", "layer", "task", "kind", "k", "variant"]
    wide = frame.pivot_table(index=keys, columns="subspace", values="primary_score",
                             aggfunc="mean").reset_index()
    if not {"rrr", "pca"} <= set(wide):
        return pd.DataFrame()
    rrr, pca = wide["rrr"].clip(lower=0), wide["pca"].clip(lower=0)
    wide["relative"] = (rrr - pca) / ((rrr + pca) / 2 + .05)
    return wide


_DECODING_FEATURES = [("concreteness", "Concreteness"),
                      ("sentence_position", "Sentence position"),
                      ("standardized_entropy", "Entropy"),
                      ("volume", "Volume")]


def _decoding_consensus(frame):
    """Per (model, layer, task) relative decoding AUC of every dataset (rows with all datasets)."""
    auc_rows = []
    if not frame.empty:
        for (dataset, model, layer, task), group in frame.groupby(
            ["dataset", "model", "layer", "task"], dropna=False
        ):
            mean = group.groupby("k")["relative"].mean().sort_index()
            ranks = mean.index.to_numpy(dtype=float)
            if len(ranks) < 2:
                continue
            x = np.log(ranks)
            span = x[-1] - x[0]
            if span > 0:
                auc_rows.append((dataset, model, layer, task,
                                 float(np.trapezoid(mean, x=(x - x[0]) / span))))
    auc = pd.DataFrame(auc_rows, columns=["dataset", "model", "layer", "task", "auc"])
    consensus = (auc.pivot_table(
        index=["model", "layer", "task"], columns="dataset", values="auc"
    ).dropna(subset=list(DATASETS)).reset_index()
                 if not auc.empty else pd.DataFrame())
    return consensus


def decoding_aggregate(decoding):
    """Relative decoding curves (RRR vs PCA) for four example features, one panel each."""
    with paper_style():
        figure = plt.figure(figsize=(20, 4.8))
        grid = figure.add_gridspec(1, 4, wspace=.08)
        frame = _decoding_relative(decoding)
        curve_axes = []
        for col, (task, title) in enumerate(_DECODING_FEATURES):
            ax = figure.add_subplot(grid[0, col],
                                    sharey=curve_axes[0] if curve_axes else None)
            curve_axes.append(ax)
            for dataset, linestyle in zip(DATASETS, ("-", "--", ":")):
                selected = frame[(frame["task"] == task) & (frame["dataset"] == dataset)] if not frame.empty else frame
                if len(selected):
                    mean = selected.groupby("k")["relative"].mean().sort_index()
                    decoding_relative_axis(ax, mean.index, mean, LABELS[dataset],
                                           linestyle=linestyle)
            ax.set_title(title)
            if col:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
        for ax in curve_axes:
            add_decoding_gradient(ax)
            if ax is not curve_axes[-1]:
                continue
            low, high = ax.get_ylim()
            zero = -low / (high - low)
            arrow_scale = 2 * min(zero, 1 - zero)
            for direction, color, label in (
                (1, "#6ab2e2", "RRR better"),
                (-1, "#63bc51", "PCA better"),
            ):
                ax.annotate(
                    "", xy=(1.03, zero + direction * .20 * arrow_scale),
                    xytext=(1.03, zero + direction * .04 * arrow_scale),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.5},
                    annotation_clip=False, clip_on=False,
                )
                ax.text(
                    1.07, zero + direction * .24 * arrow_scale, label,
                    transform=ax.transAxes, ha="center", va="center", rotation=90,
                    color=color, fontsize=11, clip_on=False,
                )
        handles, labels = curve_axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=3,
                      loc="upper center", bbox_to_anchor=(.5, 1.0))
        figure.subplots_adjust(left=.05, right=.975, top=.80, bottom=.15)
        return figure


def decoding_auc_scatters(decoding):
    """Cross-dataset correlations of the relative decoding AUCs (one scatter per dataset pair)."""
    with paper_style():
        figure = plt.figure(figsize=(18, 6))
        grid = figure.add_gridspec(1, 3, wspace=.28)
        frame = _decoding_relative(decoding)
        pairs = [("cars-2", "podcast"), ("podcast", "fmri"), ("cars-2", "fmri")]
        consensus = _decoding_consensus(frame)
        bound = (max(float(np.nanmax(np.abs(consensus[list(DATASETS)].to_numpy()))), .05)
                 if not consensus.empty else .05) * 1.05
        ticks = MaxNLocator(nbins=3, symmetric=True).tick_values(-bound, bound)  # few ticks: labels must not touch
        limits = (float(ticks[0]), float(ticks[-1]))
        for col, (first, second) in enumerate(pairs):
            scatter = figure.add_subplot(grid[0, col])
            matched = (consensus[[first, second]].dropna() if not consensus.empty
                       else pd.DataFrame(columns=[first, second]))
            correlation_scatter_axis(scatter, matched[first], matched[second], limits, ticks)
            if len(matched):
                if (len(matched) >= 2 and matched[first].nunique() > 1
                        and matched[second].nunique() > 1):
                    from scipy.stats import pearsonr

                    result = pearsonr(matched[first], matched[second])
                    r, p_value = result.statistic, result.pvalue
                else:
                    r, p_value = np.nan, np.nan
                scatter.set_title(
                    f"{LABELS[first]} vs {LABELS[second]}\n"
                    f"Pearson r = {r:.2f}, {format_p_value(p_value)}\n"
                    f"n = {len(matched)}",
                    fontsize=14,
                )
            if col == 0:
                scatter.legend(frameon=False, fontsize=12, loc="upper left")
            scatter.set_xlabel(f"{LABELS[first]} relative AUC")
            scatter.set_ylabel(f"{LABELS[second]} relative AUC")
        figure.subplots_adjust(left=.06, right=.99, top=.76, bottom=.14)
        return figure


def roi_curves_figure(curves, region_sets, dataset, pair, *, fraction=.99, columns=3,
                      lag_summary="mean"):
    """Raw-correlation rank curves per ROI for one iEEG dataset and model layer.

    `curves` is the frame from `load_electrode_curves`; `region_sets` maps each ROI
    name to the {(subject, electrode)} pairs it contains. Each panel shows the
    Fisher-mean RRR curve (thick), the PCA mean curve, one faint RRR line per
    subject with electrodes in the ROI, and a dashed line (per method) at the rank where the
    ROI's mean RRR and PCA curves first reach `fraction` of their own peak.
    """
    identity = pd.MultiIndex.from_frame(curves[["subject", "electrode"]])
    rows = -(-len(region_sets) // columns)
    with paper_style():
        figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows),
                                    sharey=True, squeeze=False)
        for ax, (region, members) in zip(axes.flat, region_sets.items()):
            inside = curves[identity.isin(list(members))]
            ax.set_title(region)
            if inside.empty:
                ax.text(.5, .5, "no significant electrodes", ha="center", va="center",
                        transform=ax.transAxes, color="#64748b")
                finish_axis(ax)
                continue
            rrr = inside[inside["method"] == "rrr"]
            for _, subject in rrr.groupby("subject"):
                line = subject.groupby("rank")["correlation"].apply(fisher_mean)
                ax.plot(line.index, line.values, color=COLORS["rrr"], alpha=.3, linewidth=1)
            for method in ("pca", "rrr"):
                mean = (inside[inside["method"] == method]
                        .groupby("rank")["correlation"].apply(fisher_mean))
                ax.plot(mean.index, mean.values, marker="o", markersize=3.5,
                        color=COLORS[method], linewidth=2.2 if method == "rrr" else 1.6,
                        label=method.upper())
                saturation = quantile_saturation_rank(
                    mean.index.to_numpy(), mean.to_numpy()[:, None], fraction
                )
                if np.isfinite(saturation):
                    ax.axvline(saturation, color=COLORS[method], linestyle="--",
                               linewidth=1.4)
            n_electrodes = inside[["subject", "electrode"]].drop_duplicates().shape[0]
            ax.text(.97, .05, f"n={n_electrodes}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=13, color="#64748b")
            ax.set_xscale("log", base=2)
            ax.set(xlabel="Rank $k$", ylabel="r")
            finish_axis(ax)
        for ax in list(axes.flat)[len(region_sets):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, loc="upper right", ncol=2)
        what = ("mean of electrode peak correlations" if lag_summary == "peak"
                else "mean correlation")
        figure.suptitle(f"{LABELS[dataset]} · {pair[0]} / layer {pair[1]} · {what} by ROI")
        figure.tight_layout(rect=(0, 0, 1, .95))
        return figure

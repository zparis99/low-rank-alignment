import numpy as np
import pandas as pd
import pytest

from src.utils.rrr.plot_metrics import mean_saturation_rank


def test_mean_saturation_interpolates_first_peak_fraction_crossing():
    assert mean_saturation_rank([1, 4, 8, 16], [.1, .4, .3, .5]) == pytest.approx(15.8)
    assert mean_saturation_rank([16, 1, 4, 8], [.5, .1, .4, .3]) == pytest.approx(15.8)
    assert mean_saturation_rank([1, 4, 8], [.1, .4, .3], .5) == pytest.approx(2)
    assert np.isnan(mean_saturation_rank([1], [np.nan]))
    with pytest.raises(ValueError):
        mean_saturation_rank([1], [.1], 0)


def test_final_saturation_uses_quantiles_and_averages_loso_subject_ranks():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _loso_saturation_values, _saturation_values
    from src.utils.plotting.final_figures import rrr_aggregate

    frame = pd.DataFrame([
        ("podcast", "glove", None, subject, "rrr", rank, value)
        for subject, values in (("a", [.1, .5]), ("b", [.4, .5]))
        for rank, value in zip([1, 9], values)
    ], columns=["dataset", "model", "layer", "subject", "method", "rank", "correlation"])
    expected = np.mean([8.9, 8.6])
    quantiles = {('podcast', 'glove', None, 'rrr'): ([1, 9], [[.1, .004], [.5, .005]])}
    loso_quantiles = {
        ('podcast', 'glove', None, subject, 'rrr'): ([1, 9], np.tile(values, (6, 1)).T)
        for subject, values in (("a", [.1, .5]), ("b", [.4, .5]))
    }
    assert _loso_saturation_values(loso_quantiles, "podcast", .99)["rrr"][0][1] == pytest.approx(expected)
    assert _saturation_values(quantiles, "podcast", ("rrr",), .99)["rrr"] == [(('glove', None), 8.9)]
    for mode in ("internal", "loso"):
        figure = rrr_aggregate(frame, frame, quantiles, loso_quantiles, [], ('glove', None), mode=mode)
        plt.close(figure)


def test_notebook_and_export_parse():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in ('notebooks/final_plots.py', 'src/utils/plotting/final_export.py'):
        ast.parse((root / path).read_text())


def test_significance_table_groups_models_by_relative_depth():
    from src.utils.plotting.final_figures import significance_latex

    significant = {
        ("podcast", "gpt2", 0): {"a"},
        ("podcast", "gpt2", 12): {"a", "b"},
        ("podcast", "gpt2-xl", 12): {"c"},
        ("podcast", "gpt2-xl-batch", 24): {"c"},
        ("podcast", "glove", None): {"a", "c"},
        ("cars-2", "gpt2", 3): {"d"},
    }
    table = significance_latex(significant, ["gpt2", "gpt2-xl", "glove"])
    assert r"0\% & 25\% & 50\% & 75\% & 100\%" in table
    assert "Podcast (ECoG) & gpt2 & 1 & -- & -- & -- & 2" in table
    assert " & gpt2-xl & -- & 1 & -- & -- & --" in table
    assert " & gpt2-xl-batch & -- & -- & 1 & -- & --" in table
    assert r" & glove & \multicolumn{5}{c}{2 (layerless)}" in table
    assert r" & Union & \multicolumn{5}{c}{3}" in table
    assert "Cars 2 (sEEG) & gpt2 & -- & 1 & -- & -- & --" in table
    assert table.count("Podcast") == table.count("Cars 2") == table.count("fMRI") == 1


def test_quantile_figures_keep_six_levels_and_quantile_saturation_marker():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_data import QUANTILES
    from src.utils.plotting.final_figures import average_quantile_grid, quantile_grid, sensitivity_grid

    assert QUANTILES == (.05, .10, .25, .50, .75, .90)
    quantiles = {("podcast", "gpt2", 0, "rrr"): (np.array([1, 9]), np.tile([.1, .5], (6, 1)).T)}
    internal = pd.DataFrame([
        ("podcast", "gpt2", 0, "rrr", 1, .4),
        ("podcast", "gpt2", 0, "rrr", 9, .5),
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    sample = quantile_grid(quantiles, ("gpt2", 0))
    assert len(sample.axes[0].lines) == 7
    assert sample.axes[0].lines[-1].get_xdata()[0] == pytest.approx(8.9)
    normalized = average_quantile_grid(quantiles)
    assert len(normalized.axes[0].lines) == 6
    sensitivity = sensitivity_grid(quantiles, [.5, .99])
    assert len(sensitivity.axes) == 6
    for figure in (sample, normalized, sensitivity):
        plt.close(figure)


def test_main_analysis_selectors_include_control_models():
    from src.utils.plotting.final_export import available_pairs

    frame = pd.DataFrame({"model": ["gpt2", "random-gpt2", "glove-300", "glove"]})
    quantiles = {(dataset, model, 0, method): ([1, 2], [[.1], [.2]])
                 for dataset in ("podcast", "cars-2", "fmri")
                 for model in frame["model"] for method in ("rrr", "pca")}
    assert set(available_pairs(quantiles)) == {(model, 0) for model in frame["model"]}


def test_main_saturation_includes_glove_and_random_gpt2():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _saturation_values, rrr_aggregate

    frame = pd.DataFrame([
        (dataset, model, layer, method, rank, correlation)
        for dataset in ("podcast", "cars-2", "fmri")
        for model, layers in (("random-gpt2", [0, 3]), ("glove-300", [None]), ("gpt2", [0]))
        for layer in layers for method in ("rrr", "pca")
        for rank, correlation in ((1, .1), (9, .5))
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    quantiles = {
        (dataset, model, layer, method): ([1, 9], np.tile([.1, .5], (6, 1)).T)
        for dataset in ('podcast', 'cars-2', 'fmri')
        for model, layers in (("random-gpt2", [0, 3]), ("glove-300", [None]), ("gpt2", [0]))
        for layer in layers for method in ('rrr', 'pca')
    }
    values = _saturation_values(quantiles, 'podcast', ('rrr', 'pca'), .99)
    assert {pair for pair, rank in values['rrr']} == {
        ('random-gpt2', 0), ('random-gpt2', 3), ('glove-300', None), ('gpt2', 0)
    }
    figure = rrr_aggregate(frame, frame, quantiles, {}, [], ('glove-300', None))
    plt.close(figure)


def test_loso_sensitivity_compares_three_methods_at_each_fraction():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import sensitivity_grid

    frame = pd.DataFrame([
        ("podcast", "gpt2", 0, subject, method, rank, correlation)
        for subject in ('a', 'b')
        for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
        for rank, correlation in ((1, .1), (9, .5))
    ], columns=['dataset', 'model', 'layer', 'subject', 'method', 'rank', 'correlation'])
    quantiles = {
        ('podcast', 'gpt2', 0, subject, method): ([1, 9], np.tile([.1, .5], (6, 1)).T)
        for subject in ('a', 'b')
        for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
    }
    figure = sensitivity_grid(quantiles, [.5, .99], mode='loso')
    assert len(figure.axes) == 6
    for ax, expected_mean in ((figure.axes[0], 4), (figure.axes[3], 8.9)):
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            'RRR\nbaseline', 'RRR\ntransfer', 'PCA\nbaseline'
        ]
        assert any(f'mean = {expected_mean:.1f}' in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_peak_cutoff_sensitivity_excludes_low_peaks_at_fixed_99_percent():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import sensitivity_grid

    # The weak quantile saturates later, until its peak is excluded by the cutoff.
    matrix = np.array([[.01, 0], [.1, .003], [.1, .006]])
    quantiles = {('podcast', 'gpt2', 0, method): ([1, 4, 9], matrix)
                for method in ('rrr', 'pca')}
    figure = sensitivity_grid(quantiles, (0, .005, .01), variable='floor')
    assert len(figure.axes) == 9
    for index, cutoff, expected in ((0, '0', 8.9), (3, '0.005', 8.9), (6, '0.01', 4.0)):
        ax = figure.axes[index]
        assert ax.get_ylabel() == cutoff
        assert any(f'mean = {expected:.1f}' in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_quantile_robustness_ignores_negative_peaks_and_outer_quantiles():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _quantile_robustness_values, quantile_saturation_robustness

    # Outer quantiles would cross much later; the median dips after its first crossing.
    matrix = np.array([
        [.01, .01, -.3, .1, .1, .01],
        [.01, .01, -.2, .5, .2, .01],
        [.5, .5, -.1, .3, .4, .5],
    ])
    quantiles = {('podcast', 'gpt2', 0, method): (np.array([1, 4, 10]), matrix)
                 for method in ('rrr', 'pca')}
    # 75th percentile determines the final crossing, at 9.88.
    ranks = _quantile_robustness_values(quantiles, 'podcast')
    assert ranks['rrr'][0][1] == pytest.approx(9.88)
    figure = quantile_saturation_robustness(quantiles)
    assert len(figure.axes) == 3
    plt.close(figure)
    negative = {('podcast', 'gpt2', 0, 'rrr'): ([1, 4, 10], -np.ones((3, 6)))}
    assert not _quantile_robustness_values(negative, 'podcast')


def test_explained_variance_curve_clips_negatives_and_averages_over_targets():
    from src.utils.rrr.plot_metrics import explained_variance_curve

    # Rank 1: r = (.1, -.5, nan) -> squares (.01, 0), mean .005. Rank 2: (.3, .1, .2).
    values = np.array([[.1, -.5, np.nan], [.3, .1, .2]])
    assert explained_variance_curve(values) == pytest.approx([.005, (.09 + .01 + .04) / 3])
    # Higher-dimensional target axes (electrode x lag) are flattened together.
    stacked = np.stack([values, values], axis=-1)
    assert explained_variance_curve(stacked) == pytest.approx(explained_variance_curve(values))
    assert np.isnan(explained_variance_curve(np.full((1, 2), np.nan))[0])


def test_explained_variance_saturation_is_scale_free_and_has_no_peak_floor():
    from src.utils.rrr.plot_metrics import explained_variance_saturation_rank

    ranks, curve = [1, 4, 8], np.array([.0001, .0004, .0003])
    # The monotonic envelope peaks at rank 4; 99% of that is reached at 3.97 ... of the way.
    expected = 1 + (.99 * .0004 - .0001) / (.0004 - .0001) * 3
    assert explained_variance_saturation_rank(ranks, curve) == pytest.approx(expected)
    # Tiny absolute values are not dropped by the quantile statistic's 0.005 cutoff,
    # and rescaling (sum instead of mean over targets) leaves the rank unchanged.
    assert explained_variance_saturation_rank(ranks, curve * 1000) == pytest.approx(expected)
    assert np.isnan(explained_variance_saturation_rank(ranks, [0, 0, 0]))
    assert explained_variance_saturation_rank([1, 4], [.2, .2]) == pytest.approx(1)


def test_explained_variance_figures_use_explained_variance_tables():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import (
        cross_aggregate, cross_gap_scatter, rrr_aggregate, sensitivity_grid,
    )

    ranks = np.array([1, 4, 9])
    curves = {('podcast', 'gpt2', 0, method): (ranks, np.array([[.001], [.004], [.005]]))
              for method in ('rrr', 'pca')}
    loso_curves = {('podcast', 'gpt2', 0, subject, method): (ranks, np.array([[.001], [.004], [.005]]))
                   for subject in ('a', 'b')
                   for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')}
    internal = pd.DataFrame([
        ('podcast', 'gpt2', 0, method, int(rank), float(value))
        for method in ('rrr', 'pca') for rank, value in zip(ranks, [.001, .004, .005])
    ], columns=['dataset', 'model', 'layer', 'method', 'rank', 'explained_variance'])
    loso = pd.DataFrame([
        ('podcast', 'gpt2', 0, subject, method, int(rank), float(value))
        for subject in ('a', 'b')
        for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
        for rank, value in zip(ranks, [.001, .004, .005])
    ], columns=['dataset', 'model', 'layer', 'subject', 'method', 'rank', 'explained_variance'])
    full_rows = [('podcast', 'gpt2', 0, .006)]
    for mode, frame in (('internal', internal), ('loso', loso)):
        figure = rrr_aggregate(internal, loso, curves, loso_curves, full_rows, ('gpt2', 0),
                               mode=mode, saturation_mode='explained_variance')
        assert 'Mean $r^2$\nper target' in {ax.get_ylabel() for ax in figure.axes}
        # The 0.005 quantile peak floor must not remove these small explained-variance peaks.
        assert any('mean = ' in text.get_text() for ax in figure.axes for text in ax.texts)
        plt.close(figure)
    cross = pd.DataFrame([
        (method, int(rank), float(value), 'podcast', 'cars-2', 'gpt2', 0)
        for method in ('Within RRR', 'Within PCA', 'Transfer RRR', 'Transfer PCA')
        for rank, value in zip(ranks, [.001, .004, .005])
    ], columns=['method', 'rank', 'explained_variance', 'target_dataset', 'source_dataset',
                'model', 'layer'])
    for figure in (cross_aggregate(cross, 'explained_variance'),
                   cross_gap_scatter(cross, 'explained_variance'),
                   sensitivity_grid(curves, (.5, .99), saturation_mode='explained_variance'),
                   sensitivity_grid(loso_curves, (.5, .99), mode='loso',
                                    saturation_mode='explained_variance')):
        plt.close(figure)
    with pytest.raises(ValueError):
        sensitivity_grid(curves, (0, .005), variable='floor', saturation_mode='explained_variance')
    with pytest.raises(ValueError):
        rrr_aggregate(internal, loso, curves, loso_curves, full_rows, ('gpt2', 0),
                      saturation_mode='nonsense')


def test_fmri_only_figures_draw_only_fmri_panels():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import cross_aggregate, rrr_aggregate

    ranks = np.array([1, 4, 9])
    values = [.001, .004, .005]
    curves = {('fmri', 'gpt2', 0, method): (ranks, np.array([[v] for v in values]))
              for method in ('rrr', 'pca')}
    loso_curves = {('fmri', 'gpt2', 0, subject, method): (ranks, np.array([[v] for v in values]))
                   for subject in ('a', 'b')
                   for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')}
    internal = pd.DataFrame([('fmri', 'gpt2', 0, method, int(rank), v)
                             for method in ('rrr', 'pca') for rank, v in zip(ranks, values)],
                            columns=['dataset', 'model', 'layer', 'method', 'rank',
                                     'explained_variance'])
    loso = pd.DataFrame([('fmri', 'gpt2', 0, subject, method, int(rank), v)
                         for subject in ('a', 'b')
                         for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
                         for rank, v in zip(ranks, values)],
                        columns=['dataset', 'model', 'layer', 'subject', 'method', 'rank',
                                 'explained_variance'])
    for mode in ('internal', 'loso'):
        for distributions_only in (False, True):
            figure = rrr_aggregate(internal, loso, curves, loso_curves, [('fmri', 'gpt2', 0, .006)],
                                   ('gpt2', 0), mode=mode, saturation_mode='explained_variance',
                                   datasets=('fmri',), fmri_noise_normalized=False,
                                   distributions_only=distributions_only)
            titles = {ax.get_title() for ax in figure.axes}
            assert 'Moth (fMRI)' in titles
            assert not titles & {'Podcast (ECoG)', 'Cars 2 (sEEG)'}
            plt.close(figure)
    cross = pd.DataFrame([
        (method, int(rank), v, 'fmri', source, 'gpt2', 0)
        for source in ('podcast', 'cars-2')
        for method in ('Within RRR', 'Within PCA', 'Transfer RRR', 'Transfer PCA')
        for rank, v in zip(ranks, values)
    ], columns=['method', 'rank', 'explained_variance', 'target_dataset', 'source_dataset',
                'model', 'layer'])
    for distributions_only in (False, True):
        figure = cross_aggregate(cross, 'explained_variance', targets=('fmri',),
                                 distributions_only=distributions_only)
        assert not any('→ Podcast' in ax.get_title() or '→ Cars' in ax.get_title()
                       for ax in figure.axes)
        plt.close(figure)


def test_raw_fmri_curves_come_from_the_same_summary():
    from src.utils.plotting.final_data import _summarize

    rng = np.random.default_rng(0)
    values = rng.uniform(.1, .5, size=(3, 2, 4, 1))  # [rank, fold, target, lag]
    scale = np.array([.25, .5, 1.0, 2.0])
    scaled = _summarize(values, [1, 2, 4], "mean", scale)
    plain = _summarize(values, [1, 2, 4], "mean", None)
    assert plain["ev_raw"] is None
    np.testing.assert_allclose(scaled["ev_raw"], plain["ev"])
    assert not np.allclose(scaled["ev"], scaled["ev_raw"])


def test_bootstrap_intervals_come_from_products_without_reading_results():
    from src.utils.plotting import final_bootstrap as bootstrap
    from src.utils.plotting.final_products import result_products

    ranks = np.array([1., 2., 4., 8.])
    rng = np.random.default_rng(0)
    products = {}
    for dataset in ("podcast", "cars-2", "fmri"):
        targets = pd.DataFrame({"subject": [str(i % 6) for i in range(12)],
                                "electrode": [f"A{i}" for i in range(12)]})
        per_pair = rng.uniform(0, 1, size=(4, 12, 1)) * np.array([.1, .5, .9, 1.0])[:, None, None]
        scale = np.full(12, 2.0) if dataset == "fmri" else None
        entry = result_products(dataset, ranks, per_pair, scale, targets)
        products[(dataset, "gpt2", 0)] = {
            "internal": {"rrr": entry, "pca": entry},
            "cross": {source: {"rrr": entry, "pca": entry}
                      for source in ("podcast", "cars-2", "fmri") if source != dataset}}
    result = bootstrap.internal_intervals(products, [("gpt2", [0])])
    assert {key[0] for key in result["raw_fmri"]} == {"fmri"}
    assert {key[0] for key in result["normalized"]} == {"podcast", "cars-2", "fmri"}
    auc = bootstrap.auc_intervals(products, [("gpt2", [0])])
    assert {key[0] for key in auc["raw_fmri"]} == {"fmri"}
    assert len(auc["normalized"]) == 6 * 3  # six directed pairs x three comparisons


def test_result_products_match_direct_computation():
    from src.utils.plotting.final_products import result_products, squared_variants
    from src.utils.rrr.plot_metrics import explained_variance_curve

    rng = np.random.default_rng(1)
    per_pair = rng.uniform(-.2, .8, size=(3, 8, 2))
    scale = np.array([.5, 1, 2, 1, 1, np.nan, 1, 4.])
    targets = pd.DataFrame({"subject": list("aabbccdd")})
    region_index = {"IFG": np.array([0, 1, 2]), "empty": np.array([], dtype=int)}
    products = result_products("fmri", [1, 2, 4], per_pair, scale, targets,
                               region_index=region_index)
    sums, counts, _ = products["clusters"]["normalized"]
    np.testing.assert_allclose(
        sums.sum(axis=0) / counts.sum(axis=0),
        explained_variance_curve(per_pair, scale=scale[:, None]))
    np.testing.assert_allclose(
        products["roi_curves"]["raw_fmri"]["IFG"],
        explained_variance_curve(per_pair[:, [0, 1, 2], :]))
    assert set(products["roi"]["normalized"]) == {"IFG"}  # regions without targets are skipped
    assert products["roi_counts"] == {"IFG": 3, "empty": 0}
    assert set(squared_variants(per_pair, None)) == {"normalized"}


def test_atlas_labels_are_looked_up_once_per_distinct_target_table(monkeypatch):
    from src.utils.plotting import final_data

    calls = []

    class Adapter:
        def get_electrode_coordinates(self, targets, root):
            calls.append("coordinates")
            return np.zeros((len(targets), 3))

        def get_target_atlas_labels(self, targets, root):
            calls.append("labels")
            return np.array(["IFG"] * len(targets))

    monkeypatch.setattr("src.core.registry.get_dataset_adapter", lambda name: Adapter())
    monkeypatch.setattr(final_data, "_ATLAS_LABELS", {})
    first = pd.DataFrame({"subject": ["1", "1"], "electrode": ["a", "b"]})
    again = first.copy()
    other = pd.DataFrame({"subject": ["1", "2"], "electrode": ["a", "b"]})
    final_data.target_atlas_labels("podcast", first, "root")
    final_data.target_atlas_labels("podcast", again, "root")
    assert calls == ["coordinates", "labels"]
    final_data.target_atlas_labels("podcast", other, "root")
    assert calls == ["coordinates", "labels"] * 2

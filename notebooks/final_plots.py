"""Curated paper figures from preloaded result summaries."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.utils.plotting.final_data import load_final_data, load_shuffled_internal_data
    from src.utils.plotting.final_figures import (
        average_quantile_grid, cross_aggregate, cross_gap_scatter,
        decoding_aggregate, encoding_grid, model_order, quantile_grid,
        rrr_aggregate, sensitivity_grid, significance_grid, quantile_saturation_robustness,
        significance_latex,
    )

    results_root = repo_root / "results"
    return (
        average_quantile_grid,
        cross_aggregate,
        cross_gap_scatter,
        encoding_grid,
        load_final_data,
        load_shuffled_internal_data,
        mo,
        model_order,
        quantile_grid,
        repo_root,
        results_root,
        rrr_aggregate,
        sensitivity_grid,
        significance_grid,
        significance_latex,
    )


@app.cell
def _(load_final_data, results_root):
    # The only large artifact read in this notebook. Restart to refresh results.
    (internal, loso, cross, quantiles, loso_quantiles, full_rows,
     significant, decoding, gaps, explained_variance) = load_final_data(results_root)
    return (
        cross,
        decoding,
        explained_variance,
        full_rows,
        gaps,
        internal,
        loso,
        loso_quantiles,
        quantiles,
        significant,
    )


@app.cell
def _(gaps, mo):
    directed_gaps = [gap for gap in gaps if gap.startswith("directed ")]
    directed_details = "\n".join(f"- {gap}" for gap in directed_gaps)
    other_gaps = [gap for gap in gaps if not gap.startswith("directed ")]
    other_details = "\n".join(f"- {gap}" for gap in other_gaps)
    mo.md(
        "# Final plots\n\n"
        "All saved correlations and target summaries were loaded at startup. "
        "Restart the notebook to pick up new artifacts.\n\n"
        f"**Missing or incomplete results:** {len(gaps)} total, including "
        f"{len(directed_gaps)} directed results. Available completed results are plotted.\n\n"
        f"<details><summary>Show directed result gaps</summary>\n\n{directed_details}\n\n</details>\n\n"
        f"<details><summary>Show other result gaps</summary>\n\n{other_details}\n\n</details>"
    )
    return


@app.cell
def _(mo, quantiles):
    pairs = sorted({(model, layer) for dataset, model, layer, method in quantiles
                    if all((required_dataset, model, layer, required_method) in quantiles
                           for required_dataset in ("podcast", "cars-2", "fmri")
                           for required_method in ("rrr", "pca"))},
                   key=lambda item: (item[0], item[1] is not None, item[1] or -1))
    options = {f"{model} / {'layerless' if layer is None else f'layer {layer}'}": (model, layer)
               for model, layer in pairs}
    selected_pair = mo.ui.dropdown(
        options=options,
        value=next((label for label, pair in options.items() if pair == ("gpt2", 0)),
                   next(iter(options), None)),
        label="Sample model layer",
    )
    selected_pair
    return (selected_pair,)


@app.cell
def _(loso_quantiles, mo):
    _loso_pairs = sorted({(model, layer) for dataset, model, layer, subject, method in loso_quantiles
                    if all(any(key[:3] == (required_dataset, model, layer)
                               and key[-1] == required_method for key in loso_quantiles)
                           for required_dataset in ("podcast", "cars-2", "fmri")
                           for required_method in ("RRR baseline", "RRR transfer", "PCA baseline"))},
                   key=lambda item: (item[0], item[1] is not None, item[1] or -1))
    _loso_options = {f"{model} / {'layerless' if layer is None else f'layer {layer}'}": (model, layer)
               for model, layer in _loso_pairs}
    loso_pair = mo.ui.dropdown(
        options=_loso_options,
        value=next((label for label, pair in _loso_options.items() if pair == ("gpt2", 0)),
                   next(iter(_loso_options), None)),
        label="LOSO sample model layer",
    )
    loso_pair
    return (loso_pair,)


@app.cell(hide_code=True)
def _(explained_variance, repo_root, results_root, roi_names, subspace_data):
    from src.utils.plotting.final_bootstrap import interval_tables

    # Computed (and cached) on first use. Subjects are resampled (Cars 2: depth probes); tables for
    # saturation ranks (internal, LOSO), cross-dataset AUC differences, and per-ROI saturation ranks.
    model_cis = interval_tables(
        results_root, explained_variance.loso_curves, subspace_data.MODEL_LAYER_LIST,
        {"podcast": repo_root / "data", "cars-2": repo_root / "data" / "brain-treebank",
         "fmri": repo_root / "data" / "deep-fMRI-dataset"}, roi_names)
    return interval_tables, model_cis


@app.cell(hide_code=True)
def _():
    from src.utils.plotting.final_axes import set_model_point_colors

    # Points in every distribution plot are colored by model.
    model_colors_on = set_model_point_colors(True)
    return (model_colors_on,)


@app.cell(hide_code=True)
def _(repo_root):
    repo_root  # the first cell puts the repo on sys.path
    from src.utils.plotting.final_figures import without_models

    # Main figures leave random GPT-2 out entirely (GloVe stays in); the companion figures draw
    # it as a separate distribution next to the other models.
    RANDOM_MODEL = "random-gpt2"
    MAIN_EXCLUDED = (RANDOM_MODEL,)
    return MAIN_EXCLUDED, RANDOM_MODEL, without_models


@app.cell
def _(mo):
    mo.md("""
    # Main figures: explained-variance saturation

    "
        "Each curve is the mean squared correlation ($r^2$, negatives clipped to zero) "
        "per target and rank, over all (electrode, lag) pairs. Saturation is the first "
        "rank reaching 99% of the curve's own maximum.

        **Statistics.** Layers of one model are not independent, so tests use one unit per model: layer
        values are averaged within each model, then compared with an exact paired sign-flip test across
        models (smallest possible p = 1/2^n, e.g. 0.016 for six models). In the distribution plots, small
        dots are layers, squares are model means, and the black diamond is the mean of the model means.

            **Confidence intervals**: thin whiskers on the model-mean squares are 95%
            bootstrap intervals over resampled subjects (Cars 2 (sEEG): depth probes), weighted by number of targets;
            each resample averages a model's layers in log$_2$ rank. LOSO intervals resample held-out subjects
            (only where there are at least five).

            **fMRI explained variance** is noise-normalized throughout: each voxel's $r^2$ is divided by its
            squared noise ceiling ($CC_{max}^2$, ceiling floored at 0.25; voxels without a ceiling are dropped), so
            it is a fraction of the explainable variance. Podcast (ECoG) and Cars 2 (sEEG) have no noise ceiling and use raw $r^2$.
            The appendix at the end repeats the fMRI figures with raw $r^2$.
    """)
    return


@app.cell
def _(
    MAIN_EXCLUDED,
    explained_variance,
    loso_pair,
    mo,
    model_cis,
    model_colors_on,
    rrr_aggregate,
    selected_pair,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    _cis = lambda mode: None if model_cis is None else model_cis[mode]
    _ev = without_models(explained_variance, MAIN_EXCLUDED)
    if selected_pair.value is None:
        _ev_aggregate = mo.md("No internal RRR curves are available.")
    else:
        _ev_figures = [
            mo.md("## Internal RRR aggregate"),
            rrr_aggregate(_ev.internal, _ev.loso, _ev.curves, _ev.loso_curves,
                          _ev.full_rows, selected_pair.value, mode="internal", intervals=_cis("internal"),
                          saturation_mode="explained_variance"),
        ]
        if loso_pair.value is not None:
            _ev_figures.extend([
                mo.md("## Leave-one-subject-out RRR aggregate"),
                rrr_aggregate(_ev.internal, _ev.loso, _ev.curves, _ev.loso_curves,
                              _ev.full_rows, loso_pair.value, mode="loso", intervals=_cis("loso"),
                              saturation_mode="explained_variance"),
            ])
        _ev_aggregate = mo.vstack(_ev_figures)
    _ev_aggregate
    return


@app.cell
def _(
    MAIN_EXCLUDED,
    cross_aggregate,
    cross_gap_scatter,
    explained_variance,
    mo,
    model_cis,
    model_colors_on,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    mo.vstack([
        mo.md("## Directed cross-dataset RRR aggregate"),
        cross_aggregate(without_models(explained_variance.cross, MAIN_EXCLUDED), "explained_variance",
                        intervals=None if model_cis is None else model_cis["auc"]),
        mo.md("## Internal minus transfer AUC gap correlation"),
        cross_gap_scatter(without_models(explained_variance.cross, MAIN_EXCLUDED), "explained_variance"),
    ])
    return


@app.cell
def _(
    MAIN_EXCLUDED,
    explained_variance,
    mo,
    model_colors_on,
    sensitivity_grid,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    _fractions = (.5, .75, .9, .95, .975, .99, 1.0)
    mo.vstack([
        mo.md("## Saturation sensitivity to required fraction of maximum explained variance"),
        sensitivity_grid(without_models(explained_variance.curves, MAIN_EXCLUDED), _fractions,
                         saturation_mode="explained_variance"),
        mo.md("## LOSO saturation sensitivity to required fraction of maximum explained variance"),
        sensitivity_grid(without_models(explained_variance.loso_curves, MAIN_EXCLUDED), _fractions, mode="loso",
                         saturation_mode="explained_variance"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Shuffled-embedding null control

    The internal RRR aggregate recomputed from `results/run_rrr_shuffled/`, where word embeddings
    were globally permuted against the neural data before fitting. Everything else matches the
    main internal figure (explained variance, fMRI noise-normalized, random GPT-2 left out, same
    sample model layer), so the two can be compared directly: if shuffling kills the effect, the
    curves should be near zero and the saturation ranks uninformative.
    """)
    return


@app.cell
def _(load_shuffled_internal_data, results_root):
    shuffled_ev, shuffled_gaps = load_shuffled_internal_data(results_root)
    return shuffled_ev, shuffled_gaps


@app.cell
def _(
    MAIN_EXCLUDED,
    mo,
    model_colors_on,
    rrr_aggregate,
    selected_pair,
    shuffled_ev,
    shuffled_gaps,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    _details = "\n".join(f"- {gap}" for gap in shuffled_gaps)
    _gap_note = mo.md(f"<details><summary>Shuffled result gaps ({len(shuffled_gaps)})</summary>\n\n"
                      f"{_details}\n\n</details>")
    if selected_pair.value is None or shuffled_ev.internal.empty:
        _shuffled = mo.vstack([mo.md("No shuffled internal RRR curves are available."), _gap_note])
    else:
        _sev = without_models(shuffled_ev, MAIN_EXCLUDED)
        _shuffled = mo.vstack([
            mo.md("## Internal RRR aggregate (shuffled embeddings)"),
            rrr_aggregate(_sev.internal, _sev.loso, _sev.curves, _sev.loso_curves,
                          _sev.full_rows, selected_pair.value, mode="internal",
                          saturation_mode="explained_variance"),
            _gap_note,
        ])
    _shuffled
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Random GPT-2 companions

    The main figures above leave out random GPT-2. In these companions it is drawn as its own
    distribution: a separate set of violins next to the other models, so its response profile
    can be compared directly. Only the distribution plots are shown; there are no curve plots or
    sensitivity analyses in this section.
    """)
    return


@app.cell(hide_code=True)
def _(
    RANDOM_MODEL,
    cross_aggregate,
    explained_variance,
    loso_pair,
    mo,
    model_cis,
    model_colors_on,
    rrr_aggregate,
    selected_pair,
):
    model_colors_on  # redraw when the point-color toggle changes
    _cis = lambda mode: None if model_cis is None else model_cis[mode]
    _ev = explained_variance
    if selected_pair.value is None:
        _companion = mo.md("No internal RRR curves are available.")
    else:
        _figures = [
            mo.md("## Internal RRR saturation distributions (random GPT-2 separate)"),
            rrr_aggregate(_ev.internal, _ev.loso, _ev.curves, _ev.loso_curves,
                          _ev.full_rows, selected_pair.value, mode="internal", intervals=_cis("internal"),
                          saturation_mode="explained_variance", separate_model=RANDOM_MODEL,
                          distributions_only=True),
        ]
        if loso_pair.value is not None:
            _figures.extend([
                mo.md("## Leave-one-subject-out saturation distributions (random GPT-2 separate)"),
                rrr_aggregate(_ev.internal, _ev.loso, _ev.curves, _ev.loso_curves,
                              _ev.full_rows, loso_pair.value, mode="loso", intervals=_cis("loso"),
                              saturation_mode="explained_variance", separate_model=RANDOM_MODEL,
                              distributions_only=True),
            ])
        _figures.extend([
            mo.md("## Directed cross-dataset AUC differences (random GPT-2 separate)"),
            cross_aggregate(_ev.cross, "explained_variance", separate_model=RANDOM_MODEL,
                            distributions_only=True, intervals=_cis("auc")),
        ])
        _companion = mo.vstack(_figures)
    _companion
    return


@app.cell
def _(decoding, mo):
    from src.utils.plotting import final_figures as _final_figures

    mo.vstack([mo.md("## Language decoding aggregate"),
               _final_figures.decoding_aggregate(decoding)])
    return


@app.cell
def _(encoding_grid, full_rows, mo, model_order):
    order = model_order(full_rows)
    mo.vstack([mo.md("## Full encoding-model performance"),
               encoding_grid(full_rows, order)])
    return (order,)


@app.cell
def _(encoding_grid, full_rows, order):
    encoding_grid(full_rows, order).savefig("encodingresults.png")
    return


@app.cell
def _(mo, order, significance_grid, significance_latex, significant):
    latex = significance_latex(significant, order)
    mo.vstack([
        mo.md("## Significant targets by model layer"),
        significance_grid(significant, order),
        mo.md("### Significant-target counts and dataset unions (LaTeX)"),
        mo.md("```latex\n" + latex + "\n```"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    # Appendix: target-quantile saturation

    "
        "The same analyses with saturation measured from target-correlation quantiles "
        "(the latest 99%-of-peak crossing among the 5th–95th percentile curves), to check "
        "that the conclusions also hold for lower-signal electrodes and voxels.
    """)
    return


@app.cell
def _(
    full_rows,
    internal,
    loso,
    loso_pair,
    loso_quantiles,
    mo,
    model_colors_on,
    quantiles,
    rrr_aggregate,
    selected_pair,
):
    model_colors_on  # redraw when the point-color toggle changes
    if selected_pair.value is None:
        _quantile_aggregate = mo.md("No internal RRR quantile curves are available.")
    else:
        _quantile_figures = [
            mo.md("## Internal RRR aggregate (quantile saturation)"),
            rrr_aggregate(internal, loso, quantiles, loso_quantiles,
                          full_rows, selected_pair.value, mode="internal"),
        ]
        if loso_pair.value is not None:
            _quantile_figures.extend([
                mo.md("## Leave-one-subject-out RRR aggregate (quantile saturation)"),
                rrr_aggregate(internal, loso, quantiles, loso_quantiles,
                              full_rows, loso_pair.value, mode="loso"),
            ])
        _quantile_aggregate = mo.vstack(_quantile_figures)
    _quantile_aggregate
    return


@app.cell
def _(cross, cross_aggregate, cross_gap_scatter, mo, model_colors_on):
    model_colors_on  # redraw when the point-color toggle changes
    mo.vstack([
        mo.md("## Directed cross-dataset RRR aggregate (correlation curves)"),
        cross_aggregate(cross),
        mo.md("## Internal minus transfer AUC gap correlation (correlation curves)"),
        cross_gap_scatter(cross),
    ])
    return


@app.cell
def _(mo, quantile_grid, quantiles, selected_pair):
    if selected_pair.value is None:
        quantile_output = mo.md("No selected model layer.")
    else:
        quantile_output = mo.vstack([
            mo.md("## RRR and PCA target quantiles by rank"),
            quantile_grid(quantiles, selected_pair.value),
        ])
    quantile_output
    return


@app.cell
def _(average_quantile_grid, quantiles):
    average_quantile_grid(quantiles).savefig("average-quantiles.png")
    return


@app.cell
def _(average_quantile_grid, mo, quantiles):
    mo.vstack([
        mo.md("## Mean normalized quantile curves across model layers"),
        average_quantile_grid(quantiles),
    ])
    return


@app.cell
def _(quantiles, sensitivity_grid):
    sensitivity_grid(quantiles, (0, .005, .01), variable="floor").savefig("floor-sensitivity.png")
    return


@app.cell
def _(mo, model_colors_on, quantiles, sensitivity_grid):
    model_colors_on  # redraw when the point-color toggle changes
    mo.vstack([
        mo.md("## Saturation sensitivity to minimum quantile correlation"),
        sensitivity_grid(quantiles, (0, .005, .01), variable="floor"),
        mo.md("## Saturation sensitivity to required peak performance"),
        sensitivity_grid(quantiles, (.5, .75, .9, .95, .975, .99, 1.0)),
    ])
    return


@app.cell
def _(quantiles, sensitivity_grid):
    sensitivity_grid(quantiles, (.5, .75, .9, .95, .975, .99, 1.0)).savefig("perf-sensitivity.png")
    return


@app.cell(hide_code=True)
def _(loso_quantiles, mo, model_colors_on, sensitivity_grid):
    model_colors_on  # redraw when the point-color toggle changes
    mo.vstack([
        mo.md("## LOSO saturation sensitivity to required peak performance"),
        sensitivity_grid(loso_quantiles, (.5, .75, .9, .95, .975, .99, 1.0), mode="loso"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    # Appendix: dimensionality and subspace consistency

    Per-layer distributions, PCA dimensionality of the neural targets and of the embeddings,
    ROI comparisons of RRR against PCA, and principal-angle comparisons of the RRR and PCA
    subspaces. Results or raw data that are not available on this machine are shown as
    empty panels and listed under the figure. The first run of each section is slow; its
    result is cached under `results/final_subspace_cache` and refreshed when RRR artifacts change.
    """)
    return


@app.cell
def _(repo_root):
    repo_root  # the first cell puts the repo on sys.path, so run after it
    from src.utils.plotting import final_subspace_data as subspace_data
    from src.utils.plotting import final_subspace_figures as subspace_figures

    return subspace_data, subspace_figures


@app.cell
def _(mo):
    mo.md("""
    ## Fraction of explainable variance captured (fMRI)

    Each voxel contributes $r^2 / CC_{max}^2$ at its saturation rank (the first rank reaching
    99% of its peak correlation, negatives clipped to zero). $CC_{max}$ is the per-voxel
    noise ceiling, floored at 0.25. There is no noise ceiling for the iEEG datasets.
    """)
    return


@app.cell
def _(mo, repo_root, results_root, subspace_data, subspace_figures):
    _gaps = []
    _fractions = subspace_data.attempt(_gaps, "fMRI noise-ceiling fractions", lambda: subspace_data.cached(
        results_root, "fmri_noise_ceiling_fractions",
        lambda: subspace_data.fmri_noise_ceiling_fractions(
            results_root, repo_root / "data" / "deep-fMRI-dataset")))
    mo.vstack([mo.md("\n".join(_gaps)) if _fractions is None else
               subspace_figures.noise_ceiling_histograms(_fractions, subspace_data.MODEL_LAYER_LIST)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## PCA dimensionality of the neural targets

    Cumulative variance explained by the principal components of the z-scored significant
    targets over the GloVe-covered words (iEEG: electrode-lag pairs; fMRI: voxels over the
    training-story TRs). Dashed lines mark 80%, 90% and 95% of the variance.
    """)
    return


@app.cell
def _(mo, repo_root, results_root, subspace_data, subspace_figures):
    _gaps = []
    _neural = {
        _dataset: subspace_data.attempt(_gaps, f"{_dataset} neural PCA", lambda: subspace_data.cached(
            results_root, f"neural_pca-{_dataset}",
            (lambda: subspace_data.fmri_neural_pca(results_root, repo_root)) if _dataset == "fmri" else
            (lambda: subspace_data.ieeg_neural_pca(results_root, repo_root, _dataset))))
        for _dataset in ("podcast", "cars-2", "fmri")
    }
    mo.vstack([subspace_figures.neural_pca_figure(_neural),
               mo.md("\n".join(f"- {_gap}" for _gap in _gaps))])
    return


@app.cell
def _(mo):
    mo.md("""
    ## PCA dimensionality of the word embeddings

    The same curves for every model layer's z-scored embeddings over each dataset's
    GloVe-covered words (Podcast (ECoG) and Cars 2 (sEEG): the cached epoch words; Moth (fMRI): the training-story words).
    """)
    return


@app.cell
def _(mo, repo_root, results_root, subspace_data, subspace_figures):
    _gaps, _figures = [], []
    for _dataset in ("podcast", "cars-2", "fmri"):
        _result = subspace_data.attempt(_gaps, f"{_dataset} embedding PCA", lambda: subspace_data.cached(
            results_root, f"embedding_pca-{_dataset}",
            lambda: subspace_data.embedding_pca(repo_root, _dataset)))
        if _result is not None:
            _gaps.extend(_result.get("gaps", []))
            _figures.append(subspace_figures.embedding_pca_figure(
                _result, subspace_data.MODEL_LAYER_LIST, _dataset))
    mo.vstack([*_figures, mo.md("\n".join(f"- {_gap}" for _gap in _gaps))])
    return


@app.cell
def _(mo):
    mo.md("""
    ## ROI comparison of RRR and PCA

    Mean $r^2$ per target within each bihemispheric ROI, with the normalized curves of every
    model layer and the paired RRR vs. PCA saturation ranks.
    """)
    return


@app.cell
def _(repo_root, results_root, subspace_data):
    from src.utils.atlas.core import bihemispheric_region_groups
    from src.utils.plotting.roi_saturation import load_roi_curves

    roi_names = list(bihemispheric_region_groups())
    _data = repo_root / "data"
    roi_curves, roi_counts, roi_gaps = subspace_data.cached(
        results_root, "roi_curves",
        lambda: load_roi_curves(
            results_root,
            {"podcast": _data, "cars-2": _data / "brain-treebank", "fmri": _data / "deep-fMRI-dataset"},
            regions=roi_names))
    return load_roi_curves, roi_counts, roi_curves, roi_gaps, roi_names


@app.cell
def _(
    MAIN_EXCLUDED,
    mo,
    model_cis,
    model_colors_on,
    roi_counts,
    roi_curves,
    roi_gaps,
    roi_names,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    from src.utils.plotting.roi_saturation import roi_normalized_curves, roi_saturation_distributions

    _main_curves = without_models(roi_curves, MAIN_EXCLUDED, index=2)
    mo.vstack([
        mo.md("### Normalized mean explained variance by ROI"),
        roi_normalized_curves(_main_curves, roi_counts, roi_names),
        mo.md("### Saturation rank by ROI"),
        roi_saturation_distributions(_main_curves, roi_names, roi_counts,
                                     intervals=None if model_cis is None else model_cis["roi"]),
        mo.md(f"{len(roi_gaps)} model layers had no results and are omitted."),
    ])
    return roi_normalized_curves, roi_saturation_distributions


@app.cell(hide_code=True)
def _(
    RANDOM_MODEL,
    mo,
    model_cis,
    model_colors_on,
    roi_counts,
    roi_curves,
    roi_names,
    roi_saturation_distributions,
):
    model_colors_on  # redraw when the point-color toggle changes
    mo.vstack([
        mo.md("### Saturation rank by ROI (random GPT-2 separate)"),
        roi_saturation_distributions(roi_curves, roi_names, roi_counts,
                                     separate_model=RANDOM_MODEL,
                                     intervals=None if model_cis is None else model_cis["roi"]),
    ])
    return


@app.cell
def _(mo):
    SAMPLE_MODEL, SAMPLE_LAYER = "gpt2-xl-batch", 24
    mo.md(f"""
    ## Subspace consistency of RRR and PCA

    Subspaces are compared in raw-embedding coordinates using their first $r$ columns.
    **Overlap** is the mean squared cosine of the principal angles (1 = identical subspaces;
    chance is $r/d$). Leave-one-subject-out (LOSO): the held-out subject's RRR factor against
    the factor learned on the other subjects (RRR) or against the PCA basis (PCA).
    Cross-dataset: whole-dataset bases of two datasets. Sample model layer:
    {SAMPLE_MODEL}, layer {SAMPLE_LAYER}. The aggregate figures use GPT-2, GPT-2 XL and RoBERTa Large.
    """)
    return SAMPLE_LAYER, SAMPLE_MODEL


@app.cell
def _(SAMPLE_LAYER, SAMPLE_MODEL, repo_root, results_root, subspace_data):
    subspace = subspace_data.load_subspace_data(results_root, repo_root, SAMPLE_MODEL, SAMPLE_LAYER)
    subspace_table = subspace_data.saturation_split_table(subspace["summaries"])
    return subspace, subspace_table


@app.cell
def _(SAMPLE_LAYER, SAMPLE_MODEL, mo, subspace, subspace_figures):
    mo.vstack([
        subspace_figures.subspace_consistency_figure(subspace, SAMPLE_MODEL, SAMPLE_LAYER),
        mo.md("### LOSO principal angles (squared cosines)"),
        subspace_figures.loso_principal_angle_figure(subspace, SAMPLE_MODEL, SAMPLE_LAYER),
        mo.md(f"<details><summary>Show {len(subspace['gaps'])} missing subspace results</summary>\n\n"
              + "\n".join(f"- {_gap}" for _gap in subspace["gaps"]) + "\n\n</details>"),
    ])
    return


@app.cell
def _(
    MAIN_EXCLUDED,
    mo,
    subspace,
    subspace_data,
    subspace_figures,
    without_models,
):
    _main_summaries = without_models(subspace["summaries"], MAIN_EXCLUDED)
    mo.vstack([
        mo.md("### LOSO RRR vs. PCA consistency aligned to the saturation rank"),
        subspace_figures.aligned_delta_figure(_main_summaries),
        mo.md("### Below vs. above the saturation rank (one line per layer)"),
        subspace_figures.saturation_split_figure(subspace_data.saturation_split_table(_main_summaries)),
    ])
    return


@app.cell(hide_code=True)
def _(RANDOM_MODEL, mo, subspace_figures, subspace_table):
    mo.vstack([
        mo.md("### Below vs. above the saturation rank (random GPT-2 separate)"),
        subspace_figures.saturation_split_figure(subspace_table, separate_model=RANDOM_MODEL),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Appendix: fMRI without noise normalization

    The main figures report fMRI explained variance as a fraction of the explainable variance
    ($r^2 / CC_{max}^2$ per voxel). Here the main analyses are repeated with raw $r^2$ for fMRI, to show
    that the conclusions do not depend on the normalization. Podcast (ECoG) and Cars 2 (sEEG) are identical to the
    main figures, and random GPT-2 is left out as there.
    """)
    return


@app.cell(hide_code=True)
def _(load_final_data, results_root):
    explained_variance_raw = load_final_data(results_root, fmri_noise_normalized=False)[9]
    return (explained_variance_raw,)


@app.cell(hide_code=True)
def _(
    explained_variance_raw,
    interval_tables,
    repo_root,
    results_root,
    roi_names,
    subspace_data,
):
    model_cis_raw = interval_tables(
        results_root, explained_variance_raw.loso_curves, subspace_data.MODEL_LAYER_LIST,
        {"podcast": repo_root / "data", "cars-2": repo_root / "data" / "brain-treebank",
         "fmri": repo_root / "data" / "deep-fMRI-dataset"}, roi_names, fmri_noise_normalized=False)
    return (model_cis_raw,)


@app.cell(hide_code=True)
def _(
    MAIN_EXCLUDED,
    cross_aggregate,
    explained_variance_raw,
    loso_pair,
    mo,
    model_cis_raw,
    model_colors_on,
    rrr_aggregate,
    selected_pair,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    _raw = without_models(explained_variance_raw, MAIN_EXCLUDED)
    _cis = lambda mode: None if model_cis_raw is None else model_cis_raw[mode]
    if selected_pair.value is None:
        _raw_figures = mo.md("No internal RRR curves are available.")
    else:
        _figures = [
            mo.md("## Internal RRR aggregate (raw fMRI r\u00b2)"),
            rrr_aggregate(_raw.internal, _raw.loso, _raw.curves, _raw.loso_curves,
                          _raw.full_rows, selected_pair.value, mode="internal",
                          saturation_mode="explained_variance", intervals=_cis("internal"),
                          fmri_noise_normalized=False),
        ]
        if loso_pair.value is not None:
            _figures.extend([
                mo.md("## Leave-one-subject-out RRR aggregate (raw fMRI r\u00b2)"),
                rrr_aggregate(_raw.internal, _raw.loso, _raw.curves, _raw.loso_curves,
                              _raw.full_rows, loso_pair.value, mode="loso",
                              saturation_mode="explained_variance", intervals=_cis("loso"),
                              fmri_noise_normalized=False),
            ])
        _figures.extend([
            mo.md("## Directed cross-dataset RRR aggregate (raw fMRI r\u00b2)"),
            cross_aggregate(_raw.cross, "explained_variance", intervals=_cis("auc")),
        ])
        _raw_figures = mo.vstack(_figures)
    _raw_figures
    return


@app.cell(hide_code=True)
def _(load_roi_curves, repo_root, results_root, roi_names, subspace_data):
    _data = repo_root / "data"
    roi_curves_raw, roi_counts_raw, _roi_gaps_raw = subspace_data.cached(
        results_root, "roi_curves_raw_fmri",
        lambda: load_roi_curves(
            results_root,
            {"podcast": _data, "cars-2": _data / "brain-treebank", "fmri": _data / "deep-fMRI-dataset"},
            regions=roi_names, fmri_noise_normalized=False))
    return roi_counts_raw, roi_curves_raw


@app.cell(hide_code=True)
def _(
    MAIN_EXCLUDED,
    mo,
    model_cis_raw,
    model_colors_on,
    roi_counts_raw,
    roi_curves_raw,
    roi_names,
    roi_normalized_curves,
    roi_saturation_distributions,
    without_models,
):
    model_colors_on  # redraw when the point-color toggle changes
    _main_raw = without_models(roi_curves_raw, MAIN_EXCLUDED, index=2)
    mo.vstack([
        mo.md("## Normalized mean explained variance by ROI (raw fMRI r\u00b2)"),
        roi_normalized_curves(_main_raw, roi_counts_raw, roi_names),
        mo.md("## Saturation rank by ROI (raw fMRI r\u00b2)"),
        roi_saturation_distributions(_main_raw, roi_names, roi_counts_raw,
                                     intervals=None if model_cis_raw is None else model_cis_raw["roi"]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Saturation rank as a function of lag (iEEG)

    For every model layer, each lag (-2 to 2 s) gets its own explained-variance rank curve over all
    electrodes at that lag (folds Fisher-averaged, negatives clipped, mean $r^2$); its saturation rank
    is the first rank reaching 99% of that curve's own maximum. Faint lines are individual RRR model
    layers (colored by model). The black line is the mean over RRR model layers and the red line the mean
    over PCA model layers (PCA layers not shown), both averaged in log$_2$ rank. Shaded bands are 95%
    bootstrap intervals of those means: clusters (Podcast (ECoG): subjects; Cars 2 (sEEG): depth probes) are resampled
    with replacement, shared by all layers, and the per-lag curves and saturation ranks recomputed in each
    resample (pointwise, not corrected across lags).

    Stars along the top mark lags where RRR saturates at a significantly lower rank than PCA: at each lag
    a model's layers are averaged in log$_2$ rank, then RRR is compared with PCA by a one-sided exact
    paired sign-flip test across models (as in the distribution plots), with Benjamini-Hochberg FDR control
    at 0.05 across the lags of each dataset.
    """)
    return


@app.cell(hide_code=True)
def _(MAIN_EXCLUDED, results_root):
    from src.utils.plotting.final_lag_saturation import (
        lag_saturation_figure, lag_saturation_significance, load_lag_saturation,
        load_lag_saturation_intervals,
    )

    # Cached in results/.final_lag_saturation_cache*.pkl (first computation takes a few minutes).
    lag_saturation = load_lag_saturation(results_root)
    lag_saturation_intervals = load_lag_saturation_intervals(results_root, exclude_models=MAIN_EXCLUDED)
    return (
        lag_saturation,
        lag_saturation_figure,
        lag_saturation_intervals,
        lag_saturation_significance,
    )


@app.cell(hide_code=True)
def _(
    MAIN_EXCLUDED,
    lag_saturation,
    lag_saturation_figure,
    lag_saturation_intervals,
    lag_saturation_significance,
    mo,
    without_models,
):
    _lag_saturation = without_models(lag_saturation, MAIN_EXCLUDED)
    _significance = lag_saturation_significance(_lag_saturation)
    _counts = "\n".join(
        f"- **{dataset}**: {int(flagged.sum())} of {len(flagged)} lags significant (RRR < PCA, BH FDR 0.05)"
        for dataset, (_, _, _, flagged) in _significance.items())
    mo.vstack([
        lag_saturation_figure(_lag_saturation, lag_saturation_intervals, _significance),
        mo.md(_counts),
    ])
    return


if __name__ == "__main__":
    app.run()

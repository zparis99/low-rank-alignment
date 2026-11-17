"""Batch export of every figure in notebooks/final_plots.py.

`--saturation-mode explained_variance` (default) writes the main figures: saturation is
the rank reaching 99% of the maximum mean r^2 per target. `--saturation-mode quantile`
writes the target-quantile appendix versions into a `quantile/` subfolder. `--all` writes both
of those plus the raw-fMRI-r^2 appendix (`raw_fmri/` subfolder) in one run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


RANDOM_MODEL = "random-gpt2"


def _layer(value):
    if value.lower() == "none":
        return None
    try:
        layer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("layer must be an integer or 'none'") from error
    if layer < 0:
        raise argparse.ArgumentTypeError("layer must be nonnegative or 'none'")
    return layer


def available_pairs(quantiles, *, loso=False):
    """Use the same eligibility criteria as the notebook's example dropdowns."""
    from src.utils.plotting.final_data import DATASETS

    methods = ("RRR baseline", "RRR transfer", "PCA baseline") if loso else ("rrr", "pca")
    present = {(key[0], key[1], key[2], key[-1]) for key in quantiles}
    return sorted(
        {(key[1], key[2]) for key in quantiles
         if all((dataset, key[1], key[2], method) in present
                for dataset in DATASETS for method in methods)},
        key=lambda pair: (pair[0], pair[1] is not None,
                          pair[1] if pair[1] is not None else -1),
    )


def _subspace_jobs(raw_fmri_only=False):
    """Appendix figures of dimensionality and subspace consistency (same data as the notebook).

    Datasets whose results or raw data are missing are drawn as empty panels. With
    `raw_fmri_only`, only the two ROI figures of the raw-fMRI-r^2 appendix.
    """
    from src.utils.plotting import final_subspace_data as data
    from src.utils.plotting import final_subspace_figures as figures

    root = Path(__file__).resolve().parents[3]
    results, gaps = root / "results", []
    model, layer = "gpt2-xl-batch", 24

    def neural():
        return {dataset: data.attempt(gaps, f"{dataset} neural PCA", lambda: data.cached(
            results, f"neural_pca-{dataset}",
            (lambda: data.fmri_neural_pca(results, root)) if dataset == "fmri" else
            (lambda: data.ieeg_neural_pca(results, root, dataset))))
            for dataset in ("podcast", "cars-2", "fmri")}

    def embedding(dataset):
        result = data.cached(results, f"embedding_pca-{dataset}",
                             lambda: data.embedding_pca(root, dataset))
        return figures.embedding_pca_figure(result, data.MODEL_LAYER_LIST, dataset)

    def main_summaries(summaries):
        from src.utils.plotting.final_figures import without_models

        return without_models(summaries, (RANDOM_MODEL,))

    def subspace():
        # One load shared by the four subspace figures.
        if "subspace" not in _CACHE:
            _CACHE["subspace"] = data.load_subspace_data(results, root, model, layer)
        return _CACHE["subspace"]

    def roi(raw_fmri=False):
        from src.utils.atlas.core import bihemispheric_region_groups
        from src.utils.plotting.roi_saturation import load_roi_curves

        names, raw = list(bihemispheric_region_groups()), root / "data"
        key = "roi_raw_fmri" if raw_fmri else "roi"
        if key not in _CACHE:
            _CACHE[key] = (names, *data.cached(
                results, "roi_curves_raw_fmri" if raw_fmri else "roi_curves",
                lambda: load_roi_curves(
                    results, {"podcast": raw, "cars-2": raw / "brain-treebank",
                              "fmri": raw / "deep-fMRI-dataset"}, regions=names,
                    **({"fmri_noise_normalized": False} if raw_fmri else {}))))
        return _CACHE[key]

    def roi_figure(name, separate=False, raw_fmri=False):
        from src.utils.plotting import roi_saturation
        from src.utils.plotting.final_figures import without_models

        names, curves, counts, _ = roi(raw_fmri)
        if not separate:
            curves = without_models(curves, (RANDOM_MODEL,), index=2)
        options = {"separate_model": RANDOM_MODEL} if separate else {}
        return (roi_saturation.roi_normalized_curves(curves, counts, names, **options)
                if name == "normalized"
                else roi_saturation.roi_saturation_distributions(curves, names, counts, **options))

    if raw_fmri_only:
        return [
            ("roi_normalized_curves", lambda: roi_figure("normalized", raw_fmri=True)),
            ("roi_saturation_distributions", lambda: roi_figure("saturation", raw_fmri=True)),
        ]
    jobs = [
        ("fmri_noise_ceiling_histograms", lambda: figures.noise_ceiling_histograms(
            data.cached(results, "fmri_noise_ceiling_fractions",
                        lambda: data.fmri_noise_ceiling_fractions(
                            results, root / "data" / "deep-fMRI-dataset")),
            data.MODEL_LAYER_LIST)),
        ("neural_pca_dimensionality", lambda: figures.neural_pca_figure(neural())),
        *((f"embedding_pca_{dataset}", lambda dataset=dataset: embedding(dataset))
          for dataset in ("podcast", "cars-2", "fmri")),
        ("roi_normalized_curves", lambda: roi_figure("normalized")),
        ("roi_saturation_distributions", lambda: roi_figure("saturation")),
        ("roi_saturation_distributions_random_gpt2", lambda: roi_figure("saturation", True)),
        ("subspace_consistency", lambda: figures.subspace_consistency_figure(subspace(), model, layer)),
        ("loso_principal_angles", lambda: figures.loso_principal_angle_figure(subspace(), model, layer)),
        ("loso_aligned_delta", lambda: figures.aligned_delta_figure(
            main_summaries(subspace()["summaries"]))),
        ("loso_saturation_split", lambda: figures.saturation_split_figure(
            data.saturation_split_table(main_summaries(subspace()["summaries"])))),
        ("loso_saturation_split_random_gpt2", lambda: figures.saturation_split_figure(
            data.saturation_split_table(subspace()["summaries"]), separate_model=RANDOM_MODEL)),
    ]
    return jobs


_CACHE = {}


def export_figures(data, output_dir, formats, example_pair, loso_pair,
                   saturation_mode="explained_variance", shuffled=None):
    """Save and close figures one at a time to keep export memory bounded.

    Explained-variance mode writes the main figures plus the figures that do not
    depend on the saturation mode (decoding, encoding, significance, gaps) to
    `output_dir`. Quantile mode writes its appendix figures to `output_dir/quantile`.
    `saturation_mode="raw_fmri"` (data loaded with `fmri_noise_normalized=False`) writes the
    raw-fMRI-r^2 appendix to `output_dir/raw_fmri`. `shuffled` is the
    `(ExplainedVarianceData, gaps)` pair from `load_shuffled_internal_data`; when given (or
    missing/empty) the shuffled-embedding internal aggregate is exported (or reported as a gap)
    in explained-variance mode.
    """
    import matplotlib.pyplot as plt
    from src.utils.plotting import final_figures as figures

    (internal, loso, cross, quantiles, loso_quantiles, full_rows,
     significant, decoding, gaps, explained_variance) = data
    fractions = (.5, .75, .9, .95, .975, .99, 1.0)
    output_dir = Path(output_dir)
    order = figures.model_order(full_rows)
    gaps = list(gaps)
    if saturation_mode == "explained_variance":
        # Main figures leave random GPT-2 out; the companion figures draw it as its own
        # distribution next to the other models.
        ev = figures.without_models(explained_variance, (RANDOM_MODEL,))
        full = explained_variance
        jobs = [
            ("rrr_internal_aggregate", lambda: figures.rrr_aggregate(
                ev.internal, ev.loso, ev.curves, ev.loso_curves, ev.full_rows, example_pair,
                mode="internal", saturation_mode=saturation_mode)),
            ("rrr_internal_aggregate_random_gpt2", lambda: figures.rrr_aggregate(
                full.internal, full.loso, full.curves, full.loso_curves, full.full_rows,
                example_pair, mode="internal", saturation_mode=saturation_mode,
                separate_model=RANDOM_MODEL, distributions_only=True)),
            ("rrr_loso_aggregate", lambda: figures.rrr_aggregate(
                ev.internal, ev.loso, ev.curves, ev.loso_curves, ev.full_rows, loso_pair,
                mode="loso", saturation_mode=saturation_mode)),
            ("rrr_loso_aggregate_random_gpt2", lambda: figures.rrr_aggregate(
                full.internal, full.loso, full.curves, full.loso_curves, full.full_rows,
                loso_pair, mode="loso", saturation_mode=saturation_mode,
                separate_model=RANDOM_MODEL, distributions_only=True)),
            ("rrr_cross_dataset_aggregate", lambda: figures.cross_aggregate(
                ev.cross, saturation_mode)),
            ("rrr_cross_dataset_aggregate_random_gpt2", lambda: figures.cross_aggregate(
                full.cross, saturation_mode, separate_model=RANDOM_MODEL,
                distributions_only=True)),
            ("rrr_cross_dataset_auc_gap", lambda: figures.cross_gap_scatter(
                ev.cross, saturation_mode)),
            ("saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
                ev.curves, fractions, saturation_mode=saturation_mode)),
            ("loso_saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
                ev.loso_curves, fractions, mode="loso", saturation_mode=saturation_mode)),
            ("language_decoding_aggregate", lambda: figures.decoding_aggregate(decoding)),
            ("encoding_performance", lambda: figures.encoding_grid(full_rows, order)),
            ("significant_targets", lambda: figures.significance_grid(significant, order)),
        ]
        if shuffled is not None:
            shuffled_ev, shuffled_gaps = shuffled
            gaps.extend(shuffled_gaps)
            if shuffled_ev.internal.empty:
                gaps.append("shuffled internal aggregate skipped: no shuffled results")
            else:
                sev = figures.without_models(shuffled_ev, (RANDOM_MODEL,))
                jobs.insert(1, ("rrr_internal_aggregate_shuffled", lambda: figures.rrr_aggregate(
                    sev.internal, sev.loso, sev.curves, sev.loso_curves, sev.full_rows,
                    example_pair, mode="internal", saturation_mode=saturation_mode)))
    elif saturation_mode == "quantile":
        output_dir = output_dir / "quantile"
        jobs = [
            ("rrr_internal_aggregate", lambda: figures.rrr_aggregate(
                internal, loso, quantiles, loso_quantiles, full_rows, example_pair,
                mode="internal")),
            ("rrr_loso_aggregate", lambda: figures.rrr_aggregate(
                internal, loso, quantiles, loso_quantiles, full_rows, loso_pair,
                mode="loso")),
            ("rrr_cross_dataset_aggregate", lambda: figures.cross_aggregate(cross)),
            ("rrr_cross_dataset_auc_gap", lambda: figures.cross_gap_scatter(cross)),
            ("rrr_pca_target_quantiles", lambda: figures.quantile_grid(quantiles, example_pair)),
            ("mean_normalized_quantiles", lambda: figures.average_quantile_grid(quantiles)),
            ("quantile_saturation_robustness", lambda: figures.quantile_saturation_robustness(quantiles)),
            ("saturation_sensitivity_floor", lambda: figures.sensitivity_grid(
                quantiles, (0, .005, .01), variable="floor")),
            ("saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
                quantiles, fractions)),
            ("loso_saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
                loso_quantiles, fractions, mode="loso")),
        ]
    elif saturation_mode == "raw_fmri":
        output_dir = output_dir / "raw_fmri"
        raw = figures.without_models(explained_variance, (RANDOM_MODEL,))
        jobs = [
            ("rrr_internal_aggregate", lambda: figures.rrr_aggregate(
                raw.internal, raw.loso, raw.curves, raw.loso_curves, raw.full_rows, example_pair,
                mode="internal", saturation_mode="explained_variance",
                fmri_noise_normalized=False)),
            ("rrr_loso_aggregate", lambda: figures.rrr_aggregate(
                raw.internal, raw.loso, raw.curves, raw.loso_curves, raw.full_rows, loso_pair,
                mode="loso", saturation_mode="explained_variance",
                fmri_noise_normalized=False)),
            ("rrr_cross_dataset_aggregate", lambda: figures.cross_aggregate(
                raw.cross, "explained_variance")),
            *_subspace_jobs(raw_fmri_only=True),
        ]
    else:
        raise ValueError(f"unknown saturation mode {saturation_mode!r}")
    if saturation_mode == "explained_variance":
        jobs.extend(_subspace_jobs())
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, build) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Building {name}...", flush=True)
        figure = build()
        try:
            for extension in dict.fromkeys(formats):
                path = output_dir / f"{name}.{extension}"
                print(f"Saving {path}...", flush=True)
                figure.savefig(path, dpi=300, bbox_inches="tight")
                paths.append(path)
                print(f"Saved {path}", flush=True)
        finally:
            plt.close(figure)
    if saturation_mode == "explained_variance":
        table_path = output_dir / "significant_targets.tex"
        print(f"Saving {table_path}...", flush=True)
        table_path.write_text(figures.significance_latex(significant, order) + "\n")
        paths.append(table_path)
        gaps_path = output_dir / "result_gaps.txt"
        print(f"Saving {gaps_path}...", flush=True)
        gaps_path.write_text("\n".join(gaps) + ("\n" if gaps else ""))
        paths.append(gaps_path)
    print(f"Export complete: {len(paths)} files in {output_dir}", flush=True)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for exported figures (existing filenames are replaced)")
    parser.add_argument("--format", nargs="+", choices=("pdf", "png", "svg"),
                        default=["pdf", "png", "svg"], help="Output formats (default: pdf png svg)")
    parser.add_argument("--saturation-mode", choices=("explained_variance", "quantile"),
                        default="explained_variance",
                        help="explained_variance (default): saturation is the rank reaching 99%% of "
                             "the maximum mean r^2 per target. quantile: the target-quantile "
                             "appendix versions, written to <output-dir>/quantile.")
    parser.add_argument("--all", action="store_true",
                        help="Export every notebook figure: the explained-variance figures to "
                             "<output-dir>, the quantile appendix to <output-dir>/quantile and the "
                             "raw-fMRI-r^2 appendix to <output-dir>/raw_fmri. --saturation-mode is ignored.")
    parser.add_argument("--example-model", default="gpt2")
    parser.add_argument("--example-layer", type=_layer, default=0,
                        help="Example layer, or 'none' for a layerless model (default: 0)")
    parser.add_argument("--loso-example-model", help="Override the LOSO example model")
    parser.add_argument("--loso-example-layer", type=_layer, default=argparse.SUPPRESS,
                        help="Override the LOSO example layer, or 'none'")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from src.utils.plotting.final_data import load_final_data, load_shuffled_internal_data

    results_root = Path(__file__).resolve().parents[3] / "results"
    print(f"Loading result summaries from {results_root} (using the cache when valid)...", flush=True)
    data = load_final_data(results_root)
    print("Result summaries loaded. Validating example model/layers...", flush=True)
    example_pair = (args.example_model, args.example_layer)
    loso_pair = (args.loso_example_model or args.example_model,
                 getattr(args, "loso_example_layer", args.example_layer))
    curves = (data[3], data[4]) if args.saturation_mode == "quantile" and not args.all else (
        data[9].curves, data[9].loso_curves)
    for label, pair, quantiles, is_loso in (
        ("internal", example_pair, curves[0], False),
        ("LOSO", loso_pair, curves[1], True),
    ):
        pairs = available_pairs(quantiles, loso=is_loso)
        if pair not in pairs:
            available = ", ".join(f"{model}/{'none' if layer is None else layer}"
                                  for model, layer in pairs) or "none"
            parser.error(f"Unavailable {label} example {pair}. Available model/layer pairs: {available}")
    print(f"Examples: internal={example_pair}, LOSO={loso_pair}", flush=True)
    print(f"Missing or incomplete results: {len(data[8])}; exporting available results.", flush=True)
    if not args.all:
        shuffled = (load_shuffled_internal_data(results_root)
                    if args.saturation_mode == "explained_variance" else None)
        return export_figures(data, args.output_dir, args.format, example_pair, loso_pair,
                              args.saturation_mode, shuffled)
    paths = export_figures(data, args.output_dir, args.format, example_pair, loso_pair,
                           "explained_variance", load_shuffled_internal_data(results_root))
    paths += export_figures(data, args.output_dir, args.format, example_pair, loso_pair, "quantile")
    print("Loading raw-fMRI-r^2 result summaries...", flush=True)
    raw_data = load_final_data(results_root, fmri_noise_normalized=False)
    paths += export_figures(raw_data, args.output_dir, args.format, example_pair, loso_pair, "raw_fmri")
    return paths

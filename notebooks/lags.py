import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import sys
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.run_rrr import Artifact

    return Artifact, Path, np, plt


@app.cell
def _(Artifact, Path):
    art = Artifact(mode="internal", dataset="podcast", model="gpt2-xl-batch", layer=12, results_root=Path("results"))
    return (art,)


@app.cell
def _(art):
    m = art.load_rank_matrix(data_root="data")
    return (m,)


@app.cell
def _(m):
    m.targets
    return


@app.cell
def _(m):
    m.values.shape
    return


@app.cell
def _(np):
    from src.utils.rrr.plot_metrics import mean_saturation_rank

    threshold = 0.99
    # Interpolated rank at which each electrode first reaches `threshold` of its peak
    # (same linear-in-rank interpolation as final_plots, via mean_saturation_rank).
    def per_electrode_min_saturation_rank(per_target_corrs, ranks, threshold):
        _ranks = np.asarray(ranks, dtype=float)
        _out = np.empty(per_target_corrs.shape[1])
        for _e in range(per_target_corrs.shape[1]):
            _curve = per_target_corrs[:, _e]
            # Preserve previous behavior: electrodes with a negative peak saturate at the first rank.
            if np.max(_curve) < 0:
                _out[_e] = _ranks[0]
            else:
                _out[_e] = mean_saturation_rank(_ranks, _curve, threshold)
        return _out


    def min_saturation_rank(per_elec_means, threshold):
        min_rank_idx = []
        for curr_curve in per_elec_means:
            max_val = curr_curve.max()
            for _j, curr_val in enumerate(curr_curve):
                if curr_val > threshold * max_val:
                    min_rank_idx.append(_j)
        return min_rank_idx

    return min_saturation_rank, per_electrode_min_saturation_rank, threshold


@app.cell
def _(m, np, plt):
    all_z = np.arctanh(m.values).mean((1)).clip(0)
    var_curve = np.square(np.tanh(all_z))
    # plt.semilogx(m.ranks, var_curve, label="rrr")
    plt.semilogx(m.ranks, var_curve.mean(1), label="rrr")
    # m_pca = art.load_rank_matrix("pca", data_root="data")
    # all_z = np.arctanh(m_pca.values).mean((1, 3)).clip(0)
    # var_curve = np.square(np.tanh(all_z)).sum(1)
    # plt.semilogx(m_pca.ranks, var_curve, label="pca")
    # plt.legend()
    # plt.xlabel("Ranks")
    # plt.ylabel("Cumulative r^2")
    # plt.show()
    return (var_curve,)


@app.cell
def _(m, per_electrode_min_saturation_rank, plt, var_curve):
    outs = per_electrode_min_saturation_rank(var_curve.mean(1), m.ranks, .99)
    plt.semilogy(outs)
    return


@app.cell
def _(best_ordered, m, plt):
    plt.semilogx(m.ranks, m.values[:, :, best_ordered[200], :].mean((1, 2)).T)
    return


@app.cell
def _(m, np):
    best_ordered = np.flip(np.argsort(m.values.mean((1, 3)).max(0)))
    return (best_ordered,)


@app.cell
def _(m):
    m.targets
    return


@app.cell
def _(best_ordered, m, plt):
    plt.plot(m.values[:, :, best_ordered[400:500]].mean((1, 2)).T)
    return


@app.cell
def _(Artifact, Path, plt):
    cars_art = Artifact(mode="internal", dataset="brain-treebank-cars-2", model="gpt2", layer=0, results_root=Path("results"))
    m_cars = cars_art.load_rank_matrix(data_root="data/brain-treebank/")
    plt.plot(m_cars.lags, m_cars.values.mean(axis=(1, 2)).T)
    return (m_cars,)


@app.cell
def _(m_cars):
    m_cars.targets
    return


@app.cell
def _(np, plt):
    def plot_electrode_glass_brain(
        coords,
        values,
        log_scale=False,
        cmap="viridis",
        vmin=None,
        vmax=None,
        marker_size=30,
        alpha=1.0,
        figsize=(8, 7),
        views=("l", "r", "y", "z"),
        title=None,
        colorbar_label=None,
    ):
        """Plot electrodes on a nilearn glass brain, colored by `values`.

        coords: [n, 3] MNI coordinates. values: [n] numeric array parallel to coords.
        log_scale: use a log color scale (all plotted values must be > 0) instead of linear.
        """
        from matplotlib import colors as mcolors
        from matplotlib import ticker as mticker
        from matplotlib.cm import ScalarMappable
        from nilearn import plotting

        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape [n, 3], got {coords.shape}")
        if values.shape != (coords.shape[0],):
            raise ValueError(f"values must have shape ({coords.shape[0]},), got {values.shape}")

        # Drop electrodes with missing coordinates or values.
        keep = np.isfinite(coords).all(axis=1) & np.isfinite(values)
        coords, values = coords[keep], values[keep]

        vmin = np.nanmin(values) if vmin is None else vmin
        vmax = np.nanmax(values) if vmax is None else vmax
        if log_scale:
            if vmin <= 0:
                raise ValueError("log_scale requires vmin > 0; pass a positive vmin or filter non-positive values")
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        mappable = ScalarMappable(norm=norm, cmap=cmap)
        colors = mappable.to_rgba(values)

        colors[:, 3] = alpha

        # 2x2 grid of single-view glass brains (row-major order of `views`) plus a thin colorbar row.
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(3, 2, height_ratios=[10, 10, 0.6], hspace=0.1, wspace=0.05)
        displays = []
        for i, view in enumerate(views):
            ax = fig.add_subplot(grid[i // 2, i % 2])
            display = plotting.plot_glass_brain(None, display_mode=view, axes=ax)
            display.add_markers(coords, marker_color=colors, marker_size=marker_size)
            displays.append(display)
        if title:
            fig.suptitle(title)
        cbar_ax = fig.add_subplot(grid[2, :])

        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
        if log_scale:
            cbar.ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # drop cluttered minor tick labels
        if colorbar_label:
            cbar.set_label(colorbar_label)
        return fig

    return (plot_electrode_glass_brain,)


@app.cell
def _(
    m,
    per_electrode_min_saturation_rank,
    plot_electrode_glass_brain,
    plt,
    threshold,
):
    _min_saturation_rank = per_electrode_min_saturation_rank(m.values.mean((1, 3)), m.ranks, threshold)
    plot_electrode_glass_brain(m.targets[["x", "y", "z"]].values, _min_saturation_rank, log_scale=True)
    plt.show()
    return


@app.cell
def _(
    m_cars,
    per_electrode_min_saturation_rank,
    plot_electrode_glass_brain,
    plt,
    threshold,
):
    _min_saturation_rank = per_electrode_min_saturation_rank(m_cars.values.mean((1, 3)), m_cars.ranks, threshold)
    plot_electrode_glass_brain(m_cars.targets[["x", "y", "z"]].values, _min_saturation_rank, log_scale=True)
    plt.show()
    return


@app.cell
def _():
    import os

    return


@app.cell
def _(Artifact, Path):
    _art = Artifact(mode="internal", dataset="podcast", model="glove-300", layer=None, results_root=Path("results"))
    _art.load_rank_matrix()
    return


@app.cell
def _(Artifact, Path, plt):
    runs = Artifact.list_available(results_root="results", dataset="podcast", mode="internal")
    per_model_layer_means = []
    for _run in runs:
        _model = _run["model"]
        _layer = _run["layer"]
        _art = Artifact(mode="internal", dataset="podcast", model=_model, layer=_layer, results_root=Path("results"))
        _rank_matrix = _art.load_rank_matrix(data_root="data/")
        _means = _rank_matrix.values.mean((1, 2, 3))
        per_model_layer_means.append(_means)
        plt.plot(_means, label=f"{_model}-{_layer}")
    plt.legend()
    plt.show()
    return (per_model_layer_means,)


@app.cell
def _(min_saturation_rank, per_model_layer_means, threshold):
    min_saturation_rank(per_model_layer_means, threshold)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

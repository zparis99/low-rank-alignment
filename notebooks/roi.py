import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import configparser
    import importlib.util
    import sys
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.run_rrr import Artifact

    FMRI_DATA_ROOT = repo_root / "data" / "deep-fMRI-dataset"
    PYCORTEX_DB = FMRI_DATA_ROOT / "data" / "ds003020" / "derivative" / "pycortex-db"


    def ensure_pycortex_filestore(filestore):
        """Make pycortex's user config point its filestore at `filestore`; write it if not.

        pycortex reads this config once, at first import, so this must run before
        `import cortex`. Returns True if the config file was written or changed.
        """
        filestore = str(Path(filestore).resolve())
        if not Path(filestore).is_dir():
            raise FileNotFoundError(f"pycortex db not found: {filestore}")
        # Load cortex/appdirs.py standalone (importing the package would read the config).
        spec = importlib.util.find_spec("cortex")
        appdirs_path = Path(spec.submodule_search_locations[0]) / "appdirs.py"
        appdirs_spec = importlib.util.spec_from_file_location("_pycortex_appdirs", appdirs_path)
        appdirs = importlib.util.module_from_spec(appdirs_spec)
        appdirs_spec.loader.exec_module(appdirs)
        cfg_path = Path(appdirs.user_data_dir("pycortex")) / "options.cfg"

        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)
        if cfg.get("basic", "filestore", fallback=None) == filestore:
            changed = False
        else:
            if not cfg.has_section("basic"):
                cfg.add_section("basic")
            cfg.set("basic", "filestore", filestore)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w") as handle:
                cfg.write(handle)
            changed = True

        if "cortex" in sys.modules:
            import cortex

            if str(Path(cortex.database.default_filestore).resolve()) != filestore:
                raise RuntimeError(
                    "pycortex was imported before its filestore was configured; "
                    "restart the marimo kernel so the updated config is read."
                )
        return changed


    pycortex_config_written = ensure_pycortex_filestore(PYCORTEX_DB)
    return Artifact, FMRI_DATA_ROOT, Path, np, plt, repo_root


@app.cell
def _(FMRI_DATA_ROOT, Path, np, plt):
    def plot_fmri_flatmap(
        subject,
        voxel_index,
        values,
        data_root=FMRI_DATA_ROOT,
        log_scale=False,
        cmap="viridis",
        vmin=None,
        vmax=None,
        alpha_values=None,
        alpha_vmin=0.0,
        alpha_vmax=None,
        with_curvature=True,
        with_colorbar=True,
        title=None,
        colorbar_label=None,
        alpha_label=None,
        figsize=(10, 6),
    ):
        """Plot per-voxel `values` for one fMRI subject on a pycortex flat map.

        voxel_index: [n] 0-based indices into the subject's `mask_thick` cortical voxels
        (the `voxel_index` column of `RankMatrix.targets`). values: [n] array parallel to it.
        Voxels not listed (or with non-finite values) are left unplotted.
        log_scale: color on a log10 scale (all plotted values must be > 0); vmin/vmax are
        given in data units and the colorbar is labelled in powers of ten.

        alpha_values: optional [n] array parallel to `values` that sets each voxel's opacity
        (a two-dimensional color scheme). It maps linearly from fully transparent at
        `alpha_vmin` to fully opaque at `alpha_vmax` (default: its maximum). The colorbar
        is replaced by a corner legend of the 2D scheme.
        """
        import json

        import cortex

        voxel_index = np.asarray(voxel_index, dtype=int)
        values = np.asarray(values, dtype=float)
        if values.shape != voxel_index.shape or voxel_index.ndim != 1:
            raise ValueError(f"voxel_index and values must be matching 1D arrays, got {voxel_index.shape} and {values.shape}")
        if alpha_values is not None:
            alpha_values = np.asarray(alpha_values, dtype=float)
            if alpha_values.shape != values.shape:
                raise ValueError(f"alpha_values must have shape {values.shape}, got {alpha_values.shape}")

        xfm_path = Path(data_root) / "data" / "ds003020" / "derivative" / "subject_xfms.json"
        xfm_names = json.loads(xfm_path.read_text()) if xfm_path.exists() else {}
        xfm = xfm_names.get(subject, f"{subject}_auto")

        mask = cortex.db.get_mask(subject, xfm, "thick")
        if voxel_index.size and (voxel_index.min() < 0 or voxel_index.max() >= mask.sum()):
            raise ValueError(f"{subject} has {int(mask.sum())} cortical voxels but voxel_index spans {voxel_index.min()}..{voxel_index.max()}")
        if log_scale and np.any(values[np.isfinite(values)] <= 0):
            raise ValueError("log_scale requires all plotted values > 0")

        full = np.full(int(mask.sum()), np.nan)
        full[voxel_index] = values
        finite = full[np.isfinite(full)]
        vmin = np.nanmin(finite) if vmin is None else vmin
        vmax = np.nanmax(finite) if vmax is None else vmax
        if log_scale:
            full, vmin, vmax = np.log10(full), np.log10(vmin), np.log10(vmax)

        fig = plt.figure(figsize=figsize)
        if alpha_values is None:
            volume = cortex.Volume(full, subject, xfm, mask=mask, vmin=vmin, vmax=vmax, cmap=cmap)
            cortex.quickflat.make_figure(
                volume, fig=fig, with_curvature=with_curvature, with_colorbar=with_colorbar,
                with_rois=False, with_labels=False, with_sulci=False,
            )
            if with_colorbar and len(fig.axes) > 1:
                cbar_ax = fig.axes[-1]
                if log_scale:
                    powers = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
                    cbar_ax.set_xticks(powers)
                    cbar_ax.set_xticklabels([f"$10^{{{int(p)}}}$" for p in powers])
                if colorbar_label:
                    cbar_ax.set_xlabel(colorbar_label)
        else:
            a_full = np.full(int(mask.sum()), np.nan)
            a_full[voxel_index] = alpha_values
            alpha_vmax = np.nanmax(alpha_values) if alpha_vmax is None else alpha_vmax
            if not alpha_vmax > alpha_vmin:
                raise ValueError("alpha_vmax must be greater than alpha_vmin")
            cmap_obj = plt.get_cmap(cmap)
            rgba = cmap_obj(np.clip((full - vmin) / (vmax - vmin), 0, 1))
            alpha = np.clip((a_full - alpha_vmin) / (alpha_vmax - alpha_vmin), 0, 1)
            alpha[~(np.isfinite(full) & np.isfinite(a_full))] = 0.0
            channels = [
                cortex.Volume((rgba[:, i] * 255).astype(np.uint8), subject, xfm, mask=mask, vmin=0, vmax=255)
                for i in range(3)
            ]
            volume = cortex.VolumeRGB(
                *channels, subject, xfm,
                alpha=cortex.Volume(alpha, subject, xfm, mask=mask, vmin=0, vmax=1),
            )
            cortex.quickflat.make_figure(
                volume, fig=fig, with_curvature=with_curvature, with_colorbar=False,
                with_rois=False, with_labels=False, with_sulci=False,
            )
            # Corner legend of the 2D scheme: color along x, opacity along y.
            main_axes = list(fig.axes)
            legend_ax = fig.add_axes([0.04, 0.06, 0.2, 0.22])
            xs, ys = np.linspace(0, 1, 128), np.linspace(0, 1, 64)
            grid = cmap_obj(np.tile(xs, (len(ys), 1)))
            grid[..., 3] = ys[:, None]
            legend_ax.imshow(grid, origin="lower", aspect="auto", extent=(vmin, vmax, alpha_vmin, alpha_vmax))
            if log_scale:
                powers = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
                legend_ax.set_xticks(powers)
                legend_ax.set_xticklabels([f"$10^{{{int(p)}}}$" for p in powers])
            legend_ax.tick_params(labelsize=7)
            legend_ax.set_xlabel(colorbar_label or "value", fontsize=8)
            legend_ax.set_ylabel(alpha_label or "opacity", fontsize=8)
        if title:
            fig.suptitle(title)
        return fig


    return (plot_fmri_flatmap,)


@app.cell
def _(Artifact, FMRI_DATA_ROOT, repo_root):
    _art = Artifact(mode="internal", dataset="deep-fmri", model="llama3-8b", layer=16, results_root=repo_root / "results")
    m_llama = _art.load_rank_matrix(data_root=FMRI_DATA_ROOT)
    return (m_llama,)


@app.cell
def _(
    load_fmri_noise_ceiling,
    m_llama,
    np,
    per_target_saturation_rank,
    plot_fmri_flatmap,
    plt,
    smooth_over_neighbors,
):
    _sub = "UTS01"
    SMOOTH_RADIUS_MM = 5.0
    _rows = (m_llama.targets["subject"] == _sub).to_numpy()
    _voxels = m_llama.targets.loc[_rows, "voxel_index"].to_numpy()
    _xyz = m_llama.targets.loc[_rows, ["x", "y", "z"]].to_numpy()
    # Average the correlation curves (and ceilings) over a spherical window, then take the saturation rank.
    _curves = smooth_over_neighbors(m_llama.values[:, 0, _rows], _xyz, SMOOTH_RADIUS_MM)
    _cc_max = smooth_over_neighbors(load_fmri_noise_ceiling(_sub, _voxels), _xyz, SMOOTH_RADIUS_MM)
    _sat = per_target_saturation_rank(_curves, m_llama.ranks)
    CC_MAX_FLOOR = 0.25
    _cc_norm = _curves.max(axis=0) / np.maximum(_cc_max, CC_MAX_FLOOR)
    plot_fmri_flatmap(
        _sub,
        _voxels,
        _sat,
        log_scale=True,
        cmap="magma_r",
        alpha_values=_cc_norm,
        alpha_vmax=1.0,
        title=f"{_sub} llama3-8b layer 16: saturation rank, {SMOOTH_RADIUS_MM:g} mm window",
        colorbar_label="saturation rank",
        alpha_label="peak CC_norm (r / CC_max)",
    )
    plt.show()
    return (CC_MAX_FLOOR,)


@app.cell
def _(np):
    from src.utils.rrr.plot_metrics import mean_saturation_rank


    def per_target_saturation_rank(curves, ranks, threshold=0.99):
        """Saturation rank for every target, given its metric-vs-rank curve.

        curves: [n_ranks, n_targets] (e.g. voxels, electrodes, or electrode-lag pairs
        flattened into one axis). ranks: [n_ranks]. Returns [n_targets]: the
        linearly interpolated rank at which each curve first reaches `threshold` of its
        own peak (see `mean_saturation_rank`). Targets whose peak is negative saturate
        at the first rank; targets with no finite values get NaN.
        """
        curves = np.asarray(curves, dtype=float)
        ranks = np.asarray(ranks, dtype=float)
        if curves.ndim != 2 or curves.shape[0] != len(ranks):
            raise ValueError(f"curves must have shape [{len(ranks)}, n_targets], got {curves.shape}")
        out = np.empty(curves.shape[1])
        for _t in range(curves.shape[1]):
            _curve = curves[:, _t]
            if np.max(_curve) < 0:
                out[_t] = ranks[0]
            else:
                out[_t] = mean_saturation_rank(ranks, _curve, threshold)
        return out


    return (per_target_saturation_rank,)


@app.cell
def _(Path, np, repo_root):
    def load_fmri_noise_ceiling(subject, voxel_index, results_root=repo_root / "results"):
        """Per-voxel noise-ceiling CC_max (Schoppe et al. 2016) for `voxel_index` of `subject`.

        Reads `results/fmri_noise_ceiling/<subject>.npz` (written by
        `scripts/fmri/compute_noise_ceiling.py`), which covers every `mask_thick` voxel, so it is
        indexed by the same `voxel_index` as `RankMatrix.targets`. NaN where the voxel's
        signal power is non-positive.
        """
        path = Path(results_root) / "fmri_noise_ceiling" / f"{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/fmri/compute_noise_ceiling.py")
        with np.load(path) as archive:
            return archive["cc_max"][np.asarray(voxel_index, dtype=int)].astype(float)


    return (load_fmri_noise_ceiling,)


@app.cell
def _(np):
    def smooth_over_neighbors(values, coords, radius):
        """Average `values` over all targets within `radius` of each target (itself included).

        values: [..., n_targets] (e.g. the [n_ranks, n_targets] curves, or a per-target vector).
        coords: [n_targets, 3] positions for ONE subject / electrode set, in the units of `radius`
        (mm for the `x`, `y`, `z` target columns). Uniform weights over a spherical window,
        averaging the values themselves (NaNs are ignored; all-NaN windows give NaN). Smooth
        before computing any per-target statistic such as the saturation rank.
        """
        from scipy import sparse
        from scipy.spatial import cKDTree

        values = np.asarray(values, dtype=float)
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or values.shape[-1] != coords.shape[0]:
            raise ValueError(f"values must end in n_targets={coords.shape[0]}, got {values.shape}")
        pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
        n = coords.shape[0]
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        window = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        flat = values.reshape(-1, n)
        finite = np.isfinite(flat)
        totals = np.asarray(window @ np.where(finite, flat, 0.0).T).T
        counts = np.asarray(window @ finite.T.astype(float)).T
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(counts > 0, totals / counts, np.nan)
        return smoothed.reshape(values.shape)


    return (smooth_over_neighbors,)


@app.cell
def _(
    CC_MAX_FLOOR,
    load_fmri_noise_ceiling,
    m_llama,
    np,
    per_target_saturation_rank,
    plot_fmri_flatmap,
    plt,
    smooth_over_neighbors,
):
    import marimo as mo

    _sub = "UTS01"
    SMOOTH_RADII_MM = [3.0, 5.0, 8.0, 12.0]
    _rows = (m_llama.targets["subject"] == _sub).to_numpy()
    _voxels = m_llama.targets.loc[_rows, "voxel_index"].to_numpy()
    _xyz = m_llama.targets.loc[_rows, ["x", "y", "z"]].to_numpy()
    _raw_cc_max = load_fmri_noise_ceiling(_sub, _voxels)
    _figs = []
    for _radius in SMOOTH_RADII_MM:
        _curves = smooth_over_neighbors(m_llama.values[:, 0, _rows], _xyz, _radius)
        _cc_max = smooth_over_neighbors(_raw_cc_max, _xyz, _radius)
        _fig = plot_fmri_flatmap(
            _sub,
            _voxels,
            per_target_saturation_rank(_curves, m_llama.ranks),
            log_scale=True,
            cmap="magma_r",
            alpha_values=_curves.max(axis=0) / np.maximum(_cc_max, CC_MAX_FLOOR),
            alpha_vmax=1.0,
            title=f"{_sub} llama3-8b layer 16: saturation rank, {_radius:g} mm window",
            colorbar_label="saturation rank",
            alpha_label="peak CC_norm",
            figsize=(8, 4.8),
        )
        _figs.append(_fig)
        plt.close(_fig)
    mo.vstack(_figs)
    return


@app.cell
def _(
    Artifact,
    CC_MAX_FLOOR,
    FMRI_DATA_ROOT,
    load_fmri_noise_ceiling,
    np,
    repo_root,
):
    import yaml

    # The configured fMRI cross-model/layer set (same union for every model's config).
    _cfg = yaml.safe_load(
        (repo_root / "configs/fmri/within_fmri_cross_model_layer/gpt2.yml").read_text()
    )
    MODEL_LAYERS = []
    for _entry in _cfg["significance_selection"]["artifacts"]:
        _layers = _entry["layer"]
        _layers = [None] if _layers is None else list(np.atleast_1d(_layers))
        MODEL_LAYERS.append((_entry["model"], [None if _l is None else int(_l) for _l in _layers]))

    explained_variance = {}  # (model, layer) -> [n_voxels] r^2 / CC_max^2 at each voxel's saturation rank
    _targets = None
    _cc_max = None
    for _model, _layers in MODEL_LAYERS:
        for _layer in _layers:
            _artifact = Artifact(
                mode="internal", dataset="deep-fmri", model=_model, layer=_layer,
                results_root=repo_root / "results",
            )
            # Select the corrected pooled-training-story input-scaling variant (as in plots.py).
            _variants = [
                _v for _v in sorted(_artifact._read_json(_artifact._leaf() / "manifest.json", default={}).get("variants", {}))
                if "input_scaling-pooled-training-stories-v1" in _v.split("/")
            ]
            if len(_variants) != 1:
                raise ValueError(f"{_model}/layer-{_layer}: expected one pooled-training-stories variant, found {_variants}")
            _artifact.variant = _variants[0]
            _matrix = _artifact.load_rank_matrix(data_root=FMRI_DATA_ROOT)
            if _targets is None:
                _targets = _matrix.targets[["subject", "voxel_index"]].reset_index(drop=True)
                _cc_max = np.full(len(_targets), np.nan)
                for _sub in _targets["subject"].unique():
                    _rows = (_targets["subject"] == _sub).to_numpy()
                    _cc_max[_rows] = load_fmri_noise_ceiling(_sub, _targets.loc[_rows, "voxel_index"].to_numpy())
            elif not _matrix.targets[["subject", "voxel_index"]].reset_index(drop=True).equals(_targets):
                raise ValueError(f"{_model} layer {_layer}: voxel targets differ from the first artifact")
            _curves = _matrix.values[:, 0, :]  # single fold (fMRI held-out story)
            # Saturation rank = interpolated first crossing of 99% of the voxel's peak; the next
            # tested rank at or above it is the first tested rank whose r reaches that threshold.
            _reached = _curves >= 0.99 * _curves.max(axis=0)
            _r_sat = _curves[_reached.argmax(axis=0), np.arange(_curves.shape[1])]
            _r_sat = np.clip(_r_sat, 0.0, None)  # R^2: clip negative correlations to zero first
            explained_variance[(_model, _layer)] = _r_sat**2 / np.maximum(_cc_max, CC_MAX_FLOOR) ** 2
            del _matrix, _curves, _reached

    # Voxels without a defined ceiling (non-positive signal power) are dropped.
    VALID_VOXELS = np.isfinite(_cc_max)
    return MODEL_LAYERS, VALID_VOXELS, explained_variance


@app.cell
def _(MODEL_LAYERS, VALID_VOXELS, explained_variance, np, plt):
    X_MAX = 1.5
    _bins = np.linspace(0.0, X_MAX, 61)
    _DISPLAY_NAMES = {
        "gpt2": "GPT-2", "gpt2-xl-batch": "GPT-2 XL", "gpt-j-6b": "GPT-J 6B",
        "roberta-large": "RoBERTa Large", "llama3-8b": "Llama 3 8B",
        "random-gpt2": "Random GPT-2", "glove-300": "GloVe 300D",
    }
    _n_cols = max(len(_layers) for _, _layers in MODEL_LAYERS)
    _fig, _axes = plt.subplots(
        len(MODEL_LAYERS), _n_cols, figsize=(2.6 * _n_cols, 1.9 * len(MODEL_LAYERS)),
        sharex=True, sharey=True, squeeze=False,
    )
    for _i, (_model, _layers) in enumerate(MODEL_LAYERS):
        for _j in range(_n_cols):
            _ax = _axes[_i, _j]
            if _j >= len(_layers):
                _ax.axis("off")
                continue
            _vals = explained_variance[(_model, _layers[_j])][VALID_VOXELS]
            _ax.hist(
                np.clip(_vals, None, X_MAX), bins=_bins,
                weights=np.full(len(_vals), 1.0 / len(_vals)),
                color="#5B8DB8", edgecolor="white", linewidth=0.3,
            )
            _ax.axvline(np.median(_vals), color="#D1495B", lw=1.4)
            _name = _DISPLAY_NAMES.get(_model, _model)
            _title = _name if _layers[_j] is None else f"{_name}, Layer {_layers[_j]}"
            _ax.set_title(_title, fontsize=9, fontweight="bold", color="0.2")
            _ax.text(
                0.97, 0.95, f"Median: {np.median(_vals):.2f}\n% > 1: {np.mean(_vals > 1):.1%}",
                transform=_ax.transAxes, ha="right", va="top", fontsize=7, color="0.25",
            )
            _ax.tick_params(labelsize=7, labelbottom=True, colors="0.3")
            _ax.spines[["top", "right"]].set_visible(False)
            _ax.grid(axis="y", color="0.9", linewidth=0.6)
            _ax.set_axisbelow(True)
    _fig.supxlabel(r"Fraction of explainable variance captured, $r^2 / CC_{max}^2$ at saturation rank (values $>$ 1.5 pooled in last bin)")
    _fig.supylabel("Fraction of voxels")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(Artifact, components_needed, np, pca_cumulative_variance, repo_root):
    from src.utils.shared.embeddings import load_embeddings
    from src.utils.shared.ieeg import load_precomputed_epochs

    # Podcast RRR targets: every significant (electrode, lag) pair in the internal RRR artifact
    # (identical for all model layers because they share one cross-model/layer significance union).
    _artifact = Artifact(
        mode="internal", dataset="podcast", model="gpt2", layer=0, results_root=repo_root / "results"
    )
    _matrix = _artifact.load_rank_matrix()
    _target_mask = np.isfinite(_matrix.values).all(axis=(0, 1))  # [electrode, lag]
    _electrodes = _matrix.targets

    # Rows: cached words that are GloVe-covered (the RRR word set), offset 0.
    _, _word_df = load_embeddings(
        "data/", "transformer", dataset="podcast", limit_to_glove=True, glove_dim=300,
        model_dir="gpt2", layer=0,
    )
    _in_glove = _word_df["in_glove"].to_numpy()

    _blocks = []
    _selection = None
    for _sub in _electrodes["subject"].unique():
        _rows = (_electrodes["subject"] == _sub).to_numpy()
        _cache = load_precomputed_epochs(int(_sub), "data/")
        if _selection is None:
            _selection = _cache["selection"]
        elif not np.array_equal(_selection, _cache["selection"]):
            raise ValueError(f"subject {_sub} has a different word selection")
        _idx = [_cache["ch_names"].index(_e) for _e in _electrodes.loc[_rows, "electrode"]]
        _data = _cache["epoch_data"].read_flat(electrode_indices=_idx).reshape(len(_selection), len(_idx), -1)
        _blocks.append(_data[_in_glove[_selection]][:, _target_mask[_rows]])  # [words, targets]
        del _data

    podcast_pca_cumulative, _n_targets, _n_constant = pca_cumulative_variance(_blocks)
    del _blocks
    podcast_pca_components_needed = components_needed(podcast_pca_cumulative)
    print(f"{len(podcast_pca_cumulative)} words x {_n_targets} targets ({_n_constant} constant, zeroed)")
    print({f"{_t:.0%}": _k for _t, _k in podcast_pca_components_needed.items()})

    return (
        load_embeddings,
        load_precomputed_epochs,
        podcast_pca_components_needed,
        podcast_pca_cumulative,
    )


@app.cell
def _(plot_pca_curve, podcast_pca_components_needed, podcast_pca_cumulative):
    plot_pca_curve(podcast_pca_cumulative, podcast_pca_components_needed, "Podcast: PCA of significant (electrode, lag) targets")
    return


@app.cell
def _(np, plt):
    PCA_THRESHOLDS = (0.80, 0.90, 0.95)


    def pca_cumulative_variance(blocks):
        """Cumulative explained-variance curve for z-scored features pooled column-wise over `blocks`.

        blocks: list of [n_words, n_targets_i] arrays sharing the same rows (e.g. one per subject).
        Every target is z-scored (constant targets are zeroed), then the PCA eigenvalue spectrum is
        taken from the words x words Gram matrix, whose nonzero eigenvalues equal the covariance
        eigenvalues without forming an n_targets x n_targets matrix. At most n_words components exist.
        Returns (cumulative fraction of variance [n_words], n_targets, n_constant).
        """
        n_words = blocks[0].shape[0]
        gram = np.zeros((n_words, n_words))
        n_targets = 0
        n_constant = 0
        for block in blocks:
            if not np.isfinite(block).all():
                raise ValueError("non-finite neural values")
            std = block.std(axis=0, dtype=np.float64)
            constant = std == 0
            z = (block - block.mean(axis=0, dtype=np.float64)) / np.where(constant, 1.0, std)
            z[:, constant] = 0.0
            z = z.astype(np.float32)
            gram += (z @ z.T).astype(np.float64)
            n_targets += block.shape[1]
            n_constant += int(constant.sum())
        eigenvalues = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
        return np.cumsum(eigenvalues) / eigenvalues.sum(), n_targets, n_constant


    def components_needed(cumulative, thresholds=PCA_THRESHOLDS):
        """Smallest number of components whose cumulative variance reaches each threshold."""
        return {t: int(np.searchsorted(cumulative, t) + 1) for t in thresholds}


    def plot_pca_curve(cumulative, needed, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        k = np.arange(1, len(cumulative) + 1)
        ax.plot(k, cumulative, color="#5B8DB8", lw=2)
        for (threshold, n), color in zip(needed.items(), ["#E4A02B", "#D1495B", "#7B4B94"]):
            ax.axvline(n, color=color, ls="--", lw=1.2, label=f"{threshold:.0%} of variance: {n} components")
        ax.set_xscale("log")
        ax.set_xlim(1, k[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative fraction of variance explained")
        ax.set_title(title, fontsize=10, fontweight="bold", color="0.2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        plt.close(fig)
        return fig


    return components_needed, pca_cumulative_variance, plot_pca_curve


@app.cell
def _(
    MODEL_LAYERS,
    components_needed,
    load_embeddings,
    load_precomputed_epochs,
    np,
    pca_cumulative_variance,
):
    # Podcast rows: the cached, GloVe-covered words used for the Podcast neural PCA (offset 0).
    _selection = load_precomputed_epochs(1, "data/")["selection"]

    embedding_pca_cumulative = {}  # (model, layer) -> cumulative variance fraction, one entry per dimension
    embedding_pca_components_needed = {}
    for _model, _layers in MODEL_LAYERS:
        for _layer in _layers:
            if _model == "glove-300":
                _kwargs = {"embedding_getter_name": "glove", "dim": 300}
            else:
                _kwargs = {"embedding_getter_name": "transformer", "model_dir": _model, "layer": _layer}
            _embeddings, _words = load_embeddings(
                "data/", dataset="podcast", limit_to_glove=True, glove_dim=300, **_kwargs
            )
            _rows = _words["in_glove"].to_numpy()[_selection]
            _x = np.asarray(_embeddings)[_selection][_rows]
            # Z-score each dimension; at most `n_dims` components carry variance.
            _cumulative, _, _ = pca_cumulative_variance([_x])
            _cumulative = _cumulative[: _x.shape[1]]
            embedding_pca_cumulative[(_model, _layer)] = _cumulative
            embedding_pca_components_needed[(_model, _layer)] = components_needed(_cumulative)
            del _embeddings, _x
    print(f"{len(embedding_pca_cumulative)} model layers, {_rows.sum()} words")

    return embedding_pca_components_needed, embedding_pca_cumulative


@app.cell
def _(
    MODEL_LAYERS,
    embedding_pca_components_needed,
    embedding_pca_cumulative,
    np,
    plt,
):
    _DISPLAY_NAMES = {
        "gpt2": "GPT-2", "gpt2-xl-batch": "GPT-2 XL", "gpt-j-6b": "GPT-J 6B",
        "roberta-large": "RoBERTa Large", "llama3-8b": "Llama 3 8B",
        "random-gpt2": "Random GPT-2", "glove-300": "GloVe 300D",
    }
    _COLORS = ["#E4A02B", "#D1495B", "#7B4B94"]
    _n_cols = max(len(_layers) for _, _layers in MODEL_LAYERS)
    _fig, _axes = plt.subplots(
        len(MODEL_LAYERS), _n_cols, figsize=(2.6 * _n_cols, 1.9 * len(MODEL_LAYERS)),
        sharey=True, squeeze=False,
    )
    for _i, (_model, _layers) in enumerate(MODEL_LAYERS):
        for _j in range(_n_cols):
            _ax = _axes[_i, _j]
            if _j >= len(_layers):
                _ax.axis("off")
                continue
            _cumulative = embedding_pca_cumulative[(_model, _layers[_j])]
            _needed = embedding_pca_components_needed[(_model, _layers[_j])]
            _ax.plot(np.arange(1, len(_cumulative) + 1), _cumulative, color="#5B8DB8", lw=1.6)
            for (_threshold, _n), _color in zip(_needed.items(), _COLORS):
                _ax.axvline(_n, color=_color, ls="--", lw=1)
            _ax.text(
                0.97, 0.05, "\n".join(f"{_t:.0%}: {_n}" for _t, _n in _needed.items()),
                transform=_ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.25",
            )
            _name = _DISPLAY_NAMES.get(_model, _model)
            _title = _name if _layers[_j] is None else f"{_name}, Layer {_layers[_j]}"
            _ax.set_title(_title, fontsize=9, fontweight="bold", color="0.2")
            _ax.set_xscale("log")
            _ax.set_xlim(1, len(_cumulative))
            _ax.set_ylim(0, 1.02)
            _ax.tick_params(labelsize=7, colors="0.3")
            _ax.spines[["top", "right"]].set_visible(False)
            _ax.grid(color="0.92", linewidth=0.6)
            _ax.set_axisbelow(True)
    _fig.supxlabel("Number of principal components (dashed lines: 80%, 90%, 95% of variance)")
    _fig.supylabel("Cumulative fraction of variance explained")
    _fig.tight_layout()
    plt.close(_fig)
    _fig

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(repo_root):
    from src.utils.atlas.core import bihemispheric_region_groups
    from src.utils.plotting.roi_saturation import (
        load_roi_curves, roi_normalized_curves, roi_saturation_distributions,
    )

    # Mean r^2 per target curves, restricted to each ROI, for every model layer (slow:
    # reads every internal RRR/PCA artifact and labels all targets with the atlas).
    ROI_NAMES = list(bihemispheric_region_groups())
    _data = repo_root / "data"
    roi_curves, roi_counts, roi_gaps = load_roi_curves(
        repo_root / "results",
        {"podcast": _data, "cars-2": _data / "brain-treebank",
         "fmri": _data / "deep-fMRI-dataset"},
        regions=ROI_NAMES,
    )
    print(f"{len(roi_gaps)} missing model layers")
    return (
        ROI_NAMES,
        roi_counts,
        roi_curves,
        roi_normalized_curves,
        roi_saturation_distributions,
    )


@app.cell
def _(ROI_NAMES, plt, roi_counts, roi_curves, roi_normalized_curves):
    _fig = roi_normalized_curves(roi_curves, roi_counts, ROI_NAMES)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(ROI_NAMES, plt, roi_curves, roi_saturation_distributions):
    _fig = roi_saturation_distributions(roi_curves, ROI_NAMES)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(np, repo_root):
    from src.utils.plotting.final_data import _rrr as rrr_artifact
    import pandas as pd

    SUBSPACE_RESULTS_ROOT = repo_root / "results"


    def subspace_ranks(max_rank):
        """Every power of two up to (and including) `max_rank` if it is itself one."""
        return [2 ** _p for _p in range(int(np.floor(np.log2(max_rank))) + 1)]


    def to_raw_coordinates(factor, scaler):
        """Express a standardized-v1 factor in raw-embedding coordinates (factor / scale)."""
        if scaler.get("coordinates") != "standardized-v1":
            raise ValueError(f"unsupported scaler coordinates {scaler.get('coordinates')!r}")
        return np.asarray(factor) / np.asarray(scaler["scale"]).reshape(-1, 1)


    def mean_principal_angle(a, b):
        """Mean principal angle (degrees) between the column spaces of `a` and `b` [d, r]."""
        qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
        qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
        cosines = np.linalg.svd(qa.T @ qb, compute_uv=False)
        return float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).mean())


    def mean_squared_cosine(a, b):
        """Subspace overlap in [0, 1]: mean squared cosine of the principal angles,
        ||Qa^T Qb||_F^2 / r, for the column spaces of `a` and `b` [d, r]."""
        qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
        qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
        return float(np.sum((qa.T @ qb) ** 2) / min(qa.shape[1], qb.shape[1]))


    def index_by_selector(collection):
        return {tuple(sorted(dict(key).items())): value for key, value in collection.items()}


    def loso_subspace_overlaps(dataset, model, layer):
        """Per-subject mean squared cosine vs rank for an iEEG leave-one-subject-out artifact.

        RRR: held-out-subject factor (`baseline`) vs the factor learned on the other subjects
        (`train_subjects`), matched by fold. PCA: the same held-out-subject RRR factor vs the
        dataset's PCA basis (the `baseline` PCA factor of that subject/fold). Everything is compared in raw-embedding coordinates, using the
        first r columns of each factor. Returns [subject, method, rank, overlap], averaged over
        folds.
        """
        artifact = rrr_artifact(SUBSPACE_RESULTS_ROOT, "leave_one_subject_out", dataset, model, layer
                        ).load_artifacts(["input_factors", "input_scalers"])
        factors, scalers = index_by_selector(artifact.input_factors), index_by_selector(artifact.input_scalers)

        def raw(subject, fold, phase, method):
            key = tuple(sorted({"fold": fold, "held_out_subject": subject,
                                "loso_phase": phase, "method": method}.items()))
            return to_raw_coordinates(factors[key], scalers[key]) if key in factors else None

        subjects = sorted({dict(k)["held_out_subject"] for k in factors}, key=str)
        folds = sorted({dict(k)["fold"] for k in factors})
        ranks = subspace_ranks(next(iter(factors.values())).shape[1])
        rows = []
        for subject in subjects:
            for method, pairs in (
                ("RRR", lambda f: [((subject, f, "baseline", "rrr"), (subject, f, "train_subjects", "rrr"))]),
                ("PCA", lambda f: [((subject, f, "baseline", "rrr"), (subject, f, "baseline", "pca"))]),
            ):
                overlaps = {rank: [] for rank in ranks}
                for fold in folds:
                    for left, right in pairs(fold):
                        a, b = raw(*left), raw(*right)
                        if a is None or b is None:
                            continue
                        for rank in ranks:
                            overlaps[rank].append(mean_squared_cosine(a[:, :rank], b[:, :rank]))
                rows.extend((str(subject), method, rank, float(np.mean(v))) for rank, v in overlaps.items() if v)
        return pd.DataFrame(rows, columns=["subject", "method", "rank", "overlap"])


    def cross_dataset_basis(dataset, model, layer, method, rank):
        """Whole-dataset basis [d, rank] in raw-embedding coordinates.

        iEEG: the `full_fit` factor truncated to `rank` columns. fMRI: the internal artifact's
        rank-specific factor for `rank`.
        """
        if dataset == "fmri":
            artifact = rrr_artifact(SUBSPACE_RESULTS_ROOT, "internal", dataset, model, layer
                            ).load_artifacts(["input_factors", "input_scalers"])
            factor = index_by_selector(artifact.input_factors)[(("method", method), ("rank", rank))]
            scaler = index_by_selector(artifact.input_scalers)[()]
            return to_raw_coordinates(factor, scaler)
        artifact = rrr_artifact(SUBSPACE_RESULTS_ROOT, "full_fit", dataset, model, layer
                        ).load_artifacts(["input_factors", "input_scalers"])
        key = (("method", method),)
        return to_raw_coordinates(index_by_selector(artifact.input_factors)[key],
                                  index_by_selector(artifact.input_scalers)[key])[:, :rank]


    def cross_subspace_overlaps(pair, model, layer, ranks):
        """Mean squared cosine (subspace overlap) vs rank between two datasets' whole-dataset bases.

        Four comparisons per rank, with `pair = (first, second)`: PCA vs PCA, RRR vs RRR,
        and both directions of RRR vs PCA (one dataset's RRR against the other's PCA).
        """
        first, second = pair
        rows = []
        for rank in ranks:
            basis = {(d, m): cross_dataset_basis(d, model, layer, m, rank)
                     for d in pair for m in ("rrr", "pca")}
            comparisons = {
                "PCA vs PCA": (basis[first, "pca"], basis[second, "pca"]),
                "RRR vs RRR": (basis[first, "rrr"], basis[second, "rrr"]),
                f"RRR ({first}) vs PCA ({second})": (basis[first, "rrr"], basis[second, "pca"]),
                f"RRR ({second}) vs PCA ({first})": (basis[second, "rrr"], basis[first, "pca"]),
            }
            rows.extend((" - ".join(pair), label, rank, mean_squared_cosine(a, b))
                        for label, (a, b) in comparisons.items())
        return pd.DataFrame(rows, columns=["pair", "comparison", "rank", "overlap"])


    return (
        SUBSPACE_RESULTS_ROOT,
        cross_dataset_basis,
        cross_subspace_overlaps,
        index_by_selector,
        loso_subspace_overlaps,
        pd,
        rrr_artifact,
        subspace_ranks,
        to_raw_coordinates,
    )


@app.cell
def _(
    cross_dataset_basis,
    cross_subspace_overlaps,
    loso_subspace_overlaps,
    pd,
    subspace_ranks,
):
    SAMPLE_MODEL, SAMPLE_LAYER = "gpt2-xl-batch", 24
    CROSS_PAIRS = [("podcast", "cars-2"), ("podcast", "fmri"), ("cars-2", "fmri")]

    subspace_loso = {
        _dataset: loso_subspace_overlaps(_dataset, SAMPLE_MODEL, SAMPLE_LAYER)
        for _dataset in ("podcast", "cars-2")
    }
    SAMPLE_DIM = cross_dataset_basis("podcast", SAMPLE_MODEL, SAMPLE_LAYER, "pca", 1).shape[0]
    _ranks = subspace_ranks(SAMPLE_DIM)
    subspace_cross = pd.concat(
        [cross_subspace_overlaps(_pair, SAMPLE_MODEL, SAMPLE_LAYER, _ranks) for _pair in CROSS_PAIRS],
        ignore_index=True,
    )

    return (
        SAMPLE_DIM,
        SAMPLE_LAYER,
        SAMPLE_MODEL,
        subspace_cross,
        subspace_loso,
    )


@app.cell
def _(
    SAMPLE_DIM,
    SAMPLE_LAYER,
    SAMPLE_MODEL,
    loso_saturation_ranks,
    np,
    plt,
    subspace_cross,
    subspace_loso,
    subspace_ranks,
):
    _COLORS = {"RRR": "#2563EB", "PCA": "#9CA3AF"}
    _TITLES = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}

    def _style(ax, title):
        ax.set_xscale("log", base=2)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)

    subspace_fig, _axes = plt.subplots(2, 3, figsize=(13, 8.5), sharey=True)
    for _ax, (_dataset, _frame) in zip(_axes[0], subspace_loso.items()):
        for _method, _group in _frame.groupby("method"):
            _per_rank = _group.groupby("rank")["overlap"]
            _mean, _sem = _per_rank.mean(), _per_rank.sem()
            _ax.plot(_mean.index, _mean.values, "o-", color=_COLORS[_method], label=_method)
            _ax.fill_between(_mean.index, _mean - _sem, _mean + _sem, color=_COLORS[_method], alpha=0.2, lw=0)
        _ax.plot(subspace_ranks(SAMPLE_DIM), np.array(subspace_ranks(SAMPLE_DIM)) / SAMPLE_DIM,
                    color="black", linestyle=(0, (1, 2)), linewidth=1.2, label="Chance (r / d)")
        _saturation = float(np.median(list(loso_saturation_ranks[_dataset].values())))
        _ax.axvline(_saturation, color="red", linestyle="--", linewidth=1.3,
                    label=f"Saturation rank (median over subjects): {_saturation:.1f}")
        _ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style(_ax, f"LOSO: {_TITLES[_dataset]} (held-out subject vs. other subjects)")
    _axes[0, 2].axis("off")
    for _ax, (_pair, _frame) in zip(_axes[1], subspace_cross.groupby("pair", sort=False)):
        _first, _second = _pair.split(" - ")
        _styles = {  # comparison -> (color, linestyle)
            "PCA vs PCA": ("#6B7280", "-"),
            "RRR vs RRR": ("#2563EB", "-"),
            f"RRR ({_first}) vs PCA ({_second})": ("#D97706", "--"),
            f"RRR ({_second}) vs PCA ({_first})": ("#059669", ":"),
        }
        for _label, _group in _frame.groupby("comparison", sort=False):
            _color, _linestyle = _styles[_label]
            _ax.plot(_group["rank"], _group["overlap"], marker="o", markersize=4, color=_color,
                     linestyle=_linestyle, label=_label.replace(_first, _TITLES[_first]).replace(_second, _TITLES[_second]))
        _ax.plot(subspace_ranks(SAMPLE_DIM), np.array(subspace_ranks(SAMPLE_DIM)) / SAMPLE_DIM,
                    color="black", linestyle=(0, (1, 2)), linewidth=1.2, label="Chance (r / d)")
        _ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        _style(_ax, f"Cross-dataset: {_TITLES[_first]} vs. {_TITLES[_second]}")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("Mean squared cosine (subspace overlap)")
    for _ax in _axes[1]:
        _ax.set_xlabel("Rank")
    subspace_fig.suptitle(f"Subspace consistency: {SAMPLE_MODEL}, layer {SAMPLE_LAYER}")
    subspace_fig.tight_layout(h_pad=2.5)
    plt.close(subspace_fig)
    subspace_fig

    return


@app.cell
def _(
    SUBSPACE_RESULTS_ROOT,
    index_by_selector,
    np,
    rrr_artifact,
    to_raw_coordinates,
):
    def principal_angles(a, b):
        """All principal angles (degrees, ascending) between the column spaces of `a` and `b`."""
        qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
        qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
        cosines = np.linalg.svd(qa.T @ qb, compute_uv=False)
        return np.sort(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))))


    def loso_principal_angle_vectors(dataset, model, layer, ranks):
        """Full principal-angle vectors for an iEEG LOSO artifact, same comparisons as
        `loso_subspace_overlaps`: RRR = held-out-subject factor vs the other-subjects factor,
        PCA = held-out-subject factor vs the PCA basis, fold-matched, in raw coordinates.

        Returns {(method, rank): array [n_subjects, rank]} of ascending angles averaged
        over folds (sorted angle index k is averaged across folds), plus the subject list.
        """
        artifact = rrr_artifact(SUBSPACE_RESULTS_ROOT, "leave_one_subject_out", dataset, model, layer
                                ).load_artifacts(["input_factors", "input_scalers"])
        factors, scalers = index_by_selector(artifact.input_factors), index_by_selector(artifact.input_scalers)

        def raw(subject, fold, phase, method):
            key = tuple(sorted({"fold": fold, "held_out_subject": subject,
                                "loso_phase": phase, "method": method}.items()))
            return to_raw_coordinates(factors[key], scalers[key])

        subjects = sorted({dict(k)["held_out_subject"] for k in factors}, key=str)
        folds = sorted({dict(k)["fold"] for k in factors})
        out = {}
        for method, other in (("RRR", ("train_subjects", "rrr")), ("PCA", ("baseline", "pca"))):
            for rank in ranks:
                out[(method, rank)] = np.stack([
                    np.mean([principal_angles(raw(s, f, "baseline", "rrr")[:, :rank],
                                              raw(s, f, *other)[:, :rank]) for f in folds], axis=0)
                    for s in subjects])
        return out, subjects


    return (loso_principal_angle_vectors,)


@app.cell
def _(SAMPLE_LAYER, SAMPLE_MODEL, loso_principal_angle_vectors):
    DISTRIBUTION_RANKS = [4, 16, 64, 256]
    loso_vectors = {
        _dataset: loso_principal_angle_vectors(_dataset, SAMPLE_MODEL, SAMPLE_LAYER, DISTRIBUTION_RANKS)
        for _dataset in ("podcast", "cars-2")
    }

    return DISTRIBUTION_RANKS, loso_vectors


@app.cell
def _(DISTRIBUTION_RANKS, SAMPLE_LAYER, SAMPLE_MODEL, loso_vectors, np, plt):
    _names = {"podcast": "Podcast", "cars-2": "Cars 2"}
    _colors = {"RRR": "#2563EB", "PCA": "#9CA3AF"}
    loso_distribution_fig, _axes = plt.subplots(2, len(DISTRIBUTION_RANKS), figsize=(14, 7),
                                                sharey=True, squeeze=False)
    for _row, (_dataset, (_vectors, _subjects)) in zip(_axes, loso_vectors.items()):
        for _ax, _rank in zip(_row, DISTRIBUTION_RANKS):
            for _method in ("PCA", "RRR"):
                _angles = _vectors[(_method, _rank)]  # [subjects, rank]
                _x = np.arange(1, _rank + 1)
                for _subject_angles in _angles:
                    _ax.plot(_x, _subject_angles, color=_colors[_method], alpha=0.3, lw=0.8)
                _ax.plot(_x, np.median(_angles, axis=0), color=_colors[_method], lw=2.2, label=f"{_method} (median)")
            _ax.set_title(f"{_names[_dataset]}, rank {_rank}", fontsize=11)
            _ax.set_ylim(0, 90)
            _ax.spines[["top", "right"]].set_visible(False)
            _ax.grid(color="0.9", linewidth=0.6)
            _ax.set_axisbelow(True)
            _ax.set_xlabel("Principal angle index (sorted ascending)")
        _row[0].set_ylabel("Principal angle (degrees)")
    _axes[0, 0].legend(frameon=False, fontsize=9)
    loso_distribution_fig.suptitle(
        f"LOSO principal angles, all {SAMPLE_MODEL} layer {SAMPLE_LAYER} (thin: one line per held-out subject; "
        "RRR: held-out vs other subjects, PCA: held-out RRR vs PCA basis)")
    loso_distribution_fig.tight_layout()
    plt.close(loso_distribution_fig)
    loso_distribution_fig

    return


@app.cell
def _(SAMPLE_LAYER, SAMPLE_MODEL, SUBSPACE_RESULTS_ROOT, np):
    from src.utils.plotting.final_data import _loso_one as loso_one_curves
    from src.utils.rrr.plot_metrics import explained_variance_saturation_rank

    SATURATION_FRACTION = 0.99
    # Saturation rank per held-out subject: first rank reaching 99% of the LOSO RRR-transfer
    # explained-variance curve's own maximum (as in the explained-variance main figures).
    loso_saturation_ranks = {}
    for _dataset in ("podcast", "cars-2"):
        _, _, _ev = loso_one_curves(SUBSPACE_RESULTS_ROOT, _dataset, SAMPLE_MODEL, SAMPLE_LAYER)
        loso_saturation_ranks[_dataset] = {
            _subject: explained_variance_saturation_rank(_ranks, np.asarray(_curve).reshape(len(_ranks), -1)[:, 0],
                                                         SATURATION_FRACTION)
            for (_subject, _label), (_ranks, _curve) in _ev.items() if _label == "RRR transfer"
        }

    return (
        SATURATION_FRACTION,
        explained_variance_saturation_rank,
        loso_one_curves,
        loso_saturation_ranks,
    )


@app.cell
def _(
    SATURATION_FRACTION,
    SUBSPACE_RESULTS_ROOT,
    cross_dataset_basis,
    explained_variance_saturation_rank,
    loso_one_curves,
    loso_subspace_overlaps,
    np,
):
    import time
    from scripts.utilities.check_final_significance import MODEL_LAYERS as FINAL_MODEL_LAYERS

    # Random GPT-2 and GloVe are kept out of the aggregate for now.
    AGGREGATE_MODELS = ("gpt2", "gpt2-xl", "roberta-large")
    AGGREGATE_DATASETS = ("podcast", "cars-2")


    def loso_layer_summary(dataset, model, layer):
        """Chance-normalized RRR-minus-PCA LOSO overlap vs rank, plus the saturation rank.

        excess = (overlap - r/d) / (1 - r/d) per method (0 = chance, 1 = identical), averaged over
        held-out subjects; delta = excess(RRR) - excess(PCA). Saturation rank = median over
        subjects of the 99%-of-peak RRR-transfer explained-variance rank. Ranks r >= d are dropped.
        """
        overlaps = loso_subspace_overlaps(dataset, model, layer)
        dim = cross_dataset_basis(dataset, model, layer, "pca", 1).shape[0]
        overlaps = overlaps[overlaps["rank"] < dim].copy()
        chance = overlaps["rank"] / dim
        overlaps["excess"] = (overlaps["overlap"] - chance) / (1 - chance)
        by_method = overlaps.groupby(["method", "rank"])["excess"].mean().unstack(0)
        _, _, ev = loso_one_curves(SUBSPACE_RESULTS_ROOT, dataset, model, layer)
        saturation = float(np.median([
            explained_variance_saturation_rank(ranks, np.asarray(curve).reshape(len(ranks), -1)[:, 0],
                                               SATURATION_FRACTION)
            for (_, label), (ranks, curve) in ev.items() if label == "RRR transfer"]))
        return {"ranks": by_method.index.to_numpy(), "rrr": by_method["RRR"].to_numpy(),
                "pca": by_method["PCA"].to_numpy(), "saturation": saturation, "dim": dim}


    return (
        AGGREGATE_DATASETS,
        AGGREGATE_MODELS,
        FINAL_MODEL_LAYERS,
        loso_layer_summary,
    )


@app.cell
def _(
    AGGREGATE_DATASETS,
    AGGREGATE_MODELS,
    FINAL_MODEL_LAYERS,
    loso_layer_summary,
):
    loso_layer_summaries, loso_summary_gaps = {}, []
    for _key in AGGREGATE_MODELS:
        _model, _layers = FINAL_MODEL_LAYERS[_key]
        for _layer in _layers:
            for _dataset in AGGREGATE_DATASETS:
                try:
                    loso_layer_summaries[(_dataset, _model, _layer)] = loso_layer_summary(_dataset, _model, _layer)
                except Exception as _error:
                    loso_summary_gaps.append((_dataset, _model, _layer, f"{type(_error).__name__}: {_error}"[:150]))

    return (loso_layer_summaries,)


@app.cell
def _(AGGREGATE_DATASETS, loso_layer_summaries, np, plt):
    ALIGNED_GRID = np.arange(-5, 5.01, 0.25)  # log2(rank / saturation rank)


    def aligned_delta(summary, grid=ALIGNED_GRID):
        """RRR - PCA excess overlap interpolated onto log2(rank / saturation rank).

        NaN outside the layer's evaluated rank range (no extrapolation)."""
        x = np.log2(summary["ranks"] / summary["saturation"])
        delta = summary["rrr"] - summary["pca"]
        return np.interp(grid, x, delta, left=np.nan, right=np.nan)


    _model_colors = {"gpt2": "#2563EB", "gpt2-xl-batch": "#D97706", "roberta-large": "#059669"}
    _model_names = {"gpt2": "GPT-2", "gpt2-xl-batch": "GPT-2 XL", "roberta-large": "RoBERTa Large"}
    _titles = {"podcast": "Podcast", "cars-2": "Cars 2"}
    aligned_fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _ax, _dataset in zip(_axes, AGGREGATE_DATASETS):
        _stack = []
        for (_d, _model, _layer), _summary in loso_layer_summaries.items():
            if _d != _dataset:
                continue
            _curve = aligned_delta(_summary)
            _stack.append(_curve)
            _ax.plot(ALIGNED_GRID, _curve, color=_model_colors[_model], alpha=0.35, lw=0.9)
        _stack = np.array(_stack)
        with np.errstate(all="ignore"):
            _mean = np.nanmean(_stack, axis=0)
        _enough = np.isfinite(_stack).sum(axis=0) >= max(3, len(_stack) // 2)
        _ax.plot(ALIGNED_GRID[_enough], _mean[_enough], color="black", lw=2.5, label="Mean over layers")
        _ax.axhline(0, color="0.4", lw=1)
        _ax.axvline(0, color="red", linestyle="--", lw=1.3, label="Saturation rank")
        _ax.set_title(f"{_titles[_dataset]} LOSO (n = {len(_stack)} layers)", fontsize=11)
        _ax.set_xlabel("log\u2082(rank / saturation rank)")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(color="0.9", linewidth=0.6)
        _ax.set_axisbelow(True)
    _axes[0].set_ylabel("RRR \u2212 PCA excess overlap over chance")
    for _model, _color in _model_colors.items():
        _axes[0].plot([], [], color=_color, label=_model_names[_model])
    _axes[0].legend(frameon=False, fontsize=8, loc="best")
    aligned_fig.suptitle("Held-out RRR subspace vs. PCA: consistency across subjects, aligned to saturation rank")
    aligned_fig.tight_layout()
    plt.close(aligned_fig)
    aligned_fig

    return


@app.cell
def _(loso_layer_summaries, np, pd):
    from src.utils.plotting.final_axes import format_p_value
    from src.utils.rrr.plot_metrics import paired_sign_flip

    # Per layer: mean RRR - PCA excess overlap over the evaluated ranks below / above the saturation
    # rank (each evaluated power-of-two rank weighted equally).
    _rows = []
    for (_dataset, _model, _layer), _s in loso_layer_summaries.items():
        _delta = _s["rrr"] - _s["pca"]
        _below, _above = _s["ranks"] < _s["saturation"], _s["ranks"] > _s["saturation"]
        _rows.append((_dataset, _model, _layer, _s["saturation"],
                      _delta[_below].mean() if _below.any() else np.nan,
                      _delta[_above].mean() if _above.any() else np.nan,
                      int(_below.sum()), int(_above.sum())))
    saturation_split = pd.DataFrame(_rows, columns=["dataset", "model", "layer", "saturation_rank",
                                                    "below", "above", "n_ranks_below", "n_ranks_above"])
    saturation_split.round(3)

    return format_p_value, paired_sign_flip, saturation_split


@app.cell
def _(
    AGGREGATE_DATASETS,
    format_p_value,
    np,
    paired_sign_flip,
    plt,
    saturation_split,
):
    _model_colors = {"gpt2": "#2563EB", "gpt2-xl-batch": "#D97706", "roberta-large": "#059669"}
    _model_names = {"gpt2": "GPT-2", "gpt2-xl-batch": "GPT-2 XL", "roberta-large": "RoBERTa Large"}
    _titles = {"podcast": "Podcast", "cars-2": "Cars 2"}
    split_fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for _ax, _dataset in zip(_axes, AGGREGATE_DATASETS):
        _frame = saturation_split[saturation_split["dataset"] == _dataset].dropna(subset=["below", "above"])
        for _, _row in _frame.iterrows():
            _ax.plot([0, 1], [_row["below"], _row["above"]], color=_model_colors[_row["model"]],
                     marker="o", markersize=4, alpha=0.5, lw=1)
        _ax.plot([0, 1], [_frame["below"].mean(), _frame["above"].mean()], color="black", marker="D", lw=2.5, zorder=5)
        _ax.axhline(0, color="0.4", lw=1, linestyle="--")
        _p_below = paired_sign_flip(_frame["below"], np.zeros(len(_frame)), alternative="greater")
        _p_above = paired_sign_flip(_frame["above"], np.zeros(len(_frame)), alternative="less")
        _ax.text(0, 1.0, f"> 0: {format_p_value(_p_below)}", transform=_ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        _ax.text(1, 1.0, f"< 0: {format_p_value(_p_above)}", transform=_ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        _ax.set_xticks([0, 1], ["Below saturation rank", "Above saturation rank"])
        _ax.set_xlim(-0.4, 1.4)
        _ax.set_title(f"{_titles[_dataset]} LOSO (n = {len(_frame)} layers)", fontsize=11)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(axis="y", color="0.9", linewidth=0.6)
        _ax.set_axisbelow(True)
    _axes[0].set_ylabel("Mean RRR \u2212 PCA excess overlap over chance")
    for _model, _color in _model_colors.items():
        _axes[0].plot([], [], color=_color, label=_model_names[_model])
    _axes[0].plot([], [], color="black", marker="D", label="Mean over layers")
    _axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    split_fig.suptitle("RRR vs. PCA subspace consistency, below vs. above saturation rank (one line per layer)")
    split_fig.tight_layout()
    plt.close(split_fig)
    split_fig

    return


if __name__ == "__main__":
    app.run()

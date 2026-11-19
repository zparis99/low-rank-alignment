"""Data for the distribution, PCA-dimensionality, ROI and subspace-consistency figures.

Everything here reads artifacts or raw data and returns plain arrays, dictionaries and
tables for `final_subspace_figures`. Expensive results are cached on disk next to the
other final-plot caches and are recomputed whenever an RRR artifact manifest changes.
Each dataset is computed independently so a dataset whose results or raw data are not
present (for example fMRI leave-one-subject-out on a machine without them) yields a
recorded gap instead of failing the others.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utilities.check_final_significance import MODEL_LAYERS
from src.core.registry import resolve_dataset_context
from src.utils.plotting.final_data import DATASETS, _loso_one, _read_cache, _rrr, _write_cache
from src.utils.rrr.plot_metrics import explained_variance_saturation_rank

CACHE_NAME = "final_subspace_cache"
CC_MAX_FLOOR = 0.25
PCA_THRESHOLDS = (0.80, 0.90, 0.95)
SATURATION_FRACTION = 0.99
FMRI_SESSIONS = (1, 2, 3, 4, 5)
# Dataset-specific raw-data roots and iEEG stimulus selections used by the RRR configs.
IEEG_SETTINGS = {
    "podcast": {"dataset": "podcast", "root": "data", "per_dataset_config": None},
    "cars-2": {"dataset": "brain_treebank", "root": "data/brain-treebank",
               "per_dataset_config": {"movie": "cars-2", "rereference": "laplacian"}},
}
# (model directory, layers) for every configured model layer, in figure order.
MODEL_LAYER_LIST = [(model, list(layers)) for model, layers in MODEL_LAYERS.values()]
# Models in the LOSO aggregates; the figures draw random GPT-2 separately (or not at all).
LOSO_MODEL_KEYS = ("gpt2", "gpt2-xl", "roberta-large", "glove", "random-gpt2")
LOSO_DATASETS = tuple(DATASETS)


@lru_cache(maxsize=None)
def _fingerprint(results_root):
    """Fingerprint of the RRR/significance artifact manifests only.

    Unlike the main final-plots cache this ignores plotting and test code, so editing a figure
    or statistics function never forces the slow recomputation. Delete `results/final_subspace_cache`
    after changing a computation in this module.
    """
    import hashlib

    root = Path(results_root)
    digest = hashlib.blake2b(digest_size=20)
    for category in ("run_rrr", "run_significance", "fmri_noise_ceiling"):
        for path in sorted((root / category).rglob("manifest.json")):
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def cached(results_root, name, compute):
    """Return `compute()`, reusing the pickle cached under the current artifact fingerprint.

    The fingerprint (all artifact manifests) is read once per process, so restart the
    notebook to pick up new results.
    """
    path = Path(results_root) / CACHE_NAME / f"{name}.pkl"
    fingerprint = _fingerprint(str(results_root))
    value = _read_cache(path, fingerprint)
    if value is None:
        value = compute()
        _write_cache(path, fingerprint, value)
    return value


# --- explained variance relative to the noise ceiling (fMRI) ----------------------------


def load_fmri_noise_ceiling(results_root, subject, voxel_index):
    """Per-voxel noise ceiling CC_max for `voxel_index` of `subject` (NaN if undefined)."""
    path = Path(results_root) / "fmri_noise_ceiling" / f"{subject}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/fmri/compute_noise_ceiling.py")
    with np.load(path) as archive:
        return archive["cc_max"][np.asarray(voxel_index, dtype=int)].astype(float)


def fmri_noise_ceiling_fractions(results_root, data_root):
    """Fraction of explainable variance captured per fMRI voxel for every model layer.

    Returns `{(model, layer): [n_valid_voxels] r^2 / CC_max^2}` where r is the voxel's
    correlation at its saturation rank (the first tested rank reaching 99% of its own
    peak, negatives clipped to zero) and CC_max is floored at `CC_MAX_FLOOR`. Voxels
    without a defined noise ceiling are dropped.
    """
    fractions, targets, cc_max = {}, None, None
    for model, layers in MODEL_LAYER_LIST:
        for layer in layers:
            matrix = _rrr(Path(results_root), "internal", "fmri", model, layer
                          ).load_rank_matrix(data_root=data_root)
            identifiers = matrix.targets[["subject", "voxel_index"]].reset_index(drop=True)
            if targets is None:
                targets = identifiers
                cc_max = np.full(len(targets), np.nan)
                for subject in targets["subject"].unique():
                    rows = (targets["subject"] == subject).to_numpy()
                    cc_max[rows] = load_fmri_noise_ceiling(
                        results_root, subject, targets.loc[rows, "voxel_index"].to_numpy())
            elif not identifiers.equals(targets):
                raise ValueError(f"{model} layer {layer}: voxel targets differ from the first artifact")
            curves = matrix.values[:, 0, :]  # one held-out-story fold
            reached = curves >= SATURATION_FRACTION * curves.max(axis=0)
            saturated = np.clip(curves[reached.argmax(axis=0), np.arange(curves.shape[1])], 0.0, None)
            fractions[(model, layer)] = saturated ** 2 / np.maximum(cc_max, CC_MAX_FLOOR) ** 2
    valid = np.isfinite(cc_max)
    return {key: value[valid] for key, value in fractions.items()}


# --- PCA dimensionality of the neural targets and of the embeddings ----------------------


def pca_cumulative_variance(blocks):
    """Cumulative explained-variance curve for z-scored features pooled column-wise.

    `blocks` is an iterable of [n_rows, n_targets_i] arrays sharing rows (one per subject);
    it may be a generator so only one block is in memory at a time. Every column is
    z-scored (constant columns are zeroed) and the eigenvalue spectrum comes from the
    rows x rows Gram matrix, whose nonzero eigenvalues equal the covariance eigenvalues.
    Returns (cumulative fraction [n_rows], n_targets, n_constant).
    """
    gram = None
    n_targets = n_constant = 0
    for block in blocks:
        block = np.asarray(block, dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("non-finite values")
        std = block.std(axis=0, dtype=np.float64)
        constant = std == 0
        block -= block.mean(axis=0, dtype=np.float64).astype(np.float32)
        block /= np.where(constant, 1.0, std).astype(np.float32)
        block[:, constant] = 0.0
        if gram is None:
            gram = np.zeros((block.shape[0], block.shape[0]))
        gram += (block @ block.T).astype(np.float64)
        n_targets += block.shape[1]
        n_constant += int(constant.sum())
        del block
    eigenvalues = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
    return np.cumsum(eigenvalues) / eigenvalues.sum(), n_targets, n_constant


def feature_pca_cumulative_variance(x):
    """`pca_cumulative_variance` for one [n_rows, n_dims] matrix with n_rows >> n_dims.

    Takes the eigenvalues of the n_dims x n_dims covariance of the z-scored columns (the same
    nonzero eigenvalues as the rows x rows Gram matrix) so memory does not scale with n_rows.
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite values")
    std = x.std(axis=0, dtype=np.float64)
    constant = std == 0
    z = (x - x.mean(axis=0, dtype=np.float64).astype(np.float32)) / np.where(constant, 1.0, std).astype(np.float32)
    z[:, constant] = 0.0
    eigenvalues = np.clip(np.linalg.eigvalsh((z.T @ z).astype(np.float64))[::-1], 0.0, None)
    return np.cumsum(eigenvalues) / eigenvalues.sum()


def components_needed(cumulative, thresholds=PCA_THRESHOLDS):
    """Smallest number of components whose cumulative variance reaches each threshold."""
    return {t: int(np.searchsorted(cumulative, t) + 1) for t in thresholds}


def _pca_summary(cumulative, **extra):
    return {"cumulative": cumulative, "needed": components_needed(cumulative), **extra}


def _ieeg_context(dataset, subjects, repo_root):
    settings = IEEG_SETTINGS[dataset]
    return resolve_dataset_context(
        dataset=settings["dataset"], subject_ids=[int(s) for s in subjects],
        per_dataset_config=settings["per_dataset_config"],
        data_root=str(Path(repo_root) / settings["root"]),
    )


def _ieeg_word_rows(dataset, repo_root, model_kwargs, subjects):
    """(embeddings, cached word selection, in-GloVe flags) for an iEEG dataset's word set."""
    from src.utils.shared.embeddings import load_embeddings
    from src.utils.shared.ieeg import compute_shared_word_selection

    settings = IEEG_SETTINGS[dataset]
    root = str(Path(repo_root) / settings["root"])
    context = _ieeg_context(dataset, subjects, repo_root)
    embeddings, words = load_embeddings(
        root, dataset=settings["dataset"], dataset_context=context,
        limit_to_glove=True, glove_dim=300, **model_kwargs)
    selection = compute_shared_word_selection(
        [int(s) for s in subjects], root, dataset=settings["dataset"], dataset_context=context)
    return np.asarray(embeddings), selection, words["in_glove"].to_numpy()


def _ieeg_subjects(results_root, dataset):
    matrix = _rrr(Path(results_root), "internal", dataset, "gpt2", 0).load_rank_matrix()
    return matrix, sorted(matrix.targets["subject"].unique())


def ieeg_neural_pca(results_root, repo_root, dataset):
    """PCA of the significant (electrode, lag) targets over the cached, GloVe-covered words."""
    from src.utils.shared.ieeg import compute_shared_word_selection, load_precomputed_epochs

    settings = IEEG_SETTINGS[dataset]
    root = str(Path(repo_root) / settings["root"])
    matrix, subjects = _ieeg_subjects(results_root, dataset)
    context = _ieeg_context(dataset, subjects, repo_root)
    # Significant targets are identical for every model layer (one shared significance union).
    target_mask = np.isfinite(matrix.values).all(axis=(0, 1))  # [electrode, lag]
    electrodes = matrix.targets
    _, _, in_glove = _ieeg_word_rows(dataset, repo_root, {
        "embedding_getter_name": "transformer", "model_dir": "gpt2", "layer": 0}, subjects)
    # Words cached for every subject (the RRR runs' shared word selection), GloVe-covered only.
    selection = compute_shared_word_selection(
        [int(s) for s in subjects], root, dataset=settings["dataset"], dataset_context=context)
    selection = selection[in_glove[selection]]
    blocks = []
    for subject in subjects:
        rows = (electrodes["subject"] == subject).to_numpy()
        cache = load_precomputed_epochs(
            int(subject), root, dataset=settings["dataset"], dataset_context=context)
        position = {int(w): i for i, w in enumerate(cache["selection"])}
        word_rows = np.array([position[int(w)] for w in selection])
        indices = [cache["ch_names"].index(name) for name in electrodes.loc[rows, "electrode"]]
        data = cache["epoch_data"].read_flat(electrode_indices=indices)
        data = data.reshape(len(cache["selection"]), len(indices), -1)
        blocks.append(data[word_rows][:, target_mask[rows]])
        del data
    cumulative, n_targets, n_constant = pca_cumulative_variance(blocks)
    return _pca_summary(cumulative, n_targets=n_targets, n_constant=n_constant)


def fmri_neural_pca(results_root, repo_root):
    """PCA of the significant fMRI voxels over the training-story TRs, pooled over subjects."""
    import h5py

    from src.utils.shared.fmri import load_fmri_responses, resolve_fmri_stories, response_path

    data_root = str(Path(repo_root) / "data" / "deep-fMRI-dataset")
    matrix = _rrr(Path(results_root), "internal", "fmri", "gpt2", 0).load_rank_matrix(
        data_root=data_root)
    train, _, _ = resolve_fmri_stories(data_root, list(FMRI_SESSIONS))
    lengths = []
    for story in train:
        with h5py.File(response_path(data_root, "UTS01", story), "r") as handle:
            lengths.append(handle["data"].shape[0])
    stops = np.cumsum(lengths)
    slices = {s: slice(int(stop - n), int(stop)) for s, n, stop in zip(train, lengths, stops)}
    significant = matrix.targets[np.isfinite(matrix.values).all(axis=(0, 1))]
    def blocks():  # one subject's responses at a time (all subjects would need ~20 GB)
        for subject in sorted(significant["subject"].unique()):
            with h5py.File(response_path(data_root, subject, train[0]), "r") as handle:
                mask = np.zeros(handle["data"].shape[1], dtype=bool)
            mask[significant.loc[significant["subject"] == subject, "voxel_index"].to_numpy()] = True
            yield load_fmri_responses(data_root, train, slices, [subject], {subject: mask})

    cumulative, n_targets, n_constant = pca_cumulative_variance(blocks())
    return _pca_summary(cumulative, n_targets=n_targets, n_constant=n_constant)


def fmri_word_embeddings(repo_root, model, layer):
    """Training-story word embeddings (GloVe-covered words only) for one fMRI model layer."""
    from src.utils.shared.fmri import load_story_embeddings, load_story_word_data, resolve_fmri_stories
    from src.utils.shared.embeddings import get_glove_embeddings

    data_root = str(Path(repo_root) / "data" / "deep-fMRI-dataset")
    train, _, _ = resolve_fmri_stories(data_root, list(FMRI_SESSIONS))
    # The configured models' directory names are the artifact model names (GloVe: glove-300).
    stories = load_story_embeddings(
        str(Path(repo_root) / "data" / "fmri" / "stimuli"), model, train, layer, "mean",
        fmri_repo_root=data_root, glove_data_root=str(Path(repo_root) / "data"),
        limit_to_glove=True, glove_dim=300)
    frames, _ = load_story_word_data(data_root, train)
    covered = [get_glove_embeddings(str(Path(repo_root) / "data"), frames[s], 300)[1]["in_glove"]
               .to_numpy() for s in train]
    return np.concatenate([stories[s][c] for s, c in zip(train, covered)])


def embedding_pca(repo_root, dataset):
    """`{(model, layer): PCA summary}` of each model layer's embeddings over a dataset's words.

    A model layer whose embeddings are not available maps to None and is listed in `gaps`.
    """
    summaries, rows, gaps = {}, None, []
    for model, layers in MODEL_LAYER_LIST:
        for layer in layers:
            try:
                if dataset == "fmri":
                    x = fmri_word_embeddings(repo_root, model, layer)
                else:
                    kwargs = ({"embedding_getter_name": "glove", "dim": 300} if model == "glove-300"
                              else {"embedding_getter_name": "transformer",
                                    "model_dir": model, "layer": layer})
                    subjects = {"podcast": range(1, 10), "cars-2": (3, 7, 10)}[dataset]
                    embeddings, selection, in_glove = _ieeg_word_rows(
                        dataset, repo_root, kwargs, list(subjects))
                    x = embeddings[selection][in_glove[selection]]
                summaries[(model, layer)] = _pca_summary(feature_pca_cumulative_variance(x))
                rows = len(x)
            except (FileNotFoundError, OSError) as error:
                summaries[(model, layer)] = None
                gaps.append(f"{dataset} {model} layer {layer}: {type(error).__name__}")
    return {"summaries": summaries, "n_words": rows, "gaps": gaps}


# --- subspace consistency -----------------------------------------------------------------


def subspace_ranks(max_rank):
    """Every power of two up to (and including) `max_rank` if it is itself one."""
    return [2 ** p for p in range(int(np.floor(np.log2(max_rank))) + 1)]


def to_raw_coordinates(factor, scaler):
    """Express a standardized-v1 factor in raw-embedding coordinates (factor / scale)."""
    if scaler.get("coordinates") != "standardized-v1":
        raise ValueError(f"unsupported scaler coordinates {scaler.get('coordinates')!r}")
    return np.asarray(factor) / np.asarray(scaler["scale"]).reshape(-1, 1)


def _orthonormal(a):
    return np.linalg.qr(np.asarray(a, dtype=np.float64))[0]


def _overlap_of_orthonormal(qa, qb):
    """`mean_squared_cosine` for bases that are already orthonormal."""
    return float(np.sum((qa.T @ qb) ** 2) / min(qa.shape[1], qb.shape[1]))


def mean_squared_cosine(a, b):
    """Subspace overlap in [0, 1]: mean squared cosine of the principal angles of `a`, `b`."""
    return _overlap_of_orthonormal(_orthonormal(a), _orthonormal(b))


def squared_cosines(a, b):
    """Squared cosines of all principal angles between the column spaces of `a` and `b`.

    Sorted descending, i.e. smallest principal angle first. Their mean is `mean_squared_cosine`.
    """
    cosines = np.linalg.svd(_orthonormal(a).T @ _orthonormal(b), compute_uv=False)
    return np.sort(np.clip(cosines, 0.0, 1.0) ** 2)[::-1]


def index_by_selector(collection):
    return {tuple(sorted(dict(key).items())): value for key, value in collection.items()}


def _loso_factors(results_root, dataset, model, layer):
    """Accessor `raw(subject, fold, phase, method, rank)` for an LOSO artifact's factors.

    Factors are returned in raw-embedding coordinates with exactly `rank` columns, or None
    if absent. iEEG artifacts store one full factor per fold (truncated to `rank` columns);
    fMRI artifacts have a single fold (`None`), one factor per rank, and one shared scaler.
    Also returns the held-out subjects, the folds, the largest stored rank and the embedding
    dimension `dim`.
    """
    artifact = _rrr(Path(results_root), "leave_one_subject_out", dataset, model, layer
                    ).load_artifacts(["input_factors", "input_scalers"])
    factors = index_by_selector(artifact.input_factors)
    scalers = index_by_selector(artifact.input_scalers)
    per_rank = dataset == "fmri"

    def raw(subject, fold, phase, method, rank):
        selector = {"held_out_subject": subject, "loso_phase": phase, "method": method}
        if per_rank:
            key = tuple(sorted({**selector, "rank": rank}.items()))
            if key not in factors:
                return None
            if () not in scalers:
                raise ValueError("the fMRI LOSO artifact has no saved input scaler; rerun with refresh")
            return to_raw_coordinates(factors[key], scalers[()])
        key = tuple(sorted({**selector, "fold": fold}.items()))
        if key not in factors:
            return None
        # Truncate before rescaling: the stored factor is [d, d] but only `rank` columns are needed.
        return to_raw_coordinates(factors[key][:, :rank], scalers[key])

    subjects = sorted({dict(k)["held_out_subject"] for k in factors}, key=str)
    if per_rank:
        folds = [None]
        max_rank = max(dict(k)["rank"] for k in factors if "rank" in dict(k))
    else:
        folds = sorted({dict(k)["fold"] for k in factors})
        max_rank = next(iter(factors.values())).shape[1]
    dim = next(iter(factors.values())).shape[0]
    return raw, subjects, folds, max_rank, dim


# Per held-out subject: RRR = held-out factor vs the factor learned on the other subjects;
# PCA = held-out RRR factor vs the PCA basis. Both fold-matched, in raw-embedding coordinates.
_LOSO_COMPARISONS = {"RRR": ("train_subjects", "rrr"), "PCA": ("baseline", "pca")}


def loso_comparison_pairs(raw, subject, fold, rank, methods=tuple(_LOSO_COMPARISONS)):
    held_out = raw(subject, fold, "baseline", "rrr", rank)
    return {name: (held_out, raw(subject, fold, *_LOSO_COMPARISONS[name], rank))
            for name in methods}


def _loso_overlaps_from_factors(raw, subjects, folds, max_rank, rank_limit=None):
    """`loso_subspace_overlaps` from loaded factors; ranks >= `rank_limit` are skipped.

    The held-out factor is orthonormalized once per (subject, fold, rank) and shared by the
    RRR and PCA comparisons.
    """
    ranks = [rank for rank in subspace_ranks(max_rank) if rank_limit is None or rank < rank_limit]
    rows = []
    for subject in subjects:
        overlaps = {(method, rank): [] for method in _LOSO_COMPARISONS for rank in ranks}
        for fold in folds:
            for rank in ranks:
                held_out = raw(subject, fold, "baseline", "rrr", rank)
                if held_out is None:
                    continue
                q_held_out = _orthonormal(held_out)
                for method, other in _LOSO_COMPARISONS.items():
                    b = raw(subject, fold, *other, rank)
                    if b is not None:
                        overlaps[(method, rank)].append(
                            _overlap_of_orthonormal(q_held_out, _orthonormal(b)))
        rows.extend((str(subject), method, rank, float(np.mean(v)))
                    for (method, rank), v in overlaps.items() if v)
    return pd.DataFrame(rows, columns=["subject", "method", "rank", "overlap"])


def loso_subspace_overlaps(results_root, dataset, model, layer):
    """Per-subject mean squared cosine vs rank, averaged over folds: [subject, method, rank, overlap]."""
    raw, subjects, folds, max_rank, _ = _loso_factors(results_root, dataset, model, layer)
    return _loso_overlaps_from_factors(raw, subjects, folds, max_rank)


def loso_squared_cosine_vectors(results_root, dataset, model, layer, ranks):
    """`{(method, rank): [n_subjects, rank]}` squared cosines of the sorted principal angles
    (smallest angle first), averaged over folds."""
    raw, subjects, folds, *_ = _loso_factors(results_root, dataset, model, layer)
    out = {}
    for method in _LOSO_COMPARISONS:
        for rank in ranks:
            out[(method, rank)] = np.stack([
                np.mean([squared_cosines(*loso_comparison_pairs(raw, subject, fold, rank, (method,))[method])
                         for fold in folds], axis=0)
                for subject in subjects])
    return out


def loso_saturation_ranks(results_root, dataset, model, layer):
    """Per held-out subject: first rank reaching 99% of the RRR-transfer explained-variance peak."""
    _, _, ev = _loso_one(Path(results_root), dataset, model, layer)
    return {subject: explained_variance_saturation_rank(
                ranks, np.asarray(curve).reshape(len(ranks), -1)[:, 0], SATURATION_FRACTION)
            for (subject, label), (ranks, curve) in ev.items() if label == "RRR transfer"}


def cross_dataset_basis(results_root, dataset, model, layer, method, rank):
    """Whole-dataset basis [d, rank] in raw-embedding coordinates.

    iEEG: the `full_fit` factor truncated to `rank` columns. fMRI: the internal artifact's
    rank-specific factor for `rank`.
    """
    if dataset == "fmri":
        artifact = _rrr(Path(results_root), "internal", dataset, model, layer
                        ).load_artifacts(["input_factors", "input_scalers"])
        factor = index_by_selector(artifact.input_factors)[(("method", method), ("rank", rank))]
        return to_raw_coordinates(factor, index_by_selector(artifact.input_scalers)[()])
    artifact = _rrr(Path(results_root), "full_fit", dataset, model, layer
                    ).load_artifacts(["input_factors", "input_scalers"])
    key = (("method", method),)
    return to_raw_coordinates(index_by_selector(artifact.input_factors)[key],
                              index_by_selector(artifact.input_scalers)[()])[:, :rank]


def cross_subspace_overlaps(results_root, pair, model, layer, ranks):
    """Overlap vs rank between two datasets' whole-dataset bases: PCA/PCA, RRR/RRR, RRR/PCA both ways."""
    first, second = pair
    rows = []
    for rank in ranks:
        basis = {(d, m): cross_dataset_basis(results_root, d, model, layer, m, rank)
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


def loso_layer_summary(results_root, dataset, model, layer):
    """Chance-normalized RRR-minus-PCA LOSO overlap vs rank, plus the saturation rank.

    excess = (overlap - r/d) / (1 - r/d) per method (0 = chance, 1 = identical), averaged
    over held-out subjects. Saturation rank = median over subjects of the per-subject
    saturation rank. Ranks r >= d are dropped.
    """
    # One artifact load; ranks >= dim are dropped, so never compute them (the full-rank
    # decompositions are by far the most expensive).
    raw, subjects, folds, max_rank, dim = _loso_factors(results_root, dataset, model, layer)
    overlaps = _loso_overlaps_from_factors(raw, subjects, folds, max_rank, rank_limit=dim)
    chance = overlaps["rank"] / dim
    overlaps["excess"] = (overlaps["overlap"] - chance) / (1 - chance)
    by_method = overlaps.groupby(["method", "rank"])["excess"].mean().unstack(0)
    saturation = float(np.median(list(loso_saturation_ranks(
        results_root, dataset, model, layer).values())))
    return {"ranks": by_method.index.to_numpy(), "rrr": by_method["RRR"].to_numpy(),
            "pca": by_method["PCA"].to_numpy(), "saturation": saturation, "dim": dim}


def saturation_split_table(summaries):
    """Mean RRR - PCA excess overlap over the ranks below / above each layer's saturation rank."""
    rows = []
    for (dataset, model, layer), s in summaries.items():
        delta = s["rrr"] - s["pca"]
        below, above = s["ranks"] < s["saturation"], s["ranks"] > s["saturation"]
        rows.append((dataset, model, layer, s["saturation"],
                     delta[below].mean() if below.any() else np.nan,
                     delta[above].mean() if above.any() else np.nan))
    return pd.DataFrame(rows, columns=["dataset", "model", "layer", "saturation_rank",
                                       "below", "above"])


def attempt(gaps, label, compute):
    """`compute()`, or None with the failure recorded in `gaps`."""
    try:
        return compute()
    except Exception as error:  # a missing artifact or raw dataset must not block the rest
        gaps.append(f"{label}: {type(error).__name__}: {error}"[:200])
        return None


def load_subspace_data(results_root, repo_root, sample_model, sample_layer,
                       distribution_ranks=(4, 16, 64, 256)):
    """Everything the subspace figures need, per dataset, with failures collected in `gaps`.

    Returns a dict with `loso` (overlap tables), `cross` (overlap table), `cosines`
    (squared-cosine vectors of the principal angles), `saturation` (median LOSO saturation rank), `summaries`
    (per-layer LOSO summaries over `LOSO_MODEL_KEYS`) and `gaps`.
    """
    gaps = []
    out = {"gaps": gaps, "loso": {}, "cosines": {}, "saturation": {}, "summaries": {},
           "ranks": list(distribution_ranks)}
    for dataset in LOSO_DATASETS:
        sample = (dataset, sample_model, sample_layer)
        out["loso"][dataset] = attempt(gaps, f"LOSO overlap {sample}", lambda: cached(
            results_root, "loso_overlap-%s-%s-%s" % sample,
            lambda: loso_subspace_overlaps(results_root, *sample)))
        out["cosines"][dataset] = attempt(gaps, f"LOSO squared cosines {sample}", lambda: cached(
            results_root, "loso_cosines-%s-%s-%s" % sample,
            lambda: loso_squared_cosine_vectors(
                results_root, *sample, list(distribution_ranks))))
        for key in LOSO_MODEL_KEYS:
            model, layers = MODEL_LAYERS[key]
            for layer in layers:
                label = (dataset, model, layer)
                summary = attempt(gaps, f"LOSO summary {label}", lambda: cached(
                    results_root, "loso_summary-%s-%s-%s" % label,
                    lambda: loso_layer_summary(results_root, *label)))
                if summary is not None:
                    out["summaries"][label] = summary
        # The summary's saturation is this same median, so only recompute it if the sample
        # layer is not among the summarized ones.
        if sample in out["summaries"]:
            out["saturation"][dataset] = out["summaries"][sample]["saturation"]
        else:
            out["saturation"][dataset] = attempt(gaps, f"LOSO saturation {sample}", lambda: cached(
                results_root, "loso_saturation-%s-%s-%s" % sample,
                lambda: float(np.median(list(loso_saturation_ranks(results_root, *sample).values())))))
    dim = cross_dataset_basis(results_root, "podcast", sample_model, sample_layer, "pca", 1).shape[0]
    out["dim"] = dim
    frames = []
    for pair in (("podcast", "cars-2"), ("podcast", "fmri"), ("cars-2", "fmri")):
        frame = attempt(gaps, f"cross-dataset overlap {pair}", lambda: cached(
            results_root, "cross_overlap-%s-%s-%s-%s" % (*pair, sample_model, sample_layer),
            lambda: cross_subspace_overlaps(
                results_root, pair, sample_model, sample_layer, subspace_ranks(dim))))
        if frame is not None:
            frames.append(frame)
    out["cross"] = pd.concat(frames, ignore_index=True) if frames else None
    return out

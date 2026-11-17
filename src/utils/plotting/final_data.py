"""Read artifacts once for the curated results notebook.

The returned values are ordinary tables, arrays, and dictionaries. Plot functions
never receive an Artifact or open a result file.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_significance import Artifact as SignificanceArtifact
from scripts.run_language_decoding import Artifact as DecodingArtifact
from scripts.utilities.check_final_significance import MODEL_LAYERS
from src.utils.rrr.plot_metrics import (
    explained_variance_curve, fisher_mean, rank_quantiles,
)


DATASETS = {"podcast": "podcast", "cars-2": "brain-treebank-cars-2", "fmri": "deep-fmri"}
QUANTILES = (.05, .10, .25, .50, .75, .90)
LABELS = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
PAIRS = tuple((model, layer) for model, layers in MODEL_LAYERS.values() for layer in layers)
FMRI_VARIANT = "input_scaling-pooled-training-stories-v1"
CC_MAX_FLOOR = 0.25  # noise ceilings below this are floored (as in the fMRI histograms)
CACHE_VERSION = 4
LAG_SUMMARIES = ("mean", "peak")
SATURATION_MODES = ("explained_variance", "quantile")


class ExplainedVarianceData(NamedTuple):
    """Explained-variance counterparts of the correlation curve tables.

    The frames mirror `internal`, `loso` and `cross` with an `explained_variance`
    column (mean r^2 per target) in place of `correlation`. `curves` and
    `loso_curves` mirror the quantile dictionaries, keyed the same way, with a
    (ranks, curve) pair. `full_rows` holds (dataset, model, layer, mean r^2) for the
    full-model reference.
    """

    internal: pd.DataFrame
    loso: pd.DataFrame
    cross: pd.DataFrame
    curves: dict
    loso_curves: dict
    full_rows: list

@lru_cache(maxsize=None)
def _noise_ceiling(results_root, subject):
    path = Path(results_root) / "fmri_noise_ceiling" / f"{subject}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/fmri/compute_noise_ceiling.py")
    with np.load(path) as archive:
        return archive["cc_max"].astype(float)


def _ev_scale(results_root, dataset, targets, normalize=True):
    """Per-target divisor of the explained variance: squared (floored) noise ceiling for fMRI.

    fMRI explained variance is always reported as a fraction of the explainable variance,
    r^2 / max(CC_max, CC_MAX_FLOOR)^2 per voxel; voxels without a defined ceiling are
    left out. iEEG has no noise ceiling (returns None: plain r^2), and neither does fMRI with
    `normalize=False` (the appendix's raw-r^2 version).
    """
    if dataset != "fmri" or not normalize:
        return None
    subjects = targets["subject"].astype(str).to_numpy()
    ceiling = np.full(len(targets), np.nan)
    for subject in np.unique(subjects):
        rows = subjects == subject
        ceiling[rows] = _noise_ceiling(str(results_root), subject)[
            targets.loc[rows, "voxel_index"].to_numpy().astype(int)]
    return np.maximum(ceiling, CC_MAX_FLOOR) ** 2


def _cache_fingerprint(results_root, lag_summary="mean"):
    """Track artifact manifests and summary code without opening large payloads."""
    digest = hashlib.blake2b(digest_size=20)
    sources = (Path(__file__), Path(__file__).parents[1] / "rrr" / "plot_metrics.py")
    manifests = (
        path for category in ("run_rrr", "run_significance", "run_language_decoding")
        for path in (results_root / category).rglob("manifest.json")
    )
    for path in sorted((*sources, *manifests)):
        stat = path.stat()
        digest.update(str(path.relative_to(results_root) if path.is_relative_to(results_root)
                          else path).encode())
        digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
    digest.update(str(CACHE_VERSION).encode())
    digest.update(lag_summary.encode())
    return digest.hexdigest()


def _read_cache(path, fingerprint):
    try:
        with path.open("rb") as handle:
            if handle.read(len(fingerprint)) != fingerprint.encode():
                return None
            return pickle.load(handle)
    except (FileNotFoundError, EOFError, OSError, pickle.UnpicklingError):
        return None


def _write_cache(path, fingerprint, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".final_plots_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(fingerprint.encode())
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _variant(artifact, dataset, mode, source=None):
    index = artifact._read_json(artifact._leaf() / "manifest.json", default={})
    candidates = list(index.get("variants", {}))
    if dataset == "fmri" and mode in {"internal", "cross_dataset"}:
        candidates = [v for v in candidates if FMRI_VARIANT in v.split("/")]
    if dataset == "cars-2":
        candidates = [v for v in candidates if "rereference-laplacian" in v.split("/")]
    if source is not None:
        candidates = [v for v in candidates if DATASETS[source] in v.split("/")]
    candidates = [v for v in candidates if artifact._read_json(
        artifact._leaf() / v / "manifest.json", default={}
    ).get("complete")]
    if len(candidates) != 1:
        raise ValueError(f"expected one completed {mode} variant, found {candidates}")
    artifact.variant = candidates[0]
    return artifact


def _rrr(results_root, mode, dataset, model, layer, source=None, shuffled_embeddings=False):
    artifact = RRRArtifact(
        mode=mode, dataset=DATASETS[dataset], model=model, layer=layer,
        results_root=results_root, shuffled_embeddings=shuffled_embeddings,
    )
    return _variant(artifact, dataset, mode, source)


def _groups(manifest, size):
    if manifest is None:
        return np.arange(size), size
    if len(manifest) != size:
        raise ValueError(f"target manifest has {len(manifest)} rows for {size} metrics")
    if "electrode" in manifest:
        columns = ["subject", "electrode"]
    elif "voxel_index" in manifest:
        columns = ["subject", "voxel_index"]
    else:
        return np.arange(size), size
    groups = manifest.groupby(columns, sort=False).ngroup().to_numpy()
    return groups, int(groups.max()) + 1


def _collapse(values, manifest):
    values = np.asarray(values, dtype=float).reshape(-1)
    groups, count = _groups(manifest, len(values))
    z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
    finite = np.isfinite(z)
    totals = np.bincount(groups[finite], weights=z[finite], minlength=count)
    sizes = np.bincount(groups[finite], minlength=count)
    return np.tanh(np.divide(totals, sizes, out=np.full(count, np.nan), where=sizes > 0))


def _peak_over_lags(values, manifest):
    """Each electrode's largest (signed) correlation across its lags."""
    values = np.asarray(values, dtype=float).reshape(-1)
    groups, count = _groups(manifest, len(values))
    peaks = np.full(count, np.nan)
    finite = np.isfinite(values)
    np.fmax.at(peaks, groups[finite], values[finite])
    return peaks


def _electrode_values(folds, manifest, lag_summary):
    """Per-electrode values from one per-target array per evaluation fold.

    "mean" Fisher-averages each fold's lags and returns one array per fold. "peak"
    first Fisher-averages the folds at every lag, then takes the maximum over lags,
    and returns a single array. fMRI targets have no lag structure; callers pass
    "mean" for them.
    """
    if lag_summary == "mean":
        return [_collapse(values, manifest) for values in folds]
    if lag_summary == "peak":
        stacked = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in folds])
        return [_peak_over_lags(fisher_mean(stacked, axis=0), manifest)]
    raise ValueError(f"lag_summary must be one of {LAG_SUMMARIES}, got {lag_summary!r}")


def _summary_for(dataset, lag_summary):
    return "mean" if dataset == "fmri" else lag_summary


def _manifest_for(artifact, selector):
    choices = artifact.target_manifest
    keys = {k: str(v) for k, v in selector.items() if k in {"method", "loso_phase", "held_out_subject"}}
    for encoded, frame in choices.items():
        found = {k: str(v) for k, v in dict(encoded).items()}
        if all(found.get(k) == value for k, value in keys.items()):
            return frame
    return choices.get(())


def _metric_name(dataset, method, *, phase=None):
    if dataset == "fmri":
        return "corr_per_target"
    if phase == "transfer":
        return "transfer_orth_corr_per_target"
    return "nr_orth_corr_per_target" if method == "rrr" else "pca_corr_per_target"


def _fisher(values, axis=None):
    """`fisher_mean` that stays quiet for all-NaN slices (missing target sets)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fisher_mean(values, axis=axis)


def _load_matrix(artifact, method, metric, selectors=None):
    """Standard `Artifact.load_rank_matrix` call; None when the method has no metrics."""
    try:
        return artifact.load_rank_matrix(method, metric=metric, selectors=selectors)
    except ValueError as error:
        if str(error).startswith("no per-target rank metrics"):
            return None
        raise


def _lagged(matrix):
    """RankMatrix values as [rank, fold, target, lag]; fMRI gets one trivial lag."""
    values = np.asarray(matrix.values, dtype=float)
    return values if values.ndim == 4 else values[..., None]


def _summarize(values, ranks, lag_summary, ev_scale=None):
    """Every figure summary of one method's [rank, fold, target, lag] correlations.

    Returns `fold_values` (the mean correlation per rank and fold; one column for
    "peak"), the target-quantile curves used in quantile saturation mode, and `ev`,
    the explained-variance curve used in explained-variance mode. `ev` always
    spans all (target, lag) pairs, so it does not depend on `lag_summary`. `ev_scale`
    ([target], see `_ev_scale`) divides each target's r^2 (fMRI noise ceiling).
    """
    ranks = np.asarray(ranks, dtype=float)
    if lag_summary == "mean":
        targets = _fisher(values, axis=3)
        fold_values = _fisher(values, axis=(2, 3))
    elif lag_summary == "peak":
        peaks = np.fmax.reduce(_fisher(values, axis=1), axis=2)
        targets = peaks[:, None, :]
        fold_values = _fisher(peaks, axis=1)[:, None]
    else:
        raise ValueError(f"lag_summary must be one of {LAG_SUMMARIES}, got {lag_summary!r}")
    per_target = _fisher(targets, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quantiles = rank_quantiles(ranks, per_target, QUANTILES)
    return {
        "ranks": ranks,
        "fold_values": fold_values,
        "quantiles": (ranks, quantiles),
        "ev": explained_variance_curve(
            _fisher(values, axis=1),
            scale=None if ev_scale is None else np.asarray(ev_scale)[:, None]),
    }


def _curve_rows(summary):
    return [(int(rank), float(value))
            for rank, row in zip(summary["ranks"], summary["fold_values"])
            for value in row if np.isfinite(value)]


def _subject_summaries(matrix, values, lag_summary, ev_scale=None):
    """Per-subject RRR curves from a pooled internal run (the LOSO full-training control)."""
    if "subject" not in matrix.targets:
        return [], {}, {}
    target_subjects = matrix.targets["subject"].astype(str).to_numpy()
    curves, quantiles, ev = [], {}, {}
    for subject in np.unique(target_subjects):
        rows = target_subjects == subject
        summary = _summarize(values[:, :, rows, :], matrix.ranks, lag_summary,
                             None if ev_scale is None else ev_scale[rows])
        curves.extend(
            (subject, "RRR baseline", int(rank), float(_fisher(row)))
            for rank, row in zip(summary["ranks"], summary["fold_values"])
            if np.isfinite(row).any()
        )
        quantiles[(subject, "RRR baseline")] = summary["quantiles"]
        ev[(subject, "RRR baseline")] = (summary["ranks"], summary["ev"])
    return curves, quantiles, ev


def _internal_one(results_root, dataset, model, layer, lag_summary="mean",
                  fmri_noise_normalized=True, shuffled_embeddings=False):
    artifact = _rrr(results_root, "internal", dataset, model, layer,
                    shuffled_embeddings=shuffled_embeddings).load_artifacts(
        ["target_manifest", "full_metrics"]
    )
    lag_summary = _summary_for(dataset, lag_summary)
    curves, quantiles, ev, rrr = [], {}, {}, None
    for method in ("rrr", "pca"):
        matrix = _load_matrix(artifact, method, _metric_name(dataset, method))
        if matrix is None:
            continue
        values = _lagged(matrix)
        scale = _ev_scale(results_root, dataset, matrix.targets, fmri_noise_normalized)
        summary = _summarize(values, matrix.ranks, lag_summary, scale)
        curves.extend((method, rank, value) for rank, value in _curve_rows(summary))
        quantiles[method] = summary["quantiles"]
        ev[method] = (summary["ranks"], summary["ev"])
        if method == "rrr":
            rrr = (matrix, values, scale)
    if not curves:
        raise ValueError("no internal rank correlations")
    full_folds = []
    for encoded, metrics in artifact.full_metrics.items():
        selector = dict(encoded)
        if dataset != "fmri" and selector.get("method") != "rrr":
            continue
        name = "corr_per_target" if dataset == "fmri" else "full_corr_per_target"
        if name in metrics:
            full_folds.append(metrics[name])
    full = (_electrode_values(full_folds, _manifest_for(artifact, {"method": "rrr"}), lag_summary)
            if full_folds else [])
    full_values = fisher_mean(np.stack(full), axis=0) if full else np.array([])
    full_ev = np.nan
    if full_folds:
        pairs = _fisher(np.stack([np.asarray(f, dtype=float).reshape(-1) for f in full_folds]),
                        axis=0)
        full_scale = rrr[2] if rrr is not None else None
        if full_scale is not None and len(full_scale) != pairs.shape[0]:
            raise ValueError("full-model targets do not match the rank-curve targets")
        full_ev = float(explained_variance_curve(pairs[None, :], scale=full_scale)[0])
    subject = (_subject_summaries(rrr[0], rrr[1], lag_summary, rrr[2]) if rrr is not None
               else ([], {}, {}))
    return curves, quantiles, ev, full_values, full_ev, subject


def _loso_one(results_root, dataset, model, layer, lag_summary="mean",
              fmri_noise_normalized=True):
    artifact = _rrr(results_root, "leave_one_subject_out", dataset, model, layer).load_artifacts(
        ["target_manifest"]
    )
    names = {("rrr", "transfer"): "RRR transfer", ("pca", "baseline"): "PCA baseline"}
    lag_summary = _summary_for(dataset, lag_summary)
    curves, quantiles, ev = [], {}, {}
    for (method, phase), label in names.items():
        held_out = sorted({dict(key)["held_out_subject"] for key in artifact.target_manifest
                           if dict(key).get("method") == method
                           and dict(key).get("loso_phase") == phase
                           and "held_out_subject" in dict(key)}, key=str)
        for subject in held_out:
            matrix = _load_matrix(
                artifact, method, _metric_name(dataset, method, phase=phase),
                {"loso_phase": phase, "held_out_subject": subject},
            )
            if matrix is None:
                continue
            summary = _summarize(_lagged(matrix), matrix.ranks, lag_summary,
                                 _ev_scale(results_root, dataset, matrix.targets,
                                           fmri_noise_normalized))
            curves.extend((str(subject), label, rank, value)
                          for rank, value in _curve_rows(summary))
            quantiles[(str(subject), label)] = summary["quantiles"]
            ev[(str(subject), label)] = (summary["ranks"], summary["ev"])
    return curves, quantiles, ev


def _cross_one(results_root, target, source, model, layer, internal_curves,
               internal_ev, lag_summary="mean", fmri_noise_normalized=True):
    artifact = _rrr(results_root, "cross_dataset", target, model, layer, source)
    lag_summary = _summary_for(target, lag_summary)
    records, ev_records = [], []
    for method in ("rrr", "pca"):
        label = f"Transfer {method.upper()}"
        if target == "fmri":
            candidates = ("corr_per_target",)
        elif method == "rrr":
            candidates = ("nr_orth_corr_per_target", "nr_corr_per_target")
        else:
            candidates = ("pca_corr_per_target",)
        matrix = next((m for m in (_load_matrix(artifact, method, name)
                                   for name in candidates) if m is not None), None)
        if matrix is None:
            continue
        summary = _summarize(_lagged(matrix), matrix.ranks, lag_summary,
                             _ev_scale(results_root, target, matrix.targets,
                                       fmri_noise_normalized))
        records.extend((label, rank, value) for rank, value in _curve_rows(summary))
        ev_records.extend((label, int(rank), float(value))
                          for rank, value in zip(summary["ranks"], summary["ev"]))
    for method, rank, value in internal_curves:
        records.append((f"Within {method.upper()}", rank, value))
    for method, (ranks, curve) in internal_ev.items():
        ev_records.extend((f"Within {method.upper()}", int(rank), float(value))
                          for rank, value in zip(ranks, curve))
    required = {"Within RRR", "Within PCA", "Transfer RRR", "Transfer PCA"}
    frame = pd.DataFrame(records, columns=["method", "rank", "correlation"])
    if not required.issubset(frame["method"]):
        raise ValueError("incomplete four-method directed curves")
    frame["fisher_z"] = np.arctanh(np.clip(frame.pop("correlation"), -1 + 1e-7, 1 - 1e-7))
    frame = frame.groupby(["method", "rank"], as_index=False)["fisher_z"].mean()
    frame["correlation"] = np.tanh(frame.pop("fisher_z"))
    ev_frame = pd.DataFrame(ev_records, columns=["method", "rank", "explained_variance"])
    return frame, ev_frame


def _significance_one(results_root, dataset, model, layer):
    mode = "held_out_session" if dataset == "fmri" else "sig_lags"
    variant = "rereference-laplacian" if dataset == "cars-2" else "default"
    artifact = SignificanceArtifact(
        mode=mode, dataset=DATASETS[dataset], model=model, layer=layer,
        variant=variant, results_root=results_root,
    )
    if not artifact.manifest().get("complete"):
        raise ValueError("incomplete significance artifact")
    targets = artifact.load_targets()
    return {(str(subject), str(row.target_id)) for subject, frame in targets.items()
            for row in frame.itertuples() if bool(row.significant)}


def _decoding_summaries(results_root):
    root = Path(results_root) / "run_language_decoding"
    rows = []
    for mode in ("within_dataset", "cross_dataset"):
        for path in root.glob(f"{mode}/*/offset-*/*/layer-*/manifest.json"):
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
                identity = index["identity"]
                variants = index["variants"]
            except (ValueError, KeyError, OSError):
                continue
            for variant in variants:
                try:
                    artifact = DecodingArtifact(
                        mode=identity["mode"], dataset=identity["dataset"],
                        offset=identity["offset"], model=identity["model"],
                        layer=identity["layer"], variant=variant, results_root=results_root,
                    )
                    summary = artifact.load_field("summary").copy()
                    config = artifact.manifest().get("configuration", {})
                except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError):
                    continue
                summary["dataset"] = next((d for d, root_name in DATASETS.items()
                                           if root_name == identity["dataset"]), identity["dataset"])
                summary["model"] = identity["model"]
                summary["layer"] = identity["layer"]
                summary["label_offset"] = int(config.get("label_offset", 0))
                summary["variant"] = variant
                summary["generation"] = (
                    "cross_dataset" if mode == "cross_dataset" or variant.startswith("source-")
                    else "within_dataset"
                )
                rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_electrode_curves(results_root, dataset, model, layer, lag_summary="mean"):
    """Per-electrode internal RRR and PCA correlation by rank for one iEEG model layer.

    Each electrode's (electrode, lag) correlations are Fisher-averaged across lags,
    then across evaluation folds. Returns a frame with columns subject, electrode,
    method, rank, correlation, covering the targets stored in the internal artifact.
    """
    if dataset == "fmri":
        raise ValueError("electrode curves are only defined for iEEG datasets")
    artifact = _rrr(results_root, "internal", dataset, model, layer).load_artifacts(
        ["rank_metrics", "target_manifest"]
    )
    per_fold = defaultdict(list)
    for encoded, metrics in artifact.rank_metrics.items():
        selector = dict(encoded)
        method, rank = selector.get("method"), selector.get("rank")
        name = _metric_name(dataset, method) if method in {"rrr", "pca"} else None
        if rank is None or name not in metrics:
            continue
        per_fold[(method, int(rank))].append(metrics[name])
    frames = []
    for (method, rank), folds in sorted(per_fold.items()):
        manifest = _manifest_for(artifact, {"method": method})
        values = _electrode_values(folds, manifest, lag_summary)
        groups, count = _groups(manifest, len(manifest))
        first = np.full(count, -1)
        first[groups[::-1]] = np.arange(len(groups))[::-1]
        identity = manifest.iloc[first][["subject", "electrode"]].reset_index(drop=True)
        identity["subject"] = identity["subject"].astype(int)
        identity["electrode"] = identity["electrode"].astype(str)
        identity["method"], identity["rank"] = method, rank
        identity["correlation"] = fisher_mean(np.stack(values), axis=0)
        frames.append(identity)
    if not frames:
        raise ValueError("no per-electrode rank correlations")
    return pd.concat(frames, ignore_index=True)


def load_shuffled_internal_data(results_root, lag_summary="mean", fmri_noise_normalized=True):
    """Explained-variance internal RRR inputs from the shuffled-embedding null control.

    Reads `results/run_rrr_shuffled/` (internal mode only) and returns
    (`ExplainedVarianceData`, gaps). Only `internal`, `curves` and `full_rows` are populated
    (LOSO and cross tables are empty), matching what `rrr_aggregate(mode="internal",
    saturation_mode="explained_variance")` reads. Not cached.
    """
    results_root = Path(results_root)
    ev_rows, ev_curves, ev_full_rows, gaps = [], {}, [], []
    for dataset in DATASETS:
        for model, layer in PAIRS:
            identity = (dataset, model, layer)
            try:
                _, _, internal_ev, _, full_ev, _ = _internal_one(
                    results_root, *identity, lag_summary, fmri_noise_normalized,
                    shuffled_embeddings=True)
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"shuffled internal {identity}: {error}")
                continue
            ev_curves.update({(*identity, method): (ranks, curve[:, None])
                              for method, (ranks, curve) in internal_ev.items()})
            ev_rows.extend((dataset, model, layer, method, int(rank), float(value))
                           for method, (ranks, curve) in internal_ev.items()
                           for rank, value in zip(ranks, curve))
            if np.isfinite(full_ev):
                ev_full_rows.append((*identity, full_ev))
    empty = lambda columns: pd.DataFrame(columns=columns)
    data = ExplainedVarianceData(
        internal=pd.DataFrame(ev_rows, columns=[
            "dataset", "model", "layer", "method", "rank", "explained_variance"]),
        loso=empty(["dataset", "model", "layer", "subject", "method", "rank", "explained_variance"]),
        cross=empty(["method", "rank", "explained_variance", "target_dataset",
                     "source_dataset", "model", "layer"]),
        curves=ev_curves, loso_curves={}, full_rows=ev_full_rows,
    )
    return data, gaps


def load_final_data(results_root, lag_summary="mean", fmri_noise_normalized=True):
    """Eagerly load all requested figures' inputs and return separate plain values.

    Returns (internal, loso, cross, quantiles, loso_quantiles, full_rows,
    significant, decoding, gaps, explained_variance). All rank results are read
    through `Artifact.load_rank_matrix`; the first nine values are the quantile-mode
    inputs and `explained_variance` (an `ExplainedVarianceData`) holds the
    explained-variance mode's.

    `lag_summary` chooses how iEEG (electrode, lag) targets become electrode values
    in the quantile-mode inputs: "mean" averages across lags, "peak" takes each
    electrode's best lag. Explained variance always uses every (electrode, lag)
    pair. fMRI and decoding results are unaffected.

    fMRI explained variance is the fraction of explainable variance, r^2 / CC_max^2 per
    voxel, unless `fmri_noise_normalized=False` (raw r^2, for the appendix); the cache is
    kept separately for each setting. Correlation-mode inputs are never normalized.
    """
    if lag_summary not in LAG_SUMMARIES:
        raise ValueError(f"lag_summary must be one of {LAG_SUMMARIES}, got {lag_summary!r}")
    results_root = Path(results_root)
    suffix = ("" if lag_summary == "mean" else f"_{lag_summary}")
    suffix += "" if fmri_noise_normalized else "_raw_fmri"
    cache_path = results_root / f".final_plots_cache{suffix}.pkl"
    fingerprint = _cache_fingerprint(results_root, lag_summary + suffix)
    cached = _read_cache(cache_path, fingerprint)
    if cached is not None:
        return cached
    internal_rows, loso_rows, cross_rows, full_rows = [], [], [], []
    ev_internal_rows, ev_loso_rows, ev_cross_rows, ev_full_rows = [], [], [], []
    quantiles, loso_quantiles, significant, gaps = {}, {}, {}, []
    ev_curves, ev_loso_curves = {}, {}
    for dataset in DATASETS:
        for model, layer in PAIRS:
            identity = (dataset, model, layer)
            subject_curves, subject_quantiles, subject_ev = [], {}, {}
            curves, internal_ev = [], {}
            try:
                (curves, matrices, internal_ev, full, full_ev,
                 (subject_curves, subject_quantiles, subject_ev)) = _internal_one(
                    results_root, *identity, lag_summary, fmri_noise_normalized
                )
                quantiles.update({(*identity, method): value for method, value in matrices.items()})
                internal_rows.extend((dataset, model, layer, *row) for row in curves)
                ev_curves.update({(*identity, method): (ranks, curve[:, None])
                                  for method, (ranks, curve) in internal_ev.items()})
                ev_internal_rows.extend(
                    (dataset, model, layer, method, int(rank), float(value))
                    for method, (ranks, curve) in internal_ev.items()
                    for rank, value in zip(ranks, curve)
                )
                if full.size:
                    full_rows.append((dataset, model, layer, full))
                if np.isfinite(full_ev):
                    ev_full_rows.append((dataset, model, layer, full_ev))
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"internal {identity}: {error}")
                curves, internal_ev = [], {}
            try:
                values, matrices, loso_ev = _loso_one(results_root, *identity, lag_summary,
                                                  fmri_noise_normalized)
                transfer_subjects = {subject for subject, method, _, _ in values
                                     if method == "RRR transfer"}
                pca_subjects = {subject for subject, method, _, _ in values
                                if method == "PCA baseline"}
                control_subjects = {subject for subject, _, _, _ in subject_curves}
                matched_subjects = transfer_subjects & pca_subjects & control_subjects
                values = [row for row in values if row[0] in matched_subjects]
                values.extend(row for row in subject_curves if row[0] in matched_subjects)
                matrices = {key: value for key, value in matrices.items()
                            if key[0] in matched_subjects}
                matrices.update({key: value for key, value in subject_quantiles.items()
                                 if key[0] in matched_subjects})
                loso_ev = {key: value for key, value in loso_ev.items()
                           if key[0] in matched_subjects}
                loso_ev.update({key: value for key, value in subject_ev.items()
                                if key[0] in matched_subjects})
                loso_quantiles.update({(*identity, *key): value for key, value in matrices.items()})
                loso_rows.extend((dataset, model, layer, *row) for row in values)
                ev_loso_curves.update({(*identity, *key): (ranks, curve[:, None])
                                       for key, (ranks, curve) in loso_ev.items()})
                ev_loso_rows.extend(
                    (dataset, model, layer, subject, method, int(rank), float(value))
                    for (subject, method), (ranks, curve) in loso_ev.items()
                    for rank, value in zip(ranks, curve)
                )
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"LOSO {identity}: {error}")
            try:
                significant[identity] = _significance_one(results_root, *identity)
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"significance {identity}: {error}")
            for source in DATASETS:
                if source == dataset:
                    continue
                try:
                    frame, ev_frame = _cross_one(
                        results_root, dataset, source, model, layer, curves, internal_ev,
                        lag_summary, fmri_noise_normalized,
                    )
                    for table in (frame, ev_frame):
                        table["target_dataset"], table["source_dataset"] = dataset, source
                        table["model"], table["layer"] = model, layer
                    cross_rows.append(frame)
                    ev_cross_rows.append(ev_frame)
                except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                    gaps.append(f"directed {source} → {dataset} {model}/layer-{layer}: {error}")
    internal = pd.DataFrame(internal_rows, columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    loso = pd.DataFrame(loso_rows, columns=["dataset", "model", "layer", "subject", "method", "rank", "correlation"])
    cross = pd.concat(cross_rows, ignore_index=True) if cross_rows else pd.DataFrame(
        columns=["method", "rank", "correlation", "target_dataset", "source_dataset", "model", "layer"]
    )
    explained_variance = ExplainedVarianceData(
        internal=pd.DataFrame(ev_internal_rows, columns=[
            "dataset", "model", "layer", "method", "rank", "explained_variance"]),
        loso=pd.DataFrame(ev_loso_rows, columns=[
            "dataset", "model", "layer", "subject", "method", "rank", "explained_variance"]),
        cross=pd.concat(ev_cross_rows, ignore_index=True) if ev_cross_rows else pd.DataFrame(
            columns=["method", "rank", "explained_variance", "target_dataset",
                     "source_dataset", "model", "layer"]),
        curves=ev_curves, loso_curves=ev_loso_curves, full_rows=ev_full_rows,
    )
    decoding = _decoding_summaries(results_root)
    result = (internal, loso, cross, quantiles, loso_quantiles, full_rows,
              significant, decoding, gaps, explained_variance)
    _write_cache(cache_path, fingerprint, result)
    return result

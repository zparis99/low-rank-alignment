"""Unified client entry point for dataset-specific reduced-rank regression."""

from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd

from src.core.artifacts import (
    ArtifactError,
    ArtifactStore,
    canonical_significance_selection,
    normalize_dataset,
    normalize_model,
    readable_variant,
)
from src.core.config import (
    FMRICrossDatasetRRRConfig,
    RRRConfig,
    ScriptMode,
    SignificanceSelectionConfig,
)
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)


FMRI_DATASETS = {"fmri", "deep_fmri", "deep-fmri"}


def _is_fmri(dataset):
    return dataset in FMRI_DATASETS


def _strict_branch(payload: dict, cls):
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    from src.utils.shared.config import dict_to_dataclass
    return dict_to_dataclass(payload, cls)


@dataclass
class Config:
    """One client-facing RRR configuration for every neural dataset."""

    STRICT_CONFIG: ClassVar[bool] = True

    dataset: str = "podcast"
    mode: ScriptMode = ScriptMode.INTERNAL
    results_root: str = "results"
    plot: bool = True
    refresh: bool = False
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    significance_selection: SignificanceSelectionConfig = field(
        default_factory=SignificanceSelectionConfig
    )
    rrr_source: dict | None = None
    views: list[dict] = field(default_factory=list)
    multi_view_name: str | None = None
    subspace_aggregation: str | None = None
    # Null control: randomly permute which word's embedding is paired with which
    # neural response (internal mode only). Results get a distinct Artifact variant.
    shuffle_embeddings: bool = False
    shuffle_seed: int = 0
    ieeg: dict | None = field(default_factory=dict)
    fmri: dict | None = None

    @property
    def modality(self):
        return "fmri" if _is_fmri(self.dataset) else "ieeg"

    def validate(self):
        if isinstance(self.mode, str):
            self.mode = ScriptMode(self.mode)
        selected = self.fmri if self.modality == "fmri" else self.ieeg
        inactive = self.ieeg if self.modality == "fmri" else self.fmri
        if selected is None:
            raise ValueError(f"dataset={self.dataset!r} requires a {self.modality} block")
        if inactive not in (None, {}):
            raise ValueError("Only the dataset-selected modality block may be configured")
        shared = {
            ScriptMode.INTERNAL,
            ScriptMode.CROSS_DATASET,
            ScriptMode.LEAVE_ONE_SUBJECT_OUT,
            ScriptMode.MULTI_VIEW,
        }
        if self.modality == "fmri" and self.mode not in shared:
            raise ValueError(f"RRR mode {self.mode.value!r} is unsupported for fMRI")
        if self.mode in {
            ScriptMode.CROSS_DATASET, ScriptMode.CROSS_TEMPORAL_WINDOWS
        } and not self.rrr_source:
            raise ValueError(f"mode={self.mode.value} requires rrr_source")
        if self.mode == ScriptMode.MULTI_VIEW and not self.multi_view_name:
            raise ValueError("mode=multi_view requires multi_view_name")
        if self.shuffle_embeddings and self.mode != ScriptMode.INTERNAL:
            raise ValueError("shuffle_embeddings is supported only for mode=internal")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")

    @staticmethod
    def _legacy_source(source):
        return deepcopy(source) if source else None

    def selected_config(self):
        self.validate()
        payload = deepcopy(self.fmri if self.modality == "fmri" else self.ieeg) or {}
        if self.modality == "fmri" and "data" in payload:
            payload["fmri"] = payload.pop("data")
        aggregation = self.subspace_aggregation
        if (
            aggregation is None
            and self.modality == "fmri"
            and self.mode in {ScriptMode.INTERNAL, ScriptMode.MULTI_VIEW}
        ):
            aggregation = "prediction_weighted"
        payload.update({
            "results_root": self.results_root,
            "plot": self.plot,
            "refresh": self.refresh,
            "rank_grid": self.rank_grid,
            "significance_selection": deepcopy(self.significance_selection),
            "subspace_aggregation": aggregation,
            "shuffle_embeddings": self.shuffle_embeddings,
            "shuffle_seed": self.shuffle_seed,
        })
        converted_views = []
        for view in self.views:
            item = {key: deepcopy(value) for key, value in view.items() if key != "source"}
            source = view.get("source")
            if not source:
                raise ValueError(f"RRR view {view.get('name')!r} requires source")
            item["fmri_rrr_source" if _is_fmri(source.get("dataset"))
                 else "ieeg_rrr_source"] = deepcopy(source)
            converted_views.append(item)
        payload["views"] = converted_views
        payload["multi_view_name"] = self.multi_view_name
        if self.modality == "fmri":
            payload["source_mode"] = {
                ScriptMode.INTERNAL: "within_fmri",
                ScriptMode.CROSS_DATASET: "cross_dataset",
                ScriptMode.LEAVE_ONE_SUBJECT_OUT: "leave_one_subject_out",
                ScriptMode.MULTI_VIEW: "multi_view",
            }[self.mode]
            if self.rrr_source:
                if _is_fmri(self.rrr_source.get("dataset")):
                    raise ValueError("fMRI cross-dataset RRR requires an iEEG rrr_source")
                payload["ieeg_rrr_source"] = self._legacy_source(self.rrr_source)
            return _strict_branch(payload, FMRICrossDatasetRRRConfig)

        payload.setdefault("encoding", {})["dataset"] = self.dataset
        payload["multi_view_name"] = self.multi_view_name
        payload["mode"] = (
            ScriptMode.CROSS_SUBJECT if self.mode == ScriptMode.INTERNAL else self.mode
        )
        if self.rrr_source:
            key = "fmri_rrr_source" if _is_fmri(self.rrr_source.get("dataset")) else "ieeg_rrr_source"
            payload[key] = self._legacy_source(self.rrr_source)
            if (
                key == "fmri_rrr_source"
                and payload[key].get("subspace_aggregation") is None
            ):
                payload[key]["subspace_aggregation"] = "prediction_weighted"
            payload["cross_source_format"] = (
                "rank_bases" if key == "fmri_rrr_source" else "fold_factors"
            )
        return _strict_branch(payload, RRRConfig)


@dataclass(frozen=True)
class UnitSelector:
    method: str = "rrr"
    subject: int | str | None = None
    region: str | None = None
    temporal_window: str | None = None
    source_temporal_window: str | None = None
    target_temporal_window: str | None = None
    held_out_subject: int | str | None = None
    loso_phase: str | None = None
    cross_dataset_source: str | None = None

    def mapping(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


class RankMatrix(NamedTuple):
    """Per-target rank metrics laid out as one array.

    `values` is [rank, fold, electrode, lag] for iEEG or [rank, fold, voxel] for
    fMRI. `targets` labels the electrode/voxel axis (row i describes
    `values[:, :, i]`); `lags` holds lag seconds for iEEG and is None for fMRI.
    `folds` lists the stored fold selectors (None when results have no folds, which
    gives a fold axis of length 1). Combinations that were not computed are NaN.
    """

    values: np.ndarray
    ranks: np.ndarray
    folds: list
    targets: pd.DataFrame
    lags: np.ndarray | None


_RANK_METRIC_CANDIDATES = {
    "rrr": ("nr_orth_corr_per_target", "corr_per_target"),
    "pca": ("pca_corr_per_target", "corr_per_target"),
}


def assemble_rank_matrix(rank_metrics, manifest, method, metric=None):
    """Stack per-target rank metrics into a RankMatrix (pure; no I/O)."""
    entries = {}
    for encoded, metrics in rank_metrics.items():
        selector = dict(encoded)
        if selector.get("method") != method or "rank" not in selector:
            continue
        name = metric or next((c for c in _RANK_METRIC_CANDIDATES.get(method, ())
                               if c in metrics), None)
        if name is None or name not in metrics:
            continue
        entries[(int(selector["rank"]), selector.get("fold"))] = np.asarray(
            metrics[name], dtype=float
        ).reshape(-1)
    if not entries:
        raise ValueError(f"no per-target rank metrics found for method {method!r}"
                         + (f" and metric {metric!r}" if metric else ""))
    for key, values in entries.items():
        if len(values) != len(manifest):
            raise ValueError(
                f"rank {key[0]} fold {key[1]} has {len(values)} targets but the "
                f"manifest has {len(manifest)}; narrow `selectors` to one target set"
            )
    ranks = sorted({rank for rank, _ in entries})
    folds = sorted({fold for _, fold in entries}, key=lambda f: (f is None, f))
    if "lag_index" in manifest:
        electrode_id = manifest.groupby(["subject", "electrode"], sort=False).ngroup().to_numpy()
        lag_id = manifest["lag_index"].to_numpy()
        shape = (len(ranks), len(folds), int(electrode_id.max()) + 1, int(lag_id.max()) + 1)
        targets = manifest.drop_duplicates(["subject", "electrode"])[
            ["subject", "electrode"]].reset_index(drop=True)
        lags = (manifest.drop_duplicates("lag_index").sort_values("lag_index")["lag"]
                .to_numpy())
    else:
        shape = (len(ranks), len(folds), len(manifest))
        targets = manifest[[c for c in ("subject", "voxel_index") if c in manifest]
                           ].reset_index(drop=True)
        lags = None
    values = np.full(shape, np.nan)
    for (rank, fold), array in entries.items():
        i, j = ranks.index(rank), folds.index(fold)
        if lags is None:
            values[i, j] = array
        else:
            values[i, j][electrode_id, lag_id] = array
    return RankMatrix(values, np.asarray(ranks), folds, targets, lags)


def _dataset_adapter_name(dataset):
    """Map an Artifact dataset identity to its registered dataset adapter name."""
    if dataset == "podcast":
        return "podcast"
    if dataset.startswith("brain-treebank-"):
        return "brain_treebank"
    raise ValueError(f"no dataset adapter for Artifact dataset {dataset!r}")


class Artifact(ArtifactStore):
    """Unified storage and loading contract for iEEG and fMRI RRR results."""

    SCRIPT = "run_rrr"
    COMPACT_STORAGE = True
    SELECTOR_ORDER = (
        "subject", "method", "region", "temporal_window", "source_temporal_window",
        "target_temporal_window", "held_out_subject", "loso_phase",
        "cross_dataset_source", "basis", "delay",
        "rank", "fold",
        "name",
    )
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "target_manifest": True, "input_factors": False,
        "input_scalers": False,
        "target_factors": False, "full_metrics": True,
        "rank_metrics": True, "fold_results": False, "mean_results": True,
        "metadata": False, "unit_metadata": False, "diagnostics": False,
        "plots": False, "mode_metadata": False
    }

    def __init__(self, *, mode="internal", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None,
                 significance_selection=None, subspace_aggregation=None,
                 rereference=None, results_root="results",
                 shuffled_embeddings=False):
        if variant is not None and (
            significance_selection is not None
            or subspace_aggregation is not None
            or rereference is not None
        ):
            raise ValueError(
                "RRR Artifact variant is mutually exclusive with "
                "significance_selection, subspace_aggregation, and rereference"
            )
        if (
            significance_selection is not None
            or subspace_aggregation is not None
            or rereference is not None
        ):
            extras = {}
            if rereference not in (None, "common_average"):
                extras["rereference"] = rereference
            if dataset == "deep-fmri":
                extras["input_scaling"] = "pooled-training-stories-v1"
            if mode == "cross_dataset":
                extras["transfer_scaling"] = (
                    "source-then-target-reduced-pre-fir-v1"
                    if dataset == "deep-fmri"
                    else "source-then-target-reduced-v1"
                )
            variant = readable_variant(
                {
                    "significance_selection": significance_selection,
                    "subspace_aggregation": subspace_aggregation,
                },
                extras=extras or None,
            )
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)
        self.shuffled_embeddings = bool(shuffled_embeddings)
        if mode == "full_fit":
            self.FIELD_CONTRACT = {
                **type(self).FIELD_CONTRACT,
                "full_metrics": False,
                "rank_metrics": False,
                "mean_results": False,
            }
        self.requested_significance_selection = significance_selection
        self.requested_subspace_aggregation = subspace_aggregation
        self.requested_rereference = rereference

    def _manifest(self):
        manifest = super()._manifest()
        if self.requested_significance_selection is not None:
            saved = manifest.get("configuration", {}).get("significance_selection")
            if saved is None:
                raise ArtifactError(
                    "Selected RRR Artifact does not record significance_selection"
                )
            requested = canonical_significance_selection(
                self.requested_significance_selection
            )
            if canonical_significance_selection(saved) != requested:
                raise ArtifactError(
                    "Selected RRR Artifact significance_selection does not match "
                    "the requested selection"
                )
        if self.requested_subspace_aggregation is not None:
            saved = manifest.get("configuration", {}).get(
                "subspace_aggregation", "extrinsic"
            )
            if saved != self.requested_subspace_aggregation:
                raise ArtifactError(
                    "Selected RRR Artifact subspace_aggregation does not match "
                    "the requested method"
                )
        if self.requested_rereference is not None:
            saved = (
                manifest.get("configuration", {})
                .get("encoding", {})
                .get("per_dataset_config", {})
                .get("rereference", "common_average")
            )
            if saved != self.requested_rereference:
                raise ArtifactError(
                    "Selected RRR Artifact rereference does not match the "
                    "requested mode"
                )
        return manifest

    @classmethod
    def list_available(cls, results_root="results", dataset=None, mode=None):
        """Return every saved RRR Artifact identity under ``results_root``.

        Each entry is a dict with ``mode``, ``dataset``, ``offset``, ``model``,
        ``layer`` (``None`` for ``layer-none``), ``variant``, and ``complete``,
        sorted by those fields. Entries come from the per-leaf variant index.
        If ``dataset`` (e.g. ``"podcast"``) or ``mode`` (e.g. ``"internal"``) is
        given, only matching entries are listed.
        """
        base = Path(results_root) / cls.SCRIPT
        entries = []
        pattern = f"{mode or '*'}/{dataset or '*'}/offset-*/*/layer-*/manifest.json"
        for index_path in base.glob(pattern):
            layer_dir = index_path.parent
            model_dir = layer_dir.parent
            offset_dir = model_dir.parent
            dataset_dir = offset_dir.parent
            mode_dir = dataset_dir.parent
            layer = layer_dir.name.removeprefix("layer-")
            index = cls._read_json(index_path, default={})
            for variant, info in index.get("variants", {}).items():
                entries.append({
                    "mode": mode_dir.name,
                    "dataset": dataset_dir.name,
                    "offset": int(offset_dir.name.removeprefix("offset-")),
                    "model": model_dir.name,
                    "layer": None if layer == "none" else int(layer),
                    "variant": variant,
                    "complete": bool(info.get("complete")),
                })
        return sorted(
            entries,
            key=lambda e: (
                e["mode"], e["dataset"], e["offset"], e["model"],
                -1 if e["layer"] is None else e["layer"], e["variant"],
            ),
        )

    @classmethod
    def from_ieeg_config(cls, config, dataset_context):
        enc = config.encoding
        is_multi_view = config.mode == ScriptMode.MULTI_VIEW
        model = (
            config.multi_view_name
            if is_multi_view
            else enc.per_model_config.get("model_dir", enc.model)
        )
        mode = "internal" if config.mode == ScriptMode.CROSS_SUBJECT else config.mode.value
        selected_rereference = getattr(dataset_context, "rereference", None)
        artifact = cls(
            mode=mode,
            dataset=normalize_dataset(enc.dataset, dataset_context.stimulus_id),
            offset=enc.offset,
            model=normalize_model(model, glove_dim=enc.per_model_config.get("dim")),
            layer=(
                None
                if is_multi_view or enc.model == "glove"
                else enc.per_model_config.get("layer")
            ),
            significance_selection=(
                config.significance_selection if is_multi_view else None
            ),
            subspace_aggregation=(
                config.subspace_aggregation if is_multi_view else None
            ),
            rereference=(selected_rereference if is_multi_view else None),
            results_root=config.results_root,
            shuffled_embeddings=config.shuffle_embeddings,
        )
        artifact.variant_extras = {}
        rereference = selected_rereference
        if rereference not in (None, "common_average"):
            artifact.variant_extras["rereference"] = rereference
        if config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
            artifact.variant_extras["loso_contract"] = "orthogonal-held-out-v2"
        if config.mode == ScriptMode.CROSS_DATASET:
            artifact.variant_extras["transfer_scaling"] = "source-then-target-reduced-v1"
        return artifact

    @classmethod
    def from_fmri_config(cls, config):
        mode = "internal" if config.source_mode == "within_fmri" else config.source_mode
        is_multi_view = config.source_mode == "multi_view"
        artifact = cls(
            mode=mode,
            dataset="deep-fmri",
            offset=0,
            model=(config.multi_view_name if is_multi_view
                   else normalize_model(config.model_dir)),
            layer=None if is_multi_view else config.layer,
            significance_selection=(config.significance_selection if is_multi_view else None),
            subspace_aggregation=(config.subspace_aggregation if is_multi_view else None),
            results_root=config.results_root,
            shuffled_embeddings=config.shuffle_embeddings,
        )
        artifact.variant_extras = {
            "input_scaling": "pooled-training-stories-v1",
        }
        if is_multi_view:
            artifact.variant_extras["multiview-fir"] = "view-delay-v1"
        if config.source_mode == "cross_dataset":
            artifact.variant_extras["transfer_scaling"] = (
                "source-then-target-reduced-pre-fir-v1"
            )
        if config.source_mode == "leave_one_subject_out":
            artifact.variant_extras["loso_contract"] = "orthogonal-held-out-fmri-v1"
        return artifact

    @staticmethod
    def _selector(unit=None, **extra):
        result = unit.mapping() if isinstance(unit, UnitSelector) else dict(unit or {})
        result.update({key: value for key, value in extra.items() if value is not None})
        return result

    def save_factors(self, *, unit=None, fold=None, method=None, rank=None,
                     input_factors=None, factors=None, target_factors=None,
                     input_scaler=None, delay=None, basis=None, overwrite=False):
        value = input_factors if input_factors is not None else factors
        selector = self._selector(
            unit, fold=fold, method=method, rank=rank, delay=delay, basis=basis
        )
        self.save_field("input_factors", value, selector=selector, overwrite=overwrite)
        if input_scaler is not None:
            self.save_field(
                "input_scalers", input_scaler, selector=selector, overwrite=overwrite
            )
        if target_factors is not None:
            self.save_field("target_factors", target_factors, selector=selector,
                            overwrite=overwrite)

    @staticmethod
    def _input_scaler_selectors(factor_selector):
        """Resolve legacy factor-local scalers before the shared source scaler."""
        selector = dict(factor_selector or {})
        candidates = [selector]
        if "rank" in selector:
            # Old rank-producing artifacts sometimes stored one scaler per method.
            candidates.append({key: value for key, value in selector.items()
                               if key != "rank"})
        # Current full-fit/fMRI artifacts store the one ambient scaler globally:
        # rank and method branch only after this transformation has been applied.
        candidates.append({})
        unique = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def has_input_scaler(self, factor_selector=None):
        return any(
            self.has_field("input_scalers", selector=selector)
            for selector in self._input_scaler_selectors(factor_selector)
        )

    def load_input_scaler(self, factor_selector=None):
        for selector in self._input_scaler_selectors(factor_selector):
            if self.has_field("input_scalers", selector=selector):
                return self.load_field("input_scalers", selector=selector)
        raise KeyError(
            f"Artifact has no input scaler for factor selector {factor_selector or {}}"
        )

    def load_input_factors(self, method="rrr", rank=None, selectors=None):
        filters = dict(selectors or {})
        filters["method"] = method
        entries = self.load_collection("input_factors", filters)
        ranked = [(dict(selector), value) for selector, value in entries.items()
                  if dict(selector).get("rank") == rank]
        if ranked:
            return [np.asarray(value) for _, value in ranked]
        values = []
        for selector, value in entries.items():
            mapping = dict(selector)
            if "rank" in mapping or "fold" not in mapping:
                continue
            array = np.asarray(value)
            values.append(array if rank is None else array[:, :int(rank)])
        return values

    def save_rank_metrics(self, *, unit=None, method=None, rank, fold=None,
                          metrics, overwrite=False):
        self.save_field("rank_metrics", metrics,
                        selector=self._selector(unit, method=method, rank=int(rank), fold=fold),
                        overwrite=overwrite)

    def load_rank_metrics(self, *, unit=None, method=None, rank, fold=None):
        return self.load_field("rank_metrics",
                               selector=self._selector(unit, method=method,
                                                       rank=int(rank), fold=fold))

    def load_factors(self, *, unit, fold):
        selector = self._selector(unit, fold=int(fold))
        result = {"input_factors": self.load_field("input_factors", selector=selector)}
        if self.has_input_scaler(selector):
            result["input_scaler"] = self.load_input_scaler(selector)
        if self.has_field("target_factors", selector=selector):
            result["target_factors"] = self.load_field("target_factors", selector=selector)
        return result

    def save_full_metrics(self, *, metrics, unit=None, fold=None, overwrite=False):
        self.save_field("full_metrics", metrics,
                        selector=self._selector(unit, fold=fold), overwrite=overwrite)

    def save_summary(self, *, summary, unit=None, overwrite=False):
        frame = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)
        if "k" in frame and "rank" not in frame:
            frame = frame.rename(columns={"k": "rank"})
        method = unit.method if isinstance(unit, UnitSelector) else None
        if method and "method" not in frame:
            frame["method"] = method
        primary = "pca" if method == "pca" else "nr"
        if "corr" not in frame and f"{primary}_corr" in frame:
            frame["corr"] = frame[f"{primary}_corr"]
        if "corr" not in frame and f"{primary}_orth_corr" in frame:
            frame["corr"] = frame[f"{primary}_orth_corr"]
        if "r2" not in frame and f"{primary}_r2" in frame:
            frame["r2"] = frame[f"{primary}_r2"]
        if "r2" not in frame and f"{primary}_orth_r2" in frame:
            frame["r2"] = frame[f"{primary}_orth_r2"]
        self.save_mean_results(results=frame, unit=unit, overwrite=overwrite)

    def save_fold_results(self, *, results, unit=None, overwrite=False):
        self.save_field("fold_results", results, selector=self._selector(unit),
                        overwrite=overwrite)

    def load_fold_results(self, selectors=None):
        frames = []
        for selector, value in self.load_collection("fold_results", selectors).items():
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for key, item in dict(selector).items():
                if key not in frame:
                    frame[key] = item
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_summary(self, *, unit):
        return self.load_field("mean_results", selector=self._selector(unit))

    def save_target_manifest(self, *, target_manifest, unit=None, overwrite=False):
        self.save_field("target_manifest", target_manifest,
                        selector=self._selector(unit), overwrite=overwrite)

    def save_diagnostics(self, *, diagnostics, unit=None, rank=None, fold=None,
                         name=None, overwrite=False):
        self.save_field("diagnostics", diagnostics,
                        selector=self._selector(
                            unit, rank=rank, fold=fold, name=name
                        ),
                        overwrite=overwrite)

    def save_unit_metadata(self, *, metadata, unit=None, overwrite=False):
        self.save_field("unit_metadata", metadata, selector=self._selector(unit),
                        overwrite=overwrite)

    def resume_plan(self, *, unit, ranks, folds, require_target_factors=False,
                    require_input_factors=True, required_metric_keys=None, refresh=False):
        completed = set()
        for fold in folds:
            factor_selector = self._selector(unit, fold=int(fold))
            factors = (not require_input_factors or
                       self.has_field("input_factors", selector=factor_selector))
            if factors and require_input_factors:
                factors = self.has_input_scaler(factor_selector)
            targets = (not require_target_factors or
                       self.has_field("target_factors", selector=factor_selector))
            for rank in ranks:
                metric_selector = self._selector(unit, rank=int(rank), fold=int(fold))
                metrics_ok = self.has_field("rank_metrics", selector=metric_selector)
                if metrics_ok and required_metric_keys:
                    metrics_ok = set(required_metric_keys).issubset(
                        self.load_field("rank_metrics", selector=metric_selector)
                    )
                if not refresh and factors and targets and metrics_ok:
                    completed.add((int(rank), int(fold)))
        ranks_by_fold = {
            int(fold): [int(rank) for rank in ranks
                        if (int(rank), int(fold)) not in completed]
            for fold in folds
        }
        return {"completed_pairs": completed, "ranks_by_fold": ranks_by_fold,
                "folds_to_build": {fold for fold, pending in ranks_by_fold.items() if pending}}

    def resume_method_plan(self, methods, ranks, refresh=False):
        completed = set()
        for method in methods:
            for rank in ranks:
                selector = self._selector(method=method, rank=int(rank))
                if not refresh and self.has_field("rank_metrics", selector=selector):
                    completed.add((method, int(rank)))
        return {"completed": completed,
                "pending": [(method, int(rank)) for method in methods for rank in ranks
                            if (method, int(rank)) not in completed]}

    def load_artifacts(self, fields, selectors=None):
        selectors = selectors or {}
        scalar = {"effective_config", "upstream_identities", "metadata"}
        for name in fields:
            setattr(self, name, self.load_field(name) if name in scalar
                    else self.load_collection(name, selectors))
        return self

    def load_rank_matrix(self, method="rrr", *, metric=None, selectors=None,
                         data_root=None):
        """Load per-target rank metrics as a single array (see `RankMatrix`).

        Returns [rank, fold, electrode, lag] for iEEG or [rank, fold, voxel] for
        fMRI. `metric` overrides the default per-target metric (RRR:
        `nr_orth_corr_per_target`, PCA: `pca_corr_per_target`, fMRI:
        `corr_per_target`). Modes whose results span several target sets (for
        example leave-one-subject-out) need `selectors` that pin one of them, such as
        `{"held_out_subject": "3", "loso_phase": "transfer"}`.

        For iEEG, passing `data_root` adds `x`, `y`, `z` (mm) columns to `targets`
        via the dataset adapter's `get_electrode_coordinates` (NaN where an electrode
        has none). `data_root` is the dataset's own root, as in its configs
        (`data/` for podcast, `data/brain-treebank` for Brain Treebank).
        fMRI targets get no coordinates yet.
        """
        selectors = {"method": method, **(selectors or {})}
        manifests = self.load_collection("target_manifest", {})
        wanted = {k: str(v) for k, v in selectors.items()
                  if k in {"method", "loso_phase", "held_out_subject"}}
        manifest = next(
            (frame for encoded, frame in manifests.items()
             if all({k: str(v) for k, v in dict(encoded).items()}.get(k) == v
                    for k, v in wanted.items())),
            manifests.get(()),
        )
        if manifest is None:
            raise ValueError("no target manifest matches the requested selectors")
        matrix = assemble_rank_matrix(
            self.load_collection("rank_metrics", selectors), manifest, method, metric
        )
        if data_root is not None and matrix.lags is not None:
            targets = matrix.targets.copy()
            targets[["x", "y", "z"]] = get_dataset_adapter(
                _dataset_adapter_name(self.dataset)
            ).get_electrode_coordinates(targets, str(data_root))
            matrix = matrix._replace(targets=targets)
        return matrix

    def save_artifacts(self, *, effective_config, upstream_identities,
                       plot=True, overwrite=False):
        self.save_field("effective_config", effective_config, overwrite=overwrite)
        self.save_field("upstream_identities", upstream_identities, overwrite=overwrite)

    def save_mean_results(self, *, results, unit=None, overwrite=False):
        self.save_field("mean_results", results, selector=self._selector(unit),
                        overwrite=overwrite)

    def load_mean_results(self, selectors=None):
        frames = []
        for selector, value in self.load_collection("mean_results", selectors).items():
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for key, item in dict(selector).items():
                if key not in frame:
                    frame[key] = item
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_plot(self, *, name, figure, unit=None, overwrite=False):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        self.save_field("plots", buffer.getvalue(),
                        selector=self._selector(unit, name=name), overwrite=overwrite)


def run_config(config: Config):
    selected = config.selected_config()
    if config.modality == "fmri":
        from src.utils.rrr.rrr_fmri import run_fmri
        return [run_fmri(item) for item in expand_fmri_layer_configs(selected)]
    from src.utils.rrr.rrr_ieeg import run_ieeg
    return [run_ieeg(item) for item in expand_per_model_layer_configs(selected)]


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()

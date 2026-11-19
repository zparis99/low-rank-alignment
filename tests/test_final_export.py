"""Validate batch example selection before any figures are written."""

import pytest

from src.utils.plotting import final_data, final_export


@pytest.fixture(autouse=True)
def _intervals(monkeypatch):
    """Bootstrap intervals are computed once per data set (main and raw fMRI) and passed through."""
    monkeypatch.setattr(final_export, "bootstrap_intervals",
                        lambda root, explained_variance, **kwargs: ("intervals", kwargs))


def _data():
    quantiles = {(dataset, model, layer, method): None
                 for dataset in final_data.DATASETS
                 for model, layer in (("gpt-j-6b", 21), ("glove", None))
                 for method in ("rrr", "pca")}
    loso_quantiles = {(dataset, model, layer, subject, method): None
                      for dataset in final_data.DATASETS
                      for model, layer in (("gpt-j-6b", 21), ("glove", None))
                      for subject in (1, 2)
                      for method in ("RRR baseline", "RRR transfer", "PCA baseline")}
    explained_variance = final_data.ExplainedVarianceData(
        None, None, None, dict(quantiles), dict(loso_quantiles), [])
    return (None, None, None, quantiles, loso_quantiles, [], {}, None, [], explained_variance)


@pytest.mark.parametrize("arguments,internal,loso", [
    ([], ("gpt-j-6b", 21), ("gpt-j-6b", 21)),
    (["--example-model", "glove", "--example-layer", "none"],
     ("glove", None), ("glove", None)),
    (["--loso-example-model", "glove", "--loso-example-layer", "none"],
     ("gpt-j-6b", 21), ("glove", None)),
])
def test_example_selection(monkeypatch, tmp_path, arguments, internal, loso):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    calls = []
    monkeypatch.setattr(final_export, "export_figures", lambda *args, **kwargs: calls.append(args))
    final_export.main(["--output-dir", str(tmp_path), *arguments])
    assert calls == [(data, tmp_path, ["pdf", "png", "svg"], internal, loso,
                      "explained_variance", "shuffled", ("intervals", {}))]


def test_saturation_mode_argument(monkeypatch, tmp_path):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    calls = []
    monkeypatch.setattr(final_export, "export_figures", lambda *args, **kwargs: calls.append(args))
    final_export.main(["--output-dir", str(tmp_path), "--saturation-mode", "quantile"])
    assert calls[0][5] == "quantile"
    assert calls[0][7] is None  # the quantile appendix has no intervals


def test_all_exports_every_appendix(monkeypatch, tmp_path):
    data = _data()
    loads = []
    monkeypatch.setattr(final_data, "load_final_data",
                        lambda root, **kwargs: loads.append(kwargs) or data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    calls = []
    monkeypatch.setattr(final_export, "export_figures",
                        lambda *args, **kwargs: calls.append((*args, kwargs)) or [])
    final_export.main(["--output-dir", str(tmp_path), "--all"])
    assert [call[5] for call in calls] == ["explained_variance", "quantile", "raw_fmri"]
    assert calls[0][6] == "shuffled"
    assert calls[0][7] == ("intervals", {})
    # The raw-fMRI appendix reuses the single load and the single set of intervals.
    assert calls[2][0] is data
    assert calls[2][-1] == {"intervals": ("intervals", {}), "scope": "all"}
    assert loads == [{}]


@pytest.mark.parametrize("saturation_mode", ["explained_variance", "quantile"])
@pytest.mark.parametrize("index,key", [
    (3, ("fmri", "gpt-j-6b", 21, "pca")),
    (4, ("fmri", "gpt-j-6b", 21, 1, "RRR transfer")),
])
def test_incomplete_example_rejected_before_export(monkeypatch, tmp_path, capsys, index,
                                                   key, saturation_mode):
    data = _data()
    # Examples are validated against the curves of the selected saturation mode.
    tables = (data[index] if saturation_mode == "quantile"
              else data[9].curves if index == 3 else data[9].loso_curves)
    del tables[key]
    if index == 4:
        # LOSO eligibility requires at least one subject for each method.
        del tables[("fmri", "gpt-j-6b", 21, 2, "RRR transfer")]
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    output = tmp_path / "figures"
    with pytest.raises(SystemExit) as error:
        final_export.main(["--output-dir", str(output), "--saturation-mode", saturation_mode])
    assert error.value.code == 2
    assert "Available model/layer pairs: glove/none" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("mode", sorted(final_export.REMOTE_ONLY))
def test_scopes_partition_the_jobs(mode):
    names = ["rrr_internal_aggregate", *sorted(final_export.REMOTE_ONLY[mode]), "lag_saturation"]
    jobs = [(name, name) for name in names]
    local = {name for name, _ in final_export.select_jobs(jobs, mode, "local")}
    remote = {name for name, _ in final_export.select_jobs(jobs, mode, "remote")}
    assert local | remote == set(names) and not local & remote
    assert remote == final_export.REMOTE_ONLY[mode]
    assert final_export.select_jobs(jobs, mode, "all") == jobs
    with pytest.raises(ValueError):
        final_export.select_jobs(jobs, mode, "elsewhere")


def test_every_loso_figure_is_remote():
    # Anything drawn from leave-one-subject-out or decoding/shuffled results is remote-only.
    remote = final_export.REMOTE_ONLY["explained_variance"]
    assert {"rrr_loso_aggregate", "loso_saturation_sensitivity_fraction",
            "language_decoding_aggregate", "rrr_internal_aggregate_shuffled",
            "loso_principal_angles", "subspace_consistency"} <= remote
    assert not {"rrr_internal_aggregate", "rrr_cross_dataset_aggregate", "lag_saturation",
                "significant_targets", "roi_normalized_curves"} & remote


def test_local_scope_skips_remote_only_loads(monkeypatch, tmp_path):
    data = _data()
    # No LOSO example is needed locally, even when the local results lack it.
    del data[9].loso_curves[("fmri", "gpt-j-6b", 21, 1, "RRR transfer")]
    del data[9].loso_curves[("fmri", "gpt-j-6b", 21, 2, "RRR transfer")]
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data",
                        lambda root: pytest.fail("shuffled control is remote-only"))
    calls = []
    monkeypatch.setattr(final_export, "export_figures",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    final_export.main(["--output-dir", str(tmp_path), "--scope", "local"])
    assert calls[0][0][6] is None and calls[0][1] == {"scope": "local"}
    calls.clear()
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    with pytest.raises(SystemExit):  # the remote scope does need the LOSO example
        final_export.main(["--output-dir", str(tmp_path), "--scope", "remote"])


def test_scope_forwarded_to_every_appendix(monkeypatch, tmp_path):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    calls = []
    monkeypatch.setattr(final_export, "export_figures",
                        lambda *args, **kwargs: calls.append(kwargs) or [])
    final_export.main(["--output-dir", str(tmp_path), "--all", "--scope", "remote"])
    assert [call["scope"] for call in calls] == ["remote"] * 3


def test_subspace_only_skips_data_loading_and_writes_only_subspace_figures(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("final data must not be loaded")

    monkeypatch.setattr(final_data, "load_final_data", fail)
    monkeypatch.setattr(final_export, "export_figures", fail)
    names = []
    monkeypatch.setattr(final_export, "_subspace_jobs", lambda **kwargs: [
        (name, None) for name in (*final_export.SUBSPACE_ALIGNMENT_FIGURES, "roi_normalized_curves")])
    monkeypatch.setattr(final_export, "_save_jobs",
                        lambda jobs, output_dir, formats: names.extend(job[0] for job in jobs) or [])
    final_export.main(["--output-dir", str(tmp_path), "--subspace-only"])
    assert tuple(names) == final_export.SUBSPACE_ALIGNMENT_FIGURES

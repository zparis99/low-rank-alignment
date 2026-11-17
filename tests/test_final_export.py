"""Validate batch example selection before any figures are written."""

import pytest

from src.utils.plotting import final_data, final_export


def _data():
    quantiles = {(dataset, model, layer, method): None
                 for dataset in final_data.DATASETS
                 for model, layer in (("gpt2", 0), ("glove", None))
                 for method in ("rrr", "pca")}
    loso_quantiles = {(dataset, model, layer, subject, method): None
                      for dataset in final_data.DATASETS
                      for model, layer in (("gpt2", 0), ("glove", None))
                      for subject in (1, 2)
                      for method in ("RRR baseline", "RRR transfer", "PCA baseline")}
    explained_variance = final_data.ExplainedVarianceData(
        None, None, None, dict(quantiles), dict(loso_quantiles), [])
    return (None, None, None, quantiles, loso_quantiles, [], {}, None, [], explained_variance)


@pytest.mark.parametrize("arguments,internal,loso", [
    ([], ("gpt2", 0), ("gpt2", 0)),
    (["--example-model", "glove", "--example-layer", "none"],
     ("glove", None), ("glove", None)),
    (["--loso-example-model", "glove", "--loso-example-layer", "none"],
     ("gpt2", 0), ("glove", None)),
])
def test_example_selection(monkeypatch, tmp_path, arguments, internal, loso):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    calls = []
    monkeypatch.setattr(final_export, "export_figures", lambda *args: calls.append(args))
    final_export.main(["--output-dir", str(tmp_path), *arguments])
    assert calls == [(data, tmp_path, ["pdf", "png", "svg"], internal, loso,
                      "explained_variance", "shuffled")]


def test_saturation_mode_argument(monkeypatch, tmp_path):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    calls = []
    monkeypatch.setattr(final_export, "export_figures", lambda *args: calls.append(args))
    final_export.main(["--output-dir", str(tmp_path), "--saturation-mode", "quantile"])
    assert calls[0][5] == "quantile"


def test_all_exports_every_appendix(monkeypatch, tmp_path):
    data = _data()
    loads = []
    monkeypatch.setattr(final_data, "load_final_data",
                        lambda root, **kwargs: loads.append(kwargs) or data)
    monkeypatch.setattr(final_data, "load_shuffled_internal_data", lambda root: "shuffled")
    calls = []
    monkeypatch.setattr(final_export, "export_figures",
                        lambda *args: calls.append(args) or [])
    final_export.main(["--output-dir", str(tmp_path), "--all"])
    assert [call[5] for call in calls] == ["explained_variance", "quantile", "raw_fmri"]
    assert calls[0][6] == "shuffled"
    assert loads == [{}, {"fmri_noise_normalized": False}]


@pytest.mark.parametrize("saturation_mode", ["explained_variance", "quantile"])
@pytest.mark.parametrize("index,key", [
    (3, ("fmri", "gpt2", 0, "pca")),
    (4, ("fmri", "gpt2", 0, 1, "RRR transfer")),
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
        del tables[("fmri", "gpt2", 0, 2, "RRR transfer")]
    monkeypatch.setattr(final_data, "load_final_data", lambda root, **kwargs: data)
    output = tmp_path / "figures"
    with pytest.raises(SystemExit) as error:
        final_export.main(["--output-dir", str(output), "--saturation-mode", saturation_mode])
    assert error.value.code == 2
    assert "Available model/layer pairs: glove/none" in capsys.readouterr().err
    assert not output.exists()

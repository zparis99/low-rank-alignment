import numpy as np
import pandas as pd
import pytest

from scripts.run_rrr import assemble_rank_matrix


def _ieeg_manifest():
    rows = [(s, e, l, .1 * l) for s, e in ((1, "A"), (1, "B"), (2, "A")) for l in range(4)]
    return pd.DataFrame(rows, columns=["subject", "electrode", "lag_index", "lag"])


def test_ieeg_matrix_places_each_target_by_electrode_and_lag():
    manifest = _ieeg_manifest()
    metrics = {
        (("fold", fold), ("method", "rrr"), ("rank", rank)):
            {"nr_orth_corr_per_target": np.arange(12) + 100 * rank + 1000 * fold}
        for rank in (1, 4) for fold in (0, 1)
    }
    cube = assemble_rank_matrix(metrics, manifest, "rrr")
    assert cube.values.shape == (2, 2, 3, 4)
    assert list(cube.ranks) == [1, 4] and cube.folds == [0, 1]
    assert cube.targets.to_dict("records")[2] == {"subject": 2, "electrode": "A"}
    np.testing.assert_allclose(cube.lags, [0, .1, .2, .3])
    assert cube.values[1, 1, 2, 3] == 11 + 400 + 1000     # rank 4, fold 1, third electrode, last lag
    assert cube.values[0, 0, 1, 2] == 6 + 100


def test_fmri_matrix_has_unit_fold_axis_and_voxel_rows():
    manifest = pd.DataFrame({"subject": ["a", "a", "b"], "voxel_index": [0, 1, 0]})
    metrics = {(("method", "pca"), ("rank", r)): {"corr_per_target": np.full(3, float(r))}
               for r in (1, 2)}
    cube = assemble_rank_matrix(metrics, manifest, "pca")
    assert cube.values.shape == (2, 1, 3) and cube.folds == [None] and cube.lags is None
    assert list(cube.targets.columns) == ["subject", "voxel_index"]


def test_missing_combinations_are_nan_and_size_mismatch_is_rejected():
    manifest = _ieeg_manifest()
    metrics = {(("fold", 0), ("method", "rrr"), ("rank", 1)): {"nr_orth_corr_per_target": np.ones(12)},
               (("fold", 1), ("method", "rrr"), ("rank", 2)): {"nr_orth_corr_per_target": np.ones(12)}}
    cube = assemble_rank_matrix(metrics, manifest, "rrr")
    assert np.isnan(cube.values[0, 1]).all() and np.isnan(cube.values[1, 0]).all()
    with pytest.raises(ValueError, match="targets but the manifest"):
        assemble_rank_matrix({(("method", "rrr"), ("rank", 1)): {"nr_orth_corr_per_target": np.ones(5)}},
                             manifest, "rrr")

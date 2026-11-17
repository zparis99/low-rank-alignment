import numpy as np
import pandas as pd
import pytest

from scripts.run_rrr import Artifact, RankMatrix
from src.core import registry
from src.utils.atlas.core import (
    DESTRIEUX_2009_LABELS,
    bihemispheric_region_groups,
    label_voxel_positions_by_nearest_vertex,
    normalize_freesurfer_destrieux_name,
)


def test_bihemispheric_groups_mirror_left_labels_and_skip_lumped_right():
    groups = bihemispheric_region_groups()
    assert "RIGHT" not in groups
    assert {"L G_temporal_middle", "R G_temporal_middle"} == groups["MTG"]
    for labels in groups.values():
        assert labels <= set(DESTRIEUX_2009_LABELS)


def test_freesurfer_names_map_onto_atlas_labels():
    assert normalize_freesurfer_destrieux_name("G&S_paracentral", "L") == "L G_and_S_paracentral"
    assert normalize_freesurfer_destrieux_name("S_oc-temp_med&Lingual", "R") == (
        "R S_oc-temp_med_and_Lingual")
    assert normalize_freesurfer_destrieux_name("Unknown", "L") == ""
    assert normalize_freesurfer_destrieux_name("Medial_wall", "R") == "R Medial_wall"


def test_nearest_vertex_labels_include_hemisphere_and_unlabelled_vertices():
    surfaces = {"L": np.array([[-10.0, 0, 0], [-20.0, 0, 0]]),
                "R": np.array([[10.0, 0, 0], [20.0, 0, 0]])}
    names = ["Unknown", "G_temporal_middle"]
    annotations = {"L": (np.array([1, -1]), names), "R": (np.array([1, 0]), names)}
    positions = np.array([[-9.0, 0, 0], [-19.0, 0, 0], [11.0, 0, 0], [19.0, 0, 0]])
    labels = label_voxel_positions_by_nearest_vertex(positions, surfaces, annotations)
    assert list(labels) == ["L G_temporal_middle", "", "R G_temporal_middle", ""]
    with pytest.raises(ValueError, match="annotation has"):
        label_voxel_positions_by_nearest_vertex(
            positions, surfaces, {**annotations, "L": (np.array([1]), names)})


def test_load_rank_matrix_region_keeps_matching_targets(monkeypatch):
    class Adapter:
        def get_electrode_coordinates(self, targets, data_root):
            return np.zeros((len(targets), 3))

        def get_target_atlas_labels(self, targets, data_root):
            return np.array(["L G_temporal_middle", "R G_front_inf-Opercular", "",
                             "R G_temporal_middle"], dtype=object)

    monkeypatch.setattr(registry, "dataset_registry", {"fmri": Adapter()})
    monkeypatch.setattr("scripts.run_rrr.get_dataset_adapter", registry.get_dataset_adapter)
    values = np.arange(2 * 1 * 4, dtype=float).reshape(2, 1, 4)
    targets = pd.DataFrame({"subject": ["a"] * 4, "voxel_index": range(4)})
    matrix = RankMatrix(values, np.array([1, 2]), [None], targets, None)
    artifact = Artifact.__new__(Artifact)
    artifact.dataset = "deep-fmri"
    kept = artifact._restrict_to_region(matrix, "MTG", "root")
    assert list(kept.targets["voxel_index"]) == [0, 3]
    np.testing.assert_array_equal(kept.values[:, 0], values[:, 0, [0, 3]])
    with pytest.raises(ValueError, match="unknown region"):
        artifact._restrict_to_region(matrix, "nope", "root")
    with pytest.raises(ValueError, match="needs data_root"):
        artifact._restrict_to_region(matrix, "MTG", None)

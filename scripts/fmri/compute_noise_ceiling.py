#!/usr/bin/env python
"""Compute per-voxel closed-form (Schoppe et al. 2016) noise ceilings from the fMRI test-story repeats.

Reads ``individual_repeats`` from each subject's test-story response file and
writes one ``<subject>.npz`` (signal_power, noise_power, cc_max, all voxels) per subject.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from src.utils.shared.fmri import response_path, noise_ceiling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri-repo-root", default="data/deep-fMRI-dataset")
    parser.add_argument("--story", default="wheretheressmoke")
    parser.add_argument(
        "--subjects", nargs="+", default=[f"UTS{i:02d}" for i in range(1, 10)]
    )
    parser.add_argument("--output-dir", default="results/fmri_noise_ceiling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        path = response_path(args.fmri_repo_root, subject, args.story)
        with h5py.File(path, "r") as handle:
            repeats = handle["individual_repeats"][:]
        result = noise_ceiling(repeats)
        np.savez_compressed(out_dir / f"{subject}.npz", **result)
        ceiling = result["cc_max"]
        print(
            f"{subject}: {int(result['n_repeats'])} repeats, {ceiling.size} voxels, "
            f"{int(np.isnan(ceiling).sum())} NaN, "
            f"nanmedian CC_max {np.nanmedian(ceiling):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

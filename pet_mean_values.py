#!/usr/bin/env python
"""
Compute the IDIF value for each patient: the mean PET intensity inside the
descending-aorta VOI (<patient>_roi_des_aorta.nii.gz).

The ROI masks are read from final_masks/<patient>/. The matching PET image is
looked up in a PET directory that you provide (it is not part of this folder,
since PET lives with your dynamic data). For each patient it expects:

    <PET_DIR>/<patient>/pet.nii.gz          (default layout)

Override the PET filename with --pet-name if yours differs.

Usage:
    python pet_mean_values.py --pet-dir /path/to/pet_data
    python pet_mean_values.py --pet-dir /path/to/pet_data --pet-name dynamic.nii.gz \\
                              --out aorta_roi_pet_values.csv

For a dynamic (4-D) PET series, the mean is computed per frame and the output
CSV has one pet_mean column per frame (pet_mean_f0, pet_mean_f1, ...).
Frame timing is not stored; take the frame mid-times from your PET metadata.

The ROI is defined on the CT grid. The PET may have a different grid: each ROI
voxel is mapped through the image affines to its nearest PET voxel, so the
mean is volume-weighted over the CT-space ROI. CT and PET must be
co-registered (as on a PET/CT scanner).

Author : Camila Silva, University of Bern (April 2026)
Part of the master's thesis "Deep Learning-Based Whole-Body Tumor Segmentation: Comparison of Static and Parametric 18F-FDG PET/CT".
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-dir", default=os.path.join(SCRIPT_DIR, "final_masks"),
                    help="folder with <patient>/<patient>_roi_des_aorta.nii.gz")
    ap.add_argument("--pet-dir", required=True,
                    help="folder with <patient>/<pet-name> PET images")
    ap.add_argument("--pet-name", default="pet.nii.gz",
                    help="PET filename inside each patient folder")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "aorta_roi_pet_values.csv"))
    args = ap.parse_args()

    results = []

    for patient in sorted(os.listdir(args.roi_dir)):
        roi_path = os.path.join(args.roi_dir, patient, f"{patient}_roi_des_aorta.nii.gz")
        pet_path = os.path.join(args.pet_dir, patient, args.pet_name)

        if not os.path.exists(roi_path) or not os.path.exists(pet_path):
            print(f"Skipping {patient}")
            continue

        print(f"Processing {patient}")

        roi_nii = nib.load(roi_path)
        pet_nii = nib.load(pet_path)

        roi = roi_nii.get_fdata() > 0
        pet = pet_nii.get_fdata()

        dx, dy, dz = roi_nii.header.get_zooms()[:3]

        # map ROI voxels (CT grid) -> world (mm) -> nearest PET voxel
        ijk = np.array(np.where(roi))
        world = roi_nii.affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
        pet_ijk = np.rint(np.linalg.inv(pet_nii.affine) @ world)[:3].astype(int)
        inside = np.all((pet_ijk >= 0) & (pet_ijk < np.array(pet.shape[:3])[:, None]), axis=0)
        if not inside.all():
            print(f"  {int((~inside).sum())} ROI voxels fall outside the PET FOV")
        i, j, k = pet_ijk[:, inside]

        row = {"patient_id": patient}

        # PET mean(s): handle static (3-D) and dynamic (4-D) PET
        if pet.ndim == 4:
            for f in range(pet.shape[3]):
                vals = pet[i, j, k, f]
                row[f"pet_mean_f{f}"] = vals.mean() if vals.size else np.nan
        else:
            vals = pet[i, j, k]
            row["pet_mean"] = vals.mean() if vals.size else np.nan

        # Geometry of the VOI
        voxel_volume = dx * dy * dz
        row["volume_mm3"] = np.sum(roi) * voxel_volume

        coords = np.array(np.where(roi))
        if len(coords[0]) == 0:
            row["length_mm"] = 0
            row["diameter_mm"] = 0
        else:
            z_min, z_max = coords[2].min(), coords[2].max()
            row["length_mm"] = (z_max - z_min + 1) * dz

            z_center = (z_min + z_max) // 2
            coords_slice = np.array(np.where(roi[:, :, z_center]))
            if len(coords_slice[0]) > 0:
                x_min, x_max = coords_slice[0].min(), coords_slice[0].max()
                y_min, y_max = coords_slice[1].min(), coords_slice[1].max()
                row["diameter_mm"] = max((x_max - x_min) * dx, (y_max - y_min) * dy)
            else:
                row["diameter_mm"] = 0

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print(f"\nSaved -> {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()

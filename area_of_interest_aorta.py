#!/usr/bin/env python
"""
Define the cylindrical reference VOI (the IDIF sampling volume) inside the
descending-aorta mask.

  1. Erode the mask by a small margin to move safely inside the lumen
     (reduces partial-volume contamination near the vessel wall).
  2. Place the VOI centre at the cranio-caudal midpoint of the eroded mask.
  3. Over the central 20 mm, intersect a fixed 10 mm-diameter disk with the
     eroded lumen in each slice.

Geometry follows Hu et al. (PERCIST-derived) and matches the 10 mm luminal
cylinders used by Sari et al. (2022) for aortic IDIF extraction.

The mask is internally reoriented to RAS+ (z = cranio-caudal axis), so this
works for any input orientation; the VOI is saved on the original grid.

Usage:
    python area_of_interest_aorta.py <des_aorta.nii.gz> <out_roi.nii.gz>

Author : Camila Silva, University of Bern (April 2026)
Part of the master's thesis "Deep Learning-Based Whole-Body Tumor Segmentation: Comparison of Static and Parametric 18F-FDG PET/CT".
"""

import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion

EROSION_ITERS = 2          # in-plane erosion into the lumen (voxels)
TARGET_DIAMETER_MM = 10.0  # final fixed diameter
TARGET_LENGTH_MM = 20.0    # final fixed length


def extract_roi(seg_path, out_path):
    orig = nib.load(seg_path)
    nii = nib.as_closest_canonical(orig)
    back = nib.orientations.ornt_transform(
        nib.io_orientation(nii.affine), nib.io_orientation(orig.affine)
    )
    mask = nii.get_fdata() > 0
    affine = nii.affine
    dx, dy, dz = nii.header.get_zooms()[:3]

    # Small erosion to move safely inside the lumen
    eroded = mask.copy()
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    for _ in range(EROSION_ITERS):
        new = np.zeros_like(eroded)
        for z in range(eroded.shape[2]):
            if np.any(eroded[:, :, z]):
                new[:, :, z] = binary_erosion(eroded[:, :, z], structure=structure)
        eroded = new

    coords = np.array(np.where(eroded))
    if len(coords[0]) == 0:
        print("Erosion removed the mask, skipping")
        return False

    # geometric middle of eroded mask
    z_min, z_max = coords[2].min(), coords[2].max()
    z_center = (z_min + z_max) // 2

    half_len_vox = int(round((TARGET_LENGTH_MM / dz) / 2))
    z1 = max(0, z_center - half_len_vox)
    z2 = min(mask.shape[2], z_center + half_len_vox)

    # center in x/y from eroded middle segment
    seg_coords = np.array(np.where(eroded[:, :, z1:z2]))
    if len(seg_coords[0]) == 0:
        print("No voxels in middle segment after erosion, skipping")
        return False

    # seg_coords[2] is local to z1:z2, x/y are global
    x_center = int(round(seg_coords[0].mean()))
    y_center = int(round(seg_coords[1].mean()))

    # Build fixed 10 mm diameter disk in each slice over 20 mm length
    radius_mm = TARGET_DIAMETER_MM / 2.0

    x = np.arange(mask.shape[0])[:, None]
    y = np.arange(mask.shape[1])[None, :]
    dist2_mm = ((x - x_center) * dx) ** 2 + ((y - y_center) * dy) ** 2
    disk = dist2_mm <= radius_mm ** 2

    final = np.zeros_like(mask, dtype=bool)
    for z in range(z1, z2):
        # keep ROI inside the original aorta mask
        final[:, :, z] = disk & eroded[:, :, z]

    if not np.any(final):
        print("Final ROI is empty, skipping")
        return False

    # Save ROI
    out = nib.Nifti1Image(final.astype(np.uint8), affine).as_reoriented(back)
    nib.save(out, out_path)

    voxel_volume = dx * dy * dz
    volume_mm3 = final.sum() * voxel_volume

    coords_f = np.array(np.where(final))
    z_min_f, z_max_f = coords_f[2].min(), coords_f[2].max()
    length_mm = (z_max_f - z_min_f + 1) * dz

    z_mid_f = (z_min_f + z_max_f) // 2
    slice_mask = final[:, :, z_mid_f]
    coords_s = np.array(np.where(slice_mask))

    if len(coords_s[0]) > 0:
        x_min, x_max = coords_s[0].min(), coords_s[0].max()
        y_min, y_max = coords_s[1].min(), coords_s[1].max()
        diameter_x = (x_max - x_min) * dx
        diameter_y = (y_max - y_min) * dy
        diameter_mm = max(diameter_x, diameter_y)
    else:
        diameter_mm = 0.0

    print(
        f"Saved -> {out_path}\n"
        f"  center = ({x_center}, {y_center}, z {z_center})\n"
        f"  diameter ~= {diameter_mm:.2f} mm\n"
        f"  length   ~= {length_mm:.2f} mm\n"
        f"  volume   ~= {volume_mm3:.2f} mm^3"
    )
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ok = extract_roi(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)

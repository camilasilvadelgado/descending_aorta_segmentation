#!/usr/bin/env python
"""
Isolate the descending thoracic aorta from a TotalSegmentator multilabel map.

Input : a TotalSegmentator "--ml" segmentation containing at least the
        aorta, heart and thoracic vertebrae T4-T12.
Output: a binary mask of the descending aorta.

Steps (see README):
  1. Restrict the aorta to the cranio-caudal range of the T4-T12 vertebrae.
  2. Remove the arch / ascending aorta by cutting above the top of the heart.
  3. Remove the abdominal aorta by cutting below the top of T12.
  4. Keep only the posterior half (anterior-posterior split).
  5. Keep the largest connected component.

The segmentation is internally reoriented to RAS+ (voxel index increases
towards Right, Anterior, Superior), so the cuts are correct for any input
orientation. The output mask is saved on the original grid of the input.

Usage:
    python extract_descending_aorta.py <segmentations.nii.gz> <out_des_aorta.nii.gz>

Author : Camila Silva, University of Bern (April 2026)
Part of the master's thesis "Deep Learning-Based Whole-Body Tumor Segmentation: Comparison of Static and Parametric 18F-FDG PET/CT".
"""

import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label

# TotalSegmentator label indices (v2, "total" task)
AORTA = 52
HEART = 51
VERTS = [40, 39, 38, 37, 36, 35, 34, 33, 32]  # T4 ... T12
T12 = 32

HEART_MARGIN_SLICES = 3  # slices kept above the top of the heart (arch cut)


def to_ras(img):
    """Reorient to RAS+ and return (canonical image, transform back)."""
    canon = nib.as_closest_canonical(img)
    back = nib.orientations.ornt_transform(
        nib.io_orientation(canon.affine), nib.io_orientation(img.affine)
    )
    return canon, back


def extract_descending_aorta(seg_path, out_path):
    img = nib.load(seg_path)
    canon, back = to_ras(img)
    seg = np.asarray(canon.dataobj)

    aorta = (seg == AORTA)
    heart = (seg == HEART)
    verts = np.isin(seg, VERTS)

    print("Aorta voxels:", int(aorta.sum()))

    # 1. T4-T12 CROPPING
    z_idx = np.where(verts)[2]

    if len(z_idx) == 0:
        print("No vertebrae -> empty")
        final = np.zeros_like(aorta)
    else:
        z_min, z_max = int(z_idx.min()), int(z_idx.max())
        mask = np.zeros_like(aorta, dtype=bool)
        mask[:, :, z_min:z_max + 1] = True
        cand = aorta & mask

        print("After T4-T12 crop:", int(cand.sum()))

        # 2. REMOVE ARCH (TOP CUT)
        z_heart = np.where(heart)[2]

        if len(z_heart) > 0:
            z_cut = int(np.max(z_heart)) + HEART_MARGIN_SLICES
            cand[:, :, z_cut + 1:] = 0

        print("After arch removal:", int(cand.sum()))

        # 3. REMOVE ABDOMINAL (T12 CUT)
        t12 = (seg == T12)
        z_t12 = np.where(t12)[2]

        if len(z_t12) > 0:
            z_cut_low = int(np.max(z_t12))
            cand[:, :, :z_cut_low] = 0

        print("After T12 removal:", int(cand.sum()))

        # 4. POSTERIOR SPLIT (FOR ROBUSTNESS)
        coords = np.where(cand)
        if len(coords[0]) == 0:
            print("Nothing left after cropping -> empty")
            final = np.zeros_like(cand)
        else:
            y_mid = int(np.mean(coords[1]))

            # keep posterior half only (low y = posterior in RAS+)
            cand_post = np.zeros_like(cand, dtype=bool)
            cand_post[:, :y_mid, :] = cand[:, :y_mid, :]

            lab, num = label(cand_post)
            print("After posterior split, num components:", num)

            if num > 0:
                sizes = [(lab == i).sum() for i in range(1, num + 1)]
                chosen = np.argmax(sizes) + 1
                final = (lab == chosen)
            else:
                final = np.zeros_like(cand)

    print("Final voxels:", int(final.sum()))

    out = nib.Nifti1Image(final.astype(np.uint8), canon.affine).as_reoriented(back)
    nib.save(out, out_path)
    print(f"Saved -> {out_path}")
    return bool(final.any())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ok = extract_descending_aorta(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)

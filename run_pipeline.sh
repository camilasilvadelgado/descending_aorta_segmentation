#!/bin/bash
# =============================================================================
# Descending-aorta IDIF pipeline
# =============================================================================
# For every CT in CT_scans/ this script:
#   1. Runs TotalSegmentator (aorta, heart, vertebrae T4-T12)
#        -> intermediate_files/<patient>/<patient>_segmentations.nii.gz
#   2. Isolates the descending aorta
#        -> final_masks/<patient>/<patient>_des_aorta.nii.gz
#   3. Extracts the cylindrical IDIF VOI
#        -> final_masks/<patient>/<patient>_roi_des_aorta.nii.gz
#
# Local:  bash run_pipeline.sh
# SLURM:  mkdir -p logs && sbatch run_pipeline.sh   (submit from this folder)
#
# Author : Camila Silva, University of Bern (April 2026)
# Part of the master's thesis "Deep Learning-Based Whole-Body Tumor Segmentation: Comparison of Static and Parametric 18F-FDG PET/CT".
# =============================================================================

# ---- SLURM header: EDIT for your cluster (ignored when run with bash) -------
#SBATCH --job-name=totalseg_descending_aorta
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/totalseg_descending_aorta_%j.out
#SBATCH --error=logs/totalseg_descending_aorta_%j.err

set -uo pipefail

# Repo folder. Under SLURM the script is copied to a spool dir, so use the
# submit directory; locally, use the folder this script lives in.
# Override with: SCRIPT_DIR=/path/to/repo bash run_pipeline.sh
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    DEFAULT_DIR="$SLURM_SUBMIT_DIR"
else
    DEFAULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_DIR}"

CT_DIR="${CT_DIR:-$SCRIPT_DIR/CT_scans}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-$SCRIPT_DIR/intermediate_files}"
FINAL_DIR="${FINAL_DIR:-$SCRIPT_DIR/final_masks}"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$CT_DIR" "$INTERMEDIATE_DIR" "$FINAL_DIR" "$LOG_DIR"


# Config (override via environment)
DEVICE="${DEVICE:-gpu}"              # gpu | cpu | mps
MAX_CASES="${MAX_CASES:--1}"         # -1 = all cases
FORCE="${FORCE:-0}"                  # 1 = re-run even if outputs exist
CONDA_ENV="${CONDA_ENV:-totalseg_env}"

# Environment (cluster modules: EDIT module names for your cluster, or leave
# as is — failures are ignored)
if command -v module >/dev/null 2>&1; then
    module load Anaconda3 2>/dev/null || true
    module load CUDA 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

if ! command -v TotalSegmentator >/dev/null 2>&1; then
    echo "ERROR: TotalSegmentator not found. Activate the conda env first:"
    echo "       conda activate $CONDA_ENV"
    exit 1
fi

echo "============================================"
echo " DESCENDING AORTA PIPELINE"
echo " Repo     : $SCRIPT_DIR"
echo " CT input : $CT_DIR"
echo " Device   : $DEVICE"
echo "============================================"

count=0
failed=()

# Iterate over anything in CT_scans/: a .nii/.nii.gz file, or a folder
for ct in "$CT_DIR"/*; do

    [ -e "$ct" ] || { echo "No CT scans found in $CT_DIR"; break; }

    case "$(basename "$ct")" in .*) continue ;; esac   # skip .gitkeep, .DS_Store

    if [ "$MAX_CASES" -ne -1 ] && [ "$count" -ge "$MAX_CASES" ]; then
        break
    fi

    # Determine the patient id and the actual CT input to feed TotalSegmentator.
    # Supported layouts in CT_scans/:
    #   (a) CT_scans/<patient>.nii(.gz)          -> file, id = name w/o extension
    #   (b) CT_scans/<patient>/<something>.nii(.gz) -> folder holding NIfTI(s)
    #   (c) CT_scans/<patient>/ (DICOM series)   -> folder of .dcm slices
    base="$(basename "$ct")"
    if [ -d "$ct" ]; then
        patient="$base"
        # layout b: prefer a NIfTI whose name contains "ct" (case-insensitive)
        ct_input="$(find "$ct" -maxdepth 1 -type f -iname '*ct*' \( -name '*.nii.gz' -o -name '*.nii' \) | sort | head -n 1)"
        # else fall back to any NIfTI in the folder
        [ -z "$ct_input" ] && ct_input="$(find "$ct" -maxdepth 1 -type f \( -name '*.nii.gz' -o -name '*.nii' \) | sort | head -n 1)"
        # else treat the folder itself as a DICOM series (layout c)
        [ -z "$ct_input" ] && ct_input="$ct"
    else
        case "$base" in
            *.nii.gz|*.nii) ;;
            *) echo "Skipping $base (not .nii/.nii.gz or a folder)"; continue ;;
        esac
        patient="${base%.nii.gz}"
        patient="${patient%.nii}"
        ct_input="$ct"
    fi

    inter_out="$INTERMEDIATE_DIR/$patient"
    final_out="$FINAL_DIR/$patient"
    seg_file="$inter_out/${patient}_segmentations.nii.gz"
    des_file="$final_out/${patient}_des_aorta.nii.gz"
    roi_file="$final_out/${patient}_roi_des_aorta.nii.gz"

    echo ""
    echo ">>> Processing: $patient  (input: $ct_input)"

    mkdir -p "$inter_out" "$final_out"

    # ---- 1. TotalSegmentator ------------------------------------------------
    if [ "$FORCE" = "1" ] || [ ! -f "$seg_file" ]; then
        if ! TotalSegmentator \
            -i "$ct_input" \
            -o "$seg_file" \
            --roi_subset aorta heart \
                         vertebrae_T4 vertebrae_T5 vertebrae_T6 \
                         vertebrae_T7 vertebrae_T8 vertebrae_T9 \
                         vertebrae_T10 vertebrae_T11 vertebrae_T12 \
            --ml \
            --nr_thr_saving 1 \
            --device "$DEVICE"; then
            echo "    Segmentation failed, skipping $patient"
            failed+=("$patient"); continue
        fi
    else
        echo "    segmentations exist, reusing (set FORCE=1 to recompute)"
    fi

    # ---- 2. Isolate descending aorta ----------------------------------------
    if ! python "$SCRIPT_DIR/extract_descending_aorta.py" "$seg_file" "$des_file"; then
        echo "    Descending-aorta extraction failed, skipping $patient"
        failed+=("$patient"); continue
    fi

    # ---- 3. Cylindrical IDIF VOI --------------------------------------------
    if ! python "$SCRIPT_DIR/area_of_interest_aorta.py" "$des_file" "$roi_file"; then
        echo "    VOI extraction failed, skipping $patient"
        failed+=("$patient"); continue
    fi

    echo "    Done: $patient"
    count=$((count + 1))

done

echo ""
echo "===== DONE ($count case(s) succeeded, ${#failed[@]} failed) ====="
[ ${#failed[@]} -gt 0 ] && echo "Failed: ${failed[*]}"
exit 0

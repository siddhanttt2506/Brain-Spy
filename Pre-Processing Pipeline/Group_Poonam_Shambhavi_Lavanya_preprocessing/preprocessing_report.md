# MRI Data Preprocessing Pipeline for Alzheimer's Detection  
(Optimized for HD-BET Skull Stripping & Clinical Use)

This report outlines an MRI preprocessing workflow tailored for Alzheimer's disease detection.
---

## Step-by-Step Preprocessing Workflow

### 1. Loading & Initial Inspection

Load NIfTI files (.nii, .nii.gz) using libraries like nibabel.


- Scan dimensions (e.g., 256×256×192)
- Visualize slices (axial/sagittal/coronal) to detect motion artifacts or corruption

*Tools*: nilearn.plotting, *FSLeyes*

---

### 2. Intensity Normalization (Critical for Consistency)

*Options*:
- *Min-Max Scaling*: Rescale voxels to \[0, 1\] (simple but scanner-dependent)
- *Z-Score Normalization*: Normalize to mean = 0, std = 1 (better for multi-site data)

*Example Code*:
python
def normalize_volume(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume


---

### 3. Resampling to Standard Resolution

Ensures uniform input for CNNs (e.g., 1mm³ or 128×128×128 voxels)

*Tool*: ANTs, SimpleITK for high-quality interpolation

---

### 4. Skull Stripping (*HD-BET*)

We used HD-BET because it was giving more accurate results for skull stripping. (Skull stripping is used to remove non- brain tissues ex. fat, eyes etc.)

*Command*:
bash
hd-bet -i input.nii.gz -o output_skullstripped.nii.gz -mode accurate -device cuda


---

### 5. Bias Field Correction (N4 Algorithm)

*Fixes*: Inhomogeneity artifacts (bright/dark patches from coil distortions)

*Tool*: ANTs  
*Python Example*:
python
from ants import n4_bias_field_correction
corrected = n4_bias_field_correction(skullstripped_volume)


---

### 6. Registration to MNI152 Template

*Goal*: Align all brains to a standard space for cross-subject analysis.

*Options*:
- *Affine*: Fast, rigid — good for global alignment
- *Nonlinear (SyN)*: Higher accuracy — ideal for longitudinal studies

*Tool*: ANTs or FSL (FLIRT)

---

### 7. Denoising 

*Method*: Non-Local Means (NLM) — preserves edges while reducing noise

*Example*:
python
from dipy.denoise import nlmeans
denoised = nlmeans(corrected_volume, patch_radius=1, block_radius=2)


---

## Summary

This pipeline tries ensure MRI data is standardized, clean, and optimized for downstream deep learning models in Alzheimer's research. 

---

## References

- https://deepprep.readthedocs.io/en/24.1.0/processing.html
- https://www.researchgate.net/figure/Standard-MRI-preprocessing-steps-used-in-our-pipeline-Tools-from-the-FSL-43_fig2_373845892
# Brain-Spy Project: MRI Preprocessing Pipeline
## Group: JAR

## Introduction

This document details our group's approach to preprocessing the MRI dataset for Alzheimer's Disease prediction as part of the Brain-Spy project. Our goal was to develop a robust, reproducible pipeline that standardizes the data, reduces noise, and enhances relevant brain structures for downstream machine learning tasks.

### Problem Statement

MRI datasets contain significant variability and noise due to differences in scanners, patient movement, and non-brain tissues. These artifacts can mislead models and reduce prediction accuracy, especially in neuroimaging tasks like Alzheimer's Disease classification.

### Goals of Preprocessing

- Remove irrelevant structures (e.g., skull, scalp)
- Correct scanner-induced intensity artifacts
- Normalize intensity values across scans
- Augment data to improve model generalization

## Methodology

### Preprocessing Steps

1. **Skull Stripping (Brain Extraction)**
   - **Purpose:** Remove non-brain tissues to focus analysis on relevant neuroanatomical features and reduce computational overhead.
   - **Tool:** [hd-bet](https://github.com/MIC-DKFZ/HD-BET) library, which extracts the brain from input `.nii` files.
   - **Justification:** Non-brain tissues and dark regions (e.g., CSF) introduce noise and can lead models to learn irrelevant features.

2. **Bias Field Correction**
   - **Purpose:** Correct low-frequency intensity variations (bias fields) introduced by MRI scanners.
   - **Tool:** `N4BiasFieldCorrection` algorithm from the SimpleITK library.
   - **Justification:** Ensures the model learns genuine anatomical features, not scanner artifacts.
   - Note: We only looked into N4 bias field correction, as it is widely accepted in neuroimaging literature, however we did not implement it in the final pipeline due to time constraints.

3. **Normalization**
   - **Purpose:** Standardize intensity values across subjects and scanners.
   - **Technique:** Z-score normalization: \((x - \text{mean}) / \text{std}\).
   - **Justification:** Aids stable and efficient model training.

4. **Data Augmentation**
   - **Purpose:** Increase dataset diversity and robustness.
   - **Tool:** [TorchIO](https://torchio.readthedocs.io/), a medical imaging augmentation library for PyTorch.
   - **Techniques:**
     - RandomFlip: Rotational invariance
     - RandomAffine: Simulate scanner/positional variance
     - Elastic Deformation: Model anatomical variability
     - Noise & BiasField: Robustness to distortions
     - CropOrPad: Consistent input size for 3D CNNs

### Tools and Libraries Used

- **hd-bet** for skull stripping
- **SimpleITK** for bias correction
- **NumPy/Scikit-learn** for normalization
- **TorchIO** for augmentation
- **nibabel** for MRI file I/O
- **Matplotlib** for visualization

### Parameter Choices & Justification

- Skull stripping: Default hd-bet settings, as recommended in neuroimaging literature.
- Normalization: Z-score selected for its effectiveness in MRI preprocessing.
- Augmentation: Probability and magnitude of transforms chosen to balance realism and diversity.

## Implementation

### Code Structure

- `preprocess_pipeline.py`: Main pipeline script
- `outputs/`: Output folder for processed images and visualizations


### Dependencies

- Python 3.8+
- hd-bet
- TorchIO
- nibabel
- matplotlib
- numpy

## Results

### Visualizations

- **Before and After Skull Stripping:**  
  *(Insert images here)*
- **Bias Field Correction:**  
  *(Insert images here)*
- **Normalization:**  
  *(Insert images here)*
- **Augmentation Samples:**  
  *(Insert images here)*


### Challenges Faced

- **File Format Inconsistencies:** Some `.nii` files required conversion to `.nii.gz` before processing.
- **Long Processing Time:** Skull stripping was highly computationally intensive. It took around ~ 20 minutes per scan on average.
- **Improper outputs:** Some scans had artifacts that were not removed by hd-bet, either cutting too much of the brain or leaving non-brain tissues. We have yet to find a solution for this.

## Conclusion

### Summary

We implemented a standard MRI preprocessing pipeline, incorporating best practices from neuroimaging research. Each step—skull stripping, normalization, and augmentation—contributed to cleaner, more standardized data.

### Effectiveness

- **Strengths:** Robust removal of artifacts and irrelevant tissue, improved data quality for model training.
- **Limitations:** Processing time is significant; some edge cases (e.g., low-quality scans, poor hd-bet outputs) may still present challenges.

## References

- [hd-bet: Brain Extraction Tool](https://github.com/MIC-DKFZ/HD-BET)
- [SimpleITK N4BiasFieldCorrection](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)
- [TorchIO: Medical Image Augmentation](https://torchio.readthedocs.io/)
- [Neuroimaging Preprocessing Best Practices](https://www.frontiersin.org/articles/10.3389/fninf.2017.00037/full)


## Team Members
Team JAR:
- Joel John Mathew
- Ayush Shukla
- Reebal Faakhir


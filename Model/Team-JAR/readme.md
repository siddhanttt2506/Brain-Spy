# 🧠 ADNet — Alzheimer's Disease Classification from MRI Scans

We implemented a dual-stream convolutional neural network (CNN) for multi-label classification of Alzheimer's disease (AD) from structural MRI scans. The model distinguishes between **Cognitive Normal (CN)**, **Alzheimer's Disease (AD)** and **Mild Cognitive Impairment (MCI)** patients using (partial) MRI data from the ADNI dataset. The dual-stream architecture is designed to capture both fine and coarse neuroanatomical features relevant to AD diagnosis.

This repository presents **ADNet**, a dual-branch convolutional neural network for classifying brain MRI scans into two categories:

- **Alzheimer’s Disease (AD)**
- **Cognitively Normal (CN)**

The model processes 2D axial slices from 3D `.nii` MRI files using a carefully designed preprocessing pipeline and a custom CNN architecture.

---

## 📁 Dataset

- **Source**: ADNI (Alzheimer’s Disease Neuroimaging Initiative)
- **Format**: `.nii` or `.nii.gz` files
- **Structure**: Each file represents a single subject's preprocessed MRI volume
- **Metadata**: Labels (`CN` or `AD`) are obtained from the accompanying `ADNI1_Complete_1Yr_1.5T.csv` file based on the `Image Data ID` column

---

## 🔄 Preprocessing Pipeline

1. **Load `.nii` file** using `nibabel`
2. **Extract middle axial slice** (midpoint along the z-axis)
3. **Apply CLAHE** (Contrast Limited Adaptive Histogram Equalization) to improve local contrast
4. **Pad images** to a uniform size of `184 x 256`
5. **Normalize and reshape** to `(184, 256, 1)`
6. **Match labels** from the CSV file and encode:
   - `CN` → 0 (Cognitively Normal)
   - `MCI` → 1 (Mild Cognitive Impairment)
   - `AD` → 2 (Alzheimer’s)

All processed data is stored as NumPy arrays, ready to be used for training and validation.

---

## 🧠 Model: ADNet

ADNet is a two-branch CNN designed to extract rich spatial features at different receptive field sizes.

#### CNN1:
- Uses 3×3 convolutional kernels
- 2×2 max-pooling
- Outputs a 128-dimensional embedding
- Captures localized atrophy patterns (e.g., hippocampus)

#### CNN2:
- Uses 5×5 convolutional kernels
- 3×3 max-pooling
- Outputs another 128-dimensional embedding
- Detects global neurodegeneration patterns (e.g., ventricular enlargement, cortical thinning)

#### Final Classifier:
- **Fusion:** Concatenates the outputs of both CNN branches
- Flatten → Dense (128 units) → Dropout (0.5) → Dense (3, sigmoid activation)
- Outputs a single value (logit) for binary classification

---

## ⚙️ Training Details

- **Input shape**: `(184, 256, 1)` grayscale axial slice
- **Loss function**: `Binary Cross-Entropy with Logits`
- **Optimizer**: Adam
- **Learning rate**: 1e-4
- **Evaluation metrics**: Accuracy and Loss

The model is trained using PyTorch (and optionally TensorFlow) with GPU acceleration (CUDA/Tesla T4).

---

## ✅ Results & Usage

The model can be used for early detection of Alzheimer's Disease using non-invasive MRI scans. To use this:

1. Place preprocessed `.nii` files inside the `/pre_processed_files1/` directory
2. Ensure your CSV metadata is correctly linked
3. Run preprocessing → training → evaluation pipeline

The final trained model can be saved as:

- `.pt` (for PyTorch)
- `.keras` (if using TensorFlow port)

We have not currently worked on optimizing the model due to time constraints, however our baseline approach shows overfitting, with ~64% val accuracy and ~95% train accuracy.

---

## Team members
1. Ayush Shukla 
2. Joel John Mathew
3. Reebal Faakhir

---

## References
- [A novel CNN architecture for accurate early detection and classifcation of Alzheimer’s disease using MRI data](https://www.nature.com/articles/s41598-024-53733-6) (El-Assy et al., 2024)

- Future Directions:
   - Explore more advanced architectures (e.g., ResNet, DenseNet)
   - Implement 3D CNNs for volumetric data
   - [DEMNET: A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images](https://ieeexplore.ieee.org/document/9459692) 
   - [Deep Learning-Based Segmentation in Classification of Alzheimer’s Disease](https://link.springer.com/content/pdf/10.1007/s13369-020-05193-z.pdf)
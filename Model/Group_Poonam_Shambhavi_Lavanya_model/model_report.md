# MRI Slice-Based Alzheimer's Disease Classification using ResNet50

## Introduction

This project aims to classify MRI brain scans into three diagnostic categories — Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD) — using a deep learning approach. We focus on multi-class classification using preprocessed 3D MRI scans from the ADNI dataset. We used 2D axial/sagittal/coronal slices extracted from 3D MRI scans.

### Problem Statement

The goal is to build a multi-class classification model that accurately distinguishes between CN, MCI, and AD.

### Objectives

- Preprocess and augment brain MRI scans to prepare them for deep learning.
- Use entropy-based selection to choose the most informative slices.
- Train a ResNet50 model on the top-entropy slices for multi-class classification.
- Evaluate the model using classification metrics such as accuracy, precision, recall, and F1-score.

---

## Methodology

### Model Architecture

We use a pre-trained ResNet50 architecture with the following modifications:
- Freeze early convolutional layers (`layer1` and `layer2`).
- Fine-tune deeper layers (`layer3`, `layer4`, and `fc`).
- Replace the final fully connected layer with a Dropout + 3-way output layer.

### Training Methodology

- Dataset split: 80% training, 20% validation using stratified sampling.
- Loss function: CrossEntropyLoss with class weights to handle imbalance.
- Optimizer: AdamW with `lr=1e-4` and `weight_decay=1e-4`.
- Learning Rate Scheduler: ReduceLROnPlateau (based on validation accuracy).
- Early stopping after 5 epochs without improvement.

### Hyperparameters

| Parameter          | Value         |
|-------------------|---------------|
| Slice size        | 112x112       |
| Top-K slices      | 50            |
| Batch size        | 32            |
| Epochs            | 50 (with early stopping) |
| Patience          | 5             |

### Data Augmentation

- Random horizontal flip
- Random rotation (±15 degrees)
- Color jittering (brightness, contrast)
- Random affine transformations (small translations)

---

## Implementation

### Code Structure

```
├── filtered_mri_labels.csv         # ImageID and label mapping
├── BRAIN_SPY_MODEL.ipynb                      # Full preprocessing, training, and evaluation pipeline
├── best_resnet50_multiclass.pth   # Saved best model
```

### How to Train & Evaluate

1. Run the notebook end-to-end (Kaggle / Colab / Local).
2. Ensure preprocessed MRI slices are in NIfTI format inside `Preprocessed_ADNI_flat`.
3. Output: best model weights and classification report.

### Dependencies

```bash
pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0
pip install nibabel opencv-python scikit-image scikit-optimize torchvision
```

### Hardware Used

- GPU: NVIDIA Tesla P100 (Kaggle notebook environment)
- CUDA support: Yes

---

## Results

### Final Evaluation

```
              precision    recall  f1-score   support

          CN       0.98      0.98      0.98      1210
         MCI       0.99      1.00      0.99      1210
          AD       0.98      0.97      0.98      1210

    accuracy                           0.98      3630
   macro avg       0.98      0.98      0.98      3630
weighted avg       0.98      0.98      0.98      3630
```

### Confusion Matrix

|           | Pred: CN | Pred: MCI | Pred: AD |
|-----------|----------|-----------|----------|
| **CN**    |   1191   |    3     |   16      |
| **MCI**   |    3     |   1204    |   3      |
| **AD**    |    21    |    15     | 1174     |


---

## Conclusion

### Summary

- Our approach classifies MRI slices into CN, MCI, and AD with ~98.5% accuracy.



---

## References

- https://arxiv.org/pdf/1906.01160
- https://pubmed.ncbi.nlm.nih.gov/38725868/


---

## Team Members

- Poonam Gupta 
- Shambhavi Singh
- Lavanya Srivastava

# ADNI Brain MRI Classification using 3D CNN

A deep learning approach for classifying brain MRI scans from the ADNI dataset into three diagnostic categories using 3D Convolutional Neural Networks with residual connections and attention mechanisms.

## 📋 Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Team](#team)
- [References](#references)

## 🧠 Introduction

Our model classifies brain MRI scans from the ADNI dataset into one of three diagnostic categories:

- **CN** (Cognitively Normal)
- **MCI** (Mild Cognitive Impairment)
- **AD** (Alzheimer's Disease)

### Model Objectives

- Develop an effective multi-class classification model for brain MRIs
- Implement a reproducible preprocessing and modeling pipeline
- Leverage deep learning architectures (3D CNN + residual connections + attention mechanisms) to improve classification accuracy
- Ensure model evaluation using appropriate metrics for multi-class classification

## 🔬 Methodology

### Model Architecture

A 3D Convolutional Neural Network (CNN) with:

- **Residual Blocks**: To address vanishing gradient issues
- **Channel Attention Mechanisms**: Focus on important feature channels
- **Global Average + Max Pooling**: Extract spatially invariant features
- **Dense Layers with Dropout and L2 Regularization**: Control overfitting

**Input Shape**: (96, 96, 96, 1)  
**Output**: Softmax layer for 3-class classification

### Training Methodology

- **Preprocessing**: Advanced brain extraction, histogram-based thresholding, intensity normalization, anti-aliased resizing, and smoothing
- **Train-Validation Split**: 80:20
- **Class Weighting**: Handle class imbalance using `compute_class_weight`
- **Mixed Precision Training**: Improve speed and memory efficiency

### Hyperparameters and Justification

| Hyperparameter | Value | Reason |
|---|---|---|
| Input Shape | (96, 96, 96) | Balance between resolution and memory limits |
| Batch Size | 2 | Larger volumes per GPU memory constraints |
| Epochs | 50 | Enough iterations with early stopping |
| Optimizer | AdamW | Combines Adam benefits with weight decay regularization |
| Initial Learning Rate | 0.001 | Standard starting point |
| Learning Rate Scheduler | Reduce on plateau, 0.95 decay per epoch | Adaptive learning rate control |
| L2 Regularization | 0.001 | Control overfitting in dense and convolutional layers |
| Dropout | 0.1–0.5 | Prevent neuron co-adaptation |

### Data Augmentation Techniques

Applied the following augmentations to enhance model robustness and generalization:

- 3D Random Rotations
- Random Axis Flips
- Elastic Deformations
- Gaussian Noise
- Intensity Scaling
- Gamma Correction

## 🛠️ Implementation

### Code Structure

| Module | Purpose |
|---|---|
| `ADNIDataProcessor` class | Data loading, advanced preprocessing, file matching |
| `create_advanced_3d_cnn` | Model architecture with residual and attention blocks |
| `advanced_data_augmentation` | Custom 3D augmentations |
| `train_enhanced_model` | Training orchestration with callbacks and class weights |
| `comprehensive_evaluation` | Detailed classification and per-class metrics |

### Hardware Specifications

- **System**: Windows 10
- **CPU**: Intel i8
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX 4060

## 📊 Results

The results presently are evidently suboptimal, as the model is overfitting the training data. Due to the substantial dataset size, every minor change necessitates retraining the model, which takes hours. Therefore, extensive experimentation has been ongoing, but improvements are anticipated (a variation of the model is currently running in the background and will be updated upon completion of training).

## 💡 Conclusion

The project emphasizes the careful and patient approach required when training models on image datasets. Hyperparameters in the chosen model (ResNets in this case) need to be meticulously tuned to ensure both training and validation accuracy increase without the model overfitting.

## 🔧 Installation

### Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow nibabel matplotlib scipy
```

### Required Libraries

- numpy
- pandas
- scikit-learn
- tensorflow
- nibabel
- matplotlib
- scipy

## 🚀 Usage

### To Train

1. Data is loaded and preprocessed
2. Model is trained with callbacks (early stopping, LR scheduling, checkpointing)
3. Training history is plotted

### To Evaluate

The evaluation includes:

- Classification report
- Confusion matrix
- Per-class metrics (Precision, Recall, F1-score)
- Overall Accuracy and Multi-class AUC

## 👥 Team

- **Suhani Bansal**
- **Pratyush Sandhwar**
- **Aabha Jalan**

## 📚 References

1. [Nature Scientific Reports - Deep learning approach for brain MRI classification](https://www.nature.com/articles/s41598-024-53733-6)

2. [Frontiers in Bioengineering and Biotechnology - Machine Learning Applications in Neuroimaging](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.534592/full)

3. [Frontiers in Neuroscience - Deep Learning for Alzheimer's Disease Diagnosis](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00509/full)

---

*This project is part of ongoing research in medical image analysis and deep learning applications in neuroimaging.*
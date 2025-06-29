1. Introduction

We tried a deep learning-based pipeline for classifying subjects into Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), or Cognitive Normal (CN) categories using 3D brain MRI pre processed scans. Our objective is to evaluate the effectiveness of 3D ResNet architectures implemented with the MONAI framework. MONAI is an open-source deep learning framework specifically built for medical imaging tasks.

2. Methodology

Model Architecture:
We employ a 3D ResNet (ResNet-18 and ResNet-50 variants) provided by MONAI. The network accepts volumetric MRI inputs and outputs probabilities for three classes.

Training Methodology:
The model is trained on preprocessed MRI volumes labeled via metadata extracted from corresponding XML files. We use a weighted cross-entropy loss to account for class imbalance and the Adam optimizer with a learning rate of 1e-4.

Hyperparameters:

Batch size: 2 (very small coz of memory constraints) 

Epochs: 40

Learning rate: 0.0001

Optimizer: Adam


Data Augmentation:
We apply MONAI's 3D transforms such as random flipping, random intensity shifting, and random affine transformations to increase generalization.

3. Implementation

Code Structure:

dataset.py: handles XML parsing and dataset creation.

model.py: defines MONAI-based 3D ResNet.

train.py: contains training loop and evaluation.

Training & Evaluation:
Training is performed on 80% of the data, with 10% used for validation and 10% for testing. The model outputs are evaluated at each epoch for validation accuracy.

Hardware:

Kaggle Notebook with NVIDIA T4 GPU (16 GB)

4. Conclusion

The MONAI 3D ResNet-based pipeline effectively classifies Alzheimer's disease stages using structural MRI.


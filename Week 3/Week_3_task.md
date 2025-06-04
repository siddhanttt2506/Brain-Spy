# Week 3 Assignment

## Objective
The objective of this assignment is to build and evaluate multiple neural network models for image classification using the CIFAR-10 dataset. This includes both shallow and deep architectures, ranging from fully connected networks to modern convolutional networks. The goal is to gain practical experience in training, evaluating, and comparing deep learning models in a multi-class classification setting.

## Dataset
Use CIFAR-10 dataset from `torchvision.datasets`. Preprocess the data with appropriate transformations and organize it into training, validation, and test sets using `torch.utils.data.DataLoader`.

## Tasks

### 1. Data Preprocessing
- Load the CIFAR-10 dataset.
- Apply normalization and standard preprocessing steps.
- Apply data augmentation (e.g., flip, crop, rotate).
- Create train/validation/test data loaders.

### 2. Basic Models
- Implement a simple feedforward artificial neural network (ANN).
- Implement a basic convolutional neural network (CNN).
- Train and evaluate both models on CIFAR-10.
- Plot training and validation loss curves.

### 3. Classical CNN Architectures
- Use the following models:
  - LeNet
  - AlexNet
  - VGG16,VGG19
  - ResNet-50,ResNet-150
- Train each model (fine-tune if pretrained).
- Evaluate and compare performance on the test set.

### 4. Model Evaluation
- Evaluate each model using:
  - Accuracy
  - Precision, Recall, and F1-Score
- Generate and save:
  - Confusion matrices
  - ROC-AUC curves (using One-vs-All strategy)
  - Optional: Precision-Recall curves

### 5. Comparison and Discussion
- Compare models in terms of:
  - Parameter count
  - Training time
  - Accuracy and F1-Score
  - Generalization gap (train vs test)
- Discuss overfitting, underfitting, and generalization strategies used (dropout, regularization, data augmentation, etc.)

### 6. Challenges and Insights
- Describe any training difficulties encountered.
- Discuss strategies used to overcome challenges (e.g., learning rate schedulers, early stopping).
- Note any resource constraints or observations related to batch size, GPU usage, etc.

## Deliverables
Creat a folder in your Github repository named `week3_{Your_name_and_Roll_No}/` in Week 3 Folder(create in your repo) containing:
- `cifar_classification.ipynb`: Notebook with code, plots, and answers to all questions
- `report.md`: Written report with summary, comparison table, and performance screenshots. Remember to explain your approaches clearly and maybe give references to what we taught in class. Brownie points for that ;)


## Notes
- The deadline is 8th June 2025 EOD and work in groups of three(Note that everyone should submit separately,the members of the same group may submit the same file).
- You must train all the models end-to-end: one ANN, one basic CNN, and all the advanced CNNs (ResNet, AlexNet, LeNet, VGG16).
- Try to change the number of layers used in ANN and CNN and compare the accuracy.
- You may use `torchvision.models` for pretrained architectures.
- All code and analysis must be your own. Reference sources if using external material. Remember to not directly use codes available online or from ChatGPT. It's easy to figure out if you did.

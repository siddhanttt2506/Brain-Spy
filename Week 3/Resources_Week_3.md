# Week 3: Deep Learning Concepts for CIFAR-10 Image Classification

This document provides a comprehensive explanation of the deep learning concepts and techniques required to complete the CIFAR-10 classification assignment. It covers everything from basic neural networks to advanced convolutional architectures and evaluation metrics.

---

## 1. The CIFAR-10 Dataset

- **Dataset Description**: 60,000 color images (32x32), categorized into 10 classes (e.g., airplane, car, bird, cat).
- **Train/Test Split**: 50,000 training images, 10,000 testing images.
- **Challenges**: Small image size, limited resolution, high intra-class variation.

### Preprocessing Steps
- **Normalization**: Rescale pixel values to have zero mean and unit variance per channel.
- **Augmentation** (for generalization):
  - Random horizontal flips
  - Random cropping with padding
  - Rotations and color jitter (optional)

---

## 2. Neural Network Fundamentals

### Feedforward ANN
- **Structure**: Fully connected layers only.
- **Limitation**: Ignores spatial structure in images → poor performance on raw images.
- **Usage**: Acts as a baseline in this assignment.

### Convolutional Neural Networks (CNNs)
- **Why CNNs?**: Preserve spatial features by applying filters that detect edges, textures, and shapes.
- **Core Components**:
  - **Convolution Layers**: Apply kernel filters to input feature maps.
  - **Activation (ReLU)**: Adds non-linearity.
  - **Pooling Layers**: Downsample feature maps (MaxPooling commonly used).
  - **Fully Connected (Dense) Layers**: Flattened output for classification.
  - **Dropout/BatchNorm**: Regularization for better generalization.

---

## 3. Classical CNN Architectures

### LeNet-5
- One of the earliest CNNs, designed for digit recognition.
- Structure: Conv → Pool → Conv → Pool → FC → Output
- Shallow architecture, but useful for small image datasets.

### AlexNet
- Deeper and larger than LeNet.
- Introduced ReLU, dropout, and large kernels.
- Trained on ImageNet; significantly improved image classification.

### VGG16 / VGG19
- Deep networks using 3x3 conv kernels and 2x2 max pooling.
- VGG16: 16 weight layers; VGG19: 19 layers.
- High memory/computation requirements but very effective feature extractors.

### ResNet-50 / ResNet-101 / ResNet-152
- Introduces **residual connections** to allow very deep networks to converge.
- Enables training of models >100 layers deep without vanishing gradients.
- ResNet-50: 50 layers with skip connections (Identity and Projection blocks).

---

## 4. Transfer Learning (Optional for Pretrained Models)

- Pretrained models (AlexNet, VGG, ResNet) can be loaded from `torchvision.models`.
- **Fine-tuning**: Replace final classification layers and retrain on CIFAR-10.
- Freeze earlier layers if needed to reduce training time.

---

## 5. Evaluation Metrics

### Classification Metrics
- **Accuracy**: Ratio of correctly classified images to total images.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

### Confusion Matrix
- Matrix showing counts of true/false predictions per class.
- Helps visualize which classes are often misclassified.

### ROC-AUC (One-vs-All)
- For multi-class classification, compute ROC for each class vs all others.
- AUC indicates the model's ability to distinguish between classes.

---

## 6. Regularization and Generalization

### Common Strategies
- **Dropout**: Randomly deactivates neurons during training to prevent co-adaptation.
- **Data Augmentation**: Increases data diversity, helps reduce overfitting.
- **Weight Decay (L2 Regularization)**: Penalizes large weights.
- **Batch Normalization**: Stabilizes and accelerates training.
- **Early Stopping**: Stops training once validation loss stops improving.
- **Learning Rate Schedulers**: Adjust learning rate dynamically for better convergence.

---

## 7. Model Comparison Criteria

| Metric                | Description |
|------------------------|-------------|
| **Parameter Count**   | Total number of learnable weights and biases. |
| **Training Time**     | Time taken to train per epoch or total. |
| **Accuracy/F1**       | Performance metrics on test/validation set. |
| **Generalization Gap**| Difference between training and test accuracy. Lower is better. |

---

## 8. Practical Challenges

- **Overfitting**: Model memorizes training data, poor generalization. Use dropout, regularization, and data augmentation.
- **Underfitting**: Model too simple or not trained enough. Try deeper networks or more epochs.
- **Hardware Constraints**: Training large models like ResNet-50 can be GPU-intensive.
- **Hyperparameter Tuning**: Learning rate, batch size, and number of epochs significantly affect results.

---

## Tools and Libraries

- `torchvision.datasets.CIFAR10`: Loads CIFAR-10.
- `torch.utils.data.DataLoader`: For batching and shuffling.
- `torch.nn`, `torch.nn.functional`: Define models and layers.
- `torch.optim`: Optimizers like SGD, Adam.
- `sklearn.metrics`: For accuracy, precision, recall, F1, and confusion matrix.
- `matplotlib` or `seaborn`: For visualizing loss curves and metrics.

---

## Suggested Workflow

1. Load and preprocess CIFAR-10 data.
2. Implement a feedforward ANN.
3. Implement a basic CNN and compare with ANN.
4. Train pretrained models (VGG, AlexNet, ResNet).
5. Evaluate all models using defined metrics.
6. Plot training/validation loss and accuracy curves.
7. Create confusion matrix and ROC curves.
8. Analyze generalization gap and training behavior.
9. Write the summary in `report.md`.

---

*Use this guide as a conceptual backbone for building your models, evaluating performance, and writing your report. Stick to best practices and document all design decisions and observations clearly.*

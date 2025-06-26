# Model Architecture Review for AD vs CN Classfication

## Introduction

### Brief Overview of the Model Approach
We have tried classification of Brain MRI Scans using following models:
- **Transfer Learning Models** using pretrained **VGG19** and **ResNet152**
- A **3D Convolutional Neural Network (3D CNN)** tailored for volumetric brain scan data
- A **Spatial Graph Convolutional Neural Network (SGCNN)** leveraging spatial relationships within brain structures
- A **Double CNN architecture** combining dual feature extraction paths

For Transfer Learning Models and Double CNN architecture model, we used entropy based slicing to extract key slices of brain scan data as input.
### Problem Statement
This is a **binary classification** problem aimed at distinguishing between two groups:
- **AD (Alzheimer’s Disease)**
- **CN (Cognitively Normal)**

### Model Goals and Objectives
Our key goals include:

- **Accurate classification** of subjects into AD or CN based on neuroimaging features  
- **Evaluating model generalizability** across architectures to identify the most effective solution  
- **Minimizing overfitting**, especially given the relatively limited dataset size.
- **Comparing 2D and 3D strategies** to determine the effectiveness of volumetric data processing  
- **Computationally Cheaper** as we have limited access to resources
## Methodology
Pre-Processing pipeline included Bias-Field correction, Skull-Stripping, Spatial Normalization (MNI) and Intensity Normalization by using [NPP model](#ref3)[3].
### Transfer Learning Models:
### 3D CNN Model:
### SGCNN Model:
### Double CNN Architecture Model:
Double CNN architecture was referred from [this study](#ref1)[1]. It uses a 3 channel image consisting of middle slices of scans of all three temporal dimensions (axial, coronal, sagittal). However, this along with small size of dataset led to overfitting and bad results. To overcome this, we implemented [entropy based slicing](#ref2)[2] of brain scan as input images. We took top 30 images for each scan.
<br>
Regularization, Dropout, Early Stopping, and Reduce lr on Plateau were used to reduce overfitting of model. 
## Implementation
Dataset was processed using `/Aayush/processdataset.py` which uses [nppy library](#ref4)[4] based on [NPP model](#ref3)[3].
### Double CNN Architecture:
The Double CNN model was implemented in the notebook located at: `/Aayush/novel-entropy-model.ipynb`.
<br> **Dependencies:** `tensorflow`, `nibabel`, `cv2`, `numpy`, `pandas`, `matplotlib` <br>
**Hardware Specification:** Kaggle 2xT4 GPU <br>
## Results
### Model Performance Comparison

| Model                          | Accuracy (%) |
|-------------------------------|--------------|
| VGG19 (Transfer Learning)     | 93.49        |
| ResNet152 (Transfer Learning) | xx.xx        |
| 3D CNN                        | xx.xx        |
| Double CNN                    | 98.99        |
| SGCNN                         | xx.xx        |

### Double CNN Architecture:
Double CNN had 98.99% accuracy on test dataset after training for 20 epochs. However, the model is quite prone to overfitting.<br>

<img alt="Train, Val accuracy graph" height="300" src="./Aayush/novel-entropy-model.png" title="Double CNN Training, Validation Accuracy" width="400"/>

## Conclusion
## References
<span id="ref1">[1]</span> El-Assy, A.M., Amer, H.M., Ibrahim, H.M. et al. *A novel CNN architecture for accurate early detection and classification of Alzheimer’s disease using MRI data*. Scientific Reports, 14, 3463 (2024). [https://doi.org/10.1038/s41598-024-53733-6](https://doi.org/10.1038/s41598-024-53733-6)
<br>
<span id="ref2">[2]</span> Khan, N.M., Hon, M., & Abraham, N. (2019). *Transfer Learning with intelligent training data selection for prediction of Alzheimer's Disease*. arXiv:1906.01160 [cs.CV]. [https://doi.org/10.48550/arXiv.1906.01160](https://doi.org/10.48550/arXiv.1906.01160)
<br>
<span id="ref3">[3]</span> He, X., Wang, A.Q., Sabuncu, M.R. (2023). Neural Pre-processing: A Learning Framework for End-to-End Brain MRI Pre-processing. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14227. Springer, Cham. https://doi.org/10.1007/978-3-031-43993-3_25
<br>
<span id="ref4">[4]</span> Github Repository for nppy library https://github.com/AG3106/Neural_Pre_Processing
## Team Members
1. Dabeet Das
2. Ram Daftari
3. Aayush Gajeshwar
4. Aditya Goel
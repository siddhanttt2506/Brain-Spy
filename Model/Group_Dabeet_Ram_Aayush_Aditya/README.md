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
Pre-Processing pipeline included Bias-Field correction, Skull-Stripping, Spatial Normalization (MNI) and Intensity Normalization by using [NPP model](#ref3)[3]. <br>
Along with that, due to RAM constraints on the free tiers of Kaggle and Google Colab, we employed a method of 'entropy based selection' to select the top-k (k=100 in our case) slices from each MRI scan based on their relative importance. The model was finally trained using these slices only, they amounted to roughly 25,000 2D slices. Further, 128x128 crops are extracted from the center of each slice to be appended to the design matrix X. Integer labels are used. <br>
Also, `numpy` is used to ensure shape consistency with the pretrained models.  [Slice Selection using Entropy](#ref2)[2].

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
    <br> **Dependencies:** `tensorflow`, `nibabel`, `cv2`, `numpy`
## Results
## Conclusion
## References

<span id="ref1">[1]</span> El-Assy, A.M., Amer, H.M., Ibrahim, H.M. et al. *A novel CNN architecture for accurate early detection and classification of Alzheimer’s disease using MRI data*. Scientific Reports, 14, 3463 (2024). [https://doi.org/10.1038/s41598-024-53733-6](https://doi.org/10.1038/s41598-024-53733-6)
<br>
<span id="ref2">[2]</span> Khan, N.M., Hon, M., & Abraham, N. (2019). *Transfer Learning with intelligent training data selection for prediction of Alzheimer's Disease*. arXiv:1906.01160 [cs.CV]. [https://doi.org/10.48550/arXiv.1906.01160](https://doi.org/10.48550/arXiv.1906.01160)
## Team Members
1. Dabeet Das
2. Ram Daftari
3. Aayush Gajeshwar
4. Aditya Goel

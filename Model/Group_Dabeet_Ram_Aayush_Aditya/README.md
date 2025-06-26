# Model Architecture Review for AD vs CN Classfication

## Introduction

### Brief Overview of the Model Approach
We have tried classification of Brain MRI Scans using following models:
- **Transfer Learning Models** using pretrained **VGG19** and **ResNet152**
- A **3D Convolutional Neural Network (3D CNN)** tailored for volumetric brain scan data
- A **Spatial Graph Convolutional Neural Network (SGCNN)** leveraging spatial relationships within brain structures
- A **Double CNN architecture** combining dual feature extraction paths
- A **Attention-based eXplainability** using 2D CNNs to capture correlations and classify 3D MRIs

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

#### ResNet152
- **Model Architecture**:
  - A ResNet152 model pretrained on ImageNet is used as the base.
  - All but the last five layers are frozen to retain general features and the fully connected layers were excluded.
  - Three fully connected layers (each with 4096 units) are added, followed by a sigmoid output for binary classification.

- **Training Setup**:
  - Optimizer: Adam with default learning rate.
  - Loss: Binary Crossentropy.
  - Dataset split: 80% training / 20% testing, with 10% validation from the training set.
  - Epochs: 75, Batch size: 64.

- **Results:**
  - Training and validation accuracy curves indicate a stable and improving trend, however some overfitting is evident. We wish to test this model on the full dataset before commenting further.
  - The model was evaluated on a separate test set.
  - The final train accuracy plateaus at around ``0.996-0.997`` and above whereas the validation accuracy plateaus at roughly ``0.96``. The final test accuracy is ``0.9632``.

- **The implementation of this model can be found at the path:** `/Dabeet/preprocessed-brainspy-resnet152.ipynb`.

#### VGG19 with no preprocessing
- **Summary:**
  - VGG19 base model is used with all layers frozen along with the same 3 x 4096 units dense layer setup that was used above.
  - Point to be noted, no preprocessing was done on this model's training set, Adam optimiser with a learning rate of `1e-6` was used and the model was trained for 35 epochs.
  - A regular 80-20 train-test split was used along with 10% data from the training set being reserved for validation.
  - The final test accuracy for this model stood at `0.92`.
  - It is worth to note that the validation-train accuracy plot is much more unstable compared to the previous model.

- **Implementation can be found at:** `/Dabeet/brain-spy.ipynb`.

#### VGG19 with preprocessing
- **Summary:**
   -  This model is similar to the previous one but improves on it by utilising preprocessed data. Optimizer and learning rate and fully connected layer structure also remain the same.
   -  The model is now trained for ``100 epochs`` instead of 35 and is able to achieve a final test accuracy of ``0.9349``.
   -  The graph is indicative of slight overfitting but again, we would not like to comment on that before using the entire dataset.

- **Implementation can be found at:** ``Ayush/preprocessed_data_vgg19.ipynb``.

### 3D CNN Model:
### SGCNN Model:
### Double CNN Architecture Model:
### AXIAL:
This model aims to capture and use the relationship between multiple slices by using an Attention XAI Fusion Module and create a final representation of that particular MRI scan and then use it for classification. Basically, this model tries to use global features for an entire MRI scan rather than classify eaach 2D slice by tagging it with its parent label. 
This approach replicates this [paper](#ref5)[5]  with a few tweaks.

- **Feature Extraction Module:**
  - Input: $N$ individual 2D brain slices (e.g., axial MRI slices from a 3D volume). Here, $N$ is selected using the entropy based selection as mentioned above.
  - Process: Each slice is passed through a CNN backbone (ResNet152).
  - Output: A feature tensor of shape $(N,f_dim)$  — where each slice is represented by a vector of length $(f_dim,)$

- **Attention XAI Fusion Module:**
  - Purpose: Aggregate slice-wise features into a single subject-level representation with explainability.
  - Steps:
    - Each feature vector $(f_dim,)$ is passed through a small MLP to compute attention scores.
    - A softmax is applied over the scores.
    - Each feature is multiplied by its attention weight.
    - A weighted sum of all features produces a single vector
  - Output: A single fused feature vector representing the full subject's brain scan.

- **Diagnosis Module:**
  - Input: The fused feature vector from the attention module
  - Process: Fed into a fully connected MLP classifier.
  - Output: Labels corresponding to AD/CN.

- **Implementation:**
  - At the time of writing this, the code for this model is ready but I am unable to train it on Kaggle due to an issue where the GPU is not being detected for training. A CPU based training was forced on Kaggle and estimated training time of ``6 days`` is reported.
  - Latest: Kaggle terminal seems to have crashed, only 3 epochs were executed, I am attaching the code at ``/Dabeet/axial.ipynb`` and the last reported accuracy is ``.754`` after ``3 epochs``.
  - This model seems promising and I would like to try it out with proper resources, currently I am looking to train it locally on my CPU where the training time is estimated to be ~ ``12 hours``.

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
| ResNet152 (Transfer Learning) | 96.32        |
| 3D CNN                        | xx.xx        |
| Double CNN                    | 98.99        |
| SGCNN                         | xx.xx        |
| AXIAL                         | 75.40        |

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
<span id="ref5">[5]</span> Lozupone, G., Bria, A., Fontanella, F., Meijer, F. J. A., & De Stefano, C. (2024). *AXIAL: Attention-based eXplainability for Interpretable Alzheimer's Localized Diagnosis using 2D CNNs on 3D MRI brain scans.* arXiv preprint arXiv:2407.02418.
## Team Members
1. Dabeet Das
2. Ram Daftari
3. Aayush Gajeshwar
4. Aditya Goel

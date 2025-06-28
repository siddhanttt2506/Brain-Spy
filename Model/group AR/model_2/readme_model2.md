1. Cortical Thickness-Based Alzheimer's Classification
   
   As part of our study, we explored a machine learning approach for Alzheimer's disease (AD) classification based on region-wise cortical thickness measurements extracted    from pre-processed MRI data.

2. Motivation
   
   Alzheimer’s Disease is characterized by progressive loss of neurons, particularly in specific cortical and subcortical regions of the brain. This neuronal degeneration       leads to measurable atrophy, which manifests as reduced cortical thickness in affected regions. Detecting and quantifying this thinning is thus a biologically meaningful    way to distinguish between:
   AD (Alzheimer’s Disease),
   MCI (Mild Cognitive Impairment),
   CN (Cognitive Normal).

   What is Cortical Thickness?

      Cortical thickness refers to the distance between two anatomical surfaces:
      The white matter surface (boundary between white and gray matter)
      The pial surface (boundary between gray matter and cerebrospinal fluid)

3. Approach
   
   MRI scans were skull-stripped, bias-corrected, and segmented into gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF).
   Regional cortical thickness values were extracted using FreeSurfer(an external tool).

4. Modeling:
   
   Machine learning classifiers such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM) can be trained on these features.
   The goal is to predict the subject's condition: AD, MCI, or CN.

   We propose a hybrid approach that combines the strengths of deep learning with biologically-informed feature engineering:
   3D ResNet-based deep learning, which captures rich spatial patterns and local textures from full volumetric MRI data
   Biology-driven handcrafted features, such as region-wise cortical thickness


5. Limitations
   
   Relies on accurate segmentation: our segmentation wasn't that accurate and thats why there were abnormal values of cortical thickness

6. Resources

   https://www.sciencedirect.com/science/article/pii/S2352914819303764?via%3Dihub

7. Team

   AAKARSH AND RISHITESH


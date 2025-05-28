# MRI Data Visualization

## Objective:

The primary goal of this assignment is to develop an in-depth understanding of how to read, parse, and visualize medical imaging data stored in **NIfTI** (`.nii`) and **DICOM** (`.dcm`) formats using Python. You will explore libraries such as `nibabel`, `pydicom` and `matplotlib` to perform both low-level and high-level operations on 3D volumetric data.
Medical images are typically stored in specialized formats like **DICOM** and **NIfTI**, each designed for specific use-cases in clinical and research environments. These formats contain rich metadata and require specific techniques for visualization and analysis.

## Description

In this assignment, you will:

- Learn how to read and load medical images from disk.
- Extract and interpret metadata embedded in DICOM and NIfTI files.
- Visualize image volumes along standard anatomical planes: axial, sagittal, and coronal.
- Compare and contrast the structure, metadata, and use-cases of DICOM and NIfTI formats.
- Handle potential challenges such as orientation discrepancies and slice reordering.

## Tasks and Questions to Address in the Notebook

You are expected to answer the following questions in the form of a Jupyter notebook using appropriate code cells and markdown documentation.

### 1. How can we read and load NIfTI and DICOM files in Python?

- Use nibabel to load NIfTI files (.nii, .nii.gz).
- Use pydicom to read individual DICOM slices or entire series.
- Optional: use SimpleITK for a unified interface across both formats.

### 2. What is the internal structure and metadata of these formats?

- For **NIfTI**:
  - Explore the image shape, affine transformation matrix, voxel spacing, data type, and header fields.
- For **DICOM**:
  - Extract metadata like Patient Name, Study Date, Modality, Pixel Spacing, Slice Thickness, Image Position Patient, etc.

### 3. How do we stack DICOM slices into a 3D volume?

- Demonstrate how to:
  - Read a directory of DICOM slices.
  - Sort them using Image Position or Instance Number.
  - Stack into a 3D numpy array representing the volume.

### 4. How can we visualize anatomical planes from a 3D image volume?

- Generate static plots of:
  - Axial view (top-down)
  - Coronal view (front-back)
  - Sagittal view (side-to-side)
- Add slice sliders or interactive tools (e.g., ipywidgets) for better exploration (optional).

### 5. How do we interpret image orientation?

- Explain how to use the affine matrix (NIfTI) or orientation tags (DICOM) to understand how the image is oriented in 3D space.
- Compare orientation handling in DICOM vs NIfTI.

### 6. What are the key differences between DICOM and NIfTI?

- Compare based on:
  - Intended use (clinical vs research)
  - Metadata structure
  - File formats (single file vs multiple files per scan)
  - Ease of use and tooling in Python

## Deliverables

- DEADLINE: **31/05/2025**, **EOD**
- Create a folder in your GitHub repository with the following structure:
  repo_root
  ├──week2/
  ├── read_viz.ipynb # Main notebook with code
  ├── report.md # Detailed report
  ├── Sample_Data # Data Used

The `read_viz.ipynb` notebook must contain:

- Properly structured answers to all six questions listed above.
- Clean, well-commented code.
- Visualizations of slices with axis labels, titles, and colorbars.

The `report.md` must contain:

- A description of your approach.
- Screenshots of visualizations.
- Notes on any preprocessing or assumptions made.
- Observations and challenges encountered.

The sample MRI data in `.nii` format has been uploaded in the GitHub. You are expected to find some MRI data in `.dcm` format for your submission.

## Additional notes:

- This is a graded assignment and will be judged on the results. (Brownie points for good coding practices and well documented submissions)
- Avoid using pre-built visualizer softwares like ITK-SNAP or 3D Slicer; the focus is on coding your own tools in Python.
- You are encouraged to explore additional visualization techniques (e.g., 3D rendering, volume rendering using Plotly or ITK).
- Any kind of Plagiarism either among your peers or directly copy pasting from the internet would lead to a direct 'F' grade in the task

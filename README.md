# Automated Multimodal Experimentation for Co-Spacer Engineering and Phase Selection in Quasi-2D Halide Perovskites

This repository contains the code and data files used for automated multimodal experimentation, photoluminescence analysis, X-ray diffraction analysis, and Gaussian process-based composition selection for quasi-2D halide perovskites.

## Repository Contents

This repository includes:

- Python/Jupyter notebook files for data analysis
- XRD data files used in the workflow
- Iteration folders/data files from the closed-loop experimentation process: 5 iterations
- Scripts and notebooks related to phase selection and co-spacer engineering

## Main Workflow

The workflow includes:

1. Loading and processing photoluminescence data
2. Extracting phase-related features
3. Comparing compositions across experimental iterations
4. Using automated/closed-loop experimentation to guide composition selection
5. Loading and processing XRD data

## How to Use

The notebooks can be opened and run in Google Colab or Jupyter Notebook.

To run the analysis:

1. Download or clone this repository.
2. Open the relevant notebook.
3. Upload or locate the required XRD/PL data files.
4. Update the file paths in the notebook if needed.
5. Run the notebook cells sequentially.

## Requirements

The code uses common Python packages, including:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
glob
os

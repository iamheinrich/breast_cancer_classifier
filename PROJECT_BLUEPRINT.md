# Project Blueprint: Fairness Evaluation and Augmentation of Breast Cancer Classifiers

**Version:** 0.1
**Last Updated:** 2025-05-07

**Author:** Hendrik Schulze Bröring
**Supervisor(s):** Danny Panknin
**Associated Project:** MAIBAI (Objective 3)

## 1. Introduction & Goals

### 1.1. Master's Thesis Context

This Master's Thesis focuses on the post-processing fairness evaluation of deep learning models for breast cancer detection using digital mammography. It aims to identify biases in existing models and explore strategies for mitigating these biases through model retraining with fairness-aware methodologies.

### 1.2. Contribution to MAIBAI Project

This work directly contributes to **Objective 3** of the MAIBAI consortium project: "To use explainable and traceable AI tools for disease screening, providing the capability to train and retrain the tools as necessary. To benchmark the AI tools in terms of prediction performance, robustness, fairness and uncertainty quantification, under at least three scenarios, including low versus high image quality data, validation for specific patient demographics, presence of machine-based artefacts and noise sources. To develop and validate methods for the explainability and interpretability of the trained AI tools."
(Source: [MAIBAI Project Website](https://www.maibaiproject.eu/))

### 1.3. Core Research Questions

- How do existing breast cancer classification models (specifically the adapted NYU 2019 model and selected baselines) perform in terms of fairness across different demographic subgroups (age, poverty index, manufacturer) on the OPTIMAM dataset?
- Can fairness-augmented loss functions effectively improve the fairness of these models during retraining without significantly compromising overall predictive performance?
- What is a robust and reproducible workflow for training, evaluating, and iterating on medical imaging AI models with a focus on fairness?

## 2. Technical Setup & Infrastructure

### 2.1. Primary Development Environment

- **Local Machine:** macOS Sequoia 15.3.1
- **Shell:** /bin/zsh
- **Version Control:** Git

### 2.2. Computational Resources

- **Option A (Preferred for MAIBAI GitLab Integration):**
  - **Platform:** MAIBAI GitLab instance.
  - **Compute:** Google Cloud GPU resources accessible via MAIBAI GitLab.
  - **Storage:** Google Cloud Storage (GCS) for datasets and model artifacts.
- **Option B (Contingency/Alternative Dataset):**
  - **Platform:** University High-Performance Computing (HPC) cluster.
  - **Dataset:** Potentially VINDR-MAMO for initial development/testing if OMI-DB access on GCP is delayed.
- **Experiment Tracking:** Weights & Biases (W&B).

### 2.3. Key Software & Libraries

- **Programming Language:** Python 3.x (TBD)
- **Core ML Framework:** PyTorch
- **ML Workflow Management:** PyTorch Lightning
- **Data Handling:** Pandas, NumPy, H5Py, OpenCV-Python, Pillow
- **Cloud Interaction:** `google-cloud-storage` Python client
- **Fairness Toolkit:** Doleus (custom open-source package, pip installable)
- **Original Model Dependencies (from NYU `requirements.txt` - subject to upgrade):**
  - `torch>=0.4.1` (Target: Upgrade to modern PyTorch compatible with Lightning, e.g., 1.13+ or 2.x)
  - `torchvision>=0.2.0` (Target: Upgrade)
  - `scipy>=1.0.0`
  - `imageio>=2.4.1`
  - `tqdm>=4.19.8`
- **Environment Management:** Poetry (using `pyproject.toml` and `poetry.lock`)

## 3. Data: OPTIMAM Mammography Database

### 3.1. Overview

- **Source:** OPTIMAM Mammography Image Database ([OMI-DB](https://medphys.royalsurrey.nhs.uk/omidb/))
- **Characteristics:** Large-scale database of digital mammography images and associated clinical data.
- **Access:** Via MAIBAI Google Cloud Instance, potentially using the [OMI-DB Python client](https://scicomcore.bitbucket.io/omidb/).

### 3.2. Data Acquisition & Storage

- Data will be accessed/downloaded via the OMI-DB client and is stored on the MAIBAI Google Cloud Instance.
- Raw and processed data will reside in Google Cloud Storage, organized according to OMI-DB specifics.

### 3.3. Protected Attributes for Fairness Analysis

The dataset will be sliced and analyzed based on the following metadata attributes:

- **Age:** Patient age at time of screening.
- **Index of multiple deprivation:** Indicator of relative deprivatrion across the United Kingdom:[IMD](https://data.cdrc.ac.uk/dataset/index-multiple-deprivation-imd)
- **Manufacturer:** Manufacturer of the scanner with which the mammography was captured.

### 3.4. Data Preprocessing Pipeline (Adapting from NYU Classifier)

The following preprocessing steps will be performed (likely once upfront and stored):

1.  **DICOM to PNG/HDF5 Conversion:** If necessary (original NYU model includes a DICOM conversion utility).
2.  **Breast Cropping (`src/cropping/crop_mammogram.py`):**
    - Isolate breast tissue from background using image processing techniques (erosion, dilation, connected components).
    - Store cropped images.
    - Generate an `exam_list_cropped.pkl` containing paths and metadata for cropped images.
3.  **Optimal Center Extraction (`src/optimal_centers/get_optimal_centers.py`):**
    - Calculate `best_center` for each cropped image to guide consistent augmentation.
    - Store updated exam list (e.g., `data.pkl`) with this information.
4.  **(Optional) Heatmap Generation (`src/heatmaps/run_producer.py`):**
    - If image+heatmap models are pursued, generate heatmaps using a patch-level model.
    - Store heatmaps (e.g., as HDF5 files).

### 3.5. PyTorch Lightning `DataModule`

A `LightningDataModule` will be implemented to:

- Interface with the preprocessed data on GCS.
- Load images (and optional heatmaps).
- Apply transformations from `src/data_loading/loading.py` and `src/data_loading/augmentations.py`:
  - Image flipping (`flip_image`).
  - Cropping to fixed `input_size` based on `best_center` with location/size jitter for training (`random_augmentation_best_center`).
  - Instance-wise normalization (`standard_normalize_single_image`).
- Create `DataLoader` instances for training, validation, and testing.

## 4. Models

### 4.1. Primary Model: Adapted NYU Breast Cancer Classifier

- **Source:** [NYU Breast Cancer Classifier (Wu et al., 2019)](https://github.com/nyukat/breast_cancer_classifier)
- **Original Framework:** PyTorch 0.4.1
- **Target Framework:** PyTorch Lightning (after porting to modern PyTorch).
- **Key Architectural Components (from `src/modeling/models.py`, `src/modeling/layers.py`):**
  - `ImageBreastModel` / `SingleImageBreastModel` (focus on per-image predictions).
  - `FourViewResNet` using `resnet22` (`ViewResNetV2` with `BasicBlockV2`).
  - Custom layers: `OutputLayer`, `AllViewsGaussianNoise`, `AllViewsAvgPool`.
- **Prediction Granularity:** Model will be adapted/used to predict malignancy on a per-image basis to align with fairness evaluation requirements (not necessarily requiring all 4 views for a single prediction output used in fairness metrics).

### 4.2. Baseline Models

Approximately 2-3 baseline models will be implemented in PyTorch Lightning:

1.  **Basic CNN:** A simple Convolutional Neural Network.
2.  **Standard ResNet:** e.g., ResNet-18 or ResNet-34 from `torchvision.models`.
3.  **(Optional) Logistic Regression:** On top of features extracted by a pre-trained network or simpler image features if feasible.

### 4.3. Model (Re)Training

- All models will be trained (or fine-tuned) on the preprocessed OMI-DB dataset.
- Training will be managed by PyTorch Lightning, with experiment tracking via W&B.
- Loss functions (e.g., NLLLoss or CrossEntropyLoss) and optimizers (e.g., Adam, SGD) will be defined in the `LightningModule`.

## 5. Fairness Evaluation

### 5.1. Core Performance Metrics

- True Positive Rate (TPR / Sensitivity / Recall)
- False Positive Rate (FPR / 1 - Specificity)

### 5.2. Elaborate Fairness Metrics

Constructed based on TPR and FPR:

- **Difference/Ratio of Equalized Odds:** Comparing TPR and FPR across subgroups.
- **TPR@FPR=alpha:** TPR at a fixed FPR threshold (e.g., FPR=0.1).
- **FPR@TPR=beta:** FPR at a fixed TPR threshold (e.g., TPR=0.9).

### 5.3. Aggregate Fairness Functions

To quantify disparity across subgroups for a given fairness metric:

- **Standard Deviation (Std):** Std of the metric across subgroups.
- **Range:** Max difference of the metric across subgroups.
- **Min/Max Ratio:** Ratio of min to max metric value across subgroups.

### 5.4. Fairness Determination

- **Threshold-based:** Predefined acceptable thresholds for aggregate fairness functions.
- **Stack-ranking:** Comparing models based on their fairness scores.

### 5.5. Tooling: Doleus Framework

The custom-developed `Doleus` open-source package will be used to:

- Slice datasets by protected attributes.
- Calculate the defined fairness metrics.
- Apply aggregate functions.
- Facilitate model ranking or threshold testing.
- Requires model predictions (probabilities or scores) and ground truth labels as input.

## 6. Experiments: Fairness-Augmented Retraining

### 6.1. Hypothesis

Retraining models with a loss function that incorporates a fairness penalty can improve fairness metrics for underperforming subgroups without an unacceptable drop in overall performance.

### 6.2. Methodology

- Identify models/subgroups exhibiting significant bias from the initial evaluation.
- Design a fairness-augmented loss function. This could be, for example:
  - `Loss_Total = Loss_Performance + lambda * Loss_Fairness`
  - Where `Loss_Fairness` could penalize disparities in TPR/FPR between groups, or directly optimize for one of the aggregate fairness functions.
- Implement this custom loss within the PyTorch Lightning `training_step`.
- Retrain the selected model(s) with the new loss function.
- Re-evaluate performance and fairness using the Doleus framework.
- Analyze trade-offs between performance and fairness.

## 7. Project Structure (Git Repository)

\`\`\`
/breast-cancer-fairness-thesis
|
|-- PROJECT_BLUEPRINT.md # This document
|-- THESIS_LOG.md # Ongoing research log/diary
|
|-- data_preprocessing/ # Scripts for OMI-DB interaction, initial conversion (if any)
| |-- README.md
|
|-- notebooks/ # Jupyter notebooks for EDA, result visualization
|
|-- src/ # Main Python source code
| |-- datasets.py # LightningDataModule, PyTorch Datasets
| |-- models/ # Model definitions (adapted NYU, baselines)
| | |-- nyu_classifier.py
| | |-- baseline_cnn.py
| |-- training/
| | |-- train.py # Main Pytorch Lightning training script
| | |-- configs/ # Experiment configuration files (e.g., YAML with Hydra)
| |-- evaluation/
| | |-- evaluate.py # Script for running model evaluation
| | |-- doleus_wrapper.py # Scripts for interacting with Doleus
| |-- fairness_losses.py # Custom fairness-augmented loss functions
| |-- utils.py # Common utility functions
|
|-- pyproject.toml # Poetry project configuration and dependencies
|-- poetry.lock # Poetry lock file for reproducible environments
|-- Dockerfile # Optional: for containerizing the environment
|
|-- results/ # Output from experiments (CSV, plots - managed by W&B where possible)
| |-- experiment_configs/ # Copies of configs used for key results
|
|-- tests/ # Unit/integration tests (aspirational)
|
|-- original_nyu_classifier/ # Optional: Git submodule or copy of the original NYU code for reference
| |-- src/
| |-- ...
|
|-- README.md # Main project README with setup and run instructions
|-- .gitignore
\`\`\`

## 8. Timeline & Milestones (High-Level - 3 Months @ 20hrs/week)

- **Month 1: Setup, Data Pipeline, and NYU Model Porting & Initial Training**
  - Weeks 1-2: Detailed environment setup (local, GCP/GitLab), OMI-DB access, W&B setup. **Crucial: De-risk PyTorch 0.4.1 to modern PyTorch port for NYU model (`src/modeling/models.py`, `src/modeling/layers.py`).**
  - Weeks 3-4: Implement OMI-DB preprocessing (run existing scripts). Develop `LightningDataModule` for GCS data. Get ported NYU model training in Lightning on a subset of data.
- **Month 2: Baseline Models, Full Training, Initial Fairness Evaluation**
  - Weeks 5-6: Implement and train baseline models.
  - Weeks 7-8: Train all models on the full dataset. Integrate Doleus and perform initial fairness evaluation across all models and subgroups. Analyze initial results.
- **Month 3: Fairness-Augmented Retraining & Analysis**
  - Weeks 9-10: Design and implement fairness-augmented loss. Retrain key model(s).
  - Weeks 11-12: Evaluate retrained models. Analyze performance-fairness trade-offs. Finalize results.
- **Ongoing:** Thesis writing, documentation (`THESIS_LOG.md`).

## 9. Potential Risks & Mitigation

- **Risk: PyTorch Upgrade Complexity:** Porting NYU model from PyTorch 0.4.1 is difficult and time-consuming.
  - **Mitigation:** Strict time-box (1-2 weeks) for initial porting feasibility assessment. If major issues, consult supervisor to simplify model scope or extend timeline.
- **Risk: Data Access/Pipeline Delays:** Issues with OMI-DB access or GCS integration.
  - **Mitigation:** Early testing of data access. Have VINDR-MAMO as a potential fallback for initial development (though OMI-DB is primary).
- **Risk: Doleus Integration Challenges:** Unexpected issues making Doleus work with model outputs.
  - **Mitigation:** Early, simple tests with Doleus using dummy predictions.
- **Risk: Fairness Interventions Ineffective:** Fairness-augmented loss does not yield significant improvements or severely degrades performance.
  - **Mitigation:** This is a research outcome. Document findings thoroughly. Explore simpler post-processing fairness methods if time allows.
- **Risk: Limited Time:** 20 hours/week for 3 months is tight for the full scope.
  - **Mitigation:** Prioritize ruthlessly. Focus on the NYU model and one key fairness intervention. Document progress and any de-scoping decisions.

## 10. Communication & Reporting

- Regular updates in `THESIS_LOG.md`.
- Use of W&B for tracking experiment results.
- Regular meetings with supervisor(s).

---

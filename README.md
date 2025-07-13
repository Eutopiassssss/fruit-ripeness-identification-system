# Fruit Ripeness Identification System

This repository contains all source code, notebooks, and dataset instructions for the **Fruit Ripeness Identification System**. The system uses traditional machine learning algorithms (KNN, SVM, Random Forest) and image processing techniques to classify the ripeness stage of mangoes and papayas.

---

## Repository Structure

```plaintext
/
├── dataset/            # Dataset folders and download instructions
│   ├── mango_1/
│   ├── mango_2/
│   ├── papaya_1/
│   └── papaya_2/
└── sourcecode/         # Python scripts and notebooks
    ├── augmentation.py
    ├── feature_extraction.py
    ├── knn_classification.py
    ├── rf_classification.py
    └── svm_classification.py
```

---

## 1. Source Code

All code modules live in the `sourcecode/` directory. Brief descriptions:

* **augmentation.py**
  Defines image augmentation pipelines (rotations, flips, brightness, etc.) using Albumentations.

* **feature\_extraction.py**
  Extracts color (RGB, HIS, CIELab) and texture (GLCM, LBP) features from cropped image regions.

* **knn\_classification.py**
  Trains and evaluates a K-Nearest Neighbors classifier on extracted features; outputs classification reports and confusion matrices.

* **rf\_classification.py**
  Implements Random Forest training, cross-validation, and visualization of feature importance.

* **svm\_classification.py**
  Runs Support Vector Machine experiments with different kernels (linear, RBF, sigmoid), generates multi-class ROC curves and AUC scores.

### Installation and Dependencies

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fruit-ripeness-identification-system.git
   cd fruit-ripeness-identification-system/sourcecode
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Each script supports command-line arguments. Use `--help` to view usage, for example:

   ```bash
   python knn_classification.py --help
   ```

---

## 2. Datasets Overview

The `dataset/` directory contains instructions for four sub-datasets. Raw images are **not** stored here; follow the links to download and then process with the provided notebooks.

```plaintext
dataset/
├── mango_1/
├── mango_2/
├── papaya_1/
└── papaya_2/
```

### 2.1 mango\_1

* **Source**: Kaggle
* **Link**: [https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification](https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification)
* **Description**: Mango images labeled by ripeness (unripe, ripe, overripe).

### 2.2 mango\_2

* **Source**: Roboflow Universe
* **Link**: [https://universe.roboflow.com/lab-advance-140316/140-316-rpnps](https://universe.roboflow.com/lab-advance-140316/140-316-rpnps)
* **Description**: Background-cleaned mango images at multiple resolutions for texture-based analysis.

### 2.3 papaya\_1

* **Source**: Roboflow Universe
* **Link**: [https://universe.roboflow.com/papaya-ripeness-detection/papaya-ripeness-detection](https://universe.roboflow.com/papaya-ripeness-detection/papaya-ripeness-detection)
* **Description**: Papaya images with ripeness labels under various lighting and angles.

### 2.4 papaya\_2

* **Source**: Academic dataset (UNICAMP & UEL)
* **Reference**: Pereira et al. (2018). Predicting the ripening of papaya fruit with digital imaging and random forests. *Computers and Electronics in Agriculture*, 145, 76–82. DOI: 10.1016/j.compag.2017.12.029
* **Description**: 130 JPEG samples from 57 fruits, annotated into three stages (EM1, EM2, EM3).

---

## 3. How to Use

1. **Download raw images** into `raw_images/`:

   ```plaintext
   fruit-ripeness-identification-system/
   ├── raw_images/
   │   ├── mango_1/
   │   ├── mango_2/
   │   ├── papaya_1/
   │   └── papaya_2/
   ├── dataset/
   └── sourcecode/
   ```
2. **Install dependencies**:

   ```bash
   pip install -r sourcecode/requirements.txt
   ```
3. **Run preprocessing notebooks** in `dataset/` to clean and resize images.
4. **Execute classification scripts** in `sourcecode/`, passing paths to feature CSV and model parameters. Example:

   ```bash
   python sourcecode/knn_classification.py \
     --train_csv ../dataset/mango_1/features.csv \
     --test_csv ../dataset/mango_1/features.csv
   ```

---

*This README covers both code organization and dataset instructions to ensure reproducibility and easy extension of experiments.*

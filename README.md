# Fruit Ripeness Identification System - Source Code

This folder contains the source code for the Fruit Ripeness Identification System, developed for mango ripeness stage classification using both traditional machine learning and deep learning methods.

## 1. Introduction

This project provides scripts to preprocess mango images, extract features (such as RGB, HIS, CIELab, and texture features), and perform classification using KNN, SVM, Random Forest, and MLP models. The codebase also supports image augmentation, model evaluation, and visualization.

## 2. Prerequisites

- Python 3.8 or above
- pip

**Required Python packages:**

- numpy
- pandas
- scikit-learn
- opencv-python
- matplotlib
- scikit-image
- albumentations
- tqdm


## 3. Installation Steps

Clone this repository or download the source code。

Navigate to the Sourcecode directory:
You can install all required packages with:

```bash
pip install -r requirements.txt
```

## 4. Folder Structure

dataset/
├── mango_1/
│ ├── M1_Latest_64_bgcleaned_m1.ipynb
│ ├── M1_Latest_128_bgcleaned_m1.ipynb
│ └── M1_Latest_256_bgcleaned_m1.ipynb
├── mango_2/
│ ├── M2_Latest_64_bgcleaned_m2.ipynb
│ ├── M2_Latest_128_bgcleaned_m2.ipynb
│ └── M2_Latest_256_bgcleaned_m2.ipynb
├── papaya_1/
│ ├── P1_Latest_64_bgcleaned_p1.ipynb/
│ ├── P1_Latest_128_bgcleaned_p1.ipynb/
│ └── P1_Latest_256_bgcleaned_p1.ipynb/
└── papaya_2/
├── /P2_Latest_64_bgcleaned_p2.ipynb
├── /P2_Latest_128_bgcleaned_p2.ipynb
└── /P2_Latest_256_bgcleaned_p2.ipynb

sourcecode/
├── augmentation.py
├── feature_extraction.py
├── knn_classification.py
└── rf_classification.py
└── svm_classification.py



## 5. Installation Steps
1. Clone this repository or download the source code.

2. Navigate to the Sourcecode directory:

cd Sourcecode

4. Install all dependencies:

pip install -r requirements.txt

## 6. How to Run
Feature Extraction
python feature_extraction.py --input_dir <path_to_images> --output_csv <features.csv>

KNN Classification: 
python knn_classification.py --train_csv <train.csv> --test_csv <test.csv>

SVM Classification: 
python svm_classification.py --train_csv <train.csv> --test_csv <test.csv>

Random Forest Classification:
python rf_classification.py --train_csv <train.csv> --test_csv <test.csv>

Image Augmentation:
python augmentation.py --input_dir <path_to_images> --output_dir <augmented_images_dir>



# Dataset Overview

This folder organizes the datasets used for the Fruit Ripeness Identification System. It contains four subdirectories:

```
dataset/
├── mango_1/
├── mango_2/
├── papaya_1/
└── papaya_2/
```

---

## mango\_1

- **Source**: Kaggle
- **Link**: [https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification](https://www.kaggle.com/datasets/srabon00/mango-ripening-stage-classification)
- **Description**: This dataset includes mango images categorized by ripeness stage (e.g., unripe, ripe, overripe). It is used for color-based feature extraction and classification.

---

## mango\_2

- **Source**: Roboflow Universe
- **Link**: [https://universe.roboflow.com/lab-advance-140316/140-316-rpnps](https://universe.roboflow.com/lab-advance-140316/140-316-rpnps)
- **Description**: Provided by the Lab Advance team, this dataset contains background-cleaned mango images at multiple resolutions, suitable for texture analysis and machine learning pipelines.

---

## papaya\_1

- **Source**: Roboflow Universe
- **Link**: [https://universe.roboflow.com/papaya-ripeness-detection/papaya-ripeness-detection](https://universe.roboflow.com/papaya-ripeness-detection/papaya-ripeness-detection)
- **Description**: This dataset provides papaya images with labeled ripeness stages, captured under various lighting and angles. It is used to train supervised learning models for papaya ripeness detection.

---

## papaya\_2

- **Source**: Academic dataset from UNICAMP and UEL
- **Reference**: Pereira, L.F.S., Barbon Jr, S., Valous, N.A. & Barbin, D.F. (2018). Predicting the ripening of papaya fruit with digital imaging and random forests. *Computers and Electronics in Agriculture*, 145, 76–82. DOI: 10.1016/j.compag.2017.12.029
- **Description**: This JPEG dataset contains 130 samples from 57 papaya fruits, annotated into three maturity stages (EM1, EM2, EM3). Some fruits include multiple image captures.
- **Acknowledgements**:
  - Department of Food Engineering, University of Campinas (UNICAMP), Brazil
  - Department of Computer Science, Londrina State University (UEL), Brazil
- **BibTeX**:

```bibtex
@article{pereira2018predicting,
  title={Predicting the ripening of papaya fruit with digital imaging and random forests},
  author={Pereira, Luiz Fernando Santos and Barbon Jr, Sylvio and Valous, Nektarios A and Barbin, Douglas Fernandes},
  journal={Computers and Electronics in Agriculture},
  volume={145},
  pages={76--82},
  year={2018},
  publisher={Elsevier},
  doi={10.1016/j.compag.2017.12.029}
}
```

---

## How to Use

1. **Download raw images**

   - **mango\_1**: Download and unzip from the Kaggle link above.
   - **mango\_2** and **papaya\_1**: Download from Roboflow Universe.
   - **papaya\_2**: Download via the DOI or direct contact; unzip to `raw_images/papaya_2/`.

2. **Directory structure** after placing raw images:

   ```bash
   fruit-ripeness-identification-system/
   ├── raw_images/
   │   ├── mango_1/
   │   ├── mango_2/
   │   ├── papaya_1/
   │   └── papaya_2/
   ├── dataset/
   └── sourcecode/
   ```

3. **Run preprocessing notebooks**\
   Open the desired notebook in `dataset/mango_1/`, `dataset/mango_2/`, etc., and run all cells to generate cleaned and resized images.

4. **Dependencies**\
   See `sourcecode/requirements.txt` for the full list of Python libraries. Install with:

   ```bash
   pip install -r sourcecode/requirements.txt
   ```

---

This README provides a comprehensive overview of all dataset sources, descriptions, and usage instructions for easy replication and extension of the experiments.


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


##Installation Steps
Clone this repository or download the source code。

Navigate to the Sourcecode directory:
You can install all required packages with:

```bash
pip install -r requirements.txt

## 3. Folder Structure
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
│ ├── P1_Latest_64_bgcleaned_p1.ipynb
│ ├── P1_Latest_128_bgcleaned_p1.ipynb
│ └── P1_Latest_256_bgcleaned_p1.ipynb
└── papaya_2/
├── P2_Latest_64_bgcleaned_p2.ipynb
├── P2_Latest_128_bgcleaned_p2.ipynb
└── P2_Latest_256_bgcleaned_p2.ipynb

sourcecode/
├── augmentation.py
├── feature_extraction.py
├── knn_classification.py
└── rf_classification.py
└── svm_classification.py

## 4. Installation Steps
1. Clone this repository or download the source code.

2. Navigate to the Sourcecode directory:

cd Sourcecode

4. Install all dependencies:

pip install -r requirements.txt

## How to Run
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

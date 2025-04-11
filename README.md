# PR_HW1: Naïve Bayes & Logistic Regression Classifiers

This repository contains the implementation of two classical classification models — **Naïve Bayes** and **Logistic Regression** — as part of the homework assignment for the Pattern Recognition course (Spring 2025, NYCU CS).

## 🧠 Features

- Implementation from scratch using `numpy` 
- Supports **binary and multi-class** classification.
- Supports **Gaussian Naïve Bayes** with log-posterior decision.
- Supports **Logistic Regression** with softmax for multi-class.
- Evaluation using:
  - Accuracy
  - Confusion Matrix
  - ROC Curve & AUC (for binary classification)
- 5-Fold Stratified Cross-Validation
- Configurable via YAML files

## 📂 Datasets Used

All datasets are sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php):

- Breast Cancer Wisconsin (binary)
- Ionosphere (binary)
- Iris (multi-class, 3 classes)
- Wine (multi-class, 3 classes)

## 🚀 How to Run

```bash
# Run single train/test split (70/30)
python main.py --config configs/wine.yaml

# Run 5-fold cross-validation
python main.py --config configs/breast_cancer.yaml --cv

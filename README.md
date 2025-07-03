
# Breast Cancer Classification Models

This repository contains multiple machine learning models for classifying breast cancer tumors (benign vs malignant) using the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).

## 📂 Available Notebooks

- `Cancer_logistic_regression.ipynb`
- `Cancer_svm.ipynb`
- `Cancer_kernel_svm.ipynb`
- `Cancer_knn.ipynb`
- `Cancer_naive_bayes.ipynb`
- `Cancer_decision_tree.ipynb`
- `Cancer_random_forest.ipynb`

Each notebook implements and evaluates a different classification model using the same dataset.

---

## ⚠️ **Setup Instructions**

Before running any of the model notebooks:

1️⃣ **Upload your Kaggle API token file (`kaggle.json`) to your Colab environment.**

2️⃣ **Run this setup cell first in Colab:**
```python
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d uciml/breast-cancer-wisconsin-data
!unzip breast-cancer-wisconsin-data.zip

import pandas as pd
df = pd.read_csv("data.csv")
df.head()
```

✅ This code:
- Uploads your Kaggle API file.
- Downloads and unzips the dataset.
- Loads the dataset into a pandas DataFrame for immediate use.

---

## 📝 **Dataset Information**

- Source: [Kaggle Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- Features: 30 numeric features computed from digitized images of a breast mass.
- Target: `diagnosis` — Malignant (M) or Benign (B).

---

## 🚀 **How to Use**

- After completing the setup, open any of the provided model notebooks and run the code cells in order.
- Each notebook handles model training, evaluation, and prints performance metrics (accuracy, confusion matrix, etc).

---

## 💡 **Possible Enhancements**
If you'd like to improve or extend this project, consider:
- Adding ROC curves and AUC scores to evaluate classifier performance.
- Performing feature selection or dimensionality reduction (e.g., PCA).
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Comparing all models side-by-side in a summary notebook.

---

## 🖥️ **Requirements**
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional, for visualization)

These are usually pre-installed in Google Colab.

---

## 📌 **Notes**
- The code is designed for Google Colab. If running locally, ensure you have the dataset downloaded and accessible in your working directory.
- Replace `files.upload()` and related commands if you want to automate data loading locally.

---

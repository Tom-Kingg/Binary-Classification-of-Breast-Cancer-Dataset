#  Author
From Practise-ProjectX

Binary-Classification-of-Breast-Cancer-Dataset


UCI Breast Cancer Classification Pipeline This repository contains
a complete machine learning pipeline for binary classification on the UCI Breast Cancer Wisconsin (Diagnostic) Dataset. 
The project aims to build, train, and serialize a model to predict whether a tumor is malignant or benign based on various features.

## 📁 Files Description

- **data.csv** – Dataset used for training (UCI Breast Cancer).
- **train_pipeline.py** – Script to clean data, train models, tune hyperparameters, and save the best model and scaler.
- **model_evaluate.py** – Utility file for printing evaluation metrics.
- **predict.py** – CLI-based inference script that accepts input JSON and returns prediction.
- **best_model.pkl / scaler.pkl** – Serialized best model and feature scaler.
- **sample_input.json** – Sample format of input to be passed to `predict.py`.
- **requirements.txt** – List of required Python packages.

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt



### 2.Training the pipeline
python train_pipeline.py


### 3. Run prediction
python predict.py

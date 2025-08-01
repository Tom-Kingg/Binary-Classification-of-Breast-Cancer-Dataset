cat << EOF > report.md
# Breast Cancer Diagnosis Classification – ML Project Report

## 📌 Project Overview

This project is a part of the **NITSAN Practical Paper 1**. The goal is to build a complete machine learning pipeline for **binary classification** using the **UCI Breast Cancer Wisconsin Diagnostic Dataset**.

---

## 🔍 1. Data Loading & Exploratory Data Analysis (EDA)

- The dataset was loaded using the `ucimlrepo` package.
- There were no missing values found.
- Class distribution was **imbalanced**: malignant (M) and benign (B), with B being the majority class.
- Pearson correlation heatmaps and boxplots were used to identify relationships between features.

---

## 🧹 2. Preprocessing & Feature Engineering

- The `diagnosis` column was encoded: M → 1, B → 0.
- All numeric features were standardized using `StandardScaler`.
- A new feature `area_ratio = area1 / radius1` was engineered to provide an additional perspective on tissue density per radius unit.

---

## 🤖 3. Model Training & Comparison

Two classifiers were trained:
- **Logistic Regression**
- **Random Forest Classifier**

### Evaluation Metrics on Test Set:

| Metric        | Logistic Regression | Random Forest |
|---------------|---------------------|---------------|
| Accuracy      | ~97%                | ~98.25%       |
| Precision     | High                | Higher        |
| Recall        | High                | Higher        |
| ROC-AUC Score | Very High (~0.99)   | Highest (~0.995) |

✅ **Random Forest** outperformed Logistic Regression across all metrics.

---

## 🔧 4. Hyperparameter Tuning

- `GridSearchCV` was applied to **Random Forest** using 5-fold cross-validation.
- Parameters tuned: `n_estimators` and `max_depth`
- Best Params: `n_estimators = 150`, `max_depth = 5`
- Best CV Score: ~98.5%

---

## 💾 5. Model Serialization & Inference

- The best model was serialized using `joblib` as `best_model.pkl`.
- A separate `predict.py` script was written which:
  - Loads the saved model and scaler.
  - Accepts JSON input (via CLI or stdin).
  - Outputs prediction label and confidence probability.

---

## 📁 6. Project Structure
├── Breast_cancer_classification.ipynb
├── train_pipeline.py
├── model_utils.py
├── predict.py
├── sample_input.json
├── best_model.pkl
├── scaler.pkl
├── requirements.txt
├── run.batrt.md

---

## 🧠 7. Final Decision

**Random Forest Classifier** was chosen for deployment because:

- Superior performance on evaluation metrics.
- Robust to outliers and feature importance analysis.
- Better generalization on validation folds.

---

## ✍️ Author

From **Practise-ProjectX**
EOF


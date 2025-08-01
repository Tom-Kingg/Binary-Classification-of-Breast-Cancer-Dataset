# train_pipeline.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model_evaluate import evaluate_model

# 1. Load data
data = pd.read_csv("data.csv")

# Drop 'Unnamed: 32' and 'id' if present
data = data.drop(columns=['Unnamed: 32'], errors='ignore')
data = data.drop(columns=['id'], errors='ignore')

# 2. Convert diagnosis to binary
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# 3. Feature engineering - create area_ratio
data['area_ratio'] = data['area_worst'] / (data['area_mean'] + 1e-8)

# 4. Prepare features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# 8. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

# 9. Evaluate models
evaluate_model("Logistic Regression", y_test, log_pred, log_probs)
evaluate_model("Random Forest", y_test, rf_pred, rf_probs)

# 10. Hyperparameter tuning for RF
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
best_model.fit(X_train_scaled, y_train)
print("Best Parameters:", grid.best_params_)

# 11. Save model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

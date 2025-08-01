# model_evaluate.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"----{name}----")
    print(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision   : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall      : {recall_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC     : {roc_auc_score(y_true, y_proba):.4f}")
    print()

# cross_validation.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

def cross_validate_model(model_class, X, y, k=5, is_binary=True, verbose=True):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    aucs = []
    cms = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class()
        model.fit(X_train, y_train)
        y_pred, y_score = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        accuracies.append(acc)
        cms.append(cm)

        if is_binary:
            y_score = model.predict_proba(X_test)
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
            auc = roc_auc_score(y_test, y_score)
            aucs.append(auc)

        if verbose:
            print(f"[Fold {i+1}] Accuracy = {acc:.4f}")

    print(f"\n[Summary] Mean Accuracy = {np.mean(accuracies):.4f}")
    if is_binary:
        print(f"[Summary] Mean AUC = {np.mean(aucs):.4f}")

    # 混淆矩陣平均（非標準做法，但可供參考）
    cm_total = np.sum(cms, axis=0)
    return {
        "accuracies": accuracies,
        "aucs": aucs if is_binary else None,
        "confusion_matrix_sum": cm_total
    }

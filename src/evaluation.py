import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
import seaborn as sns

# 自動建立儲存資料夾
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def evaluate_model(y_true, y_pred, class_names=None, model_name="model", save_dir="outputs"):
    print("[INFO] Accuracy:", accuracy_score(y_true, y_pred))
    print("[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    ensure_dir(save_dir)
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")
    plt.close()

def plot_roc_curve(y_true, y_scores, model_name="model", save_dir="outputs"):
    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]  # 預設抓第1類的分數

    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        print("[WARNING] ROC curve requires binary classification. Skipping.")
        return

    pos_label = unique_labels[1]
    if np.sum(y_true == pos_label) == 0:
        print("[WARNING] No positive samples in y_true. Skipping ROC plot.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ensure_dir(save_dir)
    roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_path)
    print(f"[INFO] ROC curve saved to {roc_path}")
    plt.close()

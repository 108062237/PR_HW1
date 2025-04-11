import argparse
import numpy as np

from data_loader import load_config, load_dataset
from evaluation import evaluate_model, plot_roc_curve
from cross_validation import cross_validate_model
from naive_bayes import NaiveBayesClassifier
from logistic_regression import LogisticRegressionClassifier

def run_model(model_name, model, X_train, X_test, y_train, y_test, task_type):
    print(f"\n[MODEL] Running: {model_name}")
    model.fit(X_train.values, y_train.values)
    y_pred, y_score = model.predict(X_test.values)

    evaluate_model(y_test, y_pred, class_names=[str(c) for c in np.unique(y_test)], model_name=model_name)

    if task_type == "binary":
        plot_roc_curve(y_test.values, y_score[:, 1], model_name=model_name)

def run_cross_validation(model_name, model_class, X, y, task_type, k=5):
    print(f"\n[CV] {model_name} - {k}-Fold Cross Validation")
    results = cross_validate_model(
        model_class=model_class,
        X=X.values,
        y=y.values,
        k=k,
        is_binary=(task_type == "binary")
    )
    print(f"[CV] Mean Accuracy: {np.mean(results['accuracies']):.4f}")
    if task_type == "binary":
        print(f"[CV] Mean AUC: {np.mean(results['aucs']):.4f}")
    print(f"[CV] Cumulative Confusion Matrix:\n{results['confusion_matrix_sum']}")

def main(cfg, use_cv=False):
    X_train, X_test, y_train, y_test = load_dataset(cfg)

    print(f"[INFO] Dataset: {cfg['name']}")
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Train label distribution:\n{y_train.value_counts()}")
    print(f"[INFO] Test label distribution:\n{y_test.value_counts()}")

    models = {
        "Naive Bayes": NaiveBayesClassifier,
        "Logistic Regression": LogisticRegressionClassifier
    }

    for name, model_class in models.items():
        if use_cv:
            run_cross_validation(name, model_class, X_train, y_train, cfg["type"])
        else:
            model = model_class()
            run_model(name, model, X_train, X_test, y_train, y_test, cfg["type"])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/wine.yaml', help='Path to YAML config file')
    parser.add_argument('--cv', action='store_true', help='Enable cross-validation mode')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    cfg = load_config(args.config)
    main(cfg, use_cv=args.cv)

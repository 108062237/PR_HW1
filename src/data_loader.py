import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_dataset(cfg):
    data_obj = fetch_ucirepo(id=cfg["id"])
    print(data_obj.data.targets.columns)

    X = data_obj.data.features
    y = data_obj.data.targets[cfg["target_column"]]

    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=cfg["train_test_split_ratio"],
        random_state=cfg["random_seed"],
        stratify=y
    )

    return X_train, X_test, y_train, y_test
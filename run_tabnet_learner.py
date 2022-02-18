import argparse
import json
from os import path

import mlflow
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix

from utils import load_tabnet_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed CSV containing the visits data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory where preprocessing output must be saved')
    parser.add_argument('--tabnet_config_file', type=str, required=True,
                        help='Path to the config JSON file for xgboost')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    RUN_NAME = 'TabNeT model  spatial'
    with open(args.tabnet_config_file, 'r') as f:
        params = json.load(f)

    net = TabNetClassifier(**params)
    x_train, x_test, x_val, y_train, y_test, y_val = load_tabnet_data(args.data_path, random_seed=args.random_seed)
    net.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['auc', 'balanced_accuracy'],
        weights=1
    )
    net.save_model(path.join(args.output_dir, 'tabnet_model_spatial'))
    pred = net.predict(x_test)

    cm = confusion_matrix(y_test, (pred > 0.5).astype(int))
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metric("tn", t_n)
    mlflow.log_metric("tp", t_p)
    mlflow.log_metric("fp", f_p)
    mlflow.log_metric("fn", f_n)
    mlflow.log_metric("tpr", t_p / (t_p + f_n).astype(float))
    mlflow.log_metric("tnr", t_n / (t_n + f_p).astype(float))
    mlflow.log_metric("prec", t_p / (t_p + f_p).astype(float))
    mlflow.log_metric("NPV", t_n / (t_n + f_n).astype(float))
    mlflow.end_run()

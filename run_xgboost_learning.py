import argparse
import json

import mlflow
import xgboost as xgb
from sklearn.metrics import confusion_matrix

from utils import load_visits_data


def train_forests(dtrain, dtest, dval, params, num_round=100):
    evallist = [(dval, 'e'), (dtrain, 't')]
    mlflow.start_run(run_name='ring spatial forest')
    mlflow.xgboost.autolog()
    bst = xgb.train(params, dtrain, num_round, evallist)
    # bst.save_model('trained_models/xgboost_sp.model')

    pred = bst.predict(dtest)

    cm = confusion_matrix(dtest.get_label(), (pred > 0.5).astype(int))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed CSV containing the visits data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory where preprocessing output must be saved')
    parser.add_argument('--xgboost_config_file', type=str, required=True,
                        help='Path to the config JSON file for xgboost')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Verbosity')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.xgboost_config_file, 'r') as f:
        param = json.load(f)

    dtrain, dtest, dval = load_visits_data(args.data_path, random_seed=args.random_seed)
    train_forests(dtrain, dtest, dval, param)


import argparse
import os

import h2o
import numpy as np

from utils import load_h2o_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed CSV containing the visits data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory where preprocessing output must be saved')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Verbosity')
    args = parser.parse_args()
    np.random.seed(args.random_seed)

    h2o.init(max_mem_size='10G')
    train, test, valid = load_h2o_data(args.data_path, random_seed=args.random_seed)
    predictors = ['gen_fac', 'dm', 'starting_time', 'district', 'ring', 'dist_inf',
                  'cases', 'duration', 'num_inf_in_fac', 'crowdedness_fac']
    response = 'inf'
    verbosity = 'info' if args.verbose else None
    aml = h2o.automl.H2OAutoML(max_models=10, verbosity=verbosity, seed=args.random_seed, balance_classes=True, nfolds=0)
    aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    h2o.save_model(model=aml.get_best_model(), path=os.path.join(args.output_dir, 'h2o.model'), force=True)

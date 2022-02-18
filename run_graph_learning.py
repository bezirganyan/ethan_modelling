import argparse
import os

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from data.graph_dataset import MPNNDataModule
from models.graph_models import MPNNLSTM
from utils import read_meta_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=1500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=7,
                        help='Seperator for validation and train set.')
    parser.add_argument('--smooth_window', type=int, default=None,
                        help='Window to sooth out labels data. If not provided data will stay unchanged')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to labels csv file')
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Path to graphs directory.')
    parser.add_argument('--country', type=str, default='SI',
                        help='Country abbrevation (EN, IT, etc.)')
    parser.add_argument('--data_output_dir', type=str, default='output',
                        help='Directory where output must be saved')
    args = parser.parse_args()
    labels, gs_adj, features, y = read_meta_datasets(args.window, args.labels_path,
                                                     args.graph_dir, args.country,
                                                     smooth_window=args.smooth_window)
    n_samples = len(gs_adj)
    nfeat = features[0].shape[1]

    n_nodes = gs_adj[0].shape[0]
    res_path = os.path.join(args.data_output_dir, 'results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    results = []

    for shift in list(range(0, args.ahead)):
        result = []
        exp = 0
        for test_sample in range(args.start_exp, n_samples - shift):  #
            exp += 1
            dm = MPNNDataModule(args.window, test_sample, args.sep, shift, args.batch_size,
                                args.graph_window, labels, gs_adj, features, y)
            model = MPNNLSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes,
                             window=args.graph_window, dropout=args.dropout, lr=args.lr)

            trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", patience=args.early_stop)],
                              max_epochs=args.max_epochs,
                              log_every_n_steps=1)
            trainer.fit(model, dm)
            res = trainer.test(model, dm, ckpt_path='best')
            result.append(res[0]['test_error'])

        results.append(np.mean(result))

    print(results)
    print(np.mean(results))

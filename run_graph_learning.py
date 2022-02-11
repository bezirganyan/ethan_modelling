import argparse
import os
import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.graph_models import MPNN_LSTM
from utils import generate_new_batches, AverageMeter, read_meta_datasets
import streamlit as st


def train(adj, features, y):
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(adj, features, y):
    output = model(adj, features)
    loss_test = F.mse_loss(output, y)
    return output, loss_test


if __name__ == '__main__':
    st.title('Covid district cases prediction')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
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
    parser.add_argument('--recur', default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=2,
                        help='Seperator for validation and train set.')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to labels csv file')
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Path to graphs directory.')
    parser.add_argument('--country', type=str, default='SI',
                        help='Country abbrevation (EN, IT, etc.)')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    labels, gs_adj, features, y = read_meta_datasets(args.window, args.labels_path, args.graph_dir, args.country)

    n_samples = len(gs_adj)
    nfeat = features[0].shape[1]

    n_nodes = gs_adj[0].shape[0]
    print(n_nodes)
    if not os.path.exists('results'):
        os.makedirs('results')
    result_lines = []

    for shift in list(range(0, args.ahead)):
        result = []
        exp = 0
        for test_sample in range(args.start_exp, n_samples - shift):  #
            exp += 1
            print(test_sample)

            idx_train = list(range(args.window - 1, test_sample - args.sep))

            idx_val = list(range(test_sample - args.sep, test_sample, 2))

            idx_train = idx_train + list(range(test_sample - args.sep + 1, test_sample, 2))

            adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train,
                                                                      args.graph_window, shift,
                                                                      args.batch_size, device, test_sample)
            adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val,
                                                                args.graph_window, shift, args.batch_size,
                                                                device, test_sample)
            adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [test_sample],
                                                                   args.graph_window, shift,
                                                                   args.batch_size, device, test_sample)

            n_train_batches = ceil(len(idx_train) / args.batch_size)
            n_val_batches = 1
            n_test_batches = 1

            stop = False
            while not stop:

                model = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes,
                                  window=args.graph_window, dropout=args.dropout).to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                # ------------------- Train
                best_val_acc = 1e8
                val_among_epochs = []
                train_among_epochs = []
                stop = False

                for epoch in range(args.epochs):
                    start = time.time()

                    model.train()
                    train_loss = AverageMeter()

                    # Train for one epoch
                    for batch in range(n_train_batches):
                        output, loss = train(adj_train[batch], features_train[batch], y_train[batch])
                        train_loss.update(loss.data.item(), output.size(0))

                    model.eval()
                    output, val_loss = test(adj_val[0], features_val[0], y_val[0])
                    val_loss = float(val_loss.detach().cpu().numpy())

                    if (epoch % 50 == 0):
                        print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
                              "val_loss=", "{:.5f}".format(val_loss), "time=",
                              "{:.5f}".format(time.time() - start))

                    train_among_epochs.append(train_loss.avg)
                    val_among_epochs.append(val_loss)

                    if epoch < 30 and epoch > 10:
                        if len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1:
                            # stuck= True
                            stop = False
                            break

                    if epoch > args.early_stop:
                        if len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1:  #
                            print("break")
                            # stop = True
                            break

                    stop = True

                    # --------- Remember best accuracy and save checkpoint
                    if val_loss < best_val_acc:
                        best_val_acc = val_loss
                        torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, 'model_best.pth.tar')

                    scheduler.step(val_loss)

            print("validation")
            test_loss = AverageMeter()

            # print("Loading checkpoint!")
            checkpoint = torch.load('model_best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.eval()

            output, loss = test(adj_test[0], features_test[0], y_test[0])
            o = output.cpu().detach().numpy()
            l = y_test[0].cpu().numpy()

            error = np.sum(abs(o - l)) / n_nodes
            print("test error=", "{:.5f}".format(error))
            result.append(error)

        print("{:.5f}".format(np.mean(result)) + ",{:.5f}".format(np.std(result)) + ",{:.5f}".format(
            np.sum(labels.iloc[:, args.start_exp:test_sample].mean(1))))

        result_lines.append("MPNN_LSTM," + str(shift) + ",{:.5f}".format(np.mean(result)) + ",{:.5f}".format(
            np.std(result)) + "\n")

    with open(f'results/{args.country}_results_orig.csv', 'a+') as f:
        f.writelines(result_lines)

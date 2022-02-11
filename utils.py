from os import path

import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from datetime import timedelta, datetime


def read_meta_datasets(window, labels_path, graphs_dir, country):
    labels = pd.read_csv(labels_path)
    labels = labels.set_index("name")

    sdate = datetime.strptime(labels.columns[0], "%Y-%m-%d").date()
    edate = datetime.strptime(labels.columns[-1], "%Y-%m-%d").date()
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days + 1)]
    dates = [str(date) for date in dates]
    Gs = generate_graphs_tmp(dates, graphs_dir, country)

    labels = labels.loc[list(Gs[0].nodes()), :]
    labels = labels.loc[:, dates]
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    features = generate_new_features(Gs, labels, dates, window)

    y = list()
    nodes_without_labels = set()
    for i, G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node, dates[i]])

    return labels, gs_adj, features, y


def generate_graphs_tmp(dates, graphs_dir, country):
    Gs = []
    for date in dates:
        d = pd.read_csv(path.join(graphs_dir, country + "_" + date + ".csv"), header=None)
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)

    return Gs


def generate_new_features(Gs, labels, dates, window=7, scaled=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()

    labs = labels.copy()
    nodes = Gs[0].nodes()
    for idx, G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region

        H = np.zeros([G.number_of_nodes(), window])  # +3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1) + 1

        for i, node in enumerate(G.nodes()):
            # ---- Past cases
            if (idx < window):  # idx-1 goes before the start of the labels
                if (scaled):
                    # me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i, (window - idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node]) / sd[node]
                else:
                    H[i, (window - idx):(window)] = labs.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if (scaled):
                    H[i, 0:(window)] = (labs.loc[node, dates[(idx - window):(idx)]] - me[node]) / sd[node]
                else:
                    H[i, 0:(window)] = labs.loc[node, dates[(idx - window):(idx)]]

        features.append(H)

    return features


def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]

    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i + batch_size, N) - i) * graph_window * n_nodes
        step = n_nodes * graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i + batch_size, N) - i) * n_nodes)

        # fill the input for each batch
        for e1, j in enumerate(range(i, min(i + batch_size, N))):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2, k in enumerate(range(val - graph_window + 1, val + 1)):
                adj_tmp.append(Gs[k - 1].T)
                # each feature has a size of n_nodes
                features_tmp[(e1 * step + e2 * n_nodes):(e1 * step + (e2 + 1) * n_nodes), :] = features[
                    k]  # -features[val-graph_window-1]

            if (test_sample > 0):
                # --- val is by construction less than test sample
                if (val + shift < test_sample):
                    y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val + shift]

                else:
                    y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val]


            else:
                y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val + shift]

        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

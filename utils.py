from datetime import timedelta, datetime
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xgboost as xgb
from h2o import h2o
from sklearn.model_selection import train_test_split


def read_meta_datasets(window, labels_path, graphs_dir, country, smooth_window=None):
    labels = pd.read_csv(labels_path)
    if smooth_window:
        labels = labels.rolling(smooth_window, axis=1, min_periods=0).mean()
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
    features = generate_new_features(Gs, labels, dates, window, scaled=True)

    y = list()
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
    for idx, G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region

        H = np.zeros([G.number_of_nodes(), window])  # +3+n_departments])#])#])
        me = labs.loc[:, dates[:idx]].mean(1)
        sd = np.nan_to_num(labs.loc[:, dates[:idx]].std(1)) + 1

        for i, node in enumerate(G.nodes()):
            # ---- Past cases
            if idx < window:  # idx-1 goes before the start of the labels
                if scaled:
                    # me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i, (window - idx):window] = (labs.loc[node, dates[0:idx]] - me[node]) / sd[node]
                else:
                    H[i, (window - idx):window] = labs.loc[node, dates[0:idx]]

            elif idx >= window:
                if scaled:
                    H[i, 0:window] = (labs.loc[node, dates[(idx - window):idx]] - me[node]) / sd[node]
                else:
                    H[i, 0:window] = labs.loc[node, dates[(idx - window):idx]]

        features.append(H)

    return features


def generate_new_batches_legacy(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
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

            if test_sample > 0:
                # --- val is by construction less than test sample
                if val + shift < test_sample:
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


def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    batch_size = N
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

            if test_sample > 0:
                # --- val is by construction less than test sample
                if val + shift < test_sample:
                    y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val + shift]

                else:
                    y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val]


            else:
                y_tmp[(n_nodes * e1):(n_nodes * (e1 + 1))] = y[val + shift]

        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device).unsqueeze(0))
        features_lst.append(torch.FloatTensor(features_tmp).to(device).unsqueeze(0))
        y_lst.append(torch.FloatTensor(y_tmp).to(device).unsqueeze(0))
    adj_lst = torch.stack(adj_lst, dim=0)
    features_lst = torch.stack(features_lst, dim=0)
    y_lst = torch.stack(y_lst, dim=0)

    return adj_lst, features_lst, y_lst


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_tabnet_data(data_path, test_val_size: tuple = (0.2, 0.2), random_seed: int = 42):
    data = pd.read_csv(data_path)
    data = data.drop(['node_id1', 'dm', 'facility_id', 'starting_time'], axis=1)

    data['num_inf_in_fac'] = (data['num_inf_in_fac'] - data['num_inf_in_fac'].mean()) / data['num_inf_in_fac'].std()
    data['cases'] = (data['cases'] - data['cases'].mean()) / data['cases'].std()
    data['duration'] = (data['duration'] - data['duration'].mean()) / data['duration'].std()
    data['crowdedness_fac'] = (data['crowdedness_fac'] - data['crowdedness_fac'].mean()) / data['crowdedness_fac'].std()
    data['avg_intensity'] = (data['avg_intensity'] - data['avg_intensity'].mean()) / data['avg_intensity'].std()
    data['dist_inf'] = (data['dist_inf'] - data['dist_inf'].mean()) / data['dist_inf'].std()

    data['gen_fac'] = data['gen_fac'].astype('category').cat.codes
    data['ring'] = data['ring'].astype('category').cat.codes
    data['day'] = data['day'].astype('category')
    data['district'] = data['district'].astype('category')
    print(data.columns)
    data_y = data['inf'].values
    data = data.drop('inf', axis=1)
    data = data.values

    x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=test_val_size[0],
                                                        random_state=random_seed,
                                                        shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=test_val_size[1] / (1 - test_val_size[0]),
                                                      random_state=random_seed,
                                                      shuffle=True)

    return x_train, x_test, x_val, y_train, y_test, y_val


def load_visits_data(data_path, test_val_size: tuple = (0.2, 0.2), random_seed: int = 42):
    data = pd.read_csv(data_path)
    data = data.drop(['dm', 'facility_id', 'starting_time'], axis=1)
    data[['gen_fac']] = data[['gen_fac']].apply(lambda col: pd.Categorical(col).codes)
    data[['day']] = data[['day']].apply(lambda col: pd.Categorical(col).codes)

    data_y = data['inf'].values
    data = data.drop('inf', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=test_val_size[0],
                                                        random_state=random_seed,
                                                        shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=test_val_size[1] / (1 - test_val_size[0]),
                                                      random_state=random_seed,
                                                      shuffle=True)

    dtrain = xgb.DMatrix(
        x_train[['ring', 'gen_fac', 'day', 'cases', 'duration', 'crowdedness_fac',
                 'num_inf_in_fac', 'avg_intensity', 'dist_inf']],
        label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_val[['ring', 'gen_fac', 'day', 'cases', 'duration',
                              'crowdedness_fac', 'num_inf_in_fac', 'avg_intensity', 'dist_inf']],
                       label=y_val,
                       enable_categorical=True)
    dtest = xgb.DMatrix(
        x_test[['ring', 'gen_fac', 'day', 'cases', 'duration',
                'crowdedness_fac', 'num_inf_in_fac', 'avg_intensity', 'dist_inf']],
        label=y_test, enable_categorical=True)
    return dtrain, dval, dtest


def load_h2o_data(data_path, test_val_size: tuple = (0.2, 0.2), random_seed: int = 42):
    data = pd.read_csv(data_path)
    data_h2o = h2o.H2OFrame(data)
    data_h2o['gen_fac'] = data_h2o['gen_fac'].asfactor()
    data_h2o['day'] = data_h2o['day'].asfactor()
    data_h2o['inf'] = data_h2o['inf'].asfactor()
    data_h2o['district'] = data_h2o['inf'].asfactor()
    data_h2o['ring'] = data_h2o['ring'].asfactor()

    train, test, valid = data_h2o.split_frame(ratios=[1 - sum(test_val_size), test_val_size[1]], seed=random_seed)
    return train, test, valid

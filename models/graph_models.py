import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import io
from torch_geometric.nn import GCNConv
from PIL import Image
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import Dataset


class MPNNLSTM(pl.LightningModule):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, lr=0.001):
        super(MPNNLSTM, self).__init__()
        self.lr = lr
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.rnn1 = nn.LSTM(2 * nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)

        self.fc1 = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)

        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        x, (hn1, cn1) = self.rnn1(x)

        out2, (hn2, cn2) = self.rnn2(x)

        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        skip = skip.reshape(skip.size(0), -1)

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        return x

    def training_step(self, batch, batch_idx):
        adj, features, y = batch
        output = self(adj[0], features[0])
        loss_train = F.mse_loss(output, y[0])
        error = (output - y[0]).abs().sum()/20
        metrics = {"train_loss": loss_train, "train_error": error}
        self.log_dict(metrics)
        return loss_train

    def test_step(self, batch, batch_idx):
        adj, features, y = batch
        output = self(adj[0], features[0])
        loss_test = F.mse_loss(output, y[0])
        error = (output - y[0]).abs().sum()/20
        metrics = {"test_loss": loss_test, "test_error": error}
        self.log_dict(metrics)
        return loss_test

    def validation_step(self, batch, batch_idx):
        adj, features, y = batch
        output = self(adj[0], features[0])
        loss_test = F.mse_loss(output, y[0])
        error = (output - y[0]).abs().sum()/20
        metrics = {"val_loss": loss_test, "val_error": error}
        self.log_dict(metrics)
        return loss_test

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

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


def gru_collate(batch):
    data = [item[0] for item in batch]
    lens = [item[0].shape[0] for item in batch]
    target = [item[1].unsqueeze(0) for item in batch]
    return pack_padded_sequence(pad_sequence(data, batch_first=True), lens, batch_first=True,
                                enforce_sorted=False), torch.cat(target, dim=0)


class CovDataset(Dataset):
    def __init__(self, X, y):
        super(CovDataset, self).__init__()
        self.X = X
        self.y = y
        self.indices = torch.unique(self.X[:, [0, 1]], dim=0)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        indexes = (self.X[:, 0] == self.indices[idx, 0]) & (self.X[:, 1] == self.indices[idx, 1])
        return self.X[indexes], torch.tensor([min(1, sum(self.y[indexes]))])


class GRUModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, learning_rate=0.001, batch_size=64, pos_weight=1.):
        super(GRUModel, self).__init__()
        self.pos_weight = pos_weight
        self.learning_rate = learning_rate
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
        self.train_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
        self.batch_size = batch_size

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = num_layers
        self.hidden_dim = hidden_size

        # GRU layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_().to(self.device)

        out, _ = self.gru(x, h0)
        unpacked, lens = pad_packed_sequence(out)
        unpacked_selected = unpacked[lens-1, torch.arange(unpacked.shape[1])]
        out = torch.relu(self.fc1(unpacked_selected))
        out = torch.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # loss = focal_loss(y_hat.squeeze(1), y, alpha=1, gamma=10000)
        loss = F.binary_cross_entropy(y_hat, y.float(), weight=y * self.pos_weight + 1)
        self.train_confusion.update(y_hat, y.long())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y.float(), weight=y * self.pos_weight + 1)
        # loss = focal_loss(y_hat.squeeze(1), y, alpha=1, gamma=10000)
        # pred = torch.argmax(y_hat, dim=1)
        # print(pred.shape, y.shape)
        self.val_confusion.update(y_hat, y.long())
        self.val_accuracy(y_hat.long().reshape(-1), y.long().reshape(-1))
        self.log("val_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        tb = self.logger.experiment

        # confusion matrix
        conf_mat = self.train_confusion.compute().detach().cpu().numpy().astype(np.int)
        self.train_confusion.reset()
        df_cm = pd.DataFrame(
            conf_mat,
            index=np.arange(2),
            columns=np.arange(2))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("train_confusion_matrix", im, global_step=self.current_epoch)

    def validation_epoch_end(self, outs):
        tb = self.logger.experiment

        # confusion matrix
        conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(np.int)
        self.val_confusion.reset()
        df_cm = pd.DataFrame(
            conf_mat,
            index=np.arange(2),
            columns=np.arange(2))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()
        self.log('accuracy', self.val_accuracy.compute())
        self.val_accuracy.reset()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
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

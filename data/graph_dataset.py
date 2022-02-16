from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from utils import generate_new_batches, read_meta_datasets


class MPNNDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.adj, self.features, self.y = data
        self.len = self.adj.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.adj[item][0], self.features[item][0], self.y[item][0]


class MPNNDataModule(pl.LightningDataModule):
    def __init__(self, window: int, test_sample: int, sep: int, shift: int, batch_size: int,
                 graph_window: int, labels, gs_adj, features, y):
        super().__init__()
        self.batch_size = batch_size
        self.graph_window = graph_window
        self.shift = shift
        self.window = window
        self.test_sample = test_sample
        self.sep = sep
        self.labels = labels
        self.gs_adj = gs_adj
        self.features = features
        self.y = y

    def train_dataloader(self):
        idx_train = list(range(self.window - 1, self.test_sample - self.sep))
        idx_train = idx_train + list(range(self.test_sample - self.sep + 1, self.test_sample, 2))
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        data = generate_new_batches(self.gs_adj, self.features, self.y, idx_train,
                                    self.graph_window, self.shift,
                                    self.batch_size, device, self.test_sample)
        dataset = MPNNDataset(data)
        return DataLoader(dataset, batch_size=1)

    def val_dataloader(self):
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        idx_val = list(range(self.test_sample - self.sep, self.test_sample, 2))
        data = generate_new_batches(self.gs_adj, self.features, self.y, idx_val,
                                    self.graph_window, self.shift,
                                    self.batch_size, device, self.test_sample)
        dataset = MPNNDataset(data)
        return DataLoader(dataset, batch_size=1)

    def test_dataloader(self):
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

        data =  generate_new_batches(self.gs_adj, self.features, self.y, [self.test_sample],
                                    self.graph_window, self.shift,
                                    self.batch_size, device, self.test_sample)
        dataset = MPNNDataset(data)
        return DataLoader(dataset, batch_size=1)

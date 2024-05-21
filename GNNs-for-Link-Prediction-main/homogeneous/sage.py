# -*- coding:utf-8 -*-

import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

from models import SAGE_LP
from util import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False, disjoint_train_ratio=0),
])

# dataset = Planetoid(root_path + '/data', name='citeseer', transform=transform)
# dataset = Amazon(root_path + '/data', name='photo', transform=transform)
dataset = Coauthor(root_path + '/data', name='Physics', transform=transform)
train_data, val_data, test_data = dataset[0]

print(train_data)
print(val_data)
print(test_data)


def main():
    model = SAGE_LP(dataset.num_features, 32, 64).to(device)
    aucs, aps = [], []
    for i in range(2):
        test_auc, test_ap = train(model,
                                  train_data,
                                  val_data,
                                  test_data,
                                  save_model_path=root_path + '/models/sgc.pkl')
        aucs.append(test_auc * 100)
        aps.append(test_ap * 100)
    print(f'final best auc:{np.mean(aucs):.2f}±{np.std(aucs):.2f}')
    print(f'final best ap:{np.mean(aps):.2f}±{np.std(aps):.2f}')


if __name__ == '__main__':
    main()

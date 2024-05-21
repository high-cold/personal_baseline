import argparse

import numpy as np
import torch
from spikegcl.evaluate import test
from spikegcl.dataset import get_dataset
from spikegcl.model import SpikeGCL
from spikegcl.utils import tab_printer
from torch_geometric import seed_everything
from torch_geometric.logging import log
import torch.nn.functional as F


def read_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="data/", help="Data folder"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="Physics",
        help="Datasets (Photo, Computers, CS, Physics, Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-mag). (default: Pubmed)",
    )
    parser.add_argument(
        "--hids",
        type=int,
        default=32,
        help="Hidden units for each layer. (default: 64)",
    )
    parser.add_argument(
        "--outs",
        type=int,
        default=32,
        help="Out_channels for final embedding. (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training. (default: 1e-3)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="Random seed for model. (default: 2023)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Smooth factor for surrogate learning. (default: 2.0)",
    )
    parser.add_argument(
        "--surrogate",
        nargs="?",
        default="sigmoid",
        help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')",
    )
    parser.add_argument(
        "--neuron",
        nargs="?",
        default="PLIF",
        help="Spiking neuron used for training. (IF, LIF, PLIF). (default: PLIF)",
    )
    parser.add_argument(
        "--reset",
        nargs="?",
        default="subtract",
        help="Ways to reset spiking neuron. (zero, subtract). (default: subtract)",
    )
    parser.add_argument(
        "--act",
        nargs="?",
        default="elu",
        help="Activation function. (relu, elu, None). (default: elu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-3,
        help="Voltage threshold in spiking neuron. (default: 5e-3)",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=30,
        help="Time steps for spiking neural networks. (default: 30)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability. (default: 0.5)"
    )
    parser.add_argument(
        "--dropedge",
        type=float,
        default=0.2,
        help="Edge dropout probability. (default: 0.2)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin used in ranking loss. (default: 0.0)",
    )
    parser.add_argument('--bn', action='store_true',
                        help='Whether to use batch normalization. (default: False)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Whether to perform feature shuffling augmentation. (default: False)')
    try:
        args = parser.parse_args()
        tab_printer(args)
        return args
    except:
        parser.print_help()
        exit(0)


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:, n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, val_edges_neg, test_edges_neg], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)


def cal_lp_loss(embeddings, pos_edges, neg_edges):
    pos_scores = torch.sigmoid(2 - torch.sum((embeddings[pos_edges[0]] - embeddings[pos_edges[1]]) ** 2, -1))
    neg_scores = torch.sigmoid(2 - torch.sum((embeddings[neg_edges[0]] - embeddings[neg_edges[1]]) ** 2, -1))
    loss = F.binary_cross_entropy(pos_scores.clip(0.01, 0.99), torch.ones_like(pos_scores)) + \
           F.binary_cross_entropy(neg_scores.clip(0.01, 0.99), torch.zeros_like(neg_scores))

    label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]

    preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
    auc, ap = cal_AUC_AP(preds, label)
    return loss, auc, ap


from sklearn.metrics import roc_auc_score, average_precision_score, f1_score



best_val_acc = final_test_acc = 0

from torch_geometric.utils import negative_sampling

args = read_parser()
seed_everything(args.seed)

data = get_dataset(
    root=args.root,
    name=args.dataset,
)

edge_index = data.edge_index

neg_edge = negative_sampling(data.edge_index)
pos_edges, neg_edges = mask_edges(data.edge_index, neg_edge, 0.05, 0.1)

def cal_AUC_AP(scores, trues):
    auc = roc_auc_score(trues, scores)
    ap = average_precision_score(trues, scores)
    return auc, ap



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpikeGCL(
    data.x.size(1),
    args.hids,
    args.outs,
    args.T,
    args.alpha,
    args.surrogate,
    args.threshold,
    args.neuron,
    args.reset,
    args.act,
    args.dropedge,
    args.dropout,
    bn=args.bn,
    shuffle=not args.no_shuffle,
)

print(model)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0

    # neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
    # embeds = model.encode(data.x, data.edge_index, data.edge_attr)
    # embeds = torch.cat(embeds, dim=-1)
    # z1, _ = model(data.x, data.edge_index, data.edge_attr)
    # z1 = z1.mean(0)
    # loss, auc, ap = cal_lp_loss(z1, pos_edges=pos_edges[0], neg_edges=neg_edge_train)
    # loss.backward()
    z1s, z2s = model(data.x, data.edge_index, data.edge_attr)
    for z1, z2 in zip(z1s, z2s):
        loss = model.loss(z1, z2, args.margin)
        loss.backward()
        loss_total += loss.item()
    optimizer.step()
    return loss_total


def test_LP():
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0

    neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
    # embeds = model.encode(data.x, data.edge_index, data.edge_attr)
    # embeds = torch.cat(embeds, dim=-1)
    z1, _ = model(data.x, data.edge_index, data.edge_attr)
    z1 = torch.mean(torch.stack(z1, fim), dim=0)
    loss, auc, ap = cal_lp_loss(z1, pos_edges=pos_edges[0], neg_edges=neg_edge_train)
    loss.backward()
    return loss


aucs, aps = [], []
for epoch in range(1, args.epochs + 1):
    loss = train()
    print("epoch:", epoch, loss)
    model.eval()
    with torch.no_grad():
        embeds = model.encode(data.x, data.edge_index, data.edge_attr)
        embeds = torch.cat(embeds, dim=-1)

        neg_edge_test = neg_edges[2][:, np.random.randint(0, neg_edges[2].shape[1], pos_edges[2].shape[1])]
        z1, _ = model(data.x, data.edge_index, data.edge_attr)
        z1 = torch.mean(torch.stack(z1, dim=0), dim=0)
        loss, auc, ap = cal_lp_loss(z1, pos_edges=pos_edges[2], neg_edges=neg_edges[2])
        # test_loss, test_auc, test_ap = cal_lp_loss(torch.nn.functional.normalize(embeds), pos_edges[2], neg_edges[2])
    all_test_acc, all_test_w_f1, all_test_m_f1 = test(embeds, data, data.num_classes)
    print(auc * 100, ap * 100)
    aucs.append(auc * 100)
    aps.append(ap * 100)


top_5_aucs = sorted(aucs, reverse=True)[:3]
top_5_aps = sorted(aps, reverse=True)[:3]

# 计算均值和方差
mean_aucs = np.mean(top_5_aucs)
std_aucs = np.std(top_5_aucs)

mean_aps = np.mean(top_5_aps)
std_aps = np.std(top_5_aps)

# 输出结果，保留两位小数
print(f"aucs {mean_aucs:.2f}±{std_aucs:.2f}")
print(f"aps {mean_aps:.2f}±{std_aps:.2f}")

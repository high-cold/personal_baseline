# -*- coding: utf-8 -*-
# @Time    : 2024/5/6  21:42
# @Author  : WanQiQi
# @FileName: calc_SpikeGCL.py
# @Software: PyCharm
"""
    Description:
        
"""
import argparse

import torch
import torch.nn as nn
from thop import profile
from spikingjelly.clock_driven import neuron
from spikegcl.dataset import get_dataset
from spikegcl.model import SpikeGCL
from spikegcl.utils import tab_printer
from torch_geometric import seed_everything
from torch_geometric.logging import log
from spikegcl.evaluate import test
import main
from thop import profile
from thop import clever_format
from spikegcl.neuron import PLIF
import numpy as np
def calc_params(model, x, edge_index, edge_wight, args):
    def count_encoder(m, input, output):
        # your rule here
        m.total_ops += output.shape[0] * args.outs * 4.6 * output.sum().item() / output.shape[0] / args.T

    def count_snn(m, input, output):
        # your rule here
        x, adj, _ = input
        s_o = output
        snn = PLIF()
        o = snn(s_o)
        # m.total_params += s_o.shape[-1] * x.shape[-1]
        m.total_params += sum([np.prod(p.size()) for p in model.parameters()])
        # m.total_ops += 4.6 * (x.shape[0] * s_o.shape[-1])
        m.total_ops += 3.7 * o.sum().item() / o.shape[0] / args.T / 2
        m.total_ops += 3.7 * o.sum().item() / 2

    """
       macs, params = profile(method.model,
                           inputs=(method.cache.X, method.cache.A))
    """
    model.eval()
    from torch_geometric.nn import GCNConv

    from spikegcl.model import creat_snn_layer
    energy, params = profile(model,
                             inputs=(x, edge_index, edge_wight),
                             custom_ops={GCNConv: count_snn,
                                         PLIF: count_encoder}
                             )
    energy /= args.T
    params = clever_format([params], "%.4f")
    energy = f"{energy * 1e-9} mJ"
    return energy, params


# 定义你的网络结构
# 根据每个不同的模型修改此处
# 如果有除了n_feat, n_nodes这两个参数之外的参数，需要自己读取修改代码

if __name__ == '__main__':
    args = main.read_parser()
    seed_everything(args.seed)
    data = get_dataset(
        root=args.root,
        dataset=args.dataset,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建模型
    net = SpikeGCL(
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
    e, p = calc_params(net, data.x, data.edge_index, data.edge_weight, args)
    print(f'MACS: {e}')
    print(f"Params 是 {p}")

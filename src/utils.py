import numpy as np
import torch
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max


def sum_readout(graph, node_feature):
    return scatter_add(node_feature, 
                       graph.node2graph, 
                       dim=0, 
                       dim_size=graph.batch_size)


def mean_readout(graph, node_feature):
    return scatter_mean(node_feature, 
                        graph.node2graph, 
                        dim=0, 
                        dim_size=graph.batch_size)


def max_readout(graph, node_feature):
    return scatter_max(node_feature, 
                       graph.node2graph, 
                       dim=0, 
                       dim_size=graph.batch_size)


class AtomFeature:
    
    def __init__(self, features=["symbol"]):
        self.features = features
        self.trans_funcs = [getattr(AtomFeature, f) for f in self.features]
            
    @staticmethod
    def symbol(graph):
        return F.one_hot(graph.node_type, 
                         num_classes=graph.num_node_type).float()
    
    @staticmethod
    def degree(graph):
        degree = torch.bincount(graph.edge_index[0], minlength=graph.num_node)
        return F.one_hot(degree.clip(0, 8),
                         num_classes=9).float()
    
    def transform(self, graph):
        atom_feature = [func(graph) for func in self.trans_funcs]
        return torch.hstack(atom_feature)


def get_logger(fname, log_dir):
    import logging
    import os
    
    logger = logging.getLogger(fname)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(log_dir, fname))
#     formatter = logging.Formatter("%(asctime)s >> %(message)s")
#     file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # https://github.com/DLR-RM/stable-baselines3/blob/39a4f9379a8068110c895c4bb18cb0e4e20cd69c/stable_baselines3/common/utils.py
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y




'''
def arange_interleaves(index):
    # index: [0, 0, 1, 0, 0, 1, 1, ...] 
    # index_sort: [0, 1, 4, 2, 3, 5, 6, ...]
    # count_index: [0, 1, 0, 2, 3, 1, 2, ...]
    bincount = torch.bincount(index)
    cumcount = torch.cumsum(bincount, 0)
    index_sort = index.sort(stable=True).indices.sort(stable=True).indices
    count_index = index_sort - (cumcount - bincount)[index]
    return count_index


@torch.no_grad()
def scatter_sample(probs, index):
    assert probs.shape == index.shape
    count_index = arange_interleaves(index)
    max_size = count_index.max() + 1
    num_indices = index.unique().size(0)
    
    # box_index: [0, 1, 10, 2, 3, 11, 12, ...] (e.g. max_size: 10)
    box_index = torch.arange(num_indices) * max_size
    box_index = box_index[index] + count_index

    probs_full = torch.zeros(num_indices, max_size, dtype=torch.float32)
    probs_full.view(-1)[box_index] = probs
    dist = torch.distributions.Categorical(probs=probs_full)
    
    sample_index = dist.sample()
    
    src_index = torch.zeros(num_indices, max_size, dtype=torch.int64)
    src_index.view(-1)[box_index] = torch.arange(len(index))
    src_index = src_index.gather(1, sample_index.view(-1, 1)).view(-1)
    return sample_index, src_index
'''
# @torch.no_grad()
# def scatter_sample(logits, index):
#     assert logits.shape == index.shape
#     count_index = arange_interleaves(index)
#     max_size = count_index.max() + 1
#     num_indices = index.unique().size(0)

#     # box_index: [0, 1, 10, 2, 3, 11, 12, ...] (e.g. max_size: 10)
#     box_index = torch.arange(num_indices) * max_size
#     box_index = box_index[index] + count_index

#     logits_full = torch.full((num_indices, max_size), 
#                                  -torch.inf, 
#                                  dtype=torch.float32)
#     logits_full.view(-1)[box_index] = logits
#     dist = torch.distributions.Categorical(logits=logits_full)
#     sample_index = dist.sample()
    
#     src_index = torch.zeros(num_indices, max_size, dtype=torch.int64)
#     src_index.view(-1)[box_index] = torch.arange(len(index))
#     src_index = src_index.gather(1, sample_index.view(-1, 1)).view(-1)
#     return sample_index, src_index
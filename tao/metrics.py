from __future__ import annotations

"""
Utility metrics for evaluating TAO trees, in line with the NeurIPS 2018 paper:

- Training / test error.
- Proportion of nonzero weights in oblique trees (%nonzero).
- Number of internal nodes (#splits).
- Path length statistics and inference operation counts.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .tree import DecisionTree, TreeNode


def count_internal_nodes(tree: DecisionTree) -> int:
    return sum(1 for n in tree.iter_internal_nodes())


def count_leaves(tree: DecisionTree) -> int:
    return sum(1 for n in tree.iter_leaves())


def count_nonzero_weights(tree: DecisionTree) -> int:
    total_nonzero = 0
    for node in tree.iter_internal_nodes():
        if node.oblique_weights is not None:
            total_nonzero += int(np.count_nonzero(node.oblique_weights))
    return total_nonzero


def total_weight_size(tree: DecisionTree) -> int:
    total = 0
    for node in tree.iter_internal_nodes():
        if node.oblique_weights is not None:
            total += int(node.oblique_weights.size)
    return total


@dataclass
class PathStats:
    mean_length: float
    min_length: int
    max_length: int
    mean_ops: float


def _path_length_and_ops_for_sample(
    tree: DecisionTree, x: np.ndarray
) -> Tuple[int, int]:
    """
    Compute path length (number of visited nodes) and an approximate
    number of scalar multiplications (ops) for a single sample.
    """
    length = 0
    ops = 0
    node: TreeNode = tree.root
    while True:
        length += 1
        if node.is_leaf:
            break
        if node.oblique_weights is not None:
            # One dot product: cost ≈ number of nonzeros in w.
            ops += int(np.count_nonzero(node.oblique_weights))
        else:
            # Axis-aligned: one comparison, count as 1 op.
            ops += 1
        node = node.decision(x)
    return length, ops


def compute_path_stats(tree: DecisionTree, X: np.ndarray) -> PathStats:
    X = np.asarray(X)
    n_samples = X.shape[0]
    lengths = []
    ops_list = []
    for i in range(n_samples):
        length, ops = _path_length_and_ops_for_sample(tree, X[i])
        lengths.append(length)
        ops_list.append(ops)
    lengths_arr = np.array(lengths, dtype=int)
    ops_arr = np.array(ops_list, dtype=int)
    return PathStats(
        mean_length=float(lengths_arr.mean()),
        min_length=int(lengths_arr.min()),
        max_length=int(lengths_arr.max()),
        mean_ops=float(ops_arr.mean()),
    )


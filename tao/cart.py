from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .tree import DecisionTree, TreeNode


@dataclass
class CartConfig:
    """Configuration for the simple CART-style inducer."""

    max_depth: int = 5
    min_samples_leaf: int = 5
    min_impurity_decrease: float = 0.0


def _gini_impurity(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(1.0 - np.sum(probs * probs))


def _best_axis_aligned_split(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    config: CartConfig,
) -> Tuple[Optional[int], Optional[float], float]:
    """
    Find the best (feature, threshold) pair by minimizing Gini impurity.

    Returns (feature_index, threshold, impurity_decrease).
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None, 0.0

    parent_counts = np.bincount(y, minlength=n_classes).astype(float)
    parent_impurity = _gini_impurity(parent_counts)

    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
    best_impurity_decrease: float = 0.0

    for j in range(n_features):
        feature_values = X[:, j]
        order = np.argsort(feature_values)
        sorted_x = feature_values[order]
        sorted_y = y[order]

        # Candidate splits are between distinct sorted feature values.
        unique_mask = np.diff(sorted_x) > 0
        if not np.any(unique_mask):
            continue

        left_counts = np.zeros(n_classes, dtype=float)
        right_counts = parent_counts.copy()

        for i in range(n_samples - 1):
            cls = int(sorted_y[i])
            left_counts[cls] += 1.0
            right_counts[cls] -= 1.0

            if not unique_mask[i]:
                continue

            left_n = i + 1
            right_n = n_samples - left_n
            if (
                left_n < config.min_samples_leaf
                or right_n < config.min_samples_leaf
            ):
                continue

            left_impurity = _gini_impurity(left_counts)
            right_impurity = _gini_impurity(right_counts)
            weighted_impurity = (
                left_n * left_impurity + right_n * right_impurity
            ) / n_samples
            impurity_decrease = parent_impurity - weighted_impurity

            if impurity_decrease > best_impurity_decrease:
                best_impurity_decrease = impurity_decrease
                best_feature = j
                best_threshold = 0.5 * (sorted_x[i] + sorted_x[i + 1])

    return best_feature, best_threshold, best_impurity_decrease


def build_cart_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    min_impurity_decrease: float = 0.0,
) -> DecisionTree:
    """
    Build a simple axis-aligned classification tree using a CART-style
    greedy algorithm with Gini impurity.

    This is intended to provide an initial tree structure for TAO.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and match X in length.")

    n_samples, n_features = X.shape
    n_classes = int(y.max()) + 1
    cfg = CartConfig(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
    )

    next_id = 0

    def new_node(depth: int) -> TreeNode:
        nonlocal next_id
        node = TreeNode(node_id=next_id, depth=depth)
        next_id += 1
        return node

    def majority_label(labels: np.ndarray) -> int:
        counts = np.bincount(labels, minlength=n_classes)
        return int(np.argmax(counts))

    def grow(
        indices: np.ndarray,
        depth: int,
        parent: TreeNode | None,
    ) -> TreeNode:
        node = new_node(depth=depth)
        node.parent = parent

        y_subset = y[indices]
        if depth >= cfg.max_depth:
            node.is_leaf = True
            node.prediction = majority_label(y_subset)
            return node

        if indices.size < 2 * cfg.min_samples_leaf:
            node.is_leaf = True
            node.prediction = majority_label(y_subset)
            return node

        # Check purity
        if np.all(y_subset == y_subset[0]):
            node.is_leaf = True
            node.prediction = int(y_subset[0])
            return node

        feature, threshold, impurity_decrease = _best_axis_aligned_split(
            X[indices], y_subset, n_classes=n_classes, config=cfg
        )

        if (
            feature is None
            or threshold is None
            or impurity_decrease < cfg.min_impurity_decrease
        ):
            node.is_leaf = True
            node.prediction = majority_label(y_subset)
            return node

        node.is_leaf = False
        node.feature_index = int(feature)
        node.threshold = float(threshold)

        feat_vals = X[indices, node.feature_index]
        go_right = feat_vals >= node.threshold
        left_indices = indices[~go_right]
        right_indices = indices[go_right]

        node.left = grow(left_indices, depth + 1, parent=node)
        node.right = grow(right_indices, depth + 1, parent=node)
        return node

    root_indices = np.arange(n_samples, dtype=int)
    root = grow(root_indices, depth=0, parent=None)
    return DecisionTree(root=root, n_features=n_features, n_classes=n_classes)


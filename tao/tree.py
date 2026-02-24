from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LabelArray = np.ndarray
IndexArray = np.ndarray


@dataclass
class TreeNode:
    """
    Single node in a binary decision tree.

    A node is either:
    - an internal decision node, with a split rule and left/right children, or
    - a leaf node, with a class label prediction.
    """

    node_id: int
    depth: int
    parent: Optional["TreeNode"] = None

    # Children for internal nodes
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    # Leaf parameters
    is_leaf: bool = False
    prediction: Optional[int] = None

    # Axis-aligned split (feature index and threshold); used if oblique_weights is None.
    feature_index: Optional[int] = None
    threshold: Optional[float] = None

    # Oblique split: w^T x >= bias -> right, else left.
    oblique_weights: Optional[np.ndarray] = None
    oblique_bias: Optional[float] = None

    def is_internal(self) -> bool:
        return not self.is_leaf

    def uses_axis_aligned(self) -> bool:
        return self.is_internal() and self.oblique_weights is None

    def uses_oblique(self) -> bool:
        return self.is_internal() and self.oblique_weights is not None

    def decision(self, x: np.ndarray) -> "TreeNode":
        """
        Route a single sample vector to either left or right child using this node's split.
        """
        if self.is_leaf:
            raise ValueError("Leaf nodes do not perform decisions.")
        if self.left is None or self.right is None:
            raise ValueError("Internal node must have both left and right children.")

        if self.uses_axis_aligned():
            if self.feature_index is None or self.threshold is None:
                raise ValueError("Axis-aligned node missing feature_index or threshold.")
            go_right = x[self.feature_index] >= self.threshold
        else:
            if self.oblique_weights is None or self.oblique_bias is None:
                raise ValueError("Oblique node missing weights or bias.")
            score = float(np.dot(self.oblique_weights, x))
            go_right = score >= self.oblique_bias

        return self.right if go_right else self.left


class DecisionTree:
    """
    Binary decision tree with utilities for prediction and routing.

    This class is agnostic to how the tree was learned (CART, TAO, etc.).
    """

    def __init__(self, root: TreeNode, n_features: int, n_classes: int) -> None:
        self.root = root
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

    # ------------------------------------------------------------------
    # Traversal utilities
    # ------------------------------------------------------------------
    def iter_nodes_bfs(self) -> Iterable[TreeNode]:
        """Breadth-first traversal over all nodes."""
        queue: List[TreeNode] = [self.root]
        while queue:
            node = queue.pop(0)
            yield node
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    def iter_nodes_dfs(self) -> Iterable[TreeNode]:
        """Depth-first (pre-order) traversal over all nodes."""
        stack: List[TreeNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    def iter_leaves(self) -> Iterable[TreeNode]:
        """Iterate over all leaf nodes."""
        for node in self.iter_nodes_bfs():
            if node.is_leaf:
                yield node

    def iter_internal_nodes(self) -> Iterable[TreeNode]:
        """Iterate over all internal (decision) nodes."""
        for node in self.iter_nodes_bfs():
            if node.is_internal():
                yield node

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            node = node.decision(x)
        if node.prediction is None:
            raise ValueError("Leaf node missing prediction.")
        return int(node.prediction)

    def predict(self, X: np.ndarray) -> LabelArray:
        """
        Predict class labels for a batch of samples.

        Parameters
        ----------
        X
            Array of shape (n_samples, n_features).
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features}), "
                f"got {X.shape}."
            )
        return np.array([self._predict_one(x) for x in X], dtype=int)

    # ------------------------------------------------------------------
    # Routing utilities for TAO
    # ------------------------------------------------------------------
    def route_dataset(self, X: np.ndarray) -> Dict[int, IndexArray]:
        """
        Route all samples through the tree and return, for each node_id,
        the indices of samples that reach that node.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features}), "
                f"got {X.shape}."
            )

        # Map node_id -> list of sample indices
        routing: Dict[int, List[int]] = {self.root.node_id: list(range(n_samples))}

        for node in self.iter_nodes_bfs():
            idxs = routing.get(node.node_id, [])
            if node.is_leaf or not idxs:
                continue

            indices = np.array(idxs, dtype=int)

            if node.uses_axis_aligned():
                assert node.feature_index is not None and node.threshold is not None
                feature_vals = X[indices, node.feature_index]
                go_right = feature_vals >= node.threshold
            else:
                assert node.oblique_weights is not None and node.oblique_bias is not None
                scores = X[indices] @ node.oblique_weights
                go_right = scores >= node.oblique_bias

            left_indices = indices[~go_right]
            right_indices = indices[go_right]

            if node.left is not None and left_indices.size:
                routing.setdefault(node.left.node_id, []).extend(
                    left_indices.tolist()
                )
            if node.right is not None and right_indices.size:
                routing.setdefault(node.right.node_id, []).extend(
                    right_indices.tolist()
                )

        # Convert lists to numpy arrays
        return {
            node_id: np.array(idxs, dtype=int)
            for node_id, idxs in routing.items()
        }

    def predict_from_node(
        self,
        start_node: TreeNode,
        X: np.ndarray,
        sample_indices: Sequence[int],
    ) -> Dict[int, int]:
        """
        Predict labels for a subset of samples, starting from a given node.

        This is used in TAO to compute the labels each child subtree would
        predict, assuming we force the sample into that subtree.

        Returns
        -------
        mapping
            Dict from sample index to predicted class label.
        """
        X = np.asarray(X)
        sample_indices = np.asarray(sample_indices, dtype=int)
        result: Dict[int, int] = {}
        for idx in sample_indices:
            x = X[idx]
            node = start_node
            while not node.is_leaf:
                node = node.decision(x)
            if node.prediction is None:
                raise ValueError("Leaf node missing prediction.")
            result[int(idx)] = int(node.prediction)
        return result

    # ------------------------------------------------------------------
    # Utility metrics
    # ------------------------------------------------------------------
    def misclassification_error(self, X: np.ndarray, y: LabelArray) -> float:
        """Compute the empirical misclassification error on a dataset."""
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        preds = self.predict(X)
        return float(np.mean(preds != y))


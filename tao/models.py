from __future__ import annotations

"""
High-level estimator interfaces for TAO.

These provide a scikit-learn-like API around the core CART inducer
and TAO optimizer, while staying faithful to the NeurIPS 2018 TAO
algorithm (misclassification loss, fixed tree structure, alternating
optimization over depth levels).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cart import build_cart_tree
from .tao import (
    TAOConfig,
    TAOObliqueConfig,
    tao_optimize_axis_aligned,
    tao_optimize_oblique,
)
from .tree import DecisionTree


@dataclass
class TAOAxisAlignedClassifier:
    """
    TAO-optimized axis-aligned decision tree classifier.

    Workflow:
    1. Build an initial axis-aligned CART tree (greedy growing).
    2. Run TAO (Tree Alternating Optimization) on that fixed structure to
       minimize the misclassification error (eq. (2) in the paper) by
       iterating over depth levels and solving the reduced problem (eq. (4))
       at each internal node.
    """

    max_depth: int = 5
    min_samples_leaf: int = 5
    max_tao_iter: int = 10
    tao_tol: float = 0.0

    tree_: Optional[DecisionTree] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TAOAxisAlignedClassifier":
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # 1) Initial tree via CART (axis-aligned, Gini impurity).
        tree = build_cart_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

        # 2) TAO optimization of node parameters w.r.t. misclassification error.
        cfg = TAOConfig(max_iterations=self.max_tao_iter, tol=self.tao_tol)
        self.tree_ = tao_optimize_axis_aligned(tree, X, y, cfg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.tree_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on (X, y)."""
        y = np.asarray(y, dtype=int)
        preds = self.predict(X)
        return float((preds == y).mean())


@dataclass
class TAOObliqueClassifier:
    """
    TAO-optimized oblique decision tree classifier.

    For simplicity, we start from an axis-aligned CART tree and then
    run oblique TAO, which is consistent with the paper's observation
    that TAO can also run on random or CART trees and reshape them.
    """

    max_depth: int = 5
    min_samples_leaf: int = 5
    max_tao_iter: int = 10
    tao_tol: float = 0.0

    # Oblique surrogate/regularization hyperparameters
    loss: str = "hinge"     # {"hinge", "logistic"}
    penalty: str = "l1"     # {"l1", "l2"}
    C: float = 1.0          # inverse sparsity (λ = 1 / C)

    tree_: Optional[DecisionTree] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TAOObliqueClassifier":
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Initial tree: axis-aligned CART. TAO will turn internal nodes
        # into oblique ones as needed.
        tree = build_cart_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

        cfg = TAOObliqueConfig(
            max_iterations=self.max_tao_iter,
            tol=self.tao_tol,
            loss=self.loss,
            penalty=self.penalty,
            C=self.C,
        )
        self.tree_ = tao_optimize_oblique(tree, X, y, cfg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.tree_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=int)
        preds = self.predict(X)
        return float((preds == y).mean())


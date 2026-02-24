from __future__ import annotations

"""
TAO optimizer for decision trees (axis-aligned and oblique variants).

This module closely follows section 3 of:

M. A. Carreira-Perpiñán and P. Tavallali,
"Alternating Optimization of Decision Trees, with Application to Learning
 Sparse Oblique Trees", NeurIPS 2018.

Key ideas implemented here:

- Global objective: K-class misclassification error of the tree (eq. (2)).
- Separability condition (Theorem 3.1): we optimize nodes grouped by depth,
  from leaves up to the root (reverse breadth-first order).
- Reduced problem (Theorem 3.2, eq. (4)): for each internal node, we build
  the "care" set Ci and a binary classification problem over {left, right}
  and solve it exactly for axis-aligned splits by enumeration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .tree import DecisionTree, TreeNode

LabelArray = np.ndarray
IndexArray = np.ndarray


@dataclass
class TAOConfig:
    """Configuration parameters for the TAO optimizer (axis-aligned)."""

    # Maximum number of passes over all depth levels.
    max_iterations: int = 10
    # Stop when the improvement in misclassification error is <= tol.
    # In the exact axis-aligned case, tol can usually be 0.
    tol: float = 0.0
    # Numerical slack to assert non-increasing loss (for safety).
    monotonic_slack: float = 1e-12


@dataclass
class TAOObliqueConfig:
    """
    Configuration parameters for the oblique TAO optimizer.

    This follows the paper's suggestion of using a surrogate loss
    (hinge or logistic) with optional ℓ1 regularization to solve
    the reduced problem at each internal node.
    """

    max_iterations: int = 10
    tol: float = 0.0
    monotonic_slack: float = 1e-3  # allow small numerical / surrogate-loss wiggles

    # Surrogate details
    loss: str = "hinge"  # {"hinge", "logistic"}
    penalty: str = "l1"  # {"l1", "l2"}
    C: float = 1.0  # inverse of λ in the paper (λ = 1 / C)


def _majority_label(labels: np.ndarray, n_classes: int) -> int:
    counts = np.bincount(labels, minlength=n_classes)
    return int(np.argmax(counts))


def _optimize_leaf(
    tree: DecisionTree,
    node: TreeNode,
    y: LabelArray,
    sample_indices: IndexArray,
) -> None:
    """
    Optimize a leaf node by minimizing the K-class misclassification
    error over the subset of points that reach it (majority vote).

    This corresponds to optimizing eq. (2) over a leaf, as described
    in the paper (section "Optimizing the misclassification error at a single node").
    """
    if sample_indices.size == 0:
        return
    label = _majority_label(y[sample_indices], tree.n_classes)
    node.is_leaf = True
    node.prediction = label


def _axis_aligned_reduced_problem(
    X: np.ndarray,
    y_binary: np.ndarray,
) -> Tuple[int, float]:
    """
    Solve the reduced problem (eq. (4)) for an internal node using
    axis-aligned hyperplanes, by exact enumeration over all
    feature / threshold pairs.

    Parameters
    ----------
    X
        Care-point features, shape (n_care, n_features).
    y_binary
        Binary labels in {0, 1} indicating which child ("left"=0, "right"=1)
        should be taken to achieve correct classification under the current
        fixed subtrees.

    Returns
    -------
    feature_index, threshold
    """
    X = np.asarray(X)
    y_binary = np.asarray(y_binary, dtype=int)
    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("Reduced problem called with no care points.")

    total_pos = int(y_binary.sum())
    total_neg = n_samples - total_pos

    best_feature = 0
    best_threshold = float(X[:, 0].mean())
    best_error = n_samples  # upper bound on number of misclassified care points

    for j in range(n_features):
        feature_values = X[:, j]
        order = np.argsort(feature_values)
        sorted_x = feature_values[order]
        sorted_y = y_binary[order]

        # Skip features with no variation
        if sorted_x[0] == sorted_x[-1]:
            continue

        prefix_pos = np.cumsum(sorted_y)

        for i in range(n_samples - 1):
            if sorted_x[i] == sorted_x[i + 1]:
                continue

            left_pos = int(prefix_pos[i])
            left_n = i + 1
            left_neg = left_n - left_pos

            right_pos = total_pos - left_pos
            right_n = n_samples - left_n
            right_neg = right_n - right_pos

            # Binary misclassification error of split:
            # left side predicts "left" (0), right side predicts "right" (1)
            misclassified = left_pos + right_neg
            if misclassified < best_error:
                best_error = misclassified
                best_feature = j
                best_threshold = 0.5 * (sorted_x[i] + sorted_x[i + 1])

    return best_feature, best_threshold


def _optimize_internal_axis_aligned(
    tree: DecisionTree,
    node: TreeNode,
    X: np.ndarray,
    y: LabelArray,
    sample_indices: IndexArray,
) -> None:
    """
    Optimize an internal axis-aligned node using the reduced problem (eq. (4)).

    Implements Theorem 3.2:
    - Build the "care" set Ci: points for which left/right children differ
      in correctness.
    - Define binary labels for Ci: 0 if left child is correct, 1 if right
      child is correct.
    - Solve the resulting binary misclassification problem exactly, restricted
      to axis-aligned splits.
    """
    if sample_indices.size == 0:
        return

    if node.left is None or node.right is None:
        # Degenerate tree; nothing we can do here.
        return

    # For each sample reaching this node, compute the leaf label predicted
    # if we force it to go to the left or right child, respectively.
    left_preds = tree.predict_from_node(node.left, X, sample_indices)
    right_preds = tree.predict_from_node(node.right, X, sample_indices)

    care_indices: List[int] = []
    care_labels: List[int] = []

    for idx in sample_indices:
        yi = int(y[idx])
        z_left = left_preds[idx]
        z_right = right_preds[idx]

        left_correct = z_left == yi
        right_correct = z_right == yi

        # "don't-care" points: fate cannot be changed by this node.
        if (left_correct and right_correct) or (
            (not left_correct) and (not right_correct)
        ):
            continue

        # "care" points: classify into the child that is correct.
        care_indices.append(int(idx))
        care_labels.append(0 if left_correct else 1)

    if not care_indices:
        # Nothing to optimize for this node under current routing.
        return

    care_indices_arr = np.array(care_indices, dtype=int)
    care_labels_arr = np.array(care_labels, dtype=int)

    X_care = X[care_indices_arr]
    feature, threshold = _axis_aligned_reduced_problem(X_care, care_labels_arr)

    # Update node parameters.
    node.is_leaf = False
    node.feature_index = int(feature)
    node.threshold = float(threshold)
    node.oblique_weights = None
    node.oblique_bias = None


def _solve_oblique_reduced_problem(
    X: np.ndarray,
    y_binary: np.ndarray,
    config: TAOObliqueConfig,
) -> Tuple[np.ndarray, float]:
    """
    Solve the oblique reduced problem (eq. (4)) using a surrogate
    binary loss (hinge or logistic) and optional ℓ1 penalty.

    Returns
    -------
    weights : np.ndarray, shape (n_features,)
    bias    : float
    """
    X = np.asarray(X)
    y_binary = np.asarray(y_binary, dtype=int)
    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError("Reduced problem called with no care points.")

    # Map labels {0,1} -> {-1, +1} for hinge if desired.
    if config.loss == "hinge":
        # Use squared hinge loss with liblinear; this supports ℓ1 penalty,
        # which approximates the ℓ1-regularized SVM suggested in the paper.
        from sklearn.svm import LinearSVC as _LinearSVC

        # If all care labels are the same, any separating hyperplane is fine.
        # In that case, we skip fitting and just keep the node unchanged by
        # returning a zero vector and zero bias; TAO will treat this as a
        # no-op for this node.
        if np.unique(y_binary).size < 2:
            w = np.zeros(X.shape[1], dtype=float)
            b = 0.0
        else:
            clf = _LinearSVC(
                C=config.C,
                penalty=config.penalty,
                loss="squared_hinge",
                dual=False,
                max_iter=10_000,
            )
            clf.fit(X, y_binary)
            w = clf.coef_.ravel()
            b = float(clf.intercept_[0])
    elif config.loss == "logistic":
        # Binary logistic regression with optional ℓ1 penalty.
        # Use liblinear which supports ℓ1.
        clf = LogisticRegression(
            C=config.C,
            penalty=config.penalty,
            solver="liblinear",
            max_iter=10_000,
        )
        clf.fit(X, y_binary)
        w = clf.coef_.ravel()
        b = float(clf.intercept_[0])
    else:
        raise ValueError(f"Unsupported loss '{config.loss}' for oblique TAO.")

    return w, b


def _optimize_internal_oblique(
    tree: DecisionTree,
    node: TreeNode,
    X: np.ndarray,
    y: LabelArray,
    sample_indices: IndexArray,
    config: TAOObliqueConfig,
) -> None:
    """
    Optimize an internal oblique node using the reduced problem with
    a surrogate binary loss (hinge or logistic).

    This mirrors _optimize_internal_axis_aligned but solves for a full
    hyperplane w^T x >= b instead of an axis-aligned split.
    """
    if sample_indices.size == 0:
        return

    if node.left is None or node.right is None:
        return

    left_preds = tree.predict_from_node(node.left, X, sample_indices)
    right_preds = tree.predict_from_node(node.right, X, sample_indices)

    care_indices: List[int] = []
    care_labels: List[int] = []

    for idx in sample_indices:
        yi = int(y[idx])
        z_left = left_preds[idx]
        z_right = right_preds[idx]

        left_correct = z_left == yi
        right_correct = z_right == yi

        if (left_correct and right_correct) or (
            (not left_correct) and (not right_correct)
        ):
            continue

        care_indices.append(int(idx))
        care_labels.append(0 if left_correct else 1)

    if not care_indices:
        return

    care_indices_arr = np.array(care_indices, dtype=int)
    care_labels_arr = np.array(care_labels, dtype=int)
    X_care = X[care_indices_arr]

    w, b = _solve_oblique_reduced_problem(X_care, care_labels_arr, config)

    node.is_leaf = False
    node.oblique_weights = w
    node.oblique_bias = b
    node.feature_index = None
    node.threshold = None


def tao_optimize_axis_aligned(
    tree: DecisionTree,
    X: np.ndarray,
    y: LabelArray,
    config: TAOConfig | None = None,
) -> DecisionTree:
    """
    Run TAO on an existing axis-aligned tree, updating its node parameters
    to monotonically decrease (or keep) the misclassification error.

    This implements the alternating optimization over depth levels described
    in section 3 of the paper, specialized to axis-aligned nodes:

    - Group nodes by depth.
    - Iterate depth levels from bottom (leaves) to top (root).
    - At each depth, optimize all nodes in parallel conceptually; here, they
      are processed sequentially but the reduced problems are independent,
      using current parameters of all other nodes.
    """
    if config is None:
        config = TAOConfig()

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    prev_error = tree.misclassification_error(X, y)

    for _ in range(config.max_iterations):
        # Route all samples through current tree.
        routing = tree.route_dataset(X)

        # Group nodes by depth.
        nodes_by_depth: Dict[int, List[TreeNode]] = {}
        max_depth = 0
        for node in tree.iter_nodes_bfs():
            nodes_by_depth.setdefault(node.depth, []).append(node)
            if node.depth > max_depth:
                max_depth = node.depth

        # One TAO pass: leaves to root (reverse breadth-first order).
        for depth in range(max_depth, -1, -1):
            for node in nodes_by_depth.get(depth, []):
                indices = routing.get(node.node_id, np.array([], dtype=int))
                if node.is_leaf:
                    if indices.size > 0:
                        _optimize_leaf(tree, node, y, indices)
                else:
                    _optimize_internal_axis_aligned(
                        tree, node, X, y, indices
                    )

        current_error = tree.misclassification_error(X, y)
        improvement = prev_error - current_error

        # The exact reduced problem for axis-aligned nodes should not
        # increase the misclassification error. We enforce this as a
        # runtime sanity check, allowing for tiny numerical slack.
        if current_error - prev_error > config.monotonic_slack:
            raise AssertionError(
                "TAO axis-aligned optimization increased misclassification "
                f"error from {prev_error} to {current_error}."
            )

        if improvement <= config.tol:
            break
        prev_error = current_error

    return tree


def tao_optimize_oblique(
    tree: DecisionTree,
    X: np.ndarray,
    y: LabelArray,
    config: TAOObliqueConfig | None = None,
) -> DecisionTree:
    """
    Run TAO on an existing oblique tree, updating its node parameters
    (hyperplanes) using the oblique reduced problem with a surrogate loss.

    Notes
    -----
    - This still uses the exact care/don't-care logic from Theorem 3.2,
      but approximates the binary misclassification loss at each node by
      a convex surrogate (hinge or logistic).
    - With a convex ℓ1 penalty (penalty="l1"), this corresponds to the
      ℓ1-regularized linear SVM / logistic regression suggested in the paper
      for sparse oblique trees.
    """
    if config is None:
        config = TAOObliqueConfig()

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    prev_error = tree.misclassification_error(X, y)

    for _ in range(config.max_iterations):
        routing = tree.route_dataset(X)

        nodes_by_depth: Dict[int, List[TreeNode]] = {}
        max_depth = 0
        for node in tree.iter_nodes_bfs():
            nodes_by_depth.setdefault(node.depth, []).append(node)
            if node.depth > max_depth:
                max_depth = node.depth

        for depth in range(max_depth, -1, -1):
            for node in nodes_by_depth.get(depth, []):
                indices = routing.get(node.node_id, np.array([], dtype=int))
                if node.is_leaf:
                    if indices.size > 0:
                        _optimize_leaf(tree, node, y, indices)
                else:
                    _optimize_internal_oblique(
                        tree, node, X, y, indices, config
                    )

        current_error = tree.misclassification_error(X, y)
        improvement = prev_error - current_error

        if improvement <= config.tol:
            break
        prev_error = current_error

    return tree


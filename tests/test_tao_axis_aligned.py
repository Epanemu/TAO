from __future__ import annotations

import unittest

import numpy as np

from tao.cart import build_cart_tree
from tao.models import TAOAxisAlignedClassifier
from tao.tao import TAOConfig, _optimize_internal_axis_aligned
from tao.tree import DecisionTree, TreeNode


class TestTAOAxisAligned(unittest.TestCase):
    def test_tao_does_not_increase_error(self) -> None:
        """TAO should not increase training misclassification error."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

        tree = build_cart_tree(X, y, max_depth=3, min_samples_leaf=5)
        initial_error = tree.misclassification_error(X, y)

        cfg = TAOConfig(max_iterations=5, tol=0.0)
        tao_tree = TAOAxisAlignedClassifier(
            max_depth=3, min_samples_leaf=5, max_tao_iter=5, tao_tol=0.0
        )
        tao_tree.fit(X, y)

        optimized_error = tao_tree.tree_.misclassification_error(X, y)  # type: ignore[union-attr]

        self.assertLessEqual(
            optimized_error,
            initial_error + 1e-12,
            msg="TAO increased misclassification error.",
        )

    def test_pure_tree_stays_pure(self) -> None:
        """If all labels are identical, TAO should keep a single pure leaf."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 4))
        y = np.zeros(100, dtype=int)

        clf = TAOAxisAlignedClassifier(max_depth=4, min_samples_leaf=2, max_tao_iter=5)
        clf.fit(X, y)

        tree = clf.tree_
        assert tree is not None
        # All leaves must predict 0, and error must be 0.
        for leaf in tree.iter_leaves():
            self.assertTrue(leaf.is_leaf)
            self.assertEqual(leaf.prediction, 0)

        self.assertAlmostEqual(
            tree.misclassification_error(X, y), 0.0, places=12
        )

    def test_dont_care_points_do_not_change_node(self) -> None:
        """
        If all points are 'don't-care' for a node (both children behave
        identically w.r.t. ground truth), then optimizing that node should
        not modify its split parameters.
        """
        # Construct a simple depth-1 tree:
        #   root -> left leaf (label 0), right leaf (label 0)
        root = TreeNode(node_id=0, depth=0)
        left = TreeNode(node_id=1, depth=1, is_leaf=True, prediction=0, parent=root)
        right = TreeNode(node_id=2, depth=1, is_leaf=True, prediction=0, parent=root)
        root.left = left
        root.right = right
        root.is_leaf = False
        root.feature_index = 0
        root.threshold = 0.0

        X = np.array([[-1.0], [0.5], [2.0]])
        y = np.zeros(3, dtype=int)
        tree = DecisionTree(root=root, n_features=1, n_classes=2)

        # All points are 'don't-care' because both children predict label 0,
        # which matches the ground truth y.
        indices = np.arange(X.shape[0], dtype=int)
        old_feature = root.feature_index
        old_threshold = root.threshold

        _optimize_internal_axis_aligned(tree, root, X, y, indices)

        self.assertEqual(root.feature_index, old_feature)
        self.assertEqual(root.threshold, old_threshold)


if __name__ == "__main__":
    unittest.main()


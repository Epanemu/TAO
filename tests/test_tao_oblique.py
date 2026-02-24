from __future__ import annotations

import unittest

import numpy as np

from tao.cart import build_cart_tree
from tao.models import TAOObliqueClassifier
from tao.tao import TAOObliqueConfig, tao_optimize_oblique


class TestTAOOblique(unittest.TestCase):
    def test_oblique_tao_not_worse_than_cart_on_simple_data(self) -> None:
        """On a simple linearly separable problem, oblique TAO should not
        significantly worsen CART's training error."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        cart_tree = build_cart_tree(X, y, max_depth=3, min_samples_leaf=5)
        cart_err = cart_tree.misclassification_error(X, y)

        cfg = TAOObliqueConfig(
            max_iterations=5,
            tol=0.0,
            loss="hinge",
            penalty="l1",
            C=1.0,
        )
        oblique_tree = tao_optimize_oblique(cart_tree, X, y, cfg)
        oblique_err = oblique_tree.misclassification_error(X, y)

        self.assertLessEqual(oblique_err, cart_err + 0.05)

    def test_sparsity_path_monotone_nonzero_weights(self) -> None:
        """As C decreases (stronger ℓ1), number of nonzero weights should
        not increase along the path, on average."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 10))
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] > 0).astype(int)

        from tao.metrics import count_nonzero_weights

        depths = 4
        cart_tree = build_cart_tree(X, y, max_depth=depths, min_samples_leaf=5)
        tree = cart_tree

        C_values = [10.0, 3.0, 1.0, 0.3, 0.1]
        nonzeros = []

        for C in C_values:
            cfg = TAOObliqueConfig(
                max_iterations=5,
                tol=0.0,
                loss="hinge",
                penalty="l1",
                C=C,
            )
            tree = tao_optimize_oblique(tree, X, y, cfg)
            nonzeros.append(count_nonzero_weights(tree))

        # Allow minor non-monotonicity but require last value to be
        # no larger than the first, ensuring sparsity trend.
        self.assertLessEqual(nonzeros[-1], nonzeros[0])


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

"""
MNIST sparse oblique TAO experiments, approximating the evaluation in
section 4 of the NeurIPS 2018 paper.

This script:
- Loads MNIST.
- Builds initial trees of various depths.
- Runs oblique TAO along a path of C values (sparsity levels).
- Logs metrics: train/test error, %nonzero weights, #splits, path stats.

Results are written to a CSV file for later plotting.
"""

import csv
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tao.cart import build_cart_tree
from tao.metrics import (
    compute_path_stats,
    count_internal_nodes,
    count_nonzero_weights,
    total_weight_size,
)
from tao.tao import TAOObliqueConfig, tao_optimize_oblique


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=0, stratify=y
    )
    # Further split training into train/validation if desired.
    return X_train, X_test, y_train, y_test


def run_path_for_depth(
    depth: int,
    C_values: Iterable[float],
    output_csv: Path,
) -> None:
    X_train, X_test, y_train, y_test = load_mnist()

    # Initial CART axis-aligned tree (paper also considers oblique initial trees).
    init_tree = build_cart_tree(
        X_train,
        y_train,
        max_depth=depth,
        min_samples_leaf=5,
    )

    tree = init_tree

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "depth",
                "C",
                "train_error",
                "test_error",
                "num_internal_nodes",
                "num_leaves",
                "num_nonzero",
                "total_weight_size",
                "percent_nonzero",
                "mean_path_len",
                "min_path_len",
                "max_path_len",
                "mean_ops",
            ]
        )

        for C in C_values:
            cfg = TAOObliqueConfig(
                max_iterations=14,
                tol=0.0,
                loss="hinge",
                penalty="l1",
                C=C,
            )
            tree = tao_optimize_oblique(tree, X_train, y_train, cfg)

            train_err = tree.misclassification_error(X_train, y_train)
            test_err = tree.misclassification_error(X_test, y_test)

            num_internal = count_internal_nodes(tree)
            num_leaves = num_internal + 1
            num_nonzero = count_nonzero_weights(tree)
            total_size = total_weight_size(tree)
            percent_nonzero = (
                0.0 if total_size == 0 else 100.0 * num_nonzero / total_size
            )

            path_stats = compute_path_stats(tree, X_train)

            writer.writerow(
                [
                    depth,
                    C,
                    train_err,
                    test_err,
                    num_internal,
                    num_leaves,
                    num_nonzero,
                    total_size,
                    percent_nonzero,
                    path_stats.mean_length,
                    path_stats.min_length,
                    path_stats.max_length,
                    path_stats.mean_ops,
                ]
            )


def main() -> None:
    output_dir = Path("results_mnist")
    output_dir.mkdir(parents=True, exist_ok=True)

    depths = [4, 6, 8, 10, 12]
    # Rough log-spaced C values similar to the paper.
    C_values = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]

    for depth in depths:
        csv_path = output_dir / f"mnist_sparse_oblique_depth{depth}.csv"
        run_path_for_depth(depth, C_values, csv_path)


if __name__ == "__main__":
    main()


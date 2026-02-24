"""
Tree Alternating Optimization (TAO) package.

This exposes high-level estimator classes and core utilities.
"""

from .models import TAOAxisAlignedClassifier
from .tree import DecisionTree, TreeNode

__all__ = [
    "TAOAxisAlignedClassifier",
    "DecisionTree",
    "TreeNode",
]


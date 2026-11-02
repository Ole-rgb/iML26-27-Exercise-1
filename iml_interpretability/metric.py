"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class (e.g. decision trees, random forests and
gradient-boosted trees), so that scores are comparable across models and datasets.
:class:`RelativeNumberOfLeaves` is a minimal example for a single decision tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset
from iml_interpretability.tree import Node, depth, feature_reuse, from_sklearn, max_features_on_path, n_leaves


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    A binary tree of depth ``d`` has at most ``2 ** d`` leaves, so the score lies in ``(0, 1]``, where
    1 means a complete tree. Lower means the tree uses fewer of the decision rules its depth would
    allow. This is only an example: replace it with your own metric, which should go beyond such
    simple size measures, and argue why it captures interpretability.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()


class StructuralInterpretability(InterpretabilityMetric):
    """
    The weights are not tuned or anything, currently we just have some ideas combined to
    a score with is offset by a decision boundary and put through a sigmoid.

    threshold: so that we can have a level of complexity that marks the point, where sigmoid returns .5
        (could also be removed and complexity could be recentered around zero, but that would mean that negative
        complexity exists which is questionable)

    sigmoid: so we get a probability of being interpretable (class 1) or not (class 0).
        We currently use .5 as decision threshold (higher means more interpretable)!

    weights: one weight per entry of :meth:`features`, matched by name. Pass ``weights`` and/or ``threshold``
        to the constructor to override the defaults below.
    """

    name = "structural_interpretability"

    DEFAULT_WEIGHTS: dict[str, float] = {
        "depth": .5,
        "log2_feature_reuse": 0.2,
        "max_features_on_path": .5,
        "log2_n_leaves": .8,
    }

    DEFAULT_THRESHOLD: float = 6.0  # TODO more or less untuned!

    def __init__(self, weights: dict[str, float] | None = None, threshold: float | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS if weights is None else weights)
        self.threshold = self.DEFAULT_THRESHOLD if threshold is None else threshold

    def _features(self, root: Node) -> dict[str, float]:
        """Scaled structural features of the tree, keyed like :attr:`weights`."""
        return {
            "depth": float(depth(root)),
            "log2_feature_reuse": float(np.log2(1 + feature_reuse(root))),
            "max_features_on_path": float(max_features_on_path(root)),
            "log2_n_leaves": float(np.log2(n_leaves(root))),
        }

    def _complexity(self, root: Node) -> float:
        """Weighted sum of :meth:`features`; 0 for a single leaf, unbounded above."""
        features = self._features(root)
        if features.keys() != self.weights.keys():
            raise ValueError(f"weights {sorted(self.weights)} do not match features {sorted(features)}")
        return float(sum(self.weights[name] * value for name, value in features.items()))

    def score(self, model: ClassifierMixin, dataset: Dataset):
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return self.score_tree(from_sklearn(model))

    def score_tree(self, root: Node) -> float:
        return float(1 / (1 + np.exp(-(self.threshold - self._complexity(root)))))

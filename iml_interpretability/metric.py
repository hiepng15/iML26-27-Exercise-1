"""Interpretability metrics for fitted models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The evaluation dataset.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """

    def summarize(self, model: ClassifierMixin, dataset: Dataset) -> dict[str, float]:
        return {self.name: self.score(model, dataset)}

    def summarize_folds(self, folds: list[tuple[ClassifierMixin, Dataset]]) -> dict[str, float]:
        """Average the metric summaries from the validation folds."""
        summaries = [self.summarize(model, dataset) for model, dataset in folds]
        return {
            name: float(np.mean([summary[name] for summary in summaries]))
            for name in summaries[0]
        }


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


class PathCost(InterpretabilityMetric):
    """Count split nodes, giving repeat uses of a feature half weight within each tree."""

    name = "path_cost_p90"
    budgets = (4, 6, 8, 12, 20)

    def per_sample(self, model: ClassifierMixin, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return raw split counts and weighted costs for the same paths."""
        if isinstance(model, DecisionTreeClassifier):
            trees = [model]
        elif isinstance(model, RandomForestClassifier):
            trees = model.estimators_
        elif isinstance(model, GradientBoostingClassifier):
            trees = model.estimators_.flat
        else:
            raise TypeError(f"{type(self).__name__} does not support {type(model).__name__}")

        features = dataset.X.to_numpy(dtype=np.float32)
        split_counts = np.zeros(len(features), dtype=int)
        weighted_costs = np.zeros(len(features))

        for tree in trees:
            paths = tree.decision_path(features, check_input=False)
            node_features = tree.tree_.feature
            for row, (start, end) in enumerate(zip(paths.indptr[:-1], paths.indptr[1:], strict=True)):
                path_features = node_features[paths.indices[start:end]]
                split_features = path_features[path_features >= 0]
                splits = len(split_features)
                repeats = splits - len(set(split_features))
                split_counts[row] += splits
                weighted_costs[row] += splits - 0.5 * repeats

        return split_counts, weighted_costs

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        _, costs = self.per_sample(model, dataset)
        return float(np.quantile(costs, 0.9, method="inverted_cdf"))

    def summarize(self, model: ClassifierMixin, dataset: Dataset) -> dict[str, float]:
        return self.summarize_folds([(model, dataset)])

    def summarize_folds(self, folds: list[tuple[ClassifierMixin, Dataset]]) -> dict[str, float]:
        """Compute budgets and percentiles from all held-out predictions together."""
        split_counts_by_fold, costs_by_fold = zip(
            *(self.per_sample(model, dataset) for model, dataset in folds), strict=True
        )
        split_counts = np.concatenate(split_counts_by_fold)
        costs = np.concatenate(costs_by_fold)
        summary = {
            "split_count_p90": float(np.quantile(split_counts, 0.9, method="inverted_cdf")),
            self.name: float(np.quantile(costs, 0.9, method="inverted_cdf")),
        }
        for budget in self.budgets:
            summary[f"split_count_within_{budget}_pct"] = float(100 * np.mean(split_counts <= budget))
            summary[f"path_cost_within_{budget}_pct"] = float(100 * np.mean(costs <= budget))
        return summary

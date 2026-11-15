"""Evaluate interpretability metrics for every (dataset, model) combination."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, stdev

import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from iml_interpretability.data import Dataset
from iml_interpretability.metric import InterpretabilityMetric


def evaluate(
    datasets: Iterable[Dataset],
    models: dict[str, ClassifierMixin],
    metrics: Iterable[InterpretabilityMetric],
    seed: int = 0,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate each model on the same stratified folds and return one row per model and dataset."""
    metrics = list(metrics)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for dataset in datasets:
        splits = list(splitter.split(dataset.X, dataset.y))
        for model_name, model in models.items():
            accuracies = []
            balanced_accuracies = []
            folds = []
            for train_indices, test_indices in splits:
                test = Dataset(dataset.name, dataset.X.iloc[test_indices], dataset.y.iloc[test_indices])
                fitted = clone(model).fit(dataset.X.iloc[train_indices], dataset.y.iloc[train_indices])
                predictions = fitted.predict(test.X)
                accuracies.append(accuracy_score(test.y, predictions))
                balanced_accuracies.append(balanced_accuracy_score(test.y, predictions))
                folds.append((fitted, test))

            row = {
                "dataset": dataset.name,
                "model": model_name,
                "accuracy": mean(accuracies),
                "balanced_accuracy": mean(balanced_accuracies),
                "balanced_accuracy_std": stdev(balanced_accuracies),
            }
            for metric in metrics:
                row.update(metric.summarize_folds(folds))
            rows.append(row)
    return pd.DataFrame(rows)

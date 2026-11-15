"""Smoke tests that run offline on a synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import Dataset, InterpretabilityMetric, PathCost, RelativeNumberOfLeaves, evaluate, get_models


class _ConstantMetric(InterpretabilityMetric):
    name = "constant"

    def score(self, model, dataset):
        return 1.0


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_evaluate_returns_one_row_per_dataset_and_model(dataset):
    models = get_models()
    results = evaluate([dataset], models, [_ConstantMetric()])
    assert len(results) == len(models)
    assert set(results.columns) == {
        "dataset", "model", "accuracy", "balanced_accuracy", "balanced_accuracy_std", "constant"
    }


def test_relative_number_of_leaves_matches_tree(dataset):
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    score = RelativeNumberOfLeaves().score(model, dataset)
    assert score == model.get_n_leaves() / 2 ** model.get_depth()
    assert 0 < score <= 1


def test_relative_number_of_leaves_rejects_non_tree_models(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        RelativeNumberOfLeaves().score(model, dataset)


def test_path_cost_discounts_repeats_within_each_tree():
    dataset = Dataset(name="example", X=pd.DataFrame({"area": [0, 1, 2]}), y=pd.Series([0, 1, 0]))
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    boosting = GradientBoostingClassifier(n_estimators=2, max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    metric = PathCost()

    tree_splits, tree_costs = metric.per_sample(tree, dataset)
    boosting_splits, boosting_costs = metric.per_sample(boosting, dataset)

    assert tree_splits[1] == 2
    assert tree_costs[1] == 1.5
    assert boosting_splits[1] == 4
    assert boosting_costs[1] == 3
    assert metric.score(tree, dataset) == 1.5

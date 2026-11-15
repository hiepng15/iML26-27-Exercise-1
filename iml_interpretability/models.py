"""Tree models at several levels of complexity."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    models: dict[str, ClassifierMixin] = {
        f"decision_tree_depth_{depth}": DecisionTreeClassifier(max_depth=depth, random_state=seed)
        for depth in (2, 4, 6)
    }
    for count in (2, 3, 5, 10):
        models[f"random_forest_{count}"] = RandomForestClassifier(
            n_estimators=count, max_depth=2, random_state=seed
        )
        models[f"gradient_boosting_{count}"] = GradientBoostingClassifier(
            n_estimators=count, max_depth=2, random_state=seed
        )
    return models

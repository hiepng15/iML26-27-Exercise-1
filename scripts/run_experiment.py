"""Run the interpretability evaluation and store the results as a CSV file.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

from iml_interpretability import PathCost, evaluate, get_models, load_rice

OUTPUT = Path("results/scores.csv")
SEED = 0


def main() -> None:
    results = evaluate(
        datasets=[load_rice()],
        models=get_models(seed=SEED),
        metrics=[PathCost()],
        seed=SEED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    columns = ["model", "balanced_accuracy", "balanced_accuracy_std", "split_count_p90", "path_cost_p90"]
    print(results[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

# Exercise 1: Quantifying Interpretability

## Research question

On the Rice dataset, does counting repeated feature uses as half a point change which tree model offers the best balance between predictive accuracy and reading cost?

## Data and models

I used the [UCI Rice dataset](https://archive.ics.uci.edu/dataset/545/rice%C2%B1cammeo%C2%B1and%C2%B1osmancik) to classify Cammeo and Osmancik grains. The file is stored at `data/Rice_Cammeo_Osmancik.arff`.

I tested single decision trees with maximum depths of 2, 4 and 6. Random forests and gradient boosting used 2, 3, 5 or 10 trees, each with a maximum depth of 2. All models used the same five stratified folds.

## Cost

`Cost = d - 0.5 × (d - u)`, where `d` is the number of split nodes visited and `u` is the number of distinct features on that path. The first use of a feature costs 1 point and each repeat costs 0.5. For an ensemble, I add the costs of its trees.

## Results

| Model | Mean balanced accuracy (%) | SD (percentage points) | Cost at 90% coverage |
|---|---:|---:|---:|
| Decision Tree, depth 2 | 92.27 | 0.59 | 2 |
| Decision Tree, depth 4 | 91.80 | 0.87 | 3 |
| Decision Tree, depth 6 | 91.33 | 1.21 | 5 |
| Random Forest, 2 trees | 91.90 | 0.58 | 3.5 |
| Random Forest, 3 trees | 90.05 | 2.14 | 5.5 |
| Random Forest, 5 trees | 91.83 | 0.75 | 9.5 |
| Random Forest, 10 trees | 91.76 | 1.20 | 19 |
| Gradient Boosting, 2 trees | 89.00 | 1.19 | 4 |
| Gradient Boosting, 3 trees | 90.61 | 1.46 | 5.5 |
| Gradient Boosting, 5 trees | 92.22 | 0.58 | 9.5 |
| Gradient Boosting, 10 trees | 92.18 | 0.42 | 18 |

For each configuration, see `path_cost_p90` and `path_cost_within_*_pct` in `results/scores.csv` for the 90% cost and coverage at each tested budget.

## Conclusion

Discounting repeated features lowered the cost of some models. The depth-2 decision tree still had the highest mean balanced accuracy (92.27%) and the lowest cost under both scoring methods (2 points). Five-tree gradient boosting scored slightly higher than the depth-4 tree (92.22% vs. 91.80%), with a higher reading cost (9.5 vs. 3 points).

## AI use

I used ChatGPT to translate and explain the assignment, improve my English, and help write and test the experiment code.

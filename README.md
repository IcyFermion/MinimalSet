### How to use
Import the moduel from file:
```
from minimal_set import MinimalSetCalc
```

Initialize the calculation class with ``MinimalSetCalc(X, y, model=RandomForestRegressor(random_state=42, n_jobs=1), cpu_cores=1, num_iterations=100,feature_keep_rate=0.5, ranking_metric='feature_importances_')``

Parameters:

- `X`: array-like, sparse matrix of shape (n_samples, n_features). The data to fit. Can be for example a list, or an array. **Required**.
- `y`: array-like of shape (n_samples,) or (n_samples, n_outputs). The target variable to try to predict in the case of supervised learning. **Required**.
- `mode`: learning model object implementing ‘fit’, with feature importance metrics, the object to use to fit the data. **Optional**.
- `cpu_cores`: int, number of CPU cores for parallel processing. default = 1. Use -1 for all available CPU cores. **Optional**.
- `num_iterations`: int, number of random cross validation permutations to produce different minimal set results. **Optional**.
- `feature_keep_rate`: float within range 0 to 1 (non inclusive). The feature keep rate at each feature reduction step, 0.2 means 20 percent of the feature will be kept each time the algorithm tries to reduce the number of features. **Optional**.
- `ranking_metric`: str, attribute name of the feature importance metric in a fitted learning model. **Optional**.
- `cv_fold`: int, number of cross validation fold for each permutation, default = 5. **Optional**.

Execute the calculation with ``execute(output_dir)`` method, default output directory is the current path. Minimal sets from each run will be output into its own csv file.




### How to use
Import the moduel from file:
```
from minimal_set import MinimalSetCalc
```

Initialize the calculation class with ``MinimalSetCalc(X, y, model=RandomForestRegressor(random_state=42, n_jobs=1), cpu_cores=1, num_iterations=100,feature_keep_rate=0.5, ranking_metric='feature_importances_', cv_fold=5, output_dir='./output', size_threshold=10)``

Parameters:

- `X`: array-like, sparse matrix of shape (n_samples, n_features). The data to fit. Can be for example a list, or an array. **Required**.
- `y`: array-like of shape (n_samples,) or (n_outputs, n_samples). The target variable to try to predict in the case of supervised learning. **Required**.
- `target_names`: array-like of shape (n_outputs,). List of target names, for correct output path formatting. **Required**.
- `model`: learning model object implementing ‘fit’, with feature importance metrics, the object to use to fit the data. **Optional**.
- `cpu_cores`: int, number of CPU cores for parallel processing. default = 1. Use -1 for all available CPU cores. **Optional**.
- `num_iterations`: int, number of random cross validation permutations to produce different minimal set results. **Optional**.
- `feature_keep_rate`: float within range 0 to 1 (non inclusive). The feature keep rate at each feature reduction step, 0.2 means 20 percent of the feature will be kept each time the algorithm tries to reduce the number of features. **Optional**.
- `ranking_metric`: str, attribute name of the feature importance metric in a fitted learning model. **Optional**.
- `cv_fold`: int, number of cross validation fold for each permutation, default = 5. **Optional**.
- `output_dir`: str, output directory, default = './output'. **Optional**.
- `size_threshold`: int, size threshold for co-dependency analysis based on minimal set calculations, minimal set smaller than this number will be filtered out, default=10. **Optional**.


Execute the calculation with ``execute(co_dependency_calc=True)`` method, use the boolean parameter `co_dependency_calc` to control weather or not to follow up the minimal set calculation with additional co-dependency analysis. By default, a follow-up co-dependency calculation will be applied to all minimal sets below the size threshold, and the results will be saved in `co_dependency/co_dependency.csv` in the output directory. A table of filtered minimal sets will be returned.




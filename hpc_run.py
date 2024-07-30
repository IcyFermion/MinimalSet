# %%
# Library import
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from minimal_set import MinimalSetCalc

# %%
# variable definition
target_feature_names = ['TotalNUE', 'GrainBiomass']
cpu_cores = 10
output_dir = './output'
data_mat_path = './data/NResponse_features.csv'
cpu_cores = int(sys.argv[1])
data_mat_path = sys.argv[2]
output_dir = sys.argv[3]
target_feature_names = sys.argv[4:]
data_matrix = pd.read_csv('./data/NResponse_features.csv', index_col=0)
# Feature data X in the shape of (n_observations, n_features)
X = data_matrix.iloc[:-2].T
# target data y in the shape of (n_targets, n_observations) or (n_observations,)
y = data_matrix.loc[target_feature_names].values


# %%
# model and calculation setup
model = RandomForestRegressor(random_state=22, n_jobs=1)
test_run = MinimalSetCalc(X, y, target_feature_names, model, cpu_cores=cpu_cores, num_iterations=32, feature_keep_rate=0.5, output_dir=output_dir)

# %%
test_result = test_run.execute()

# %%




# %%
# Library import
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from minimal_set import MinimalSetCalc


# %%
#  loading target and TF lists
cpu_cores = int(sys.argv[1])
data_mat_path = sys.argv[2]
output_dir = sys.argv[3]
target_names_path = sys.argv[4]
feature_names_path = sys.argv[5]
ts_meta_file_path = sys.argv[6]
tf_list_df = pd.read_csv(feature_names_path, names=['tf'])
tf_list = tf_list_df['tf'].values
target_list_df = pd.read_csv(target_names_path, names=['target'])
target_list = target_list_df['target'].value
# load all expression data
ts_df = pd.read_csv(data_mat_path, compression='gzip', index_col=0)

# %%
X = ts_df[tf_list]
y = ts_df[target_list[6:10]]

# %%
# calculation setup
test_run = MinimalSetCalc(X, y, target_list[6:10], 
                          cpu_cores=cpu_cores, 
                          num_iterations=100, 
                          feature_keep_rate=0.5, 
                          is_ts=True, 
                          output_dir=output_dir,
                          ts_meta_df=ts_meta_file_path)

# %%
test_result = test_run.execute()



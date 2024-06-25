import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from functools import reduce
from multiprocessing import Pool, cpu_count



class MinimalSetCalc:
    def __init__(self, 
                 X, 
                 y, 
                 model=RandomForestRegressor(random_state=42, n_jobs=1), 
                 cpu_cores=1, 
                 num_iterations=100, 
                 feature_keep_rate=0.5,
                 ranking_metric='feature_importances_',
                 cv_fold=5):
        self.X = X
        self.y = y
        self.model = model
        if (cpu_cores < 0): self.cpu_cores = cpu_count()
        else: self.cpu_cores = cpu_cores
        self.num_iterations = num_iterations
        self.feature_keep_rate = feature_keep_rate
        self.ranking_metric = ranking_metric
        self.cv_fold = cv_fold
    
    def minimal_set_calc(self, seed):
        print('minimal sets calculation for seed: {}.............'.format(seed))
        X = self.X
        y = self.y
        feature_keep_rate = self.feature_keep_rate
        ranking_metric = self.ranking_metric
        cv_splits = KFold(self.cv_fold, random_state=seed, shuffle=True)
        cv_out = cross_validate(self.model, X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
        base_score = np.mean(cv_out['train_score'])
        feature_list = X.columns
        current_feature_importance = reduce(lambda a, b: a + getattr(b, self.ranking_metric), cv_out['estimator'], 0)
        current_score = base_score
        current_test_score = base_score
        continue_flag = True
        while(continue_flag):
            keep_feature_num = int(len(feature_list)*feature_keep_rate)
            kept_features = feature_list[np.argsort(current_feature_importance)[-1*keep_feature_num:]]
            current_X = X[kept_features]
            current_cv_out = cross_validate(self.model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
            current_score = np.mean(current_cv_out['train_score'])
            if current_score < base_score:
                continue_flag = False
                break
            feature_list = kept_features
            current_feature_importance = reduce(lambda a, b: a + getattr(b, self.ranking_metric), current_cv_out['estimator'], 0)
            current_test_score = np.mean(current_cv_out['test_score'])
            if (len(feature_list) < 4):
                continue_flag = False
                break

        feature_list_index = [X.columns.get_loc(feature) for feature in feature_list]
        
        minimal_features_rmse_list = [current_test_score]
        minimal_features_length_list = [len(feature_list)]
        minimal_features_idx_list = ['; '.join(str(v) for v in feature_list_index)]
        minimal_features_importance_list = ['; '.join(str(v) for v in current_feature_importance/self.cv_fold)]
        lef_over_features = X.columns.difference(feature_list)
        while (len(lef_over_features) > 0):
            minimal_features = lef_over_features
            current_X = X[minimal_features]
            current_cv_out = cross_validate(self.model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
            current_score = np.mean(current_cv_out['train_score'])
            minimal_set_test_score = np.mean(current_cv_out['test_score'])
            current_feature_importance = reduce(lambda a, b: a + getattr(b, self.ranking_metric), current_cv_out['estimator'], 0)
            if (current_score < base_score):
                continue_flag = False
                minimal_set_test_score = 0
                break
            else:
                continue_flag = True
            while (continue_flag):
                keep_feature_num = int(len(feature_list)*feature_keep_rate)
                kept_features = minimal_features[np.argsort(current_feature_importance)[-1*keep_feature_num:]]
                current_X = X[kept_features]
                current_cv_out = cross_validate(self.model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
                current_score = np.mean(current_cv_out['train_score'])
                minimal_set_test_score = np.mean(current_cv_out['test_score'])
                if (current_score < base_score):
                    continue_flag = False
                    break
                minimal_features = kept_features
                current_feature_importance = reduce(lambda a, b: a + getattr(b, self.ranking_metric), current_cv_out['estimator'], 0)
                minimal_set_test_score = np.mean(current_cv_out['test_score'])
                if (len(minimal_features) < 4):
                    continue_flag = False
                    break
            minimal_features_index = [X.columns.get_loc(feature) for feature in minimal_features]
            minimal_features_rmse_list.append(minimal_set_test_score)
            minimal_features_idx_list.append('; '.join(str(v) for v in minimal_features_index))
            minimal_features_length_list.append(len(minimal_features_index))
            minimal_features_importance_list.append('; '.join(str(v) for v in current_feature_importance/self.cv_fold))
            lef_over_features = lef_over_features.difference(minimal_features)

        res = np.array([
            len(feature_list),
            ', '.join(str(v) for v in minimal_features_length_list),
            ', '.join(str(v) for v in minimal_features_rmse_list),
            ', '.join(str(v) for v in minimal_features_idx_list),
            ', '.join(str(v) for v in minimal_features_importance_list),
        ])

        return res
    
    def execute(self, output_dir='.'):
        with Pool(self.cpu_cores) as p:
            result_list = list(tqdm(p.imap(self.minimal_set_calc, range(self.num_iterations)), total=self.num_iterations))
        result_df_list = []
        for result, i in zip(result_list, range(self.num_iterations)):
            result_df = pd.DataFrame(index=range(len(result[1].split(', '))))
            result_df['minimal_set_size'] = [int(j) for j in result[1].split(', ')]
            result_df['minimal_set_idx'] = result[3].split(', ')
            result_df['minimal_set_acc'] = [float(j) for j in result[2].split(', ')]
            result_df['minimal_set_importances'] = result[4].split(', ')
            result_df_list.append(result_df)
            result_df.to_csv('{}/minimal_set_out_{}.csv'.format(output_dir, i), index=False)
        return result_df_list


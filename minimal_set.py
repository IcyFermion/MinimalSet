import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from functools import reduce
from multiprocessing import Pool, cpu_count
from pathlib import Path
from itertools import combinations



class MinimalSetCalc:
    def __init__(self, 
                X, 
                y,
                target_names, 
                model=RandomForestRegressor(random_state=42, n_jobs=1), 
                cpu_cores=1, 
                num_iterations=100, 
                feature_keep_rate=0.5,
                ranking_metric='feature_importances_',
                cv_fold=5,
                output_dir='./output',
                size_threshold=10,
                is_ts=False,
                ts_meta_df=None):
        self.X = X
        if len(y.shape) == 1:
            raise TypeError("Need y to be a pandas dataframe shaped as (n_samples, n_outputs)")
        self.target_names = target_names
        self.model = model
        self.size_threshold = size_threshold
        if (cpu_cores < 0): self.cpu_cores = cpu_count()
        else: self.cpu_cores = cpu_cores
        self.num_iterations = num_iterations
        self.feature_keep_rate = feature_keep_rate
        self.ranking_metric = ranking_metric
        self.cv_fold = cv_fold
        self.output_dir = output_dir
        self.is_ts = is_ts
        if is_ts:
            self.y_list = y
            self.cv_fold = 1
            self.ts_meta_df = pd.read_csv(ts_meta_df, index_col=0)
            self.ts_initialize()
            # if ts_splits is None:
            #     raise TypeError("Need train/test splits index input for time-series data")
            # if repeating_y is None:
            #     raise TypeError("Need repeating y value input for time-series data")

            # self.ts_splits = ts_splits
            # self.repeating_y_list = repeating_y
            # if len(repeating_y.shape) == 1:
            #     self.repeating_y_list = np.array([repeating_y])
        else:
            self.y_list = y.T.values
            self.repeating_y_list = y.T.values


        for name in target_names:
            Path(output_dir+'/'+name).mkdir(parents=True, exist_ok=True)
            Path(output_dir+'/'+name+'/co_dependency').mkdir(parents=True, exist_ok=True)
            Path(output_dir+'/'+name+'/minimal_sets').mkdir(parents=True, exist_ok=True)

    def ts_initialize(self):
        train_source_indices = []
        test_source_indices = []
        train_target_indices = []
        test_target_indices = []
        for ind, row in self.ts_meta_df.iterrows():
            if row['is_end']:
                continue
            next_indicies = row['next_exp_entries'].split(';')
            if row['is_test']:
                test_source_indices.extend([ind]*len(next_indicies))
                test_target_indices.extend(next_indicies)
            else:        
                train_source_indices.extend([ind]*len(next_indicies))
                train_target_indices.extend(next_indicies)
        train_source = self.X.loc[train_source_indices]
        test_source = self.X.loc[test_source_indices]
        self.X = pd.concat([train_source, test_source])
        train_target = self.y_list.loc[train_target_indices]
        test_target = self.y_list.loc[test_target_indices]
        repeating_train_target = self.y_list.loc[train_source_indices]
        repeating_test_target = self.y_list.loc[test_source_indices]
        self.y_list = pd.concat([train_target, test_target]).T.values
        self.repeating_y_list = pd.concat([repeating_train_target, repeating_test_target]).T.values
        self.ts_splits = [(np.arange(len(train_source)), np.arange(len(train_source), len(train_source)+len(test_source)))]

    def minimal_set_calc(self, seed):
        X = self.X
        # dropping self-targeting feature in it exists
        if self.current_target in X.columns:
            X = X.drop(self.current_target, axis=1)
        y = self.y
        repeating_y = self.repeating_y
        feature_keep_rate = self.feature_keep_rate
        # different setup for time-series data:
        if self.is_ts:
            # force using random forest model for now
            model = RandomForestRegressor(random_state=seed, n_jobs=1)
            ranking_metric = self.ranking_metric
            # single train/test split from the initialization input
            cv_splits = self.ts_splits
            # error if repeating y value is used as prediction
            repeating_squared_error = np.square(repeating_y[cv_splits[0][1]]-y[cv_splits[0][1]])
        else:
            model = self.model
            ranking_metric = 'feature_importances_'
            cv_splits = KFold(self.cv_fold, random_state=seed, shuffle=True)
        cv_out = cross_validate(model, X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
        base_score = np.mean(cv_out['train_score'])
        feature_list = X.columns
        current_feature_importance = reduce(lambda a, b: a + getattr(b, ranking_metric), cv_out['estimator'], 0)
        current_score = base_score
        current_test_score = base_score
        # hacky way to get the current testing error
        if self.is_ts:
            current_squared_error =  np.square(cv_out['estimator'][0].predict(X.iloc[cv_splits[0][1]])-y[cv_splits[0][1]])
        continue_flag = True
        while(continue_flag):
            keep_feature_num = int(len(feature_list)*feature_keep_rate)
            kept_features = feature_list[np.argsort(current_feature_importance)[-1*keep_feature_num:]]
            current_X = X[kept_features]
            current_cv_out = cross_validate(model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
            current_score = np.mean(current_cv_out['train_score'])
            if current_score < base_score:
                continue_flag = False
                break
            feature_list = kept_features
            current_feature_importance = reduce(lambda a, b: a + getattr(b, ranking_metric), current_cv_out['estimator'], 0)
            current_test_score = np.mean(current_cv_out['test_score'])
            # hacky way to get the current testing error
            if self.is_ts:
                current_squared_error =  np.square(current_cv_out['estimator'][0].predict(current_X.iloc[cv_splits[0][1]])-y[cv_splits[0][1]])
            if (len(feature_list) < 4):
                continue_flag = False
                break

        feature_list_index = [X.columns.get_loc(feature) for feature in feature_list]
        
        minimal_features_rmse_list = [current_test_score]
        minimal_features_length_list = [len(feature_list)]
        minimal_features_idx_list = ['; '.join(str(v) for v in feature_list_index)]
        minimal_features_importance_list = ['; '.join(str(v) for v in current_feature_importance/self.cv_fold)]

        # stats for comparison with repeating value prediction
        if self.is_ts:
            repeating_comp_ttest = stats.ttest_rel(current_squared_error, repeating_squared_error)
            repeating_value_comp_diff = [repeating_comp_ttest.statistic]
            repeating_value_comp_pval = [repeating_comp_ttest.pvalue]

        lef_over_features = X.columns.difference(feature_list)
        while (len(lef_over_features) > 0):
            minimal_features = lef_over_features
            current_X = X[minimal_features]
            current_cv_out = cross_validate(model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
            current_score = np.mean(current_cv_out['train_score'])
            minimal_set_test_score = np.mean(current_cv_out['test_score'])
            current_feature_importance = reduce(lambda a, b: a + getattr(b, ranking_metric), current_cv_out['estimator'], 0)
            # hacky way to get the current testing error
            if self.is_ts:
                current_squared_error =  np.square(current_cv_out['estimator'][0].predict(current_X.iloc[cv_splits[0][1]])-y[cv_splits[0][1]])
            if (current_score < base_score):
                continue_flag = False
                minimal_set_test_score = 0
                break
            else:
                continue_flag = True
            while (continue_flag):
                keep_feature_num = int(len(minimal_features)*feature_keep_rate)
                kept_features = minimal_features[np.argsort(current_feature_importance)[-1*keep_feature_num:]]
                current_X = X[kept_features]
                current_cv_out = cross_validate(model, current_X, y, cv=cv_splits, n_jobs=1, return_train_score=True, return_estimator=True)
                current_score = np.mean(current_cv_out['train_score'])
                minimal_set_test_score = np.mean(current_cv_out['test_score'])
                if (current_score < base_score):
                    continue_flag = False
                    break
                minimal_features = kept_features
                current_feature_importance = reduce(lambda a, b: a + getattr(b, ranking_metric), current_cv_out['estimator'], 0)
                minimal_set_test_score = np.mean(current_cv_out['test_score'])
                # hacky way to get the current testing error
                if self.is_ts:
                    current_squared_error =  np.square(current_cv_out['estimator'][0].predict(current_X.iloc[cv_splits[0][1]])-y[cv_splits[0][1]])
                if (len(minimal_features) < 4):
                    continue_flag = False
                    break
            minimal_features_index = [X.columns.get_loc(feature) for feature in minimal_features]
            minimal_features_rmse_list.append(minimal_set_test_score)
            minimal_features_idx_list.append('; '.join(str(v) for v in minimal_features_index))
            minimal_features_length_list.append(len(minimal_features_index))
            minimal_features_importance_list.append('; '.join(str(v) for v in current_feature_importance/self.cv_fold))

            # stats for comparison with repeating value prediction
            if self.is_ts:
                repeating_comp_ttest = stats.ttest_rel(current_squared_error, repeating_squared_error)
                repeating_value_comp_diff.append(repeating_comp_ttest.statistic)
                repeating_value_comp_pval.append(repeating_comp_ttest.pvalue)

            lef_over_features = lef_over_features.difference(minimal_features)

        result_df = pd.DataFrame(index=range(len(minimal_features_length_list)))
        result_df['minimal_set_size'] = minimal_features_length_list
        result_df['minimal_set_idx'] = minimal_features_idx_list
        result_df['minimal_set_acc'] = minimal_features_rmse_list
        result_df['minimal_set_importances'] = minimal_features_importance_list
        if self.is_ts:
            result_df['repeating_value_comp_diff'] = repeating_value_comp_diff
            result_df['repeating_value_comp_pval'] = repeating_value_comp_pval
        result_df.to_csv('{}/{}/minimal_sets/minimal_set_out_{}.csv'.format(self.output_dir, self.current_target, seed), index=False)

        filtered_df = result_df[result_df['minimal_set_acc'] > 0]
        filtered_df = filtered_df[filtered_df['minimal_set_size'] < self.size_threshold]
        return filtered_df
    
    def co_dependency_calc(self, filtered_df, target_name, feature_names):
        # co dependency calculation
        co_appearance_dict = {}
        solo_appearance_dict = {}
        for minimal_set_idx in filtered_df['minimal_set_idx']:
            minimal_set = [int(idx) for idx in minimal_set_idx.split('; ')]
            minimal_set.sort()
            for i in minimal_set:
                if i in solo_appearance_dict: 
                    solo_appearance_dict[i] += 1
                else:
                    solo_appearance_dict[i] = 1
            for i, j in combinations(minimal_set, 2):
                if i in co_appearance_dict:
                    if j in co_appearance_dict[i]:
                        co_appearance_dict[i][j] += 1
                    else:
                        co_appearance_dict[i][j] = 1
                else:
                    co_appearance_dict[i] = {}
                    co_appearance_dict[i][j] = 1
        
        # output into a dataframe
        co_dependency_dict = copy.deepcopy(co_appearance_dict)
        co_dependency_name_list = []
        co_dependency_value_list = []
        co_dependency_count_list = []

        for i in co_appearance_dict:
            for j in co_appearance_dict[i]:
                total_appearance = 0.0+solo_appearance_dict[i]+solo_appearance_dict[j]-co_appearance_dict[i][j]
                co_dependency_dict[i][j] = co_appearance_dict[i][j] / total_appearance
                co_dependency_name_list.append(feature_names[i] + '; ' + feature_names[j])
                co_dependency_value_list.append(co_dependency_dict[i][j])
                co_dependency_count_list.append(total_appearance)

        output_df = pd.DataFrame(index=co_dependency_name_list)
        output_df['codependency'] = co_dependency_value_list
        output_df['total_appearance'] = co_dependency_count_list
        output_df = output_df.sort_values('codependency', ascending=False)

        output_df.to_csv(self.output_dir+'/'+target_name+'/co_dependency/co_dependency.csv')

    def execute(self, co_dependency_calc=True):
        filtered_df_list = []
        for target_name, y, repeating_y in zip(self.target_names, self.y_list, self.repeating_y_list):
            self.y = y
            self.repeating_y = repeating_y
            self.current_target = target_name
            X = self.X
            # dropping self-targeting feature in it exists
            if self.current_target in X.columns and self.is_ts:
                X = X.drop(self.current_target, axis=1)
            feature_names = pd.DataFrame(index=X.columns)
            feature_names.to_csv(self.output_dir+'/'+target_name+'/feature_names.tsv', header=False)
            print('Minimal Set calculation for {}... ...'.format(target_name))
            with Pool(self.cpu_cores) as p:
                result_list = list(tqdm(p.imap(self.minimal_set_calc, range(self.num_iterations)), total=self.num_iterations))
            filtered_df = pd.concat(result_list, ignore_index=True)
            filtered_df.to_csv(self.output_dir+'/'+target_name+'/filtered_minimal_sets.csv')
            filtered_df_list.append(filtered_df)
            if co_dependency_calc:
                self.co_dependency_calc(filtered_df, target_name=target_name, feature_names=X.columns)
        return filtered_df_list


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams, SpreadTopKCriterion
from utils_locals.parser import parse

import socket
from parameters import Constant, SHARED_GRID, POSSIBLE_Y_VARIABLE


class FeatureNormalizer:
    """
    Normalizer for engineered Betfair features.
    """

    def __init__(self, predictors_col):
        self.predictors_col = list(predictors_col)

        # fitted params
        self.high_missing_cols = []
        self.medians = {}  # col -> median (for NaN fill)
        self.z_means = {}  # col -> mean for z score
        self.z_stds = {}  # col -> std for z score
        self.log1p_cols = set()  # cols that use log1p before z score

        # groups for reference
        self.mom_cols = []
        self.std_cols = []
        self.count_cols = []
        self.order_dir_mean_cols = []
        self.frac_cols = []
        self.other_z_cols = ['marketBaseRate', 'numberOfActiveRunners', 'local_dow']

        self.fitted = False

    def _detect_groups(self):
        """Detect column groups based on names in self.predictors_col."""
        cols = self.predictors_col

        self.mom_cols = [c for c in cols if "_mom_" in c]
        self.std_cols = [c for c in cols if "_std_" in c]
        self.count_cols = [c for c in cols if c.startswith("qty_count_")]
        self.order_dir_mean_cols = [
            c for c in cols
            if c.startswith("order_is_back_order_is_back_") and c not in self.std_cols
        ]
        self.frac_cols = [c for c in cols if c.endswith("_frac")]
        self.other_z_cols = self.other_z_cols + [c for c in cols if c.endswith("_m0")]

    @staticmethod
    def _zscore_col(series, mean, std):
        if std == 0 or np.isnan(std):
            std = 1.0
        return (series - mean) / std

    def normalize_ins(self, df):
        """Fit normalizer on in sample df and return normalized copy."""
        df = df.copy()

        # 1. basic missing info
        miss = df[self.predictors_col].isna().mean()
        self.high_missing_cols = [c for c in self.predictors_col if miss.get(c, 0.0) > 0.5]

        for c in self.high_missing_cols:
            if c not in df.columns: continue
            ind_col = f"{c}_missing"
            df[ind_col] = df[c].isna().astype("int8")
            if ind_col not in self.predictors_col:
                self.predictors_col.append(ind_col)
            df[c] = df[c].fillna(0)

        for c in self.predictors_col:
            if c not in df.columns: continue
            if df[c].isna().any():
                med = df[c].median()
                self.medians[c] = med
                df[c] = df[c].fillna(med)
            else:
                self.medians[c] = df[c].median()

        # 2. detect groups
        self._detect_groups()

        z_cols = set(
            self.mom_cols + self.std_cols + self.count_cols +
            self.order_dir_mean_cols + self.frac_cols + self.other_z_cols
        )

        # 3. log1p
        self.log1p_cols = set(self.std_cols)
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))

        # 4. fit mean and std
        for c in z_cols:
            if c not in df.columns: continue
            mean_c = df[c].mean()
            std_c = df[c].std(ddof=0)
            self.z_means[c] = mean_c
            self.z_stds[c] = std_c
            df[c] = self._zscore_col(df[c], mean_c, std_c)

        self.fitted = True
        return df

    def normalize_oos(self, df):
        """Normalize out of sample df using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted with normalize_ins first.")

        df = df.copy()

        # Ensure all cols exist
        for c in self.predictors_col:
            if c not in df.columns:
                df[c] = np.nan

        # 1. structural missing
        for c in self.high_missing_cols:
            if c in df.columns:
                ind_col = f"{c}_missing"
                df[ind_col] = df[c].isna().astype("int8")
                df[c] = df[c].fillna(0)

        # 2. remaining NaNs
        for c in self.predictors_col:
            if c not in df.columns: continue
            if df[c].isna().any():
                med = self.medians.get(c, 0.0)
                df[c] = df[c].fillna(med)

        # 3. apply log1p
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))

        # 4. z score
        for c in self.z_means.keys():
            if c not in df.columns: continue
            mean_c = self.z_means[c]
            std_c = self.z_stds[c]
            df[c] = self._zscore_col(df[c], mean_c, std_c)

        return df

    def get_feature_metadata(self):
        """
        Returns a DataFrame summarizing how each feature is handled.
        Contains: median, z-score params (mean/std), and boolean flags for transforms.
        """
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer is not fitted yet.")

        # Create a dictionary to hold row data for each feature
        data = {}

        # Initialize with all known predictors
        for col in self.predictors_col:
            data[col] = {
                'feature': col,
                'fill_median': self.medians.get(col, np.nan),
                'z_mean': self.z_means.get(col, np.nan),
                'z_std': self.z_stds.get(col, np.nan),
                'is_log1p': col in self.log1p_cols,
                'is_high_missing': col in self.high_missing_cols,
                'group': 'other'  # default
            }

        # Tag groups for clarity
        group_map = {
            'momentum': self.mom_cols,
            'std_dev': self.std_cols,
            'count': self.count_cols,
            'order_dir': self.order_dir_mean_cols,
            'fraction': self.frac_cols,
            'misc_z': self.other_z_cols
        }

        for group_name, cols in group_map.items():
            for c in cols:
                if c in data:
                    data[c]['group'] = group_name

        df_meta = pd.DataFrame(list(data.values()))

        # Reorder columns for readability
        cols_order = ['feature', 'group', 'fill_median', 'z_mean', 'z_std', 'is_log1p', 'is_high_missing']
        # add any extra columns that might exist (though unlikely with this logic)
        cols_order += [c for c in df_meta.columns if c not in cols_order]

        return df_meta[cols_order]


if __name__ == '__main__':
    args = parse()
    par = Params()

    if args.b == 0: # lasso model
        print("Lasso model selected", flush=True)
        par.model = LassoModelParams()
        grid = [
            ['model','alpha',np.logspace(-12, 1, 6)],
        ]
        grid = SHARED_GRID + grid
    elif args.b ==1: # random forest
        print("Random Forest model selected", flush=True)
        par.model = RandomForestModelParams()
        grid = [
            ['model','n_estimators',[50,100,1000]],
            ['model','max_depth',[5,10,None]],
        ]
        grid = SHARED_GRID + grid
    elif args.b ==2: # xgboost
        print("XGBoost model selected", flush=True)
        par.model = XGBoostModelParams()
        grid = [
            ['model','n_estimators',[50,100,1000]],
            ['model','max_depth',[3,6,10]],
            ['model','learning_rate',[0.01,0.1,0.2]],
        ]
        # grid = shared_grid + [
        #     ['model', 'n_estimators', [100, 300, 600]],
        #     ['model', 'max_depth', [3, 6]],
        #     ['model', 'learning_rate', [0.01, 0.05, 0.1]],
        #     ['model', 'subsample', [0.6, 0.8, 1.0]],
        #     ['model', 'colsample_bytree', [0.6, 0.8, 1.0]],
        # ]
        grid = SHARED_GRID + grid
    else:
        raise ValueError(f"Unsupported model type: {args.b}")
    par.update_param_grid(grid, args.a)


    if par.grid.t_definition is None:
        load_dir = f'{Constant.RES_DIR}/features'
    else:
        load_dir = f'{Constant.RES_DIR}/features_t{par.grid.t_definition}'
    df = pd.DataFrame()
    to_load = range(10)
    if socket.gethostname() == 'UML-FNQ2JDW1GV': # debugging on local machine
        to_load = [9,6,7,8]
    print('start loading from', load_dir, flush=True)
    print('Range', to_load)
    for i in to_load:
        try:
            df = pd.concat([df, pd.read_parquet(f'{load_dir}/greyhound_au_features_part_{i}.parquet')], ignore_index=False)
        except:
            print('Non-fatal ERROR: Could not read part ', i, flush=True)
    if socket.gethostname() == 'UML-FNQ2JDW1GV': # debugging on local machine
        df = df.dropna(subset='file_name')
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(10000)

    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024 ** 3
    print(f"Python process RAM usage: {ram_gb:.2f} GB")

    # list all the features available at m0 based on the suffix
    suffix_available_at_t0 = [
        "_count_2_1",
        "_count_3_1",
        "_mean_2_1",
        "_mean_3_1",
        "_m0",
        # "_m1",
        # "_m2",
        # "_m3",
        "_mom_2_1",
        "_mom_3_1",
        "_order_is_back_2_1",
        "_order_is_back_3_1",
        "_std_2_1",
        "_std_3_1"
    ]

    predictors_col = [c for c in df.columns if c.endswith(tuple(suffix_available_at_t0))]

    # define all the potential target variables
    suffix_enter = "_m0"
    suffix_leave = "_p1"
    for q_str in ['','_q_100','_q_1000']:
        df[f'delta_avg_odds{q_str}'] = (df[['best_back'+ q_str + suffix_enter, 'best_lay'+q_str + suffix_enter]].mean(axis=1) -
                                        df[['best_back'+ q_str + suffix_leave, 'best_lay'+ q_str + suffix_leave]].mean(axis=1))
        df[f'delta_back_then_lay_odds{q_str}'] = df['best_back'+q_str+suffix_enter]-df['best_lay'+q_str+suffix_leave]
        df[f'delta_lay_then_back_odds{q_str}'] = df['best_lay'+q_str+suffix_enter]-df['best_back'+q_str+suffix_leave]

        df[f'delta_start_limit_back_then_lay_odds{q_str}'] = df['best_back'+q_str+suffix_enter]-df['best_back'+q_str+suffix_leave]
        df[f'delta_start_limit_lay_then_back_odds{q_str}'] = df['best_lay'+q_str+suffix_enter]-df['best_lay'+q_str+suffix_leave]
    df['win'] = 1*df['id'] == -1

    # add features on fraction of qty
    df['total_qty_m1'] = df[['total_back_qty_m1', 'total_lay_qty_m1']].sum(axis=1)
    df['total_qty_m3'] = df[['total_back_qty_m3', 'total_lay_qty_m3']].sum(axis=1)

    col_todo = ['total_qty_m1','total_back_qty_m1','total_lay_qty_m1','total_qty_m3','total_back_qty_m3','total_lay_qty_m3']
    col_frac = []
    for col in col_todo:
        c = col+'_frac'
        df[c] = df[col]/ df.groupby('file_name')[col].transform('sum')
        col_frac.append(c)

    # compute the momentum of the frac columns (m1-m3) and save columns name
    col_frac_mom = []
    for col in [x for x in col_frac if x.endswith('_m1_frac')]:
        c = col.replace('_m1','_mom_3_1')
        df[c] =  df[col] -df[col.replace('_m1','_m3')]
        col_frac_mom.append(c)

    predictors_col = predictors_col+col_frac_mom+col_frac

    # detect the top3 runners by bid-ask
    topX = par.grid.topk_restriction
    df = df.reset_index()
    if par.grid.spread_top_k_criterion == SpreadTopKCriterion.VANILLA:
        df['spread_for_top_k'] = ((df['best_back_m1']) - (df['best_lay_m1'])).abs() # todo explore with miles why it's sometimes neg.
    elif par.grid.spread_top_k_criterion == SpreadTopKCriterion.Q100:
        df['spread_for_top_k'] = ((df['best_back_q_100_m1']) - (df['best_lay_q_100_m1'])).abs()
    ind_top_k = df.groupby('file_name')['spread_for_top_k'].rank(method='min', ascending=False) <= topX

    fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    predictors_col = predictors_col  + fixed_effect_columns


    ins_years = df['marketTime_local'].dt.year.unique()
    ins_years = [int(x) for x in ins_years if (x != par.grid.oos_year) and (x >= par.grid.start_ins_year)]
    ind_ins = df['marketTime_local'].dt.year.isin(ins_years) & ind_top_k
    ind_oos = df['marketTime_local'].dt.year == par.grid.oos_year

    # if criterion is not -1, we drop observation with too high a spread
    if par.grid.spread_restriction > -1:
        ind = df['spread_for_top_k'] <= par.grid.spread_restriction
        ind_ins = ind_ins & ind

    featuresNormalizer = FeatureNormalizer(predictors_col)
    df_ins =  featuresNormalizer.normalize_ins(df.loc[ind_ins,:])
    df_ins.to_parquet(Constant.RES_DIR+'df_ins.parquet')
    par.print_values()
    breakpoint()
    df_oos =  featuresNormalizer.normalize_oos(df.loc[ind_oos,:])

    df_ins.dropna(inplace=True, subset=[par.grid.y_var])

    # df['ym'] = df['marketTime_local'].dt.year*100+df['marketTime_local'].dt.month
    # df.groupby('ym')['file_name'].count()


    if par.model.name_ == 'random_forest':
        print('Start training Random Forest model', flush=True)
        if par.grid.y_var in ['win']:
            model_type = RandomForestClassifier
        else:
            model_type = RandomForestRegressor
        model = model_type(
            n_estimators=par.model.n_estimators,
            max_depth=par.model.max_depth,
            random_state=par.seed,
            n_jobs=-1
        )
    elif par.model.name_ == 'lasso':
        print('Start training Lasso model', flush=True)
        model = Lasso(
            fit_intercept=True,
            alpha=par.model.alpha,
            random_state=par.seed
        )
    elif par.model.name_ == 'xgboost':
        # it's ugly but XGBoost doesn't play nice with mac (but work well on the linux server) so for debugging it's simpler
        from xgboost import XGBRegressor, XGBRFClassifier
        if par.grid.y_var in ['win']:
            model_type = XGBRFClassifier
        else:
            model_type = XGBRegressor
        print('Start training XGBoost model', flush=True)
        model = model_type(
            n_estimators=par.model.n_estimators,
            max_depth=par.model.max_depth,
            learning_rate=par.model.learning_rate,
            random_state=par.seed,
            n_jobs=-1,
            verbosity=0
        )
    else:
        raise ValueError(f"Unsupported model name: {par.model.name_}")
    print('Trained model on in-sample data', flush=True)
    col_save = ['best_back' + suffix_enter, 'best_lay' + suffix_enter] + ['best_back_q_100' + suffix_enter, 'best_lay_q_100' + suffix_enter]
    col_save += ['best_back' + suffix_leave, 'best_lay' + suffix_leave] + ['best_back_q_100' + suffix_leave, 'best_lay_q_100' + suffix_leave]
    id_cols = ['file_name','id']
    if par.grid.y_var not in col_save:
        col_save.append(par.grid.y_var)


    # we do it with the merge to get back the un-normalized odds.
    df_save = df_oos[id_cols+[par.grid.y_var]].copy()

    model.fit(df_ins[predictors_col], df_ins[par.grid.y_var])
    df_save['prediction'] = model.predict(df_oos[predictors_col])
    # This was to have the important columns in the analysis, but it's too inneficient for big grids.
    # df_save = df_save.merge(df[id_cols+col_save], on=id_cols, how='left')

    print(df_save[[par.grid.y_var,'prediction']].corr())
    df_save.to_parquet(par.get_model_grid_dir(old_style =True)+'save_df.parquet')

    try:
        df_norm_meta = featuresNormalizer.get_feature_metadata()
        df_norm_meta.to_parquet(par.get_model_grid_dir(old_style=True) + 'feature_normalization_params.parquet')
        print('Saved feature normalization parameters', flush=True)
    except Exception as e:
        print(f"Warning: Could not save normalization params. Error: {e}", flush=True)

    print('Saved OOS predictions', flush=True)
    if par.model.name_ =='lasso':
        # also save the coefficents associated to each feature
        coef = model.coef_
        df_coef = pd.DataFrame({'feature':predictors_col, 'coefficient':coef})
        df_coef.to_parquet(par.get_model_grid_dir(old_style =True)+'lasso_coefficients.parquet')
        print('Saved Lasso coefficients', flush=True)
    elif par.model.name_ == 'xgboost':
        # save feature importances from XGBoost
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': predictors_col,
            'importance': importances
        }).sort_values('importance', ascending=False)

        df_importance.to_parquet(
            par.get_model_grid_dir(old_style=True) + 'xgboost_feature_importances.parquet'
        )
        print('Saved XGBoost feature importances', flush=True)
        model.save_model(par.get_model_grid_dir(old_style=True) + 'xgboost_model.json')
        print('Saved xgboost model itself', flush=True)
    print('All Saved in ', par.get_model_grid_dir(old_style=True),flush=True)
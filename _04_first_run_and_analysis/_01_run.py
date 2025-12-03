import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
import psutil
import os
from parameters import Params, LassoModelParams, RandomForestModelParams, XGBoostModelParams
from utils_locals.parser import parse

import socket
from parameters import Constant

class FeatureNormalizer:
    """
    Normalizer for engineered Betfair features.

    Usage:
        norm = FeatureNormalizer(predictors_col)
        df_ins_norm = norm.normalize_ins(df_ins)   # fit + transform
        df_oos_norm = norm.normalize_oos(df_oos)   # transform only (OOS)
    """

    def __init__(self, predictors_col):
        self.predictors_col = list(predictors_col)

        # fitted params
        self.high_missing_cols = []
        self.medians = {}          # col -> median (for NaN fill)
        self.z_means = {}          # col -> mean for z score
        self.z_stds = {}           # col -> std for z score
        self.log1p_cols = set()    # cols that use log1p before z score

        # groups for reference
        self.mom_cols = []
        self.std_cols = []
        self.count_cols = []
        self.order_dir_mean_cols = []
        self.frac_cols = []        # *_frac columns (including total/runner fractions)
        self.other_z_cols = ['marketBaseRate','numberOfActiveRunners','local_dow']     # any other columns to z score not in above groups

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
        # all fraction columns (both levels and momentum, e.g. *_m1_frac, *_m3_frac, *_mom_3_1_frac)
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

        # structural missing columns (no trades etc)
        self.high_missing_cols = [c for c in self.predictors_col if miss.get(c, 0.0) > 0.5]

        # add missing indicators and fill structural NaNs with 0
        for c in self.high_missing_cols:
            if c not in df.columns:
                continue
            ind_col = f"{c}_missing"
            df[ind_col] = df[c].isna().astype("int8")
            if ind_col not in self.predictors_col:
                self.predictors_col.append(ind_col)
            df[c] = df[c].fillna(0)

        # fill remaining NaNs with median and store
        for c in self.predictors_col:
            if c not in df.columns:
                continue
            if df[c].isna().any():
                med = df[c].median()
                self.medians[c] = med
                df[c] = df[c].fillna(med)
            else:
                # still store median for OOS consistency
                self.medians[c] = df[c].median()

        # 2. detect groups (after possibly adding *_missing indicators)
        self._detect_groups()

        # z score columns (union of all groups on original predictors)
        # now includes *_frac and *_mom_*_frac columns as well
        z_cols = set(
            self.mom_cols
            + self.std_cols
            + self.count_cols
            + self.order_dir_mean_cols
            + self.frac_cols
            + self.other_z_cols
        )

        # 3. log1p for std features only (both price and qty/prc stds)
        self.log1p_cols = set(self.std_cols)

        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))

        # 4. fit mean and std for z score columns and transform
        for c in z_cols:
            if c not in df.columns:
                continue
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
            raise RuntimeError("FeatureNormalizer must be fitted with normalize_ins before calling normalize_oos.")

        df = df.copy()

        # ensure all predictor cols exist (create if missing)
        for c in self.predictors_col:
            if c not in df.columns:
                # create as NaN so the filling logic handles it
                df[c] = np.nan

        # 1. structural missing: same rule as in sample
        for c in self.high_missing_cols:
            if c in df.columns:
                ind_col = f"{c}_missing"
                df[ind_col] = df[c].isna().astype("int8")
                # do not add to predictors_col here, assumed already there
                df[c] = df[c].fillna(0)

        # 2. remaining NaNs: use stored medians
        for c in self.predictors_col:
            if c not in df.columns:
                continue
            if df[c].isna().any():
                med = self.medians.get(c, 0.0)
                df[c] = df[c].fillna(med)

        # 3. apply log1p to std columns
        for c in self.log1p_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].clip(lower=0))

        # 4. z score using stored means and stds
        z_cols = set(self.z_means.keys())

        for c in z_cols:
            if c not in df.columns:
                continue
            mean_c = self.z_means[c]
            std_c = self.z_stds[c]
            df[c] = self._zscore_col(df[c], mean_c, std_c)

        return df

POSSIBLE_Y_VARIABLE = [
    "delta_avg_odds",
    "delta_back_then_lay_odds",
    "delta_lay_then_back_odds",
    "delta_start_limit_back_then_lay_odds",
    "delta_start_limit_lay_then_back_odds",

    "delta_avg_odds_q_100",
    "delta_back_then_lay_odds_q_100",
    "delta_lay_then_back_odds_q_100",
    "delta_start_limit_back_then_lay_odds_q_100",
    "delta_start_limit_lay_then_back_odds_q_100",

    "win"

    # "delta_avg_odds_q_1000",
    # "delta_back_then_lay_odds_q_1000",
    # "delta_lay_then_back_odds_q_1000",
    # "delta_start_limit_back_then_lay_odds_q_1000",
    # "delta_start_limit_lay_then_back_odds_q_1000"
]


if __name__ == '__main__':
    args = parse()
    par = Params()
    shared_grid = [
        ['grid','start_ins_year',[2000]],
        ['grid','y_var', POSSIBLE_Y_VARIABLE],
        ['grid','topk_restriction', [1,2,3,4]],
        ['grid','t_definition', [0,1]]
    ]

    if args.b == 0: # lasso model
        print("Lasso model selected", flush=True)
        par.model = LassoModelParams()
        grid = [
            ['model','alpha',np.logspace(-12, 1, 6)],
        ]
        grid = shared_grid + grid
    elif args.b ==1: # random forest
        print("Random Forest model selected", flush=True)
        par.model = RandomForestModelParams()
        grid = [
            ['model','n_estimators',[50,100,1000]],
            ['model','max_depth',[5,10,None]],
        ]
        grid = shared_grid + grid
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
        grid = shared_grid + grid
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

    # detect the top3 runners by total liquidity at m1
    topX = par.grid.topk_restriction
    df = df.reset_index()
    ind = df.groupby('file_name')['total_qty_m1'].rank(method='min', ascending=False) <= topX

    fixed_effect_columns = ['local_dow', 'marketBaseRate', 'numberOfActiveRunners']
    predictors_col = predictors_col  + fixed_effect_columns

    # in this first version we don't do any features on df_low and just drop them, but to explore.
    # df_low = df[~ind].copy()
    df = df[ind].copy()

    ins_years = df['marketTime_local'].dt.year.unique()
    ins_years = [int(x) for x in ins_years if (x != par.grid.oos_year) and (x >= par.grid.start_ins_year)]
    ind_ins = df['marketTime_local'].dt.year.isin(ins_years)
    ind_oos = df['marketTime_local'].dt.year == par.grid.oos_year

    featuresNormalizer = FeatureNormalizer(predictors_col)
    df_ins =  featuresNormalizer.normalize_ins(df.loc[ind_ins,:])
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
import bz2
import json
import os
import didipack as didi
import pandas as pd
from parameters import Constant
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd


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
        self.high_missing_cols = [c for c in self.predictors_col if miss[c] > 0.5]

        # add missing indicators and fill structural NaNs with 0
        for c in self.high_missing_cols:
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

        # 2. detect groups
        self._detect_groups()

        # z score columns (union of all groups on original predictors)
        z_cols = set(self.mom_cols + self.std_cols + self.count_cols + self.order_dir_mean_cols)

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

if __name__ == '__main__':
    df = pd.DataFrame()
    for i in [9,6,7,8]:
        df = pd.concat([df, pd.read_parquet(f'res/features/greyhound_au_features_part_{i}.parquet')], ignore_index=False)

    # buy at m0, sell at m1

    # list all the features available at m0 based on the suffix
    suffix_available_at_t0 = [
        "_count_2_1",
        "_count_3_1",
        "_mean_2_1",
        "_mean_3_1",
        # "_m0",
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
    pd.isna(df[predictors_col]).mean()


    # columns to log in the normaliation includes:
    # one with total_

    suffix_enter = "_m0"
    suffix_leave = "_p0"

    df['delta_avg_odds'] = df[['best_back'+suffix_enter, 'best_lay'+suffix_enter]].mean(axis=1)- df[['best_back'+suffix_leave, 'best_lay'+suffix_leave]].mean(axis=1)

    featureNormalizer = FeatureNormalizer(predictors_col)
    df =featureNormalizer.normalize_ins(df)

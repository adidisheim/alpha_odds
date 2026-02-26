"""
Model Engine — loads 88 pre-trained models and runs cross-t super-ensemble inference.

Architecture:
  V1: 7 XGBoost models per t_def (4 t_defs) = 28 models
  V2: 15 (XGBoost + LightGBM) per t_def + isotonic calibrator = 60 models
  Total: 88 model components

  V1_cross = mean(V1_t0, V1_t1, V1_t2, V1_t3)
  V2_cross = mean(V2_t0, V2_t1, V2_t2, V2_t3)
  Final = 0.20 * V1_cross + 0.80 * V2_cross
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import MODELS_DIR, V1_WEIGHT, V2_WEIGHT, V1_TOP_N, V2_TOP_N

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available — V2 will use XGBoost only")


class ModelConfig:
    """Holds a single model's artifacts."""

    def __init__(self, config_name, t_def, version):
        self.config_name = config_name
        self.t_def = t_def
        self.version = version  # "v1" or "v2"
        self.xgb_model = None
        self.lgb_model = None
        self.isotonic = None


class CrossTEnsemble:
    """
    Loads all 88 models and runs the cross-t super-ensemble.

    Model directory structure (from manifest.json):
      models/v1/t{0-3}/{config}/xgboost_model.json
      models/v2/t{0-3}/{config}/xgboost_model.json
      models/v2/t{0-3}/{config}/lightgbm_model.txt
      models/v2/t{0-3}/{config}/isotonic_calibrator.pkl
    """

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.manifest = None
        self.v1_models = {}  # t_def -> list of ModelConfig
        self.v2_models = {}  # t_def -> list of ModelConfig
        self._loaded = False

    def load(self):
        """Load manifest and all model artifacts."""
        manifest_path = self.models_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found at {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        logger.info(f"Loading models from {self.models_dir}")

        total_loaded = 0
        for t_def in range(4):
            t_key = f"t{t_def}"

            # Load V1 models
            v1_configs = self.manifest.get("v1", {}).get(t_key, [])
            self.v1_models[t_def] = []
            for config_name in v1_configs[:V1_TOP_N]:
                mc = self._load_v1_model(config_name, t_def)
                if mc:
                    self.v1_models[t_def].append(mc)
                    total_loaded += 1

            # Load V2 models
            v2_configs = self.manifest.get("v2", {}).get(t_key, [])
            self.v2_models[t_def] = []
            for config_name in v2_configs[:V2_TOP_N]:
                mc = self._load_v2_model(config_name, t_def)
                if mc:
                    self.v2_models[t_def].append(mc)
                    total_loaded += 1

            logger.info(
                f"  t{t_def}: {len(self.v1_models[t_def])} V1 + "
                f"{len(self.v2_models[t_def])} V2 models"
            )

        self._loaded = True
        logger.info(f"Loaded {total_loaded} total model components")

    def _load_v1_model(self, config_name, t_def):
        """Load a single V1 XGBoost model."""
        model_dir = self.models_dir / "v1" / f"t{t_def}" / config_name
        xgb_path = model_dir / "xgboost_model.json"

        if not xgb_path.exists():
            logger.warning(f"V1 model not found: {xgb_path}")
            return None

        mc = ModelConfig(config_name, t_def, "v1")
        mc.xgb_model = XGBClassifier()
        mc.xgb_model.load_model(str(xgb_path))
        return mc

    def _load_v2_model(self, config_name, t_def):
        """Load a single V2 model (XGBoost + optional LightGBM + isotonic)."""
        model_dir = self.models_dir / "v2" / f"t{t_def}" / config_name
        xgb_path = model_dir / "xgboost_model.json"

        if not xgb_path.exists():
            logger.warning(f"V2 XGBoost model not found: {xgb_path}")
            return None

        mc = ModelConfig(config_name, t_def, "v2")
        mc.xgb_model = XGBClassifier()
        mc.xgb_model.load_model(str(xgb_path))

        # Load LightGBM if available
        lgb_path = model_dir / "lightgbm_model.txt"
        if lgb_path.exists() and HAS_LGBM:
            mc.lgb_model = lgb.Booster(model_file=str(lgb_path))

        # Load isotonic calibrator
        iso_path = model_dir / "isotonic_calibrator.pkl"
        if iso_path.exists():
            with open(iso_path, "rb") as f:
                mc.isotonic = pickle.load(f)

        return mc

    def predict(self, feature_results):
        """
        Run the full cross-t super-ensemble.

        Args:
            feature_results: dict from FeatureComputer.compute(), keyed by t_def.
                Each contains 'features_v1', 'features_v2', 'v1_predictors', 'v2_predictors',
                'runner_ids', 'file_name', 'market_prob', 'back_odds'.

        Returns:
            DataFrame with columns:
                file_name, id, model_prob, market_prob, back_odds, edge,
                v1_cross, v2_cross, and per-t_def predictions.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # Find common runner set across all available t_defs
        available_t_defs = sorted(set(feature_results.keys()) & set(self.v1_models.keys()))
        if not available_t_defs:
            logger.warning("No features available for any t_def")
            return pd.DataFrame()

        # Use t_def with most runners as base
        base_t = available_t_defs[0]
        base_res = feature_results[base_t]
        n_runners = len(base_res["runner_ids"])

        result_df = pd.DataFrame({
            "file_name": base_res["file_name"],
            "id": base_res["runner_ids"],
            "market_prob": base_res["market_prob"].values,
            "back_odds": base_res["back_odds"].values,
        })

        # Predict V1 per t_def
        v1_preds = {}
        for t_def in available_t_defs:
            if t_def not in self.v1_models or not self.v1_models[t_def]:
                continue
            fr = feature_results[t_def]
            v1_probs = self._predict_v1_ensemble(
                fr["features_v1"], fr["v1_predictors"], t_def
            )
            if v1_probs is not None:
                v1_preds[t_def] = v1_probs
                result_df[f"V1_t{t_def}"] = v1_probs

        # Predict V2 per t_def
        v2_preds = {}
        for t_def in available_t_defs:
            if t_def not in self.v2_models or not self.v2_models[t_def]:
                continue
            fr = feature_results[t_def]
            v2_probs = self._predict_v2_ensemble(
                fr["features_v2"], fr["v2_predictors"], t_def
            )
            if v2_probs is not None:
                v2_preds[t_def] = v2_probs
                result_df[f"V2_t{t_def}"] = v2_probs

        # Cross-t averages
        if v1_preds:
            v1_cross = np.mean(list(v1_preds.values()), axis=0)
            result_df["v1_cross"] = v1_cross
        else:
            result_df["v1_cross"] = np.nan

        if v2_preds:
            v2_cross = np.mean(list(v2_preds.values()), axis=0)
            result_df["v2_cross"] = v2_cross
        else:
            result_df["v2_cross"] = np.nan

        # Final blend: V1(20%) + V2(80%)
        if v1_preds and v2_preds:
            result_df["model_prob"] = V1_WEIGHT * v1_cross + V2_WEIGHT * v2_cross
        elif v2_preds:
            result_df["model_prob"] = v2_cross
        elif v1_preds:
            result_df["model_prob"] = v1_cross
        else:
            result_df["model_prob"] = np.nan

        # Edge
        result_df["edge"] = result_df["model_prob"] - result_df["market_prob"]

        logger.info(
            f"Ensemble prediction: {n_runners} runners, "
            f"{len(v1_preds)} V1 t_defs, {len(v2_preds)} V2 t_defs"
        )

        return result_df

    def _predict_v1_ensemble(self, features_df, predictor_cols, t_def):
        """
        Run V1 ensemble: average of top-N XGBoost classifiers for a given t_def.

        Returns numpy array of probabilities (one per runner).
        """
        models = self.v1_models.get(t_def, [])
        if not models:
            return None

        # Ensure all predictor columns exist
        X = features_df.copy()
        for c in predictor_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[predictor_cols].fillna(0.0)

        probs = []
        for mc in models:
            try:
                # Use model's own expected features to avoid mismatch
                expected = mc.xgb_model.get_booster().feature_names
                if expected is not None:
                    X_model = X.copy()
                    for c in expected:
                        if c not in X_model.columns:
                            X_model[c] = 0.0
                    X_model = X_model[expected]
                else:
                    X_model = X
                p = mc.xgb_model.predict_proba(X_model)[:, 1]
                probs.append(p)
            except Exception as e:
                logger.warning(f"V1 prediction failed for {mc.config_name} t{t_def}: {e}")

        if not probs:
            return None

        return np.mean(probs, axis=0)

    def _predict_v2_ensemble(self, features_df, predictor_cols, t_def):
        """
        Run V2 ensemble: average of (XGBoost+LightGBM)/2 per config, then isotonic calibration.

        Mirrors _03_win_probability_model_v2.py exactly:
          1. XGBoost predict → xgb_prob
          2. LightGBM predict → lgb_prob
          3. ensemble_prob = 0.5 * xgb + 0.5 * lgb
          4. isotonic calibrate → calibrated_prob
          5. Average calibrated_prob across top-N configs
        """
        models = self.v2_models.get(t_def, [])
        if not models:
            return None

        X = features_df.copy()
        for c in predictor_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[predictor_cols].fillna(0.0)

        calibrated_probs = []
        for mc in models:
            try:
                # XGBoost prediction
                xgb_prob = mc.xgb_model.predict_proba(X)[:, 1]

                # LightGBM prediction (if available)
                if mc.lgb_model is not None:
                    lgb_prob = mc.lgb_model.predict(X)
                    ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob
                else:
                    ensemble_prob = xgb_prob

                # Isotonic calibration (if available)
                if mc.isotonic is not None:
                    cal_prob = mc.isotonic.predict(ensemble_prob)
                else:
                    cal_prob = ensemble_prob

                calibrated_probs.append(cal_prob)
            except Exception as e:
                logger.warning(f"V2 prediction failed for {mc.config_name} t{t_def}: {e}")

        if not calibrated_probs:
            return None

        return np.mean(calibrated_probs, axis=0)

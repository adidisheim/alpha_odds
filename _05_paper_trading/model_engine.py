"""
Model Engine — loads 88 pre-trained models and runs cross-t super-ensemble inference.

Architecture:
  V1: 7 XGBoost models per t_def (4 t_defs) = 28 models
  V2: 15 (XGBoost + LightGBM) per t_def + isotonic calibrator = 60 models
  Total: 88 model components

  V1_cross = mean(V1_t0, V1_t1, V1_t2, V1_t3)
  V2_cross = mean(V2_t0, V2_t1, V2_t2, V2_t3)
  Final = 0.30 * V1_cross + 0.70 * V2_cross
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import MODELS_DIR, V1_WEIGHT, V2_WEIGHT, V1_TOP_N, V2_TOP_N, MODEL_VARIANTS, PRIMARY_VARIANT

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

    def __init__(self, models_dir=None, v1_key="v1", v2_key="v2"):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.v1_key = v1_key  # manifest key and subdirectory for V1 models
        self.v2_key = v2_key  # manifest key and subdirectory for V2 models
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

        logger.info(f"Loading models from {self.models_dir} (v1_key={self.v1_key}, v2_key={self.v2_key})")

        total_loaded = 0
        for t_def in range(4):
            t_key = f"t{t_def}"

            # Load V1 models (using v1_key for manifest lookup and directory)
            v1_configs = self.manifest.get(self.v1_key, {}).get(t_key, [])
            self.v1_models[t_def] = []
            for config_name in v1_configs[:V1_TOP_N]:
                mc = self._load_v1_model(config_name, t_def)
                if mc:
                    self.v1_models[t_def].append(mc)
                    total_loaded += 1

            # Load V2 models (using v2_key for manifest lookup and directory)
            v2_configs = self.manifest.get(self.v2_key, {}).get(t_key, [])
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

    def load_v2_only(self, shared_v1_models, manifest):
        """Load only V2 models, reusing shared V1 models from another ensemble."""
        self.manifest = manifest
        self.v1_models = shared_v1_models

        total_v2 = 0
        for t_def in range(4):
            t_key = f"t{t_def}"
            v2_configs = manifest.get(self.v2_key, {}).get(t_key, [])
            self.v2_models[t_def] = []
            for config_name in v2_configs[:V2_TOP_N]:
                mc = self._load_v2_model(config_name, t_def)
                if mc:
                    self.v2_models[t_def].append(mc)
                    total_v2 += 1
            logger.info(
                f"  {self.v2_key} t{t_def}: {len(self.v2_models[t_def])} V2 models"
            )

        self._loaded = True
        logger.info(f"Loaded {total_v2} V2 models for {self.v2_key}")

    def _load_v1_model(self, config_name, t_def):
        """Load a single V1 XGBoost model."""
        model_dir = self.models_dir / self.v1_key / f"t{t_def}" / config_name
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
        model_dir = self.models_dir / self.v2_key / f"t{t_def}" / config_name
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
            return pd.DataFrame(), []

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
        all_model_details = []
        for t_def in available_t_defs:
            if t_def not in self.v1_models or not self.v1_models[t_def]:
                continue
            fr = feature_results[t_def]
            v1_probs, v1_details = self._predict_v1_ensemble(
                fr["features_v1"], fr["v1_predictors"], t_def
            )
            all_model_details.extend(v1_details)
            if v1_probs is not None:
                v1_preds[t_def] = v1_probs
                result_df[f"V1_t{t_def}"] = v1_probs

        # Predict V2 per t_def
        v2_preds = {}
        for t_def in available_t_defs:
            if t_def not in self.v2_models or not self.v2_models[t_def]:
                continue
            fr = feature_results[t_def]
            v2_probs, v2_details = self._predict_v2_ensemble(
                fr["features_v2"], fr["v2_predictors"], t_def
            )
            all_model_details.extend(v2_details)
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

        # Final blend: V1(30%) + V2(70%)
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

        return result_df, all_model_details

    def _predict_v1_ensemble(self, features_df, predictor_cols, t_def):
        """
        Run V1 ensemble: average of top-N XGBoost classifiers for a given t_def.

        Returns:
            (ensemble_probs, details_list) where details_list contains per-config predictions.
        """
        models = self.v1_models.get(t_def, [])
        if not models:
            return None, []

        # Ensure all predictor columns exist
        X = features_df.copy()
        for c in predictor_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[predictor_cols].fillna(0.0)

        probs = []
        details = []
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
                details.append({
                    "config": mc.config_name,
                    "version": "v1",
                    "t_def": t_def,
                    "predictions": p,
                })
            except Exception as e:
                logger.warning(f"V1 prediction failed for {mc.config_name} t{t_def}: {e}")

        if not probs:
            return None, details

        return np.mean(probs, axis=0), details

    def _predict_v2_ensemble(self, features_df, predictor_cols, t_def):
        """
        Run V2 ensemble: average of (XGBoost+LightGBM)/2 per config, then isotonic calibration.

        Mirrors _03_win_probability_model_v2.py exactly:
          1. XGBoost predict → xgb_prob
          2. LightGBM predict → lgb_prob
          3. ensemble_prob = 0.5 * xgb + 0.5 * lgb
          4. isotonic calibrate → calibrated_prob
          5. Average calibrated_prob across top-N configs

        Returns:
            (ensemble_probs, details_list) where details_list contains per-config predictions.
        """
        models = self.v2_models.get(t_def, [])
        if not models:
            return None, []

        X = features_df.copy()
        for c in predictor_cols:
            if c not in X.columns:
                X[c] = 0.0
        # Deduplicate columns (keep first occurrence) — predictor_cols may have dupes
        unique_cols = list(dict.fromkeys(predictor_cols))
        X = X[unique_cols].fillna(0.0)

        calibrated_probs = []
        details = []
        for mc in models:
            try:
                # Align features to what each model expects
                expected = mc.xgb_model.get_booster().feature_names
                if expected is not None:
                    X_model = X.reindex(columns=expected, fill_value=0.0)
                    X_np = X_model.values
                else:
                    X_np = X.values

                # XGBoost prediction
                xgb_prob = mc.xgb_model.predict_proba(X_np)[:, 1]

                # LightGBM prediction (if available)
                if mc.lgb_model is not None:
                    lgb_expected = mc.lgb_model.feature_name()
                    if lgb_expected:
                        X_lgb_np = X.reindex(columns=lgb_expected, fill_value=0.0).values
                    else:
                        X_lgb_np = X_np
                    lgb_prob = mc.lgb_model.predict(X_lgb_np)
                    ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob
                else:
                    ensemble_prob = xgb_prob

                # Isotonic calibration (if available)
                if mc.isotonic is not None:
                    cal_prob = mc.isotonic.predict(ensemble_prob)
                else:
                    cal_prob = ensemble_prob

                calibrated_probs.append(cal_prob)
                details.append({
                    "config": mc.config_name,
                    "version": "v2",
                    "t_def": t_def,
                    "predictions": cal_prob,
                })
            except Exception as e:
                logger.warning(f"V2 prediction failed for {mc.config_name} t{t_def}: {e}")

        if not calibrated_probs:
            return None, details

        return np.mean(calibrated_probs, axis=0), details


class MultiVariantEnsemble:
    """
    Runs multiple model variants sharing V1 models but with different V2 model sets.

    Produces predictions for each variant (all, n30, n5) from the same feature data,
    enabling diagnostic comparison of which features are stable in live trading.

    V1 models are loaded once and shared across all variants.
    """

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.ensembles = {}  # variant_name -> CrossTEnsemble
        self._loaded = False

    def load(self):
        """Load all variant ensembles. Variants with same v1_key share V1 models."""
        manifest_path = self.models_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found at {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Determine which variants have models available
        available_variants = {}
        for name, cfg in MODEL_VARIANTS.items():
            v1_key = cfg.get("v1_key", "v1")
            v2_key = cfg["v2_key"]
            has_v1 = v1_key in manifest and any(manifest[v1_key].get(f"t{t}", []) for t in range(4))
            has_v2 = v2_key in manifest and any(manifest[v2_key].get(f"t{t}", []) for t in range(4))
            if has_v1 and has_v2:
                available_variants[name] = cfg
            else:
                logger.warning(f"Variant '{name}' (v1={v1_key}, v2={v2_key}) missing in manifest, skipping")

        if not available_variants:
            raise RuntimeError("No model variants available in manifest")

        # Group variants by v1_key to share V1 models where possible
        v1_groups = {}
        for name, cfg in available_variants.items():
            v1_key = cfg.get("v1_key", "v1")
            v1_groups.setdefault(v1_key, []).append(name)

        # Load each V1 group
        for v1_key, variant_names in v1_groups.items():
            # First variant in group loads V1 fully
            first_name = variant_names[0]
            first_cfg = available_variants[first_name]
            ens = CrossTEnsemble(self.models_dir, v1_key=v1_key, v2_key=first_cfg["v2_key"])
            ens.load()
            self.ensembles[first_name] = ens
            shared_v1 = ens.v1_models

            # Other variants in same V1 group share V1
            for name in variant_names[1:]:
                cfg = available_variants[name]
                ens2 = CrossTEnsemble(self.models_dir, v1_key=v1_key, v2_key=cfg["v2_key"])
                ens2.load_v2_only(shared_v1, manifest)
                self.ensembles[name] = ens2

        self._loaded = True
        counts = {name: (sum(len(e.v1_models.get(t, [])) for t in range(4)),
                         sum(len(e.v2_models.get(t, [])) for t in range(4)))
                  for name, e in self.ensembles.items()}
        logger.info(f"MultiVariantEnsemble loaded: { {k: f'{v[0]}V1+{v[1]}V2' for k,v in counts.items()} }")

    @property
    def v1_models(self):
        """Access shared V1 models (for startup diagnostics)."""
        if self.ensembles:
            return next(iter(self.ensembles.values())).v1_models
        return {}

    @property
    def v2_models(self):
        """Access primary variant V2 models (for startup diagnostics)."""
        primary = self.ensembles.get(PRIMARY_VARIANT)
        if primary:
            return primary.v2_models
        if self.ensembles:
            return next(iter(self.ensembles.values())).v2_models
        return {}

    def predict(self, feature_results):
        """
        Run predictions for all variants.

        Returns:
            combined_df: DataFrame with shared columns (file_name, id, market_prob, back_odds,
                         v1_cross) plus per-variant columns (model_prob_{variant}, edge_{variant},
                         v2_cross_{variant}).
            all_model_details: list of per-config prediction details (from primary variant).
            variant_summaries: dict of variant_name -> {n_v1_tdefs, n_v2_tdefs} for diagnostics.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        primary_name = PRIMARY_VARIANT if PRIMARY_VARIANT in self.ensembles else list(self.ensembles.keys())[0]

        # Run primary variant first to get base DataFrame and model details
        primary_ens = self.ensembles[primary_name]
        primary_preds, primary_details = primary_ens.predict(feature_results)

        if primary_preds.empty:
            return pd.DataFrame(), primary_details, {}

        # Start combined DataFrame with shared columns
        combined = primary_preds[["file_name", "id", "market_prob", "back_odds"]].copy()
        if "v1_cross" in primary_preds.columns:
            combined["v1_cross"] = primary_preds["v1_cross"]

        # Add primary variant columns
        combined[f"model_prob_{primary_name}"] = primary_preds["model_prob"]
        combined[f"edge_{primary_name}"] = primary_preds["edge"]
        if "v2_cross" in primary_preds.columns:
            combined[f"v2_cross_{primary_name}"] = primary_preds["v2_cross"]

        # Backward-compat: model_prob and edge point to primary variant
        combined["model_prob"] = primary_preds["model_prob"]
        combined["edge"] = primary_preds["edge"]

        variant_summaries = {}

        # Run other variants
        for name, ens in self.ensembles.items():
            if name == primary_name:
                v1_t = sum(1 for t in range(4) if ens.v1_models.get(t))
                v2_t = sum(1 for t in range(4) if ens.v2_models.get(t))
                variant_summaries[name] = {"n_v1_tdefs": v1_t, "n_v2_tdefs": v2_t}
                continue

            try:
                preds, _ = ens.predict(feature_results)
                if not preds.empty and len(preds) == len(combined):
                    combined[f"model_prob_{name}"] = preds["model_prob"].values
                    combined[f"edge_{name}"] = preds["edge"].values
                    if "v2_cross" in preds.columns:
                        combined[f"v2_cross_{name}"] = preds["v2_cross"].values

                    v1_t = sum(1 for t in range(4) if ens.v1_models.get(t))
                    v2_t = sum(1 for t in range(4) if ens.v2_models.get(t))
                    variant_summaries[name] = {"n_v1_tdefs": v1_t, "n_v2_tdefs": v2_t}
                else:
                    logger.warning(f"Variant '{name}' produced empty or mismatched predictions")
            except Exception as e:
                logger.error(f"Variant '{name}' prediction failed: {e}", exc_info=True)

        logger.info(
            f"MultiVariant prediction: {len(combined)} runners, "
            f"variants={list(self.ensembles.keys())}"
        )

        return combined, primary_details, variant_summaries

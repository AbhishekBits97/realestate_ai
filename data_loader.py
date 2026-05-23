"""
utils/data_loader.py
=====================
Shared data loading, splitting, and preprocessing utilities
used by all models in Module 3.

Provides:
  - load_feature_matrix()     : Load Module 2 output
  - make_train_test_split()   : Stratified split
  - get_feature_groups()      : Feature group definitions
  - scale_features()          : StandardScaler / RobustScaler
  - generate_mock_features()  : Mock data for dev/testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "property": [
        "bhk_count", "area_sqft", "log_area", "floor_number",
        "floor_ratio", "is_ground_floor", "is_top_floor",
        "furnishing_encoded", "transaction_encoded", "is_owner_listing",
    ],
    "price_locality": [
        "locality_mean_price_sqft", "locality_median_price_sqft",
        "locality_listing_count", "locality_price_rank",
        "price_to_locality_median_ratio",
    ],
    "geospatial": [
        "dist_nearest_metro_km", "dist_nearest_hospital_km",
        "dist_nearest_school_km", "dist_nearest_mall_km",
        "dist_nearest_it_hub_km", "dist_nearest_airport_km",
        "metro_count_2km", "hospital_count_3km", "school_count_2km",
        "metro_accessibility_score",
    ],
    "livability": [
        "livability_index", "score_healthcare", "score_education",
        "score_metro", "score_commercial", "score_employment",
    ],
    "builder": [
        "builder_reputation_score", "builder_listing_count",
        "builder_avg_price_sqft", "builder_price_premium_pct",
    ],
    "infrastructure": [
        "infrastructure_impact_score", "impact_metro_extension",
        "impact_highway", "impact_it_sez", "impact_urban_development",
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
TARGET_COL   = "price_per_sqft"


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_mock_features(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic mock feature matrix for Module 3 development.
    Mimics the output of Module 2's feature_pipeline.py.
    """
    np.random.seed(seed)

    bhk         = np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.32, 0.38, 0.17, 0.05])
    area        = bhk * np.random.uniform(430, 620, n)
    livability  = np.random.uniform(20, 95, n)
    builder_rep = np.random.uniform(15, 95, n)
    infra_score = np.random.uniform(10, 90, n)
    dist_metro  = np.random.exponential(2.5, n)
    locality_rank = np.random.uniform(0, 1, n)

    # Price/sqft — realistic model with feature interactions
    base_price = (
        6000
        + bhk * 800
        + area * 2.5
        + livability * 80
        + builder_rep * 40
        + infra_score * 25
        - dist_metro * 300
        + locality_rank * 5000
        + np.random.normal(0, 1200, n)   # noise
    )
    base_price = np.clip(base_price, 4000, 30000)

    df = pd.DataFrame({
        # Property
        "bhk_count"               : bhk,
        "area_sqft"               : area.round(0),
        "log_area"                : np.log1p(area),
        "floor_number"            : np.random.randint(0, 20, n).astype(float),
        "floor_ratio"             : np.random.uniform(0, 1, n).round(3),
        "is_ground_floor"         : (np.random.rand(n) < 0.05).astype(int),
        "is_top_floor"            : (np.random.rand(n) < 0.08).astype(int),
        "furnishing_encoded"      : np.random.choice([0, 1, 2], n, p=[0.35, 0.35, 0.30]),
        "transaction_encoded"     : np.random.choice([0, 1, 2], n, p=[0.60, 0.35, 0.05]),
        "is_owner_listing"        : (np.random.rand(n) < 0.30).astype(int),
        # Locality
        "locality_mean_price_sqft": base_price * np.random.uniform(0.85, 1.15, n),
        "locality_median_price_sqft": base_price * np.random.uniform(0.88, 1.12, n),
        "locality_listing_count"  : np.random.randint(5, 200, n),
        "locality_price_rank"     : locality_rank.round(3),
        "price_to_locality_median_ratio": np.random.uniform(0.7, 1.4, n).round(3),
        # Geospatial
        "dist_nearest_metro_km"   : dist_metro.round(3),
        "dist_nearest_hospital_km": np.random.exponential(1.5, n).round(3),
        "dist_nearest_school_km"  : np.random.exponential(1.2, n).round(3),
        "dist_nearest_mall_km"    : np.random.exponential(3.0, n).round(3),
        "dist_nearest_it_hub_km"  : np.random.exponential(4.0, n).round(3),
        "dist_nearest_airport_km" : np.random.uniform(5, 35, n).round(3),
        "metro_count_2km"         : np.random.randint(0, 5, n),
        "hospital_count_3km"      : np.random.randint(0, 8, n),
        "school_count_2km"        : np.random.randint(0, 6, n),
        "metro_accessibility_score": np.clip(100 * np.exp(-0.5 * dist_metro), 0, 100).round(2),
        # Livability
        "livability_index"        : livability.round(2),
        "score_healthcare"        : np.random.uniform(20, 100, n).round(2),
        "score_education"         : np.random.uniform(20, 100, n).round(2),
        "score_metro"             : np.clip(100 * np.exp(-0.5 * dist_metro), 0, 100).round(2),
        "score_commercial"        : np.random.uniform(15, 95, n).round(2),
        "score_employment"        : np.random.uniform(15, 95, n).round(2),
        # Builder
        "builder_reputation_score": builder_rep.round(2),
        "builder_listing_count"   : np.random.randint(1, 150, n),
        "builder_avg_price_sqft"  : base_price * np.random.uniform(0.9, 1.1, n),
        "builder_price_premium_pct": np.random.uniform(-15, 30, n).round(2),
        # Infrastructure
        "infrastructure_impact_score": infra_score.round(2),
        "impact_metro_extension"  : np.random.uniform(0, 90, n).round(2),
        "impact_highway"          : np.random.uniform(0, 70, n).round(2),
        "impact_it_sez"           : np.random.uniform(0, 85, n).round(2),
        "impact_urban_development": np.random.uniform(0, 80, n).round(2),
        # Target
        "price_per_sqft"          : base_price.round(2),
        "price_inr"               : (base_price * area).round(-3),
        "log_price"               : np.log1p(base_price * area),
        # Meta
        "city"                    : np.random.choice(["Gurgaon", "Delhi", "Noida"], n, p=[0.6, 0.25, 0.15]),
        "locality"                : np.random.choice([
            "Sector 56","DLF Phase 5","Golf Course Road","Sohna Road",
            "Sector 82","MG Road","Cyber City","Palam Vihar","Sector 15",
        ], n),
    })
    logger.info(f"[MockData] Generated {n} feature rows | target mean: ₹{base_price.mean():,.0f}/sqft")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_feature_matrix(path: str = "data/features/feature_matrix.parquet") -> pd.DataFrame:
    """Load Module 2 feature matrix. Falls back to mock data if not found."""
    p = Path(path)
    if p.exists():
        ext = p.suffix
        df  = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
        logger.info(f"Loaded feature matrix: {df.shape} from {path}")
        return df
    logger.warning(f"Feature matrix not found at {path} — using mock data")
    return generate_mock_features(n=1000)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def make_train_test_split(
    df        : pd.DataFrame,
    target    : str = TARGET_COL,
    test_size : float = 0.20,
    val_size  : float = 0.10,
    seed      : int = 42,
) -> Dict:
    """
    Split into train / validation / test sets.
    Returns dict with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = df[features].fillna(df[features].median(numeric_only=True))
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=seed)

    logger.info(f"Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCALER
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_val  : pd.DataFrame,
    X_test : pd.DataFrame,
    method : str = "robust",
) -> Tuple:
    """Fit scaler on train, apply to val/test."""
    scaler = RobustScaler() if method == "robust" else StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_val_sc, X_test_sc, scaler

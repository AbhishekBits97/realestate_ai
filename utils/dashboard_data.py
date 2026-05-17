"""
utils/dashboard_data.py
========================
Central data loader for the Streamlit dashboard.
Loads from Module 2/3/4 outputs or falls back to rich mock data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger


LOCALITIES = [
    "Golf Course Road", "DLF Phase 5", "Sector 56", "Sohna Road",
    "Sector 82", "MG Road", "Cyber City", "Palam Vihar",
    "Sector 15", "Sector 48", "Udyog Vihar", "Nirvana Country",
    "Ardee City", "South City", "New Gurgaon Sector 82",
]

LOCALITY_PROFILES = {
    "Golf Course Road"      : dict(base=17500, roi=52, risk=22, livability=88, grade="A+"),
    "DLF Phase 5"           : dict(base=16800, roi=48, risk=25, livability=85, grade="A+"),
    "Cyber City"            : dict(base=15500, roi=30, risk=28, livability=80, grade="A"),
    "MG Road"               : dict(base=13000, roi=35, risk=30, livability=80, grade="A"),
    "Sector 48"             : dict(base=11500, roi=42, risk=35, livability=74, grade="B+"),
    "Sector 56"             : dict(base=10500, roi=38, risk=35, livability=72, grade="B+"),
    "Udyog Vihar"           : dict(base=10000, roi=33, risk=38, livability=68, grade="B"),
    "Sector 15"             : dict(base=9800,  roi=36, risk=37, livability=70, grade="B"),
    "Palam Vihar"           : dict(base=8500,  roi=40, risk=42, livability=65, grade="B"),
    "Nirvana Country"       : dict(base=9200,  roi=39, risk=40, livability=68, grade="B"),
    "Ardee City"            : dict(base=8800,  roi=38, risk=43, livability=66, grade="B"),
    "South City"            : dict(base=9000,  roi=37, risk=41, livability=67, grade="B"),
    "Sohna Road"            : dict(base=8800,  roi=45, risk=44, livability=68, grade="B"),
    "Sector 82"             : dict(base=7200,  roi=58, risk=55, livability=62, grade="C"),
    "New Gurgaon Sector 82" : dict(base=7000,  roi=62, risk=60, livability=60, grade="C"),
}

GRADE_MAP = {"A+": 6, "A": 5, "B+": 4, "B": 3, "C": 2, "D": 1}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_dashboard_data(
    feature_path: str = "data/features/feature_matrix.parquet",
    scored_path : str = "data/explanations/listings_investment_scored.parquet",
) -> Dict:
    """
    Load all data needed by the dashboard.
    Falls back to rich mock data if files are not found.
    """
    np.random.seed(42)

    # ── Try loading real data
    df = None
    for path in [scored_path, feature_path]:
        if Path(path).exists():
            try:
                df = pd.read_parquet(path)
                logger.info(f"Loaded real data: {path} ({len(df)} rows)")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    if df is None:
        df = _generate_mock_listings(n=800)

    # ── Ensure required columns
    df = _ensure_columns(df)

    # ── Locality summary stats
    locality_stats = _compute_locality_stats(df)

    # ── Time series (monthly price trends per locality)
    time_series = _generate_time_series()

    # ── ROI forecasts
    roi_forecasts = _generate_roi_forecasts()

    # ── Market KPIs
    kpis = _compute_kpis(df)

    # ── SHAP feature importance (mock or real)
    shap_importance = _load_shap_importance()

    return {
        "listings"        : df,
        "locality_stats"  : locality_stats,
        "time_series"     : time_series,
        "roi_forecasts"   : roi_forecasts,
        "kpis"            : kpis,
        "shap_importance" : shap_importance,
        "localities"      : LOCALITIES,
        "locality_profiles": LOCALITY_PROFILES,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _generate_mock_listings(n: int = 800) -> pd.DataFrame:
    np.random.seed(42)
    locality_choices = np.random.choice(LOCALITIES, n, p=[
        0.08, 0.08, 0.09, 0.08, 0.07,
        0.07, 0.05, 0.06, 0.06, 0.07,
        0.05, 0.05, 0.05, 0.07, 0.07,
    ])
    bhk = np.random.choice([1,2,3,4,5], n, p=[0.07,0.30,0.38,0.18,0.07])
    area = bhk * np.random.uniform(430, 620, n)

    price_sqft = np.array([
        LOCALITY_PROFILES.get(loc, dict(base=9500))["base"]
        * np.random.uniform(0.88, 1.15)
        for loc in locality_choices
    ])
    price_inr = price_sqft * area
    price_cr  = price_inr / 1e7

    roi_arr   = np.array([LOCALITY_PROFILES.get(loc, dict(roi=38))["roi"] for loc in locality_choices])
    risk_arr  = np.array([LOCALITY_PROFILES.get(loc, dict(risk=40))["risk"] for loc in locality_choices])
    liv_arr   = np.array([LOCALITY_PROFILES.get(loc, dict(livability=70))["livability"] for loc in locality_choices])
    grade_arr = [LOCALITY_PROFILES.get(loc, dict(grade="B"))["grade"] for loc in locality_choices]

    builders = [
        "DLF Limited", "Godrej Properties", "M3M India", "Signature Global",
        "Hero Homes", "Owner", "Bestech Group", "Emaar India",
        "Sobha Limited", "Gaur Group", "Vatika Limited", "Raheja Group",
    ]

    rec_map = {"A+": "Strong Buy", "A": "Buy", "B+": "Buy", "B": "Hold", "C": "Hold", "D": "Avoid"}

    df = pd.DataFrame({
        "locality"               : locality_choices,
        "city"                   : "Gurgaon",
        "bhk_count"              : bhk,
        "area_sqft"              : area.round(0),
        "price_per_sqft"         : price_sqft.round(0),
        "price_inr"              : price_inr.round(-3),
        "price_cr"               : price_cr.round(2),
        "lat"                    : np.random.uniform(28.38, 28.56, n),
        "lon"                    : np.random.uniform(76.95, 77.12, n),
        "livability_index"       : liv_arr + np.random.normal(0, 4, n),
        "builder_reputation_score": np.random.uniform(30, 95, n).round(1),
        "infrastructure_impact_score": np.random.uniform(20, 88, n).round(1),
        "investment_risk_score"  : risk_arr + np.random.normal(0, 6, n),
        "metro_accessibility_score": np.random.uniform(15, 98, n).round(1),
        "dist_nearest_metro_km"  : np.random.exponential(2.2, n).round(2),
        "dist_nearest_it_hub_km" : np.random.exponential(3.5, n).round(2),
        "dist_nearest_hospital_km": np.random.exponential(1.4, n).round(2),
        "roi_5yr_estimate"       : roi_arr + np.random.normal(0, 5, n),
        "investment_grade"       : grade_arr,
        "recommendation"         : [rec_map.get(g, "Hold") for g in grade_arr],
        "builder_tier"           : np.random.choice(["Tier 1","Tier 2","Tier 3","Individual"], n, p=[0.25,0.35,0.25,0.15]),
        "posted_by"              : np.random.choice(builders, n),
        "furnishing_status"      : np.random.choice(["Furnished","Semi-Furnished","Unfurnished"], n, p=[0.28,0.37,0.35]),
        "transaction_type"       : np.random.choice(["New Property","Resale"], n, p=[0.40,0.60]),
        "floor_ratio"            : np.random.uniform(0, 1, n).round(2),
        "source"                 : np.random.choice(["magicbricks","99acres"], n),
    })

    # Investment score
    df["investment_score"] = (
        df["roi_5yr_estimate"] / 80 * 100 * 0.30 +
        (100 - df["investment_risk_score"]) * 0.25 +
        df["livability_index"] * 0.20 +
        df["builder_reputation_score"] * 0.15 +
        df["infrastructure_impact_score"] * 0.10
    ).clip(0, 100).round(1)

    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing required columns with sensible defaults."""
    defaults = {
        "city"                    : "Gurgaon",
        "bhk_count"               : 3,
        "area_sqft"               : 1200.0,
        "price_per_sqft"          : 10000.0,
        "price_cr"                : 1.2,
        "livability_index"        : 65.0,
        "builder_reputation_score": 55.0,
        "infrastructure_impact_score": 45.0,
        "investment_risk_score"   : 40.0,
        "roi_5yr_estimate"        : 38.0,
        "investment_score"        : 55.0,
        "investment_grade"        : "B",
        "recommendation"          : "Hold",
        "metro_accessibility_score": 50.0,
        "dist_nearest_metro_km"   : 2.5,
        "dist_nearest_it_hub_km"  : 4.0,
        "lat"                     : 28.46,
        "lon"                     : 77.03,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df["investment_risk_score"] = df["investment_risk_score"].clip(0, 100)
    df["livability_index"]      = df["livability_index"].clip(0, 100)
    return df


def _compute_locality_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("locality")
        .agg(
            avg_price_sqft          = ("price_per_sqft", "mean"),
            median_price_sqft       = ("price_per_sqft", "median"),
            listing_count           = ("price_per_sqft", "count"),
            avg_livability          = ("livability_index", "mean"),
            avg_risk                = ("investment_risk_score", "mean"),
            avg_roi                 = ("roi_5yr_estimate", "mean"),
            avg_investment_score    = ("investment_score", "mean"),
            avg_metro_score         = ("metro_accessibility_score", "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("avg_price_sqft", ascending=False)
    )


def _generate_time_series() -> Dict:
    """Generate 36-month price history for each locality."""
    import pandas as pd
    months = pd.date_range(start="2022-01-01", periods=36, freq="MS")
    series = {}
    for loc, profile in LOCALITY_PROFILES.items():
        base   = profile["base"]
        trend  = base * 0.007
        seasonal = np.sin(np.linspace(0, 4*np.pi, 36)) * base * 0.025
        noise  = np.random.normal(0, base * 0.018, 36)
        prices = base + trend * np.arange(36) + seasonal + noise
        series[loc] = pd.DataFrame({"month": months, "avg_price_sqft": prices.round(0)})
    return series


def _generate_roi_forecasts() -> Dict:
    """Generate 12-month forward price forecasts per locality."""
    import pandas as pd
    forecasts = {}
    for loc, profile in LOCALITY_PROFILES.items():
        base  = profile["base"]
        roi_m = profile["roi"] / 100 / 12
        mean  = [base * (1 + roi_m) ** (i+1) for i in range(12)]
        upper = [v * 1.08 for v in mean]
        lower = [v * 0.92 for v in mean]
        forecasts[loc] = {
            "forecast_mean"  : [round(v) for v in mean],
            "forecast_upper" : [round(v) for v in upper],
            "forecast_lower" : [round(v) for v in lower],
            "roi_12m_pct"    : round(profile["roi"] / 5, 1),
            "roi_5yr_est_pct": profile["roi"],
        }
    return forecasts


def _compute_kpis(df: pd.DataFrame) -> Dict:
    return {
        "total_listings"   : len(df),
        "avg_price_sqft"   : round(df["price_per_sqft"].mean(), 0),
        "median_price_cr"  : round(df["price_cr"].median(), 2),
        "avg_livability"   : round(df["livability_index"].mean(), 1),
        "avg_roi_5yr"      : round(df["roi_5yr_estimate"].mean(), 1),
        "low_risk_pct"     : round((df["investment_risk_score"] < 40).mean() * 100, 1),
        "grade_A_plus_pct" : round((df["investment_grade"] == "A+").mean() * 100, 1),
        "localities_covered": df["locality"].nunique(),
    }


def _load_shap_importance() -> pd.DataFrame:
    """
    Load SHAP feature importance from Module 4's property_explanations.json.
    Falls back to labelled mock values only when that file is absent.
    """
    import json
    real_path = Path("data/explanations/property_explanations.json")
    if real_path.exists():
        try:
            with open(real_path) as f:
                explanations = json.load(f)
            shap_sums: dict = {}
            shap_counts: dict = {}
            for prop in explanations.values():
                for factor in prop.get("top_factors", []):
                    feat = factor.get("display_name", factor.get("feature", ""))
                    val  = abs(factor.get("shap_value", 0))
                    shap_sums[feat]   = shap_sums.get(feat, 0) + val
                    shap_counts[feat] = shap_counts.get(feat, 0) + 1
            if shap_sums:
                mean_shap = {k: round(shap_sums[k] / shap_counts[k], 1) for k in shap_sums}
                return pd.DataFrame(
                    {"Feature": list(mean_shap.keys()), "Mean_SHAP": list(mean_shap.values())}
                ).sort_values("Mean_SHAP", ascending=True)
        except Exception:
            pass

    # Labelled fallback — only used when Module 4 has not been run yet
    return pd.DataFrame({
        "Feature": [
            "Hospital Access", "Distance to Metro", "Builder Price Premium",
            "Floor Position", "Area (sqft)", "BHK Count",
            "Infrastructure Impact", "Builder Reputation",
            "Metro Accessibility", "Livability Index", "Locality Price Rank",
        ],
        "Mean_SHAP": [190, 280, 310, 120, 380, 420, 480, 620, 950, 1210, 1840],
    }).sort_values("Mean_SHAP", ascending=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CORRECTION UTILITIES
# Call these to fix wrong scraped values — edits flow through the DataFrame
# so every chart, table and chatbot response updates automatically.
# ─────────────────────────────────────────────────────────────────────────────

def fix_locality_price(df: pd.DataFrame, locality: str, new_median: float) -> pd.DataFrame:
    """
    Rescale all price_per_sqft values in a locality so the median equals
    new_median. Use when the scraper pulled inflated or stale prices.

    Example:
        from utils.dashboard_data import fix_locality_price
        data["listings"] = fix_locality_price(data["listings"], "Sector 56", 11200)
    """
    mask = df["locality"] == locality
    current_med = df.loc[mask, "price_per_sqft"].median()
    if current_med == 0:
        return df
    scale = new_median / current_med
    df    = df.copy()
    df.loc[mask, "price_per_sqft"] *= scale
    df.loc[mask, "price_inr"]      *= scale
    df.loc[mask, "price_cr"]        = df.loc[mask, "price_inr"] / 1e7
    logger.info(f"[Fix] {locality}: prices rescaled ×{scale:.3f} → median ₹{new_median:,}/sqft")
    return df


def fix_roi_estimate(df: pd.DataFrame, locality: str, correct_roi_pct: float) -> pd.DataFrame:
    """
    Override roi_5yr_estimate for a locality with a corrected value.
    Use when the LSTM/Prophet model produced an unrealistic forecast.

    Example:
        data["listings"] = fix_roi_estimate(data["listings"], "Sector 82", 42.0)
    """
    df = df.copy()
    df.loc[df["locality"] == locality, "roi_5yr_estimate"] = correct_roi_pct
    logger.info(f"[Fix] {locality}: roi_5yr_estimate → {correct_roi_pct}%")
    return df


def fix_risk_score(df: pd.DataFrame, locality: str, correct_risk: float) -> pd.DataFrame:
    """
    Override investment_risk_score for a locality.
    Use when the Bayesian model over/under-estimated risk due to sparse data.

    Example:
        data["listings"] = fix_risk_score(data["listings"], "New Gurgaon Sector 82", 50.0)
    """
    df = df.copy()
    df.loc[df["locality"] == locality, "investment_risk_score"] = correct_risk
    logger.info(f"[Fix] {locality}: investment_risk_score → {correct_risk}")
    return df


def remove_outliers(df: pd.DataFrame,
                    col: str = "price_per_sqft",
                    lower_pct: float = 2.0,
                    upper_pct: float = 98.0) -> pd.DataFrame:
    """
    Drop rows where col is outside the [lower_pct, upper_pct] percentile range.
    Use when the scraper captured test listings or data-entry errors.

    Example:
        data["listings"] = remove_outliers(data["listings"], "price_per_sqft", 1, 99)
    """
    lo = df[col].quantile(lower_pct / 100)
    hi = df[col].quantile(upper_pct / 100)
    before = len(df)
    df     = df[(df[col] >= lo) & (df[col] <= hi)].copy()
    logger.info(f"[Fix] Removed {before - len(df)} outliers from '{col}' (kept ₹{lo:,.0f}–₹{hi:,.0f})")
    return df

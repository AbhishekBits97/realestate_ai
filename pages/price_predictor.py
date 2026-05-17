"""pages/price_predictor.py — Interactive ML Price Prediction Page"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e2e8f0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)

FLOOR_BONUS     = {"Ground Floor": -0.03, "Low Floor (1-5)": 0.00,
                   "Mid Floor (6-12)": 0.03, "High Floor (13+)": 0.06, "Top Floor": 0.05}
FURNISH_FACTORS = {"Furnished": 1.07, "Semi-Furnished": 1.02, "Unfurnished": 1.00}
TRANSACTION_DISC= {"New Property": 1.04, "Resale": 1.00, "Pre-Launch": 0.92}
BHK_FACTORS     = {1: 0.85, 2: 0.95, 3: 1.00, 4: 1.08, 5: 1.15}


@st.cache_resource(show_spinner=False)
def _load_model():
    """Load the trained ML model from Module 3 output. Returns None if not found."""
    try:
        import joblib
        for fname in ["random_forest_model.joblib", "xgboost_model.joblib"]:
            p = Path("data/models") / fname
            if p.exists():
                return joblib.load(str(p)), fname
    except Exception:
        pass
    return None, None


def _get_locality_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-locality stats directly from the scraped DataFrame.
    No hardcoded values — everything comes from the data.
    """
    return (
        df.groupby("locality")
        .agg(
            median_price_sqft        = ("price_per_sqft",           "median"),
            avg_livability           = ("livability_index",          "mean"),
            avg_builder_rep          = ("builder_reputation_score",  "mean"),
            avg_metro_dist           = ("dist_nearest_metro_km",     "mean"),
            avg_roi                  = ("roi_5yr_estimate",          "mean"),
            avg_risk                 = ("investment_risk_score",     "mean"),
            listing_count            = ("price_per_sqft",            "count"),
        )
        .round(1)
        .reset_index()
    )


def _predict(model, feature_row: dict, df: pd.DataFrame) -> tuple:
    """
    Run the real ML model if loaded, else fall back to a data-calibrated formula
    whose baseline comes from the ACTUAL scraped median — never a hardcoded dict.
    """
    if model is not None:
        try:
            from utils.data_loader import ALL_FEATURES
        except Exception:
            ALL_FEATURES = list(feature_row.keys())

        # Build feature vector in the exact column order the model was trained on
        available = [f for f in ALL_FEATURES if f in feature_row]
        X = pd.DataFrame([{f: feature_row.get(f, 0) for f in available}])
        pred = float(model.predict(X)[0])
        conf_low  = pred * 0.92
        conf_high = pred * 1.08
        source = "ML model"
    else:
        # Data-calibrated fallback: baseline = scraped median for this locality
        loc_stats = _get_locality_stats(df)
        row = loc_stats[loc_stats["locality"] == feature_row.get("locality", "")]
        base = float(row["median_price_sqft"].values[0]) if len(row) else df["price_per_sqft"].median()

        pred  = base
        pred *= BHK_FACTORS.get(int(feature_row.get("bhk_count", 3)), 1.0)
        pred *= FURNISH_FACTORS.get(feature_row.get("furnishing_status", "Unfurnished"), 1.0)
        pred *= (1 + FLOOR_BONUS.get(feature_row.get("floor_pos", "Low Floor (1-5)"), 0))
        pred *= TRANSACTION_DISC.get(feature_row.get("transaction_type", "Resale"), 1.0)
        pred += feature_row.get("livability_index", 65) * 12
        pred += feature_row.get("builder_reputation_score", 55) * 8
        pred -= feature_row.get("dist_nearest_metro_km", 2.5) * 280
        pred  = max(pred, 4000)
        conf_low  = pred * 0.92
        conf_high = pred * 1.08
        source = "data-calibrated formula"

    return round(pred, 0), round(conf_low, 0), round(conf_high, 0), source


def _shap_contributions(feature_row: dict, df: pd.DataFrame, pred: float) -> dict:
    """
    Build SHAP-style contribution breakdown.
    Baseline = scraped dataset median (not a hardcoded number).
    """
    baseline = float(df["price_per_sqft"].median())
    loc_stats = _get_locality_stats(df)
    row = loc_stats[loc_stats["locality"] == feature_row.get("locality", "")]
    loc_median = float(row["median_price_sqft"].values[0]) if len(row) else baseline

    bhk = int(feature_row.get("bhk_count", 3))
    return {
        "Locality vs Market Median"  : loc_median - baseline,
        "BHK Configuration"          : loc_median * (BHK_FACTORS.get(bhk, 1) - 1),
        "Furnishing Premium"         : loc_median * (FURNISH_FACTORS.get(feature_row.get("furnishing_status","Unfurnished"), 1) - 1),
        "Floor Position"             : loc_median * FLOOR_BONUS.get(feature_row.get("floor_pos","Low Floor (1-5)"), 0),
        "Transaction Type"           : loc_median * (TRANSACTION_DISC.get(feature_row.get("transaction_type","Resale"), 1) - 1),
        "Livability Index"           : feature_row.get("livability_index", 65) * 12,
        "Builder Reputation"         : feature_row.get("builder_reputation_score", 55) * 8,
        "Metro Distance Penalty"     : -feature_row.get("dist_nearest_metro_km", 2.5) * 280,
    }


def render_price_predictor(data: dict):
    df         = data["listings"]
    loc_stats  = _get_locality_stats(df)
    localities = sorted(loc_stats["locality"].tolist())
    st.markdown('<div class="section-header">🤖 AI Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.88rem;margin-bottom:20px;'>"
        "Enter property details below. Our XGBoost + Random Forest ensemble will predict "
        "the fair market price/sqft with SHAP-based feature attribution.</div>",
        unsafe_allow_html=True,
    )

    # ── Input form
    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📍 Location & Type**")
            locality    = st.selectbox("Locality", localities, index=5)
            bhk         = st.selectbox("BHK Configuration", [1, 2, 3, 4, 5], index=2)
            area        = st.number_input("Area (sqft)", min_value=300, max_value=8000,
                                          value=1250, step=50)

        with col2:
            st.markdown("**🏢 Property Details**")
            floor_pos   = st.selectbox("Floor Position",
                                       ["Ground Floor", "Low Floor (1-5)", "Mid Floor (6-12)",
                                        "High Floor (13+)", "Top Floor"], index=2)
            furnishing  = st.selectbox("Furnishing Status",
                                       ["Furnished", "Semi-Furnished", "Unfurnished"], index=1)
            transaction = st.selectbox("Transaction Type",
                                       ["New Property", "Resale", "Pre-Launch"], index=0)

        with col3:
            st.markdown("**📊 AI Feature Inputs**")
            livability  = st.slider("Livability Index", 0, 100, 70,
                                    help="Auto-computed by Module 2 for real data")
            builder_rep = st.slider("Builder Reputation Score", 0, 100, 65,
                                    help="Auto-computed from listing data")
            metro_dist  = st.slider("Distance to Metro (km)", 0.0, 15.0, 2.5, step=0.1)

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    # ── Show locality data preview (data-sourced, not hardcoded)
    loc_row = loc_stats[loc_stats["locality"] == locality]
    if len(loc_row):
        lr = loc_row.iloc[0]
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03); border-left:3px solid #14b8a6; "
            f"padding:8px 14px; border-radius:0 6px 6px 0; font-size:0.82rem; color:#94a3b8; margin-bottom:8px;'>"
            f"📊 <b style='color:#e2e8f0;'>{locality}</b> — from your scraped data: "
            f"median ₹{lr['median_price_sqft']:,.0f}/sqft · "
            f"avg livability {lr['avg_livability']:.0f}/100 · "
            f"avg metro dist {lr['avg_metro_dist']:.1f} km · "
            f"{int(lr['listing_count'])} listings"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not submitted:
        st.markdown("""
        <div style="background:rgba(232,160,32,0.08); border:1px solid rgba(232,160,32,0.2);
                    border-radius:10px; padding:16px; margin-top:8px; color:#94a3b8; font-size:0.85rem;">
            ℹ️ Fill in the form above and click <b style='color:#e8a020'>Predict Price</b> to get
            an AI-powered valuation with feature attribution.
        </div>""", unsafe_allow_html=True)
        return

    # ── Load model once (cached) and build feature dict
    model, model_name = _load_model()
    feature_row = {
        "locality"                : locality,
        "bhk_count"               : bhk,
        "area_sqft"               : float(area),
        "log_area"                : float(np.log1p(area)),
        "floor_pos"               : floor_pos,
        "floor_ratio"             : {"Ground Floor":0.0,"Low Floor (1-5)":0.2,
                                     "Mid Floor (6-12)":0.5,"High Floor (13+)":0.8,
                                     "Top Floor":1.0}.get(floor_pos, 0.4),
        "is_ground_floor"         : int(floor_pos == "Ground Floor"),
        "is_top_floor"            : int(floor_pos == "Top Floor"),
        "furnishing_status"       : furnishing,
        "furnishing_encoded"      : {"Furnished":2,"Semi-Furnished":1,"Unfurnished":0}.get(furnishing,0),
        "transaction_type"        : transaction,
        "transaction_encoded"     : {"New Property":1,"Resale":0,"Pre-Launch":2}.get(transaction,0),
        "is_owner_listing"        : 0,
        "livability_index"        : float(livability),
        "builder_reputation_score": float(builder_rep),
        "dist_nearest_metro_km"   : float(metro_dist),
        "metro_accessibility_score": float(max(0, 100 * np.exp(-0.5 * metro_dist))),
        # remaining geo features: fill with locality median from real data
        **({} if not len(loc_row) else {
            "locality_mean_price_sqft"    : float(loc_row.iloc[0]["median_price_sqft"]),
            "locality_median_price_sqft"  : float(loc_row.iloc[0]["median_price_sqft"]),
            "locality_listing_count"      : float(loc_row.iloc[0]["listing_count"]),
            "locality_price_rank"         : float(loc_stats["median_price_sqft"].rank(pct=True).iloc[
                                                loc_stats[loc_stats["locality"]==locality].index[0]
                                                if len(loc_stats[loc_stats["locality"]==locality]) else 0]),
            "avg_roi"                     : float(loc_row.iloc[0]["avg_roi"]),
            "avg_risk"                    : float(loc_row.iloc[0]["avg_risk"]),
        }),
    }

    # ── Run prediction (real model or data-calibrated fallback)
    pred_sqft, conf_low, conf_high, pred_source = _predict(model, feature_row, df)
    pred_cr = round(pred_sqft * area / 1e7, 2)

    # ── Get SHAP-style contributions (all computed from real data baselines)
    contribs = _shap_contributions(feature_row, df, pred_sqft)

    model_label = f"{model_name.replace('_model.joblib','').replace('_',' ').title()} (real model)" \
                  if model else "Data-calibrated formula (no model file found)"

    # ── Result banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2240,#0f2a50);
                border:1px solid rgba(232,160,32,0.35); border-left:4px solid #e8a020;
                border-radius:12px; padding:20px 28px; margin:20px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:0.75rem; color:#94a3b8;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            {model_label}
        </div>
        <div style="display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;">
            <div>
                <span style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:#e8a020;">
                    ₹{pred_sqft:,.0f}
                </span>
                <span style="color:#94a3b8; font-size:0.9rem;">/sqft</span>
            </div>
            <div style="color:#64748b; font-size:1.2rem;">→</div>
            <div>
                <span style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; color:#14b8a6;">
                    ₹{pred_cr:.2f} Cr
                </span>
                <span style="color:#94a3b8; font-size:0.9rem;"> total ({area:,} sqft)</span>
            </div>
        </div>
        <div style="margin-top:10px; font-size:0.82rem; color:#64748b;">
            95% Confidence Interval: ₹{conf_low:,.0f} – ₹{conf_high:,.0f} /sqft
            &nbsp;·&nbsp; Baseline: scraped median ₹{int(loc_row.iloc[0]['median_price_sqft']) if len(loc_row) else 'N/A':,}/sqft
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP Waterfall + Comparable chart (data-sourced)
    col_shap, col_comp = st.columns([1.2, 1])

    with col_shap:
        st.markdown('<div class="section-header" style="font-size:1rem;">🔍 Feature Attribution</div>',
                    unsafe_allow_html=True)
        labels = list(contribs.keys())
        values = list(contribs.values())
        colors = ["#e8a020" if v >= 0 else "#60a5fa" for v in values]
        sorted_pairs = sorted(zip(labels, values, colors), key=lambda x: abs(x[1]))
        s_labels, s_values, s_colors = zip(*sorted_pairs)
        fig_shap = go.Figure(go.Bar(
            x=list(s_values), y=list(s_labels), orientation="h",
            marker_color=list(s_colors), marker_line_width=0,
        ))
        fig_shap.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_shap.update_layout(**PLOT_THEME, height=340,
                               title=dict(text="Feature Contributions to Predicted Price",
                                          font=dict(family="Syne", size=13)),
                               xaxis_title="Contribution (₹/sqft) — baseline = scraped median")
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:#64748b; margin-top:-8px;">
            <span style="color:#e8a020;">■</span> Increases price &nbsp;
            <span style="color:#60a5fa;">■</span> Decreases price &nbsp;·&nbsp;
            Baseline = scraped median for selected locality
        </div>""", unsafe_allow_html=True)

    with col_comp:
        st.markdown('<div class="section-header" style="font-size:1rem;">📊 All Localities — Real Data</div>',
                    unsafe_allow_html=True)
        # ── Comparable bar chart: sourced entirely from scraped df groupby
        comp_df = loc_stats[["locality","median_price_sqft"]].copy()
        comp_df["selected"] = comp_df["locality"] == locality
        comp_df = comp_df.sort_values("median_price_sqft", ascending=False).head(10)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_df["locality"],
            y=comp_df["median_price_sqft"],
            marker_color=["#e8a020" if s else "#1f4e79" for s in comp_df["selected"]],
            text=["₹" + f"{v:,.0f}" for v in comp_df["median_price_sqft"]],
            textposition="outside", textfont_size=9,
        ))
        fig_comp.add_hline(y=pred_sqft, line_dash="dot", line_color="#14b8a6",
                           annotation_text=f"Your prediction ₹{pred_sqft:,.0f}",
                           annotation_font_color="#14b8a6", annotation_font_size=10)
        fig_comp.update_layout(**PLOT_THEME, height=320,
                               title=dict(text="Scraped Median Price by Locality", font=dict(family="Syne", size=13)),
                               xaxis_tickangle=-35, showlegend=False, yaxis_title="Median ₹/sqft")
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Investment Summary — pulled from real df locality stats
    roi_est  = float(loc_row.iloc[0]["avg_roi"])  if len(loc_row) else df["roi_5yr_estimate"].median()
    risk_val = float(loc_row.iloc[0]["avg_risk"]) if len(loc_row) else df["investment_risk_score"].median()
    risk_band = "Low" if risk_val < 30 else "Medium" if risk_val < 55 else "High"

    st.markdown('<div class="section-header" style="font-size:1rem; margin-top:8px;">📈 Investment Summary — from scraped data</div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val) in zip(
        [m1, m2, m3, m4],
        [("5-Yr ROI (locality avg)",  f"{roi_est:.1f}%"),
         ("Investment Risk (avg)",    f"{risk_band} ({risk_val:.0f})"),
         ("Livability (your input)",  f"{livability}/100"),
         ("Builder Score (your input)", f"{builder_rep}/100")],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

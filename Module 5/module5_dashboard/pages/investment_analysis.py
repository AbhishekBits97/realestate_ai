"""pages/investment_analysis.py"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOT_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="DM Sans", color="#e2e8f0", size=12),
                  margin=dict(l=20, r=20, t=40, b=20),
                  xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                  yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))

def render_investment_analysis(data: dict):
    df        = data["listings"]
    loc_stats = data["locality_stats"]
    forecasts = data["roi_forecasts"]

    st.markdown('<div class="section-header">💰 Investment Analysis</div>', unsafe_allow_html=True)

    # ── Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_risk  = st.slider("Max Risk Score", 0, 100, 60, key="ia_risk")
    with col2:
        min_roi   = st.slider("Min 5-Year ROI %", 0, 80, 30, key="ia_roi")
    with col3:
        min_score = st.slider("Min Investment Score", 0, 100, 40, key="ia_score")

    filtered = df[
        (df["investment_risk_score"] <= max_risk) &
        (df["roi_5yr_estimate"] >= min_roi) &
        (df["investment_score"] >= min_score)
    ]
    st.markdown(f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:16px;'>"
                f"Showing <b style='color:#e8a020'>{len(filtered):,}</b> of {len(df):,} properties</div>",
                unsafe_allow_html=True)

    # ── Risk-Return scatter
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(filtered.sample(min(400, len(filtered))),
                         x="investment_risk_score", y="roi_5yr_estimate",
                         color="investment_grade", size="price_cr",
                         hover_name="locality",
                         color_discrete_map={"A+":"#059669","A":"#2563eb","B+":"#16a34a",
                                             "B":"#d97706","C":"#ea580c","D":"#dc2626"},
                         labels={"investment_risk_score":"Risk Score","roi_5yr_estimate":"5-Yr ROI %"},
                         size_max=25)
        fig.update_layout(**PLOT_THEME, height=380,
                          title=dict(text="Risk vs ROI (size = price)", font=dict(family="Syne",size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # ROI forecast for selected locality
        sel_loc = st.selectbox("Forecast locality", sorted(forecasts.keys()), key="ia_loc")
        fc = forecasts[sel_loc]
        months = list(range(1, 13))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=fc["forecast_upper"],
                                  fill=None, mode="lines", line=dict(width=0),
                                  showlegend=False))
        fig2.add_trace(go.Scatter(x=months, y=fc["forecast_lower"],
                                  fill="tonexty", mode="lines", line=dict(width=0),
                                  fillcolor="rgba(232,160,32,0.12)", showlegend=False))
        fig2.add_trace(go.Scatter(x=months, y=fc["forecast_mean"],
                                  mode="lines+markers", name="Forecast",
                                  line=dict(color="#e8a020", width=2.5)))
        fig2.update_layout(**PLOT_THEME, height=340,
                           title=dict(text=f"12-Month Price Forecast — {sel_loc}", font=dict(family="Syne",size=13)),
                           xaxis_title="Month Ahead", yaxis_title="₹/sqft",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        roi_val = fc.get("roi_5yr_est_pct", 40)
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-top:4px;">
            <div class="metric-card" style="flex:1;padding:12px;">
                <div class="metric-value" style="font-size:1.4rem;">₹{fc['forecast_mean'][-1]:,}</div>
                <div class="metric-label">12m Price Forecast</div>
            </div>
            <div class="metric-card" style="flex:1;padding:12px;">
                <div class="metric-value" style="font-size:1.4rem;">{roi_val}%</div>
                <div class="metric-label">5-Year ROI Est.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Top opportunities table
    st.markdown('<div class="section-header" style="font-size:1rem;margin-top:24px;">🏆 Top Investment Opportunities</div>', unsafe_allow_html=True)
    top = filtered.nlargest(15, "investment_score")
    display_cols = ["locality","bhk_count","price_cr","investment_score","investment_grade",
                    "recommendation","roi_5yr_estimate","investment_risk_score","livability_index"]
    available = [c for c in display_cols if c in top.columns]
    rename = {"bhk_count":"BHK","price_cr":"Price (Cr)","investment_score":"Inv. Score",
               "investment_grade":"Grade","recommendation":"Rec.","roi_5yr_estimate":"5yr ROI %",
               "investment_risk_score":"Risk","livability_index":"Livability"}
    st.dataframe(top[available].rename(columns=rename).reset_index(drop=True),
                 use_container_width=True, height=380)

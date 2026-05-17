"""pages/market_overview.py — Market Overview Dashboard Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         = dict(family="DM Sans", color="#e2e8f0", size=12),
    margin       = dict(l=20, r=20, t=40, b=20),
    xaxis        = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis        = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)

def metric_card(label, value, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls  = "delta-up" if delta >= 0 else "delta-down"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="metric-delta {cls}">{sign} {abs(delta)}{delta_label}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""


def render_market_overview(data: dict):
    kpis     = data["kpis"]
    loc_stats= data["locality_stats"]
    ts       = data["time_series"]
    df       = data["listings"]

    st.markdown('<div class="section-header">📊 Gurgaon NCR — Market Overview</div>', unsafe_allow_html=True)

    # ── KPI Row
    cols = st.columns(4)
    kpi_cards = [
        ("Total Listings Analysed", f"{kpis['total_listings']:,}", None, ""),
        ("Avg Price / sqft",        f"₹{kpis['avg_price_sqft']:,.0f}", 8.2, "% YoY"),
        ("Avg 5-Year ROI Est.",     f"{kpis['avg_roi_5yr']:.1f}%", 3.1, "%"),
        ("Avg Livability Index",    f"{kpis['avg_livability']:.0f}/100", 2.4, " pts"),
    ]
    for col, (label, val, delta, dlabel) in zip(cols, kpi_cards):
        with col:
            st.markdown(metric_card(label, val, delta, dlabel), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Locality Price Heatmap + Bar Chart
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="section-header" style="font-size:1rem;">🗺️ Price Heatmap — Locality</div>', unsafe_allow_html=True)
        fig = px.bar(
            loc_stats.sort_values("avg_price_sqft"),
            x="avg_price_sqft", y="locality",
            orientation="h",
            color="avg_price_sqft",
            color_continuous_scale=[[0,"#1f4e79"],[0.5,"#e8a020"],[1,"#ef4444"]],
            text="avg_price_sqft",
            labels={"avg_price_sqft": "Avg ₹/sqft", "locality": ""},
        )
        fig.update_traces(texttemplate="₹%{x:,.0f}", textposition="outside", textfont_size=10)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOT_THEME, height=430,
                          title=dict(text="Average Price/sqft by Locality", font=dict(family="Syne", size=14)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header" style="font-size:1rem;">📈 ROI vs Risk Matrix</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            loc_stats,
            x="avg_risk", y="avg_roi",
            size="listing_count",
            color="avg_investment_score",
            color_continuous_scale="RdYlGn",
            hover_name="locality",
            hover_data={"avg_price_sqft": ":,.0f", "avg_livability": ":.1f"},
            labels={"avg_risk": "Risk Score →", "avg_roi": "5-Year ROI % →"},
            size_max=40,
        )
        fig2.update_layout(**PLOT_THEME, height=430,
                           title=dict(text="ROI vs Risk (size = listings)", font=dict(family="Syne", size=14)),
                           coloraxis_colorbar=dict(title="Inv. Score", thickness=12, len=0.7))
        # Add quadrant lines
        fig2.add_hline(y=loc_stats["avg_roi"].mean(), line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig2.add_vline(x=loc_stats["avg_risk"].mean(), line_dash="dot", line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Price Trend + Distribution
    col3, col4 = st.columns([1.5, 1])

    with col3:
        st.markdown('<div class="section-header" style="font-size:1rem;">📉 Price Trends (36 Months)</div>', unsafe_allow_html=True)
        top_locs = loc_stats.nlargest(5, "avg_price_sqft")["locality"].tolist()
        colours  = ["#e8a020","#14b8a6","#60a5fa","#f472b6","#a78bfa"]
        fig3 = go.Figure()
        for loc, colour in zip(top_locs, colours):
            if loc in ts:
                s = ts[loc]
                fig3.add_trace(go.Scatter(
                    x=s["month"], y=s["avg_price_sqft"],
                    name=loc, line=dict(color=colour, width=2),
                    hovertemplate="%{x|%b %Y}: ₹%{y:,.0f}<extra>" + loc + "</extra>",
                ))
        fig3.update_layout(**PLOT_THEME, height=320,
                           title=dict(text="Top 5 Localities — Monthly Avg Price/sqft", font=dict(family="Syne", size=14)),
                           legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                           hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header" style="font-size:1rem;">🏢 Builder Tier Mix</div>', unsafe_allow_html=True)
        if "builder_tier" in df.columns:
            tier_counts = df["builder_tier"].value_counts()
            fig4 = go.Figure(go.Pie(
                labels=tier_counts.index,
                values=tier_counts.values,
                hole=0.55,
                marker=dict(colors=["#e8a020","#14b8a6","#60a5fa","#94a3b8"]),
                textfont=dict(size=11),
                textinfo="label+percent",
            ))
            fig4.update_layout(**PLOT_THEME, height=320,
                               title=dict(text="Builder Tier Distribution", font=dict(family="Syne", size=14)),
                               showlegend=False)
            fig4.add_annotation(text="Builders", x=0.5, y=0.5,
                                font=dict(size=12, color="#94a3b8", family="Syne"),
                                showarrow=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Row 4: Investment Grade Distribution
    st.markdown('<div class="section-header" style="font-size:1rem;">🏆 Investment Grade Distribution</div>', unsafe_allow_html=True)
    if "investment_grade" in df.columns:
        grade_counts = df["investment_grade"].value_counts().reindex(["A+","A","B+","B","C","D"]).dropna()
        grade_colors = ["#059669","#2563eb","#16a34a","#d97706","#ea580c","#dc2626"]
        fig5 = go.Figure(go.Bar(
            x=grade_counts.index,
            y=grade_counts.values,
            marker_color=grade_colors[:len(grade_counts)],
            text=grade_counts.values,
            textposition="outside",
        ))
        fig5.update_layout(**PLOT_THEME, height=240,
                           title=dict(text="Properties by Investment Grade", font=dict(family="Syne", size=14)),
                           xaxis_title="Investment Grade", yaxis_title="Number of Listings")
        st.plotly_chart(fig5, use_container_width=True)

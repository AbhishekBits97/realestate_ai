"""pages/property_explorer.py"""
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go

PLOT_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="DM Sans", color="#e2e8f0", size=12),
                  margin=dict(l=10, r=10, t=40, b=10),
                  xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                  yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))

def render_property_explorer(data: dict):
    df        = data["listings"]
    localities= sorted(df["locality"].unique().tolist())

    st.markdown('<div class="section-header">🔍 Property Explorer</div>', unsafe_allow_html=True)

    # ── Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sel_locs = st.multiselect("Locality", localities, default=localities[:5], key="pe_loc")
    with col2:
        bhk_opts = sorted(df["bhk_count"].dropna().unique().astype(int).tolist())
        sel_bhk  = st.multiselect("BHK", bhk_opts, default=bhk_opts[:3], key="pe_bhk")
    with col3:
        price_range = st.slider("Price (Cr)", 0.0, float(df["price_cr"].max()), (0.5, 5.0), key="pe_price")
    with col4:
        furnishing = st.multiselect("Furnishing", df["furnishing_status"].unique().tolist() if "furnishing_status" in df.columns else ["Any"], key="pe_furn")

    mask = df["locality"].isin(sel_locs) & df["bhk_count"].isin(sel_bhk) & \
           df["price_cr"].between(*price_range)
    filtered = df[mask]

    st.markdown(f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:16px;'>"
                f"<b style='color:#e8a020'>{len(filtered):,}</b> properties match your criteria</div>",
                unsafe_allow_html=True)

    # ── Map
    if "lat" in filtered.columns and len(filtered) > 0:
        st.markdown('<div class="section-header" style="font-size:1rem;">🗺️ Property Map</div>', unsafe_allow_html=True)
        sample = filtered.sample(min(300, len(filtered)))
        fig_map = px.scatter_mapbox(
            sample, lat="lat", lon="lon",
            color="price_per_sqft",
            color_continuous_scale="RdYlGn_r",
            size="area_sqft", size_max=18,
            hover_name="locality",
            hover_data={"price_cr":":.2f","bhk_count":True,"livability_index":":.0f"},
            zoom=11, center={"lat":28.46,"lon":77.03},
            mapbox_style="carto-darkmatter",
            labels={"price_per_sqft":"₹/sqft"},
        )
        fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=420)
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Charts row
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.box(filtered, x="bhk_count", y="price_per_sqft", color="bhk_count",
                     color_discrete_sequence=["#e8a020","#14b8a6","#60a5fa","#f472b6","#a78bfa"],
                     labels={"bhk_count":"BHK","price_per_sqft":"₹/sqft"})
        fig.update_layout(**PLOT_THEME, height=320, showlegend=False,
                          title=dict(text="Price Distribution by BHK", font=dict(family="Syne",size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.scatter(filtered.sample(min(250,len(filtered))),
                          x="area_sqft", y="price_cr",
                          color="locality", size="livability_index",
                          hover_name="locality", size_max=20,
                          labels={"area_sqft":"Area (sqft)","price_cr":"Price (Cr)"})
        fig2.update_layout(**PLOT_THEME, height=320, showlegend=False,
                           title=dict(text="Area vs Price (size = livability)", font=dict(family="Syne",size=13)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Table
    st.markdown('<div class="section-header" style="font-size:1rem;margin-top:8px;">📋 Listings</div>', unsafe_allow_html=True)
    show_cols = ["locality","bhk_count","area_sqft","price_cr","price_per_sqft",
                 "livability_index","investment_grade","recommendation","builder_tier"]
    available = [c for c in show_cols if c in filtered.columns]
    st.dataframe(filtered[available].rename(columns={
        "bhk_count":"BHK","area_sqft":"Area(sqft)","price_cr":"Price(Cr)",
        "price_per_sqft":"₹/sqft","livability_index":"Livability",
        "investment_grade":"Grade","recommendation":"Rec.","builder_tier":"Builder Tier"
    }).reset_index(drop=True), use_container_width=True, height=340)

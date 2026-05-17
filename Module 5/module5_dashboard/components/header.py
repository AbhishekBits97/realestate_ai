"""components/header.py — Top header bar"""
import streamlit as st
from datetime import datetime

def render_header():
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between;
                padding:12px 0 20px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:24px;">
        <div>
            <span style="font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#e2e8f0;">
                🏘️ AI Real Estate Investment Intelligence System
            </span>
            <span style="font-size:0.75rem; color:#64748b; margin-left:12px;">
                Gurgaon NCR · {datetime.now().strftime("%d %b %Y")}
            </span>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            <span style="background:#064e3b; color:#6ee7b7; border:1px solid #059669;
                         padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600;">
                ● LIVE
            </span>
            <span style="background:rgba(255,255,255,0.05); color:#94a3b8;
                         padding:3px 10px; border-radius:20px; font-size:0.72rem;">
                v1.0
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

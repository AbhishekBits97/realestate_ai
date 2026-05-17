"""components/sidebar.py — Navigation sidebar"""
import streamlit as st

PAGES = {
    "📊 Market Overview"    : "Market Overview",
    "🔍 Property Explorer"  : "Property Explorer",
    "💰 Investment Analysis": "Investment Analysis",
    "🤖 Price Predictor"    : "Price Predictor",
    "💬 AI Advisor"         : "AI Advisor",
    "🧠 XAI Explorer"       : "XAI Explorer",
}

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 24px;">
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
                        background:linear-gradient(135deg,#e8a020,#14b8a6);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                PropIQ
            </div>
            <div style="font-size:0.7rem; color:#64748b; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px;">
                Real Estate Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

        selected = None
        for label, key in PAGES.items():
            is_active = st.session_state.get("active_page") == key
            if st.button(label, use_container_width=True, key=f"nav_{key}",
                         type="primary" if is_active else "secondary"):
                st.session_state.active_page = key
            if is_active:
                selected = key

        if selected is None:
            selected = st.session_state.get("active_page", "Market Overview")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#64748b; line-height:1.6; padding:8px 0;">
            <b style="color:#94a3b8;">Coverage</b><br>
            Gurgaon NCR · Delhi · Noida<br><br>
            <b style="color:#94a3b8;">Models</b><br>
            XGBoost · LSTM · GNN<br>
            Prophet · Bayesian Risk<br><br>
            <b style="color:#94a3b8;">Student</b><br>
            Abhishek · 2024DA04221<br>
            M.Tech Data Science<br>
            BITS Pilani WILP
        </div>
        """, unsafe_allow_html=True)

    return selected

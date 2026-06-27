"""
app.py
=======
Module 5 — Streamlit Dashboard Entry Point

AI-Powered Real Estate Investment Intelligence System
Gurgaon NCR Property Market Analytics

Run:
  streamlit run app.py
  streamlit run app.py --server.port 8501 --theme.base dark
"""

import streamlit as st

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title     = "PropIQ — Real Estate Intelligence",
    page_icon      = "🏘️",
    layout         = "wide",
    initial_sidebar_state = "expanded",
    menu_items     = {
        "Get Help"    : "mailto:2024da04221@wilp.bits-pilani.ac.in",
        "About"       : "PropIQ — AI Real Estate Intelligence by Abhishek (2024DA04221) | M.Tech Data Science, BITS Pilani",
    }
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "module4_xai_chatbot"))
sys.path.insert(0, str(Path(__file__).parent / "module3_models"))
sys.path.insert(0, str(Path(__file__).parent / "module2_feature_engineering"))

from utils.dashboard_data import load_dashboard_data
from components.sidebar    import render_sidebar
from components.header     import render_header


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root variables ── */
:root {
    --navy      : #0f1f3d;
    --navy-light: #162847;
    --gold      : #e8a020;
    --gold-light: #f5c458;
    --teal      : #0d9488;
    --teal-light: #14b8a6;
    --slate     : #64748b;
    --bg        : #0a1628;
    --bg-card   : #111e35;
    --bg-card2  : #152038;
    --text      : #e2e8f0;
    --text-dim  : #94a3b8;
    --border    : rgba(255,255,255,0.08);
    --success   : #22c55e;
    --danger    : #ef4444;
    --warn      : #f59e0b;
    --radius    : 12px;
    --shadow    : 0 4px 24px rgba(0,0,0,0.4);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0f1f3d 50%, #071020 100%);
    background-attachment: fixed;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy-light) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Cards ── */
.prop-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.prop-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--teal));
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--gold); }
.metric-label { font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.metric-delta { font-size: 0.85rem; margin-top: 6px; }
.delta-up   { color: var(--success); }
.delta-down { color: var(--danger); }

/* ── Grade badges ── */
.grade-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}
.grade-Aplus { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.grade-A     { background: #1e3a5f; color: #93c5fd; border: 1px solid #2563eb; }
.grade-Bplus { background: #1a3a1a; color: #86efac; border: 1px solid #16a34a; }
.grade-B     { background: #3d2e00; color: #fde68a; border: 1px solid #d97706; }
.grade-C     { background: #3d1a00; color: #fed7aa; border: 1px solid #ea580c; }
.grade-D     { background: #3d0000; color: #fca5a5; border: 1px solid #dc2626; }

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    padding-bottom: 10px;
    border-bottom: 2px solid var(--gold);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    color: var(--text-dim) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gold) !important;
    color: var(--navy) !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stSlider {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold), #d4870a) !important;
    color: var(--navy) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: var(--radius) !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: var(--radius) !important; }

/* ── Chat messages ── */
.chat-user {
    background: linear-gradient(135deg, var(--navy-light), var(--bg-card2));
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.92rem;
}
.chat-bot {
    background: linear-gradient(135deg, #0d2240, #0f2a50);
    border: 1px solid rgba(232,160,32,0.2);
    border-left: 3px solid var(--gold);
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.92rem;
}
.chat-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    color: var(--text-dim);
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--slate); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "Market Overview"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_dashboard_data()


with st.spinner("🔄 Loading PropIQ Intelligence Engine..."):
    data = get_data()
    st.session_state.data_loaded = True


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
render_header()
page = render_sidebar()

# Route pages
if page == "Market Overview":
    from pages.market_overview import render_market_overview
    render_market_overview(data)

elif page == "Property Explorer":
    from pages.property_explorer import render_property_explorer
    render_property_explorer(data)

elif page == "Investment Analysis":
    from pages.investment_analysis import render_investment_analysis
    render_investment_analysis(data)

elif page == "Price Predictor":
    from pages.price_predictor import render_price_predictor
    render_price_predictor(data)

elif page == "AI Advisor":
    from pages.ai_advisor import render_ai_advisor
    render_ai_advisor(data)

elif page == "XAI Explorer":
    from pages.xai_explorer import render_xai_explorer
    render_xai_explorer(data)

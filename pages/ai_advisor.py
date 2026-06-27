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
    page_title="PropIQ — Real Estate Intelligence",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:2024da04221@wilp.bits-pilani.ac.in",
        "About": "PropIQ — AI Real Estate Intelligence by Abhishek (2024DA04221) | M.Tech Data Science, BITS Pilani",
    },
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "module4_xai_chatbot"))
sys.path.insert(0, str(Path(__file__).parent / "module3_models"))
sys.path.insert(0, str(Path(__file__).parent / "module2_feature_engineering"))

from utils.dashboard_data import load_dashboard_data
from components.sidebar import render_sidebar
from components.header import render_header


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — AI INTELLIGENCE TERMINAL
# Palette: Midnight base · Cyan signal · Amber data · Emerald gain · Rose risk
# Type: Space Grotesk (display/numerics) · Inter (body) · JetBrains Mono (data)
# Signature: glowing scan-line grid background + live-pulse data rings on KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════
   TOKEN SYSTEM
═══════════════════════════════════════════════ */
:root {
    /* Base surfaces */
    --bg-void:      #030810;
    --bg-base:      #060f1e;
    --bg-raised:    #0a1628;
    --bg-card:      #0d1d35;
    --bg-card-hi:   #112240;
    --bg-glass:     rgba(13, 29, 53, 0.72);

    /* Signals */
    --cyan:         #00d4ff;
    --cyan-dim:     rgba(0, 212, 255, 0.15);
    --cyan-glow:    rgba(0, 212, 255, 0.35);
    --amber:        #f59e0b;
    --amber-dim:    rgba(245, 158, 11, 0.14);
    --amber-glow:   rgba(245, 158, 11, 0.30);
    --emerald:      #10b981;
    --emerald-dim:  rgba(16, 185, 129, 0.13);
    --emerald-glow: rgba(16, 185, 129, 0.30);
    --rose:         #f43f5e;
    --rose-dim:     rgba(244, 63, 94, 0.13);
    --violet:       #8b5cf6;
    --violet-dim:   rgba(139, 92, 246, 0.14);

    /* Text */
    --text-hi:      #f0f6ff;
    --text-body:    #94a3b8;
    --text-dim:     #475569;
    --text-data:    #00d4ff;

    /* Structure */
    --border:       rgba(0, 212, 255, 0.10);
    --border-hi:    rgba(0, 212, 255, 0.28);
    --radius-sm:    6px;
    --radius:       10px;
    --radius-lg:    16px;
    --shadow-card:  0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0,212,255,0.06);
    --shadow-glow:  0 0 40px rgba(0, 212, 255, 0.10);
}

/* ═══════════════════════════════════════════════
   GLOBAL RESET
═══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* ═══════════════════════════════════════════════
   SIGNATURE ELEMENT — SCAN-LINE GRID BACKGROUND
   A pulsing intelligence-terminal atmosphere
═══════════════════════════════════════════════ */
.stApp {
    background-color: var(--bg-void);
    background-image:
        /* Subtle dot grid */
        radial-gradient(circle, rgba(0,212,255,0.06) 1px, transparent 1px),
        /* Depth gradient */
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,212,255,0.04) 0%, transparent 70%);
    background-size: 28px 28px, 100% 100%;
    background-attachment: fixed;
    min-height: 100vh;
}

/* Horizontal scan-line overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 212, 255, 0.012) 2px,
        rgba(0, 212, 255, 0.012) 4px
    );
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg-base) !important;
    border-right: 1px solid var(--border-hi) !important;
    box-shadow: 4px 0 32px rgba(0, 0, 0, 0.6) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-body) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar section labels */
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: var(--cyan) !important;
    margin-bottom: 8px !important;
    opacity: 0.8;
}

/* Nav buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-body) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    text-align: left !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s ease !important;
    margin-bottom: 4px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--cyan-dim) !important;
    border-color: var(--border-hi) !important;
    color: var(--cyan) !important;
    transform: translateX(3px) !important;
}

/* ═══════════════════════════════════════════════
   MAIN CONTENT AREA
═══════════════════════════════════════════════ */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1600px !important;
}

/* ═══════════════════════════════════════════════
   METRIC CARDS — Glowing data rings
═══════════════════════════════════════════════ */
.iq-metric {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 20px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s;
}

.iq-metric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: var(--radius) var(--radius) 0 0;
}

.iq-metric.cyan::before  { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.iq-metric.amber::before { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
.iq-metric.emerald::before { background: linear-gradient(90deg, transparent, var(--emerald), transparent); }
.iq-metric.rose::before  { background: linear-gradient(90deg, transparent, var(--rose), transparent); }
.iq-metric.violet::before { background: linear-gradient(90deg, transparent, var(--violet), transparent); }

.iq-metric:hover {
    border-color: var(--border-hi);
    box-shadow: var(--shadow-card);
}

/* Pulse ring — signature data-alive element */
.iq-metric::after {
    content: '';
    position: absolute;
    right: 18px; top: 20px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 0 0 var(--emerald-glow);
    animation: pulse-ring 2.8s ease-in-out infinite;
}
.iq-metric.amber::after { background: var(--amber); box-shadow: 0 0 0 0 var(--amber-glow); }
.iq-metric.cyan::after  { background: var(--cyan);  box-shadow: 0 0 0 0 var(--cyan-glow); }
.iq-metric.rose::after  { background: var(--rose);  animation: none; opacity: 0.6; }

@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    60%  { box-shadow: 0 0 0 10px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

.iq-metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
}

.iq-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-hi);
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.iq-metric-value.cyan   { color: var(--cyan); }
.iq-metric-value.amber  { color: var(--amber); }
.iq-metric-value.emerald { color: var(--emerald); }
.iq-metric-value.rose   { color: var(--rose); }

.iq-metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 400;
}
.delta-up   { color: var(--emerald); }
.delta-dn   { color: var(--rose); }
.delta-flat { color: var(--text-dim); }

/* ═══════════════════════════════════════════════
   GLASS PANELS — Content containers
═══════════════════════════════════════════════ */
.iq-panel {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-card);
    margin-bottom: 20px;
}

.iq-panel-sm {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
}

/* ═══════════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════════ */
.iq-section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

.iq-section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.8;
    display: block;
    margin-bottom: 4px;
}

.iq-section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-hi);
    letter-spacing: -0.01em;
}

.iq-section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan-glow);
    flex-shrink: 0;
}

/* ═══════════════════════════════════════════════
   PROPERTY CARDS
═══════════════════════════════════════════════ */
.prop-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.prop-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--cyan);
    opacity: 0;
    transition: opacity 0.2s;
}

.prop-card:hover {
    border-color: var(--border-hi);
    background: var(--bg-card-hi);
    transform: translateY(-1px);
    box-shadow: var(--shadow-card);
}

.prop-card:hover::before { opacity: 1; }

/* ═══════════════════════════════════════════════
   GRADE BADGES — Investment intelligence signals
═══════════════════════════════════════════════ */
.iq-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1px;
}

.iq-badge-A  {
    background: var(--emerald-dim);
    color: var(--emerald);
    border: 1px solid rgba(16,185,129,0.25);
}
.iq-badge-B  {
    background: var(--cyan-dim);
    color: var(--cyan);
    border: 1px solid rgba(0,212,255,0.22);
}
.iq-badge-C  {
    background: var(--amber-dim);
    color: var(--amber);
    border: 1px solid rgba(245,158,11,0.22);
}
.iq-badge-D  {
    background: var(--rose-dim);
    color: var(--rose);
    border: 1px solid rgba(244,63,94,0.22);
}
.iq-badge-buy    { background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(16,185,129,0.25); }
.iq-badge-hold   { background: var(--amber-dim);   color: var(--amber);   border: 1px solid rgba(245,158,11,0.22); }
.iq-badge-avoid  { background: var(--rose-dim);    color: var(--rose);    border: 1px solid rgba(244,63,94,0.22); }

/* ═══════════════════════════════════════════════
   DATA TABLE
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] th {
    background: var(--bg-raised) !important;
    color: var(--text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-hi) !important;
}

[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text-body) !important;
    border-bottom: 1px solid var(--border) !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: var(--cyan-dim) !important;
}

/* ═══════════════════════════════════════════════
   TABS — Intelligence module selector
═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 2px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-dim) !important;
    padding: 8px 18px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}

.stTabs [aria-selected="true"] {
    background: var(--cyan-dim) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--border-hi) !important;
}

.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-body) !important;
}

/* ═══════════════════════════════════════════════
   FORM CONTROLS
═══════════════════════════════════════════════ */
/* Labels */
.stSlider label, .stSelectbox label, .stMultiSelect label,
.stTextInput label, .stNumberInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-hi) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--border-hi) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
}

/* Selectboxes */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-hi) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: var(--cyan) !important;
}
.stSlider [data-baseweb="slider"] > div:first-child {
    background: var(--border) !important;
}

/* Slider value label */
.stSlider [data-testid="stTickBar"] {
    color: var(--text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--cyan) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    padding: 9px 20px !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    background: var(--cyan-dim) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 16px var(--cyan-glow) !important;
}

/* Primary CTA button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.08)) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px var(--cyan-glow) !important;
}

/* ═══════════════════════════════════════════════
   CHAT INTERFACE — AI Advisor
═══════════════════════════════════════════════ */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 4px 0;
}

.chat-user {
    align-self: flex-end;
    max-width: 75%;
    background: var(--bg-card-hi);
    border: 1px solid var(--border-hi);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-hi);
    line-height: 1.6;
}

.chat-bot {
    align-self: flex-start;
    max-width: 85%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 2px solid var(--cyan);
    border-radius: 4px 12px 12px 12px;
    padding: 14px 18px;
    font-size: 13px;
    color: var(--text-body);
    line-height: 1.7;
}

.chat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-dim);
    margin-bottom: 6px;
}

.chat-label.bot { color: var(--cyan); opacity: 0.8; }

/* Thinking animation */
.iq-thinking {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 10px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 2px solid var(--cyan);
    border-radius: 4px 12px 12px 12px;
}

.iq-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--cyan);
    animation: iq-bounce 1.1s ease-in-out infinite;
}
.iq-dot:nth-child(2) { animation-delay: 0.18s; }
.iq-dot:nth-child(3) { animation-delay: 0.36s; }

@keyframes iq-bounce {
    0%, 100% { opacity: 0.3; transform: translateY(0); }
    50%       { opacity: 1;   transform: translateY(-4px); }
}

/* Quick-question chips */
.iq-chip {
    display: inline-block;
    padding: 6px 14px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.18s;
    margin: 4px;
}
.iq-chip:hover {
    border-color: var(--border-hi);
    color: var(--cyan);
    background: var(--cyan-dim);
}

/* ═══════════════════════════════════════════════
   MAP CONTAINER
═══════════════════════════════════════════════ */
[data-testid="stDeckGlJsonChart"],
.folium-map {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════
   PLOTLY CHARTS
═══════════════════════════════════════════════ */
.js-plotly-plot {
    border-radius: var(--radius) !important;
}

.js-plotly-plot .plotly {
    background: transparent !important;
}

/* ═══════════════════════════════════════════════
   SHAP / XAI BARS
═══════════════════════════════════════════════ */
.shap-row {
    display: grid;
    grid-template-columns: 160px 1fr 80px;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}
.shap-label {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--text-body);
}
.shap-bar-wrap {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.shap-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
}
.shap-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--cyan);
    text-align: right;
}

/* ═══════════════════════════════════════════════
   STATUS INDICATORS
═══════════════════════════════════════════════ */
.iq-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.iq-status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.iq-status.live .iq-status-dot {
    background: var(--emerald);
    box-shadow: 0 0 0 0 var(--emerald-glow);
    animation: pulse-ring 2.8s ease-in-out infinite;
}
.iq-status.live { color: var(--emerald); }
.iq-status.warn .iq-status-dot { background: var(--amber); }
.iq-status.warn { color: var(--amber); }
.iq-status.err  .iq-status-dot { background: var(--rose); }
.iq-status.err  { color: var(--rose); }

/* ═══════════════════════════════════════════════
   SCORE RING (circular progress for investment score)
═══════════════════════════════════════════════ */
.iq-score-ring {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.iq-score-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
}
.iq-score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
}

/* ═══════════════════════════════════════════════
   DIVIDERS & SPACING UTILITIES
═══════════════════════════════════════════════ */
.iq-divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
}

.iq-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-data);
}

.iq-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.75;
}

/* ═══════════════════════════════════════════════
   SPINNERS & LOADING
═══════════════════════════════════════════════ */
.stSpinner > div {
    border-color: var(--cyan) transparent transparent transparent !important;
}

/* ═══════════════════════════════════════════════
   STREAMLIT NATIVE OVERRIDES
═══════════════════════════════════════════════ */
/* st.metric boxes */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 20px !important;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* Positive/negative delta colours */
[data-testid="stMetricDelta"][data-direction="positive"] { color: var(--emerald) !important; }
[data-testid="stMetricDelta"][data-direction="negative"] { color: var(--rose) !important; }

/* st.info / st.warning / st.success / st.error */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

/* st.expander */
.streamlit-expanderHeader {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-body) !important;
}

/* st.progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
    border-radius: 3px !important;
}
.stProgress > div > div {
    background: var(--border) !important;
    border-radius: 3px !important;
    height: 6px !important;
}

/* Checkbox */
.stCheckbox span { color: var(--text-body) !important; }
.stCheckbox [data-testid="stCheckbox"] div {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu          { visibility: hidden; }
footer             { visibility: hidden; }
.stDeployButton    { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ═══════════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════════ */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.iq-animate {
    animation: fadeSlideUp 0.4s ease forwards;
}

/* Staggered delays for card grids */
.iq-animate-1 { animation-delay: 0.05s; opacity: 0; }
.iq-animate-2 { animation-delay: 0.10s; opacity: 0; }
.iq-animate-3 { animation-delay: 0.15s; opacity: 0; }
.iq-animate-4 { animation-delay: 0.20s; opacity: 0; }

/* Glow pulse for live data badges */
@keyframes glow-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.iq-live-glow { animation: glow-pulse 2s ease-in-out infinite; }

</style>
""",
    unsafe_allow_html=True,
)


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
@st.cache_data(show_spinner=False, ttl=3600)
def get_data():
    return load_dashboard_data()


with st.spinner("Initialising PropIQ Intelligence Engine…"):
    data = get_data()
    st.session_state.data_loaded = True


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
render_header()
page = render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────────────
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

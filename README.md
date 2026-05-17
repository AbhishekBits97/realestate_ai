# 🏘️ Module 5 — PropIQ Streamlit Dashboard

**AI-Powered Real Estate Investment Intelligence System**
Abhishek · 2024DA04221 · M.Tech Data Science · BITS Pilani WILP

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Launch on specific port
streamlit run app.py --server.port 8501
```

Open → **http://localhost:8501**

---

## 📁 Structure

```
module5_dashboard/
├── app.py                          ← Entry point + global styles
├── requirements.txt
├── pages/
│   ├── market_overview.py          ← KPIs, heatmap, price trends
│   ├── property_explorer.py        ← Map + filter + listings table
│   ├── investment_analysis.py      ← Risk-ROI matrix, forecasts, top picks
│   ├── price_predictor.py          ← Interactive ML prediction form + SHAP
│   ├── xai_explorer.py             ← SHAP global importance, waterfall, dependence
│   └── ai_advisor.py               ← Embedded LangChain chatbot
├── components/
│   ├── header.py                   ← Top navigation bar
│   └── sidebar.py                  ← Page router sidebar
└── utils/
    └── dashboard_data.py           ← Data loader (real or mock)
```

---

## 📊 Pages

### 1. Market Overview
- 4 KPI metric cards (listings, avg price, ROI, livability)
- Locality price heatmap (horizontal bar chart)
- ROI vs Risk scatter matrix
- 36-month price trend lines (top 5 localities)
- Builder tier pie chart
- Investment grade distribution bar

### 2. Property Explorer
- Interactive map (Plotly + Mapbox, colour by price/sqft)
- Filters: locality, BHK, price range, furnishing
- Price distribution by BHK (box plots)
- Area vs Price scatter
- Full filterable listings table

### 3. Investment Analysis
- Risk / ROI / Score sliders for filtering
- Risk-Return scatter coloured by grade
- 12-month Prophet/LSTM price forecast with confidence band
- Top 15 investment opportunities table

### 4. Price Predictor
- Full input form: locality, BHK, area, floor, furnishing, transaction type
- AI feature sliders: livability, builder reputation, metro distance
- Animated result banner with predicted ₹/sqft + total ₹ Cr
- SHAP waterfall attribution chart
- Comparable locality bar chart
- Investment summary metrics

### 5. XAI Explorer
- Global SHAP bar importance chart
- SHAP dot plot (feature × direction)
- Per-property waterfall + natural language explanation
- Dependence plots (any feature pair)
- SHAP vs LIME agreement histogram

### 6. AI Advisor
- 6 quick-prompt buttons
- Multi-turn conversation with memory
- Markdown-rendered responses with tables
- Handles: ROI forecasts, risk, price prediction, comparisons, search, SHAP explain
- Clear conversation button

---

## 🎨 Design System

| Token | Value |
|-------|-------|
| Primary font | Syne (headers) |
| Body font | DM Sans |
| Background | `#0a1628` dark navy gradient |
| Accent | `#e8a020` gold |
| Secondary | `#14b8a6` teal |
| Cards | `#111e35` with gold top border |
| Charts | Plotly with transparent background |

---

## 🔗 Integration with Other Modules

The dashboard auto-loads from Module 2/3/4 outputs if present:

```
data/features/feature_matrix.parquet      ← Module 2 output
data/explanations/listings_investment_scored.parquet  ← Module 4 output
data/models/prophet_forecasts.json        ← Module 3 output
```

Falls back to rich mock data (800 Gurgaon listings) if files are absent.

#!/usr/bin/env python3
"""
generate_data.py
----------------
Reads listings_investment_scored.csv and writes data.json.
Includes: kpis, ticker, sector_rankings, price_chart, market_data, listings (560 full records).

Run locally:  python generate_data.py
Auto-run by:  .github/workflows/update-data.yml on every CSV push
"""
import csv, json, statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

CSV_PATH  = Path(__file__).parent / "listings_investment_scored.csv"
JSON_PATH = Path(__file__).parent / "data.json"

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

def sf(v):
    try: return float(v)
    except: return None

# ── Full listings (for analyser.html) ────────────────────────────────
listings = []
for r in rows:
    try:
        listings.append({
            "locality": r["locality"],
            "type": r["property_type_clean"].replace("_"," ").title(),
            "bhk": int(float(r["bhk_count"])),
            "area": round(float(r["area_sqft"])),
            "price_cr": round(float(r["price_cr"]),2),
            "price_sqft": round(float(r["price_per_sqft"])),
            "score": round(float(r["investment_score"]),1),
            "grade": r["investment_grade"],
            "rec": r["recommendation"],
            "roi": round(float(r["roi_risk_adjusted"]),1),
            "appr": round(float(r["appreciation_potential_score"]),1),
            "tier": r["builder_tier"],
            "livability": round(float(r["livability_index"]),1),
            "infra": round(float(r["infrastructure_impact_score"]),1),
            "metro_km": round(float(r["dist_nearest_metro_km"]),2),
            "hospital_km": round(float(r["dist_nearest_hospital_km"]),2),
            "school_km": round(float(r["dist_nearest_school_km"]),2),
            "mall_km": round(float(r["dist_nearest_mall_km"]),2),
            "it_hub_km": round(float(r["dist_nearest_it_hub_km"]),2),
            "airport_km": round(float(r["dist_nearest_airport_km"]),2),
            "payback": round(float(r["payback_years"]),1),
            "floor": int(float(r["floor_number"])) if r.get("floor_number") else 0,
            "total_floors": int(float(r["total_floors"])) if r.get("total_floors") else 0,
            "healthcare": round(float(r["score_healthcare"]),1),
            "education": round(float(r["score_education"]),1),
            "metro_score": round(float(r["score_metro"]),1),
            "employment": round(float(r["score_employment"]),1),
            "commercial": round(float(r["score_commercial"]),1),
            "impact_metro": round(float(r["impact_metro_extension"]),1),
            "impact_highway": round(float(r["impact_highway"]),1),
            "impact_itsez": round(float(r["impact_it_sez"]),1),
            "impact_urban": round(float(r["impact_urban_development"]),1),
        })
    except:
        pass

# ── KPIs ──────────────────────────────────────────────────────────────
all_prices  = [sf(r["price_per_sqft"]) for r in rows if sf(r.get("price_per_sqft"))]
all_scores  = [sf(r["investment_score"]) for r in rows if sf(r.get("investment_score"))]
all_roi     = [sf(r["roi_risk_adjusted"]) for r in rows if sf(r.get("roi_risk_adjusted"))]
all_appr    = [sf(r["appreciation_potential_score"]) for r in rows if sf(r.get("appreciation_potential_score"))]
all_payback = [sf(r["payback_years"]) for r in rows if sf(r.get("payback_years"))]
buy_count   = sum(1 for r in rows if r.get("recommendation")=="Buy")
total       = len(rows)

kpis = {
    "avg_price_sqft":           round(statistics.mean(all_prices)),
    "median_price_sqft":        round(statistics.median(all_prices)),
    "avg_investment_score":     round(statistics.mean(all_scores),1),
    "avg_roi_risk_adjusted":    round(statistics.mean(all_roi),1),
    "avg_appreciation_potential": round(statistics.mean(all_appr),1),
    "avg_payback_years":        round(statistics.mean(all_payback),1),
    "buy_count": buy_count, "hold_count": total-buy_count,
    "buy_pct": round(buy_count/total*100,1), "total_listings": total,
}

# ── Locality aggregates ───────────────────────────────────────────────
lp=defaultdict(list); ls=defaultdict(list); la=defaultdict(list); lr=defaultdict(list)
for r in rows:
    loc=r["locality"]
    for store,field in [(lp,"price_per_sqft"),(ls,"investment_score"),(la,"appreciation_potential_score"),(lr,"roi_risk_adjusted")]:
        v=sf(r.get(field))
        if v: store[loc].append(v)

ticker = sorted(
    [{"locality":k,"avg_price_sqft":round(statistics.mean(v))} for k,v in lp.items() if len(v)>=3],
    key=lambda x:-x["avg_price_sqft"])[:12]

sector_rankings = sorted(
    [{"locality":k,
      "investment_score":round(statistics.mean(ls[k]),1),
      "avg_price_sqft":round(statistics.mean(lp[k])) if lp[k] else 0,
      "appreciation":round(statistics.mean(la[k]),1) if la[k] else 0,
      "roi":round(statistics.mean(lr[k]),1) if lr[k] else 0,
      "listing_count":len(ls[k])}
     for k in ls if len(ls[k])>=3],
    key=lambda x:-x["investment_score"])[:10]

price_chart = sorted(
    [{"locality":k,"avg_price_sqft":round(statistics.mean(v))} for k,v in lp.items() if len(v)>=3],
    key=lambda x:-x["avg_price_sqft"])[:12]

bands={"under_1cr":0,"1_2cr":0,"2_5cr":0,"above_5cr":0}
for r in rows:
    p=sf(r.get("price_cr"))
    if p is None: continue
    if p<1: bands["under_1cr"]+=1
    elif p<2: bands["1_2cr"]+=1
    elif p<5: bands["2_5cr"]+=1
    else: bands["above_5cr"]+=1

bhk=defaultdict(int); ptypes=defaultdict(int); btiers=defaultdict(int)
for r in rows:
    v=sf(r.get("bhk_count"))
    if v: bhk[int(v)]+=1
    ptypes[r.get("property_type_clean","other")]+=1
    btiers[r.get("builder_tier","unknown")]+=1

# ── Market data (for markets.html) ───────────────────────────────────
market_data=[]
for loc in sorted(lp.keys()):
    if len(lp[loc])>=3:
        market_data.append({
            "locality":loc,
            "avg_price":round(statistics.mean(lp[loc])),
            "med_price":round(statistics.median(lp[loc])),
            "avg_score":round(statistics.mean(ls[loc]),1) if ls[loc] else 0,
            "avg_appr":round(statistics.mean(la[loc]),1) if la[loc] else 0,
            "avg_roi":round(statistics.mean(lr[loc]),1) if lr[loc] else 0,
            "count":len(lp[loc]),
            "min_price":round(min(lp[loc])),
            "max_price":round(max(lp[loc])),
        })
market_data.sort(key=lambda x:-x["avg_price"])

# ── Assemble & write ──────────────────────────────────────────────────
output={
    "meta":{"total_listings":total,"generated_at":str(date.today()),"source":CSV_PATH.name},
    "kpis":kpis,
    "price_bands":bands,
    "bhk_distribution":dict(sorted(bhk.items())),
    "property_types":dict(ptypes),
    "builder_tiers":dict(btiers),
    "ticker":ticker,
    "sector_rankings":sector_rankings,
    "price_chart":price_chart,
    "market_data":market_data,
    "listings":listings,
}

with open(JSON_PATH,"w") as f:
    json.dump(output,f,separators=(",",":"))

size=JSON_PATH.stat().st_size
print(f"data.json written — {total} listings, {size/1024:.1f} KB, generated {date.today()}")

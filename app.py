import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import brentq

st.set_page_config(
    page_title="PE Fund Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid='stMetric'] { background:#f8f9fa; border-radius:10px; padding:14px 18px; }
  [data-testid='stMetricLabel'] { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#666; }
  [data-testid='stMetricValue'] { font-size:28px; font-weight:700; }
  .block-container { padding-top:1.5rem; }
</style>
""", unsafe_allow_html=True)

TSLA_PRICES = {
    "2019-01-31":59.51,"2019-02-28":62.07,"2019-03-29":65.95,"2019-04-30":64.12,
    "2019-05-31":36.46,"2019-06-28":44.45,"2019-07-31":41.48,"2019-08-30":44.55,
    "2019-09-30":41.35,"2019-10-31":53.57,"2019-11-29":66.63,"2019-12-31":83.67,
    "2020-01-31":154.49,"2020-02-28":133.38,"2020-03-31":94.02,"2020-04-30":152.83,
    "2020-05-29":161.57,"2020-06-30":211.48,"2020-07-31":309.20,"2020-08-31":441.33,
    "2020-09-30":429.01,"2020-10-30":388.04,"2020-11-30":567.60,"2020-12-31":705.67,
    "2021-01-29":793.53,"2021-02-26":675.50,"2021-03-31":661.75,"2021-04-30":711.08,
    "2021-05-28":604.97,"2021-06-30":678.90,"2021-07-30":700.84,"2021-08-31":735.35,
    "2021-09-30":775.22,"2021-10-29":1114.00,"2021-11-30":1144.76,"2021-12-31":1056.78,
    "2022-01-31":936.72,"2022-02-28":871.44,"2022-03-31":1000.19,"2022-04-29":870.76,
    "2022-05-31":758.26,"2022-06-30":681.79,"2022-07-29":891.12,"2022-08-31":893.56,
    "2022-09-30":265.25,"2022-10-31":227.54,"2022-11-30":182.92,"2022-12-30":123.18,
    "2023-01-31":173.22,"2023-02-28":202.77,"2023-03-31":207.46,"2023-04-28":160.61,
    "2023-05-31":203.93,"2023-06-30":261.77,"2023-07-31":269.78,"2023-08-31":245.01,
    "2023-09-29":250.22,"2023-10-31":197.36,"2023-11-30":234.30,"2023-12-29":248.48,
    "2024-01-31":187.29,"2024-02-29":201.88,"2024-03-29":175.79,"2024-04-30":147.05,
    "2024-05-31":178.79,"2024-06-28":197.88,"2024-07-31":232.10,"2024-08-30":214.14,
    "2024-09-30":261.63,"2024-10-31":249.85,"2024-11-29":352.56,"2024-12-31":403.84,
    "2025-01-01":404.60,"2025-02-01":292.98,"2025-03-01":259.16,"2025-04-01":282.16,
    "2025-05-01":346.46,"2025-06-01":317.66,"2025-07-01":308.27,"2025-08-01":333.87,
    "2025-09-01":444.72,"2025-10-01":456.56,"2025-11-01":430.17,"2025-12-01":449.72,
    "2026-01-01":430.41,"2026-02-01":402.51,"2026-03-01":371.75,
}

NVDA_PRICES = {
    "2019-01-31":35.16,"2019-02-28":42.78,"2019-03-29":45.94,"2019-04-30":47.59,
    "2019-05-31":32.98,"2019-06-28":39.32,"2019-07-31":40.66,"2019-08-30":39.42,
    "2019-09-30":43.66,"2019-10-31":52.91,"2019-11-29":58.11,"2019-12-31":59.00,
    "2020-01-31":62.28,"2020-02-28":52.50,"2020-03-31":48.31,"2020-04-30":68.99,
    "2020-05-29":71.34,"2020-06-30":92.80,"2020-07-31":103.55,"2020-08-31":131.38,
    "2020-09-30":133.32,"2020-10-30":110.96,"2020-11-30":138.96,"2020-12-31":131.38,
    "2021-01-29":135.43,"2021-02-26":133.87,"2021-03-31":145.22,"2021-04-30":180.10,
    "2021-05-28":166.49,"2021-06-30":196.85,"2021-07-30":204.36,"2021-08-31":220.89,
    "2021-09-30":206.34,"2021-10-29":277.14,"2021-11-30":330.42,"2021-12-31":294.11,
    "2022-01-31":234.96,"2022-02-28":232.72,"2022-03-31":273.62,"2022-04-29":216.09,
    "2022-05-31":169.00,"2022-06-30":155.33,"2022-07-29":185.00,"2022-08-31":138.49,
    "2022-09-30":123.85,"2022-10-31":133.16,"2022-11-30":165.81,"2022-12-30":146.11,
    "2023-01-31":174.66,"2023-02-28":231.83,"2023-03-31":277.77,"2023-04-28":277.23,
    "2023-05-31":378.34,"2023-06-30":423.02,"2023-07-31":467.65,"2023-08-31":493.55,
    "2023-09-29":434.99,"2023-10-31":405.00,"2023-11-30":467.70,"2023-12-29":495.22,
    "2024-01-31":612.42,"2024-02-29":788.17,"2024-03-29":903.56,"2024-04-30":762.00,
    "2024-05-31":1064.69,"2024-06-28":1208.88,"2024-07-31":117.93,"2024-08-30":125.61,
    "2024-09-30":121.44,"2024-10-31":139.56,"2024-11-29":138.85,"2024-12-31":134.25,
    "2025-01-01":120.07,"2025-02-01":124.92,"2025-03-01":108.38,"2025-04-01":108.92,
    "2025-05-01":135.13,"2025-06-01":157.99,"2025-07-01":177.87,"2025-08-01":174.18,
    "2025-09-01":186.58,"2025-10-01":202.49,"2025-11-01":177.00,"2025-12-01":186.50,
    "2026-01-01":191.13,"2026-02-01":177.19,"2026-03-01":174.40,
}


def build_price_df(prices_dict):
    df = pd.DataFrame(list(prices_dict.items()), columns=["date", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def xirr(cashflows):
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    years = [(c[0] - t0).days / 365.25 for c in cashflows]
    cfs = [c[1] for c in cashflows]
    def npv(r):
        return sum(cf / (1 + r) ** t for cf, t in zip(cfs, years))
    return brentq(npv, -0.9999, 100.0)


@st.cache_data
def simulate_fund(ticker, name, prices_dict, num_investments, committed_m, seed):
    price_df = build_price_df(prices_dict)
    committed = committed_m * 1_000_000
    rng = np.random.default_rng(seed)
    dates = price_df.index.tolist()
    n = len(dates)

    call_idxs = sorted(rng.choice(range(n // 2), size=num_investments, replace=False))
    call_dates = [dates[i] for i in call_idxs]
    call_amounts = rng.dirichlet(np.ones(num_investments)) * committed

    investments = []
    for d, amt in zip(call_dates, call_amounts):
        px = float(price_df.loc[d, "Close"])
        investments.append({"date": d, "shares": amt / px, "cost": amt, "price_in": px})

    exits = []
    for inv in investments:
        future = [i for i, d in enumerate(dates) if d > inv["date"] and i > n // 3]
        if not future:
            exits.append({})
            continue
        sell_idx = rng.choice(future[len(future) // 3:])
        sell_date = dates[sell_idx]
        shares_sold = inv["shares"] * rng.uniform(0.50, 0.80)
        px_out = float(price_df.loc[sell_date, "Close"])
        exits.append({
            "date": sell_date,
            "shares_sold": shares_sold,
            "proceeds": shares_sold * px_out,
            "price_out": px_out,
        })

    final_px = float(price_df.iloc[-1]["Close"])
    total_called = sum(i["cost"] for i in investments)
    total_dist = sum(e.get("proceeds", 0) for e in exits)
    residual = max(
        sum((investments[k]["shares"] - exits[k].get("shares_sold", 0)) * final_px
            for k in range(len(investments))), 0
    )
    total_val = total_dist + residual

    cf_list = (
        [(i["date"], -i["cost"]) for i in investments]
        + [(e["date"], e["proceeds"]) for e in exits if e]
        + [(dates[-1], residual)]
    )
    try:
        irr = xirr(cf_list) * 100
    except Exception:
        irr = float("nan")

    nav_rows = []
    for day in dates:
        called = sum(i["cost"] for i in investments if i["date"] <= day)
        if called == 0:
            continue
        dist_so_far = sum(
            e.get("proceeds", 0) for e in exits if e.get("date") and e["date"] <= day
        )
        px_today = float(price_df.loc[day, "Close"])
        unrealized = max(
            sum((investments[k]["shares"] - exits[k].get("shares_sold", 0)) * px_today
                for k in range(len(investments)) if investments[k]["date"] <= day), 0
        )
        nav = unrealized + dist_so_far
        nav_rows.append({
            "date": day,
            "NAV ($M)": round(nav / 1e6, 3),
            "Called ($M)": round(called / 1e6, 3),
            "Distributions ($M)": round(dist_so_far / 1e6, 3),
            "MOIC": round(nav / called, 3),
        })
    nav_df = pd.DataFrame(nav_rows).set_index("date")

    wf_rows = []
    for k, inv in enumerate(investments):
        ex = exits[k]
        proceeds = ex.get("proceeds", 0)
        wf_rows.append({
            "Investment": f"Inv {k + 1}",
            "Entry Date": inv["date"].strftime("%b %Y"),
            "Cost ($M)": round(inv["cost"] / 1e6, 2),
            "Proceeds ($M)": round(proceeds / 1e6, 2),
            "P&L ($M)": round((proceeds - inv["cost"]) / 1e6, 2),
            "Entry $": round(inv["price_in"], 1),
            "Exit $": round(ex.get("price_out", final_px), 1),
            "Multiple": round(proceeds / inv["cost"], 2) if inv["cost"] else 0,
        })
    wf_df = pd.DataFrame(wf_rows)

    kpis = {
        "IRR (%)": round(irr, 2),
        "MOIC": round(total_val / total_called, 3),
        "DPI": round(total_dist / total_called, 3),
        "TVPI": round(total_val / total_called, 3),
        "RVPI": round(residual / total_called, 3),
        "Called ($M)": round(total_called / 1e6, 1),
        "Distributions ($M)": round(total_dist / 1e6, 1),
        "Residual NAV ($M)": round(residual / 1e6, 1),
        "Total Value ($M)": round(total_val / 1e6, 1),
    }
    return kpis, nav_df, wf_df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Fund Parameters")
    fund_choice = st.selectbox(
        "Fund",
        ["Tesla Growth Fund (TSLA)", "NVIDIA Tech Fund (NVDA)", "Both"]
    )
    st.divider()
    committed = st.slider("Committed Capital ($M)", 50, 500, 100, step=25)
    num_inv = st.slider("Number of investments", 3, 10, 6)
    st.divider()
    st.markdown("**Metric definitions**")
    st.caption("**IRR** — XIRR on dated cash flows")
    st.caption("**MOIC** — Total value / invested")
    st.caption("**DPI** — Cash returned / called")
    st.caption("**TVPI** — (Dist + NAV) / called")
    st.caption("**RVPI** — Residual NAV / called")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 PE Fund Performance Dashboard")
st.caption("Simulated capital calls & distributions · Real Tesla & NVIDIA price data (2019–2026) · Yahoo Finance")

# ── Compute ───────────────────────────────────────────────────────────────────
funds_to_show = []
if "Tesla" in fund_choice or "Both" in fund_choice:
    funds_to_show.append(("TSLA", "Tesla Growth Fund", TSLA_PRICES, 42))
if "NVIDIA" in fund_choice or "Both" in fund_choice:
    funds_to_show.append(("NVDA", "NVIDIA Tech Fund", NVDA_PRICES, 99))

results = {}
for ticker, name, prices, seed in funds_to_show:
    kpis, nav_df, wf_df = simulate_fund(ticker, name, prices, num_inv, committed, seed)
    results[ticker] = {"name": name, "kpis": kpis, "nav": nav_df, "wf": wf_df}

# ── KPI cards ─────────────────────────────────────────────────────────────────
for ticker, r in results.items():
    k = r["kpis"]
    st.subheader(f"{'🚗' if ticker == 'TSLA' else '🟢'} {r['name']}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("IRR", f"{k['IRR (%)']:.1f}%")
    c2.metric("MOIC", f"{k['MOIC']:.2f}x")
    c3.metric("DPI", f"{k['DPI']:.2f}x")
    c4.metric("TVPI", f"{k['TVPI']:.2f}x")
    c5.metric("Called", f"${k['Called ($M)']}M")
    c6.metric(
        "Total Value",
        f"${k['Total Value ($M)']}M",
        delta=f"+${round(k['Total Value ($M)'] - k['Called ($M)'], 1)}M",
    )
    st.divider()

# ── NAV line chart ────────────────────────────────────────────────────────────
st.subheader("NAV vs Capital Called over time")
if len(results) == 1:
    ticker, r = list(results.items())[0]
    st.line_chart(r["nav"][["NAV ($M)", "Called ($M)", "Distributions ($M)"]])
else:
    nav_combined = pd.DataFrame({
        "TSLA NAV ($M)": results["TSLA"]["nav"]["NAV ($M)"],
        "NVDA NAV ($M)": results["NVDA"]["nav"]["NAV ($M)"],
        "TSLA Called ($M)": results["TSLA"]["nav"]["Called ($M)"],
        "NVDA Called ($M)": results["NVDA"]["nav"]["Called ($M)"],
    })
    st.line_chart(nav_combined)

# ── TVPI breakdown ────────────────────────────────────────────────────────────
st.subheader("DPI · RVPI · TVPI breakdown")
tvpi_data = pd.DataFrame([
    {
        "Fund": r["name"].replace(" Fund", "").replace(" Growth", "").replace(" Tech", ""),
        "DPI (realised)": r["kpis"]["DPI"],
        "RVPI (unrealised)": r["kpis"]["RVPI"],
    }
    for r in results.values()
]).set_index("Fund")
st.bar_chart(tvpi_data)

# ── Waterfall table + P&L bar ─────────────────────────────────────────────────
st.subheader("Per-investment waterfall (P&L $M)")
for ticker, r in results.items():
    if len(results) > 1:
        st.caption(r["name"])
    wf = r["wf"].copy()
    pnl_chart = wf.set_index("Investment")[["P&L ($M)"]]
    st.bar_chart(pnl_chart)

# ── Deal log ──────────────────────────────────────────────────────────────────
st.subheader("Deal log")
for ticker, r in results.items():
    if len(results) > 1:
        st.caption(r["name"])
    wf = r["wf"].copy()
    wf["Multiple"] = wf["Multiple"].apply(lambda x: f"{x:.2f}x")
    wf["P&L ($M)"] = wf["P&L ($M)"].apply(
        lambda x: f"+${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M"
    )
    st.dataframe(wf, use_container_width=True, hide_index=True)

# ── Downloads ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📥 Download data")
dl1, dl2, dl3 = st.columns(3)

all_kpi = pd.DataFrame([{"Fund": r["name"], **r["kpis"]} for r in results.values()])
all_nav = pd.concat([r["nav"].assign(ticker=t) for t, r in results.items()])
all_wf  = pd.concat([r["wf"].assign(ticker=t) for t, r in results.items()])

dl1.download_button("⬇ KPI summary CSV", all_kpi.to_csv(index=False), "kpis.csv",        "text/csv")
dl2.download_button("⬇ NAV series CSV",  all_nav.to_csv(),            "nav_series.csv",  "text/csv")
dl3.download_button("⬇ Waterfall CSV",   all_wf.to_csv(index=False),  "waterfall.csv",   "text/csv")

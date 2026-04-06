import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="PE Dashboard", layout="wide")

# ------------------ AUTO REFRESH (SAFE) ------------------
st.sidebar.markdown("### ⏱ Auto Refresh")

refresh_rate = st.sidebar.selectbox(
    "Refresh every (seconds)",
    [0, 30, 60, 120],
    index=2
)

if refresh_rate > 0:
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_rate}">',
        unsafe_allow_html=True
    )

# ------------------ STYLING ------------------
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.section-title {
    font-size:18px;
    font-weight:600;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ XIRR ------------------
def xirr(cashflows, guess=0.1):
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]

    def npv(rate):
        return sum(
            cf / (1 + rate) ** ((d - t0).days / 365.25)
            for d, cf in cashflows
        )

    rate = guess
    for _ in range(100):
        d_npv = (npv(rate + 1e-6) - npv(rate)) / 1e-6
        if abs(d_npv) < 1e-10:
            break
        rate -= npv(rate) / d_npv

    return rate

# ------------------ LIVE DATA ------------------
@st.cache_data(ttl=300)
def get_prices(ticker):
    df = yf.download(
        ticker,
        start="2019-01-01",
        end=datetime.today().strftime("%Y-%m-%d"),
        interval="1mo",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return {}

    df = df["Close"].dropna()

    # ✅ FIXED DATE HANDLING
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index = df.index.strftime("%Y-%m-%d")

    return df.to_dict()

# ------------------ SIMULATION ------------------
@st.cache_data
def simulate_fund(prices_dict, num_investments, committed_m, seed):

    # ✅ FIXED DATE PARSING
    price_df = pd.Series(prices_dict).to_frame(name="Close")
    price_df.index = pd.to_datetime(price_df.index, errors="coerce")
    price_df = price_df.dropna().sort_index()

    if price_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    committed = committed_m * 1_000_000

    rng = np.random.default_rng(seed)
    dates = price_df.index.tolist()
    n = len(dates)

    if n < 2:
        return {}, pd.DataFrame(), pd.DataFrame()

    max_possible = max(1, n // 2)
    num_inv_safe = min(num_investments, max_possible)

    call_idxs = sorted(
        rng.choice(range(max_possible), size=num_inv_safe, replace=False)
    )

    call_dates = [dates[i] for i in call_idxs]
    call_amounts = rng.dirichlet(np.ones(num_inv_safe)) * committed

    investments = []
    for d, amt in zip(call_dates, call_amounts):
        px = float(price_df.loc[d, "Close"])
        investments.append({"date": d, "shares": amt / px, "cost": amt})

    exits = []
    for inv in investments:
        future = [i for i, d in enumerate(dates) if d > inv["date"]]
        if not future:
            exits.append({})
            continue

        sell_idx = rng.choice(future[len(future)//2:])
        sell_date = dates[sell_idx]
        shares_sold = inv["shares"] * rng.uniform(0.5, 0.8)
        px_out = float(price_df.loc[sell_date, "Close"])

        exits.append({
            "date": sell_date,
            "shares_sold": shares_sold,
            "proceeds": shares_sold * px_out,
        })

    final_px = float(price_df.iloc[-1]["Close"])

    total_called = sum(i["cost"] for i in investments)
    total_dist = sum(e.get("proceeds", 0) for e in exits)

    residual = sum(
        (investments[k]["shares"] - exits[k].get("shares_sold", 0)) * final_px
        for k in range(len(investments))
    )

    total_val = total_dist + residual

    cf_list = (
        [(i["date"], -i["cost"]) for i in investments]
        + [(e["date"], e["proceeds"]) for e in exits if e]
        + [(dates[-1], residual)]
    )

    try:
        irr = xirr(cf_list) * 100
    except:
        irr = float("nan")

    # NAV
    nav_data = []
    for d in dates:
        px = float(price_df.loc[d, "Close"])

        unreal = sum(
            (investments[k]["shares"] - exits[k].get("shares_sold", 0)) * px
            for k in range(len(investments))
            if investments[k]["date"] <= d
        )

        dist = sum(
            e.get("proceeds", 0)
            for e in exits
            if e.get("date") and e["date"] <= d
        )

        nav_data.append({
            "date": d,
            "NAV": (unreal + dist) / 1e6
        })

    nav_df = pd.DataFrame(nav_data).set_index("date")

    # Waterfall
    wf = []
    for i, inv in enumerate(investments):
        proceeds = exits[i].get("proceeds", 0)
        wf.append({
            "Investment": f"Inv {i+1}",
            "P&L": (proceeds - inv["cost"]) / 1e6
        })

    wf_df = pd.DataFrame(wf).set_index("Investment")

    kpis = {
        "IRR": round(irr, 2),
        "MOIC": round(total_val / total_called, 2),
        "DPI": round(total_dist / total_called, 2),
        "TVPI": round(total_val / total_called, 2),
    }

    return kpis, nav_df, wf_df

# ------------------ UI ------------------
st.title("📊 PE Fund Dashboard (Live Data)")

with st.sidebar:
    st.header("Fund Settings")
    committed = st.slider("Committed Capital ($M)", 50, 500, 100)
    num_inv = st.slider("Number of Investments", 3, 10, 5)

# ------------------ LOAD DATA ------------------
TSLA_PRICES = get_prices("TSLA")
NVDA_PRICES = get_prices("NVDA")

if not TSLA_PRICES or not NVDA_PRICES:
    st.error("⚠️ Failed to load market data. Try refreshing.")
    st.stop()

# ------------------ RUN ------------------
tsla_kpi, tsla_nav, tsla_wf = simulate_fund(TSLA_PRICES, num_inv, committed, 42)
nvda_kpi, nvda_nav, nvda_wf = simulate_fund(NVDA_PRICES, num_inv, committed, 99)

# ------------------ KPI ------------------
st.subheader("📊 Fund Comparison")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🚗 Tesla Fund")
    st.metric("IRR", f"{tsla_kpi['IRR']}%")
    st.metric("MOIC", f"{tsla_kpi['MOIC']}x")
    st.metric("DPI", f"{tsla_kpi['DPI']}x")
    st.metric("TVPI", f"{tsla_kpi['TVPI']}x")

with c2:
    st.markdown("### 🟢 NVIDIA Fund")
    st.metric("IRR", f"{nvda_kpi['IRR']}%")
    st.metric("MOIC", f"{nvda_kpi['MOIC']}x")
    st.metric("DPI", f"{nvda_kpi['DPI']}x")
    st.metric("TVPI", f"{nvda_kpi['TVPI']}x")

# ------------------ NAV ------------------
st.subheader("📈 NAV Comparison")

nav_compare = pd.DataFrame({
    "TSLA": tsla_nav["NAV"],
    "NVDA": nvda_nav["NAV"]
})

st.line_chart(nav_compare)

# ------------------ DPI / RVPI ------------------
st.subheader("📊 DPI vs RVPI")

tvpi_df = pd.DataFrame({
    "DPI": [tsla_kpi["DPI"], nvda_kpi["DPI"]],
    "RVPI": [
        tsla_kpi["TVPI"] - tsla_kpi["DPI"],
        nvda_kpi["TVPI"] - nvda_kpi["DPI"]
    ]
}, index=["TSLA", "NVDA"])

st.bar_chart(tvpi_df)

# ------------------ WATERFALL ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚗 Tesla Waterfall")
    st.bar_chart(tsla_wf)

with col2:
    st.markdown("### 🟢 NVIDIA Waterfall")
    st.bar_chart(nvda_wf)

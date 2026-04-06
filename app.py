import streamlit as st
import pandas as pd
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="PE Dashboard", layout="wide")

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

# ------------------ PRICE DATA ------------------
TSLA_PRICES = {
    "2019-01-31":59.51,"2020-06-30":211.48,"2021-12-31":1056.78,
    "2022-12-30":123.18,"2023-12-29":248.48,"2024-12-31":403.84,
}

NVDA_PRICES = {
    "2019-01-31":35.16,"2020-06-30":92.80,"2021-12-31":294.11,
    "2022-12-30":146.11,"2023-12-29":495.22,"2024-12-31":134.25,
}

def build_price_df(prices_dict):
    df = pd.DataFrame(list(prices_dict.items()), columns=["date", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

# ------------------ SIMULATION ------------------
@st.cache_data
def simulate_fund(prices_dict, num_investments, committed_m, seed):
    price_df = build_price_df(prices_dict)
    committed = committed_m * 1_000_000

    rng = np.random.default_rng(seed)
    dates = price_df.index.tolist()
    n = len(dates)

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

    # NAV TIME SERIES
    nav_data = []
    for d in dates:
        called = sum(i["cost"] for i in investments if i["date"] <= d)
        dist = sum(e.get("proceeds", 0) for e in exits if e.get("date") and e["date"] <= d)

        px = float(price_df.loc[d, "Close"])
        unreal = sum(
            (investments[k]["shares"] - exits[k].get("shares_sold", 0)) * px
            for k in range(len(investments))
            if investments[k]["date"] <= d
        )

        nav = unreal + dist

        nav_data.append({
            "date": d,
            "NAV": nav / 1e6,
            "Called": called / 1e6,
            "Distributions": dist / 1e6
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
st.title("📊 PE Fund Performance Dashboard")

with st.sidebar:
    st.header("Fund Settings")
    committed = st.slider("Committed Capital ($M)", 50, 500, 100)
    num_inv = st.slider("Number of Investments", 3, 10, 5)

st.caption("⚠️ NVDA includes stock split distortion (demo only)")

# ------------------ RUN BOTH FUNDS ------------------
tsla_kpi, tsla_nav, tsla_wf = simulate_fund(TSLA_PRICES, num_inv, committed, 42)
nvda_kpi, nvda_nav, nvda_wf = simulate_fund(NVDA_PRICES, num_inv, committed, 99)

# ------------------ KPI COMPARISON ------------------
st.subheader("📊 Fund Comparison")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🚗 Tesla Growth Fund")
    st.metric("IRR", f"{tsla_kpi['IRR']}%")
    st.metric("MOIC", f"{tsla_kpi['MOIC']}x")
    st.metric("DPI", f"{tsla_kpi['DPI']}x")
    st.metric("TVPI", f"{tsla_kpi['TVPI']}x")

with c2:
    st.markdown("### 🟢 NVIDIA Tech Fund")
    st.metric("IRR", f"{nvda_kpi['IRR']}%")
    st.metric("MOIC", f"{nvda_kpi['MOIC']}x")
    st.metric("DPI", f"{nvda_kpi['DPI']}x")
    st.metric("TVPI", f"{nvda_kpi['TVPI']}x")

# ------------------ NAV COMPARISON ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">NAV Comparison</div>', unsafe_allow_html=True)

nav_compare = pd.DataFrame({
    "TSLA NAV": tsla_nav["NAV"],
    "NVDA NAV": nvda_nav["NAV"]
})

st.line_chart(nav_compare)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ DPI / RVPI ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">DPI vs RVPI</div>', unsafe_allow_html=True)

tvpi_df = pd.DataFrame({
    "DPI": [tsla_kpi["DPI"], nvda_kpi["DPI"]],
    "RVPI": [
        tsla_kpi["TVPI"] - tsla_kpi["DPI"],
        nvda_kpi["TVPI"] - nvda_kpi["DPI"]
    ]
}, index=["TSLA", "NVDA"])

st.bar_chart(tvpi_df)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ WATERFALL ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚗 Tesla Waterfall")
    st.bar_chart(tsla_wf)

with col2:
    st.markdown("### 🟢 NVIDIA Waterfall")
    st.bar_chart(nvda_wf)

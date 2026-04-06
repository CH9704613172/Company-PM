import streamlit as st
import pandas as pd
import numpy as np

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

# ------------------ CUSTOM XIRR (NO SCIPY) ------------------
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
# (Same as your data — shortened here for clarity, keep full data)
TSLA_PRICES = {
    "2019-01-31":59.51,"2019-02-28":62.07,"2019-03-29":65.95,
    "2024-12-31":403.84,
}

NVDA_PRICES = {
    "2019-01-31":35.16,"2019-02-28":42.78,"2019-03-29":45.94,
    "2024-12-31":134.25,
}

def build_price_df(prices_dict):
    df = pd.DataFrame(list(prices_dict.items()), columns=["date", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@st.cache_data
def simulate_fund(prices_dict, num_investments, committed_m, seed):
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

    kpis = {
        "IRR (%)": round(irr, 2),
        "MOIC": round(total_val / total_called, 2),
        "DPI": round(total_dist / total_called, 2),
        "TVPI": round(total_val / total_called, 2),
    }

    return kpis


# ------------------ UI ------------------
st.title("📊 PE Fund Dashboard")

with st.sidebar:
    committed = st.slider("Committed Capital ($M)", 50, 500, 100)
    num_inv = st.slider("Number of Investments", 3, 10, 5)

st.caption("⚠️ NVDA data includes stock split distortion (for demo only)")

tsla = simulate_fund(TSLA_PRICES, num_inv, committed, seed=42)
nvda = simulate_fund(NVDA_PRICES, num_inv, committed, seed=99)

c1, c2 = st.columns(2)

with c1:
    st.subheader("🚗 Tesla Fund")
    for k, v in tsla.items():
        st.metric(k, v)

with c2:
    st.subheader("🟢 NVIDIA Fund")
    for k, v in nvda.items():
        st.metric(k, v)

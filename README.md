Private Equity Fund Performance Dashboard
A Python + Streamlit-based Private Equity dashboard that simulates fund performance using real market data and computes key PE metrics such as IRR, MOIC, DPI, TVPI, and RVPI.
This project replicates how a Private Equity fund tracks capital calls, distributions, and portfolio valuation over time, providing an end-to-end view of fund performance.


Project Overview
Most finance dashboards show what happened.
This tool is designed to show how Private Equity funds actually work behind the scenes.


It simulates:
Capital calls (investments)
Partial exits (distributions)
Residual portfolio value (NAV)
Fund-level performance metrics

Built using real historical price data of Tesla & NVIDIA (2019–2026)

🎯 Key Features
✅ Simulates Private Equity fund lifecycle
✅ Tracks capital deployment & exits
✅ Computes industry-standard PE metrics
✅ Generates NAV over time
✅ Provides investment-level waterfall analysis
✅ Interactive dashboard with parameter controls
✅ Downloadable datasets (CSV)


📈 Key Metrics Explained
IRR (Internal Rate of Return)
Time-weighted return based on cash flows (XIRR)
MOIC (Multiple of Invested Capital)
Total Value ÷ Invested Capital
DPI (Distributed to Paid-In Capital)
Cash returned to investors
TVPI (Total Value to Paid-In Capital)
(Distributions + NAV) ÷ Paid-In
RVPI (Residual Value to Paid-In Capital)
Unrealized portfolio value



🛠 Tech Stack
Python
Streamlit
Pandas / NumPy
SciPy (XIRR using Brent method)


How It Works

Input Parameters
Committed Capital ($50M – $500M)
Number of investments
Fund selection (Tesla / NVIDIA / Both)

Simulation Logic
Randomized capital allocation using Dirichlet distribution
Entry based on historical prices
Partial exits (50%–80%) at future dates
Remaining holdings marked-to-market


Outputs
Fund KPIs (IRR, MOIC, etc.)
NAV vs Capital Called chart
TVPI breakdown (DPI + RVPI)
Investment-level waterfall (P&L)


Deal log
📊 Dashboard Highlights
📈 NAV vs Capital Called visualization
📊 DPI / RVPI / TVPI breakdown
💰 Investment-level P&L waterfall



💡 What I Learned
PE performance is driven more by cash flow timing than price movements
IRR is highly sensitive to exit timing
Diversification does not guarantee risk reduction in concentrated funds
Real-world funds balance realized (DPI) and unrealized (RVPI) returns


Private Equity Fund Accounting
NAV Calculation & Validation
Portfolio Valuation (MTM)
Investor Reporting (LP/GP)
Performance Measurement

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Doctor Copper V1",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .regime-risk-on  { color: #00d084; font-size: 1.4rem; font-weight: 700; }
    .regime-risk-off { color: #ff4b4b; font-size: 1.4rem; font-weight: 700; }
    .regime-neutral  { color: #ffd700; font-size: 1.4rem; font-weight: 700; }
    .small-label { color: #9da5b4; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Parameters")

lookback_days = st.sidebar.slider("Lookback window (days)", 180, 730, 365, 30)
zscore_window = st.sidebar.slider("Z-score rolling window (days)", 10, 60, 20, 5)
zscore_threshold = st.sidebar.slider("Risk-ON/OFF z-score threshold", 0.3, 1.5, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Cross-asset signals**")
use_dxy   = st.sidebar.checkbox("DXY (USD Index)",        value=True)
use_oil   = st.sidebar.checkbox("Oil (WTI Crude)",        value=True)
use_rates = st.sidebar.checkbox("10Y Treasury Yield",     value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Doctor Copper V1** — Cross-asset regime classifier\n\n"
    "Uses copper price z-scores + macro signals (DXY, Oil, Rates) "
    "to classify Risk-ON / Risk-OFF regimes and forecast equity returns.\n\n"
    "[GitHub](https://github.com/junshin0922/doctor-copper-V1)"
)

@st.cache_data(ttl=3600)
def load_data(lookback_days):
    end   = datetime.today()
    start = end - timedelta(days=lookback_days + 60)  # extra buffer for rolling calcs

    tickers = {
        "Copper":   "HG=F",
        "SPX":      "^GSPC",
        "DXY":      "DX-Y.NYB",
        "Oil":      "CL=F",
        "Rates_10Y":"^TNX",
    }
    raw = yf.download(list(tickers.values()), start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = list(tickers.keys())
    raw = raw.dropna(how="all").ffill().dropna()
    return raw

with st.spinner("Fetching market data…"):
    try:
        df_raw = load_data(lookback_days)
        data_ok = True
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        data_ok = False

if not data_ok:
    st.stop()

df = df_raw.copy()

df["Cu_ret"]    = df["Copper"].pct_change()
df["Cu_zscore"] = (
    df["Cu_ret"].rolling(zscore_window).mean() /
    df["Cu_ret"].rolling(zscore_window).std()
)

df["DXY_signal"]   = -df["DXY"].pct_change().rolling(zscore_window).mean()   # DXY up → risk-off
df["Oil_signal"]   =  df["Oil"].pct_change().rolling(zscore_window).mean()
df["Rates_signal"] =  df["Rates_10Y"].diff().rolling(zscore_window).mean()   # rising rates = risk-on (growth)

active_signals = [df["Cu_zscore"]]
if use_dxy:   active_signals.append(df["DXY_signal"]   / df["DXY_signal"].std())
if use_oil:   active_signals.append(df["Oil_signal"]   / df["Oil_signal"].std())
if use_rates: active_signals.append(df["Rates_signal"] / df["Rates_signal"].std())

df["composite"] = sum(active_signals) / len(active_signals)

def classify(z, thresh):
    if z >  thresh: return "RISK-ON"
    if z < -thresh: return "RISK-OFF"
    return "NEUTRAL"

df["regime"] = df["composite"].apply(lambda z: classify(z, zscore_threshold))

df = df.iloc[-lookback_days:]

df["SPX_fwd5"] = df["SPX"].pct_change(5).shift(-5) * 100

regime_returns = df.groupby("regime")["SPX_fwd5"].agg(["mean", "count", "std"]).rename(
    columns={"mean": "Avg 5d Return (%)", "count": "Observations", "std": "Volatility (%)"}
).round(3)

st.title("🔴 Doctor Copper V1")
st.caption("Cross-asset macro regime classifier · copper z-score + DXY · Oil · Rates")
st.markdown("---")

latest       = df.iloc[-1]
current_reg  = latest["regime"]
current_z    = latest["Cu_zscore"]
current_comp = latest["composite"]

reg_class = {
    "RISK-ON":  "regime-risk-on",
    "RISK-OFF": "regime-risk-off",
    "NEUTRAL":  "regime-neutral",
}[current_reg]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="small-label">Current Regime</div>
        <div class="{reg_class}">{current_reg}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.metric("Copper Z-Score",   f"{current_z:.2f}")

with c3:
    st.metric("Composite Signal", f"{current_comp:.2f}")

with c4:
    pct_risk_on = (df["regime"] == "RISK-ON").mean() * 100
    st.metric("Risk-ON % (period)", f"{pct_risk_on:.0f}%")

st.markdown("")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.45, 0.3, 0.25],
    vertical_spacing=0.04,
    subplot_titles=("S&P 500 with Regime Overlay", "Copper Z-Score & Composite Signal", "Signal Components"),
)

regime_colors = {"RISK-ON": "rgba(0,208,132,0.12)", "RISK-OFF": "rgba(255,75,75,0.12)", "NEUTRAL": "rgba(255,215,0,0.06)"}
prev_reg, start_idx = df["regime"].iloc[0], df.index[0]
for i, (idx, row) in enumerate(df.iterrows()):
    if row["regime"] != prev_reg or i == len(df) - 1:
        fig.add_vrect(
            x0=start_idx, x1=idx,
            fillcolor=regime_colors[prev_reg],
            line_width=0, row=1, col=1
        )
        prev_reg, start_idx = row["regime"], idx

fig.add_trace(go.Scatter(
    x=df.index, y=df["SPX"],
    line=dict(color="#4c8bf5", width=1.8),
    name="S&P 500", showlegend=True
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["Cu_zscore"],
    line=dict(color="#ff6b35", width=1.5),
    name="Copper Z-Score"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["composite"],
    line=dict(color="#ffffff", width=1.5, dash="dot"),
    name="Composite Signal"
), row=2, col=1)

for thresh, color in [(zscore_threshold, "rgba(0,208,132,0.5)"), (-zscore_threshold, "rgba(255,75,75,0.5)")]:
    fig.add_hline(y=thresh, line=dict(color=color, width=1, dash="dash"), row=2, col=1)
fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1), row=2, col=1)

signal_map = {"DXY": (use_dxy, df["DXY_signal"], "#ffd700"),
              "Oil": (use_oil, df["Oil_signal"], "#00d084"),
              "Rates": (use_rates, df["Rates_signal"], "#c084fc")}
for name, (active, series, color) in signal_map.items():
    if active:
        fig.add_trace(go.Scatter(
            x=df.index, y=series / series.std(),
            line=dict(color=color, width=1.2),
            name=f"{name} signal"
        ), row=3, col=1)

fig.update_layout(
    height=680,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9da5b4", size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#1e2530")
fig.update_yaxes(showgrid=True, gridcolor="#1e2530", zeroline=False)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📊 Regime → Equity Return Analysis")
st.caption("Average 5-day forward S&P 500 return by regime classification")

col_a, col_b = st.columns([1, 2])

with col_a:
    def color_regime(val):
        if isinstance(val, str):
            if val == "RISK-ON":  return "color: #00d084; font-weight:700"
            if val == "RISK-OFF": return "color: #ff4b4b; font-weight:700"
            if val == "NEUTRAL":  return "color: #ffd700; font-weight:700"
        return ""

    styled = regime_returns.style.applymap(color_regime).format({
        "Avg 5d Return (%)": "{:.2f}%",
        "Volatility (%)": "{:.2f}%",
        "Observations": "{:.0f}",
    })
    st.dataframe(styled, use_container_width=True)

with col_b:
    colors_bar = {
        "RISK-ON":  "#00d084",
        "RISK-OFF": "#ff4b4b",
        "NEUTRAL":  "#ffd700",
    }
    bar_fig = go.Figure(go.Bar(
        x=regime_returns.index,
        y=regime_returns["Avg 5d Return (%)"],
        marker_color=[colors_bar.get(r, "#888") for r in regime_returns.index],
        text=regime_returns["Avg 5d Return (%)"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
    ))
    bar_fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9da5b4", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(showgrid=True, gridcolor="#1e2530", zeroline=True, zerolinecolor="#555"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("### 🗓 Regime History (last 60 days)")
recent = df.tail(60)[["regime", "Cu_zscore", "composite", "SPX"]].copy()
recent.index = recent.index.strftime("%Y-%m-%d")

def highlight_regime(row):
    c = {"RISK-ON": "#0a2e1a", "RISK-OFF": "#2e0a0a", "NEUTRAL": "#2e2a0a"}.get(row["regime"], "")
    return [f"background-color: {c}"] * len(row)

st.dataframe(
    recent.style.apply(highlight_regime, axis=1).format({
        "Cu_zscore": "{:.3f}",
        "composite": "{:.3f}",
        "SPX": "{:,.0f}",
    }),
    use_container_width=True,
    height=300,
)

st.markdown("---")
st.caption("Data: Yahoo Finance · Built with Streamlit + Plotly · Doctor Copper V1 by Jun Shin")

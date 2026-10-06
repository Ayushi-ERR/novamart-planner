"""
NovaMart Demand Planner — Release 1 Prototype
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaMart Demand Planner",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label { font-size: 0.75rem; opacity: 0.7; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-size: 1.5rem; font-weight: 700; }
.badge-critical { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-low      { background:#ffa50022; color:#ff8c00; border:1px solid #ff8c0055;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-ok       { background:#21c35422; color:#21c354; border:1px solid #21c35455;
                  border-radius:4px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.section-header { font-size: 1rem; font-weight: 700; letter-spacing: 0.02em;
                  text-transform: uppercase; opacity: 0.5; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
AVG_UNIT_PRICE     = 45     # $ per unit
SALES_PER_STAFF_HR = 1800   # $ sales per staff hour
# Stock buffer by store type — Type A (large) holds more; Type C (small) holds less
STOCK_WEEKS = {"A": 3.5, "B": 2.8, "C": 2.0}


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/retail_data.csv", parse_dates=["date"])
    df["total_markdown"] = df[
        ["markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5"]
    ].fillna(0).sum(axis=1)
    df["holiday_name"] = df["holiday_name"].replace("None", "").fillna("")
    return df

df       = load_data()
last_date   = df["date"].max()
all_stores  = sorted(df["store_id"].unique())
all_depts   = sorted(df["department"].unique())


# ── Forecast helper ───────────────────────────────────────────────────────────
def generate_forecast(data: pd.DataFrame, weeks_ahead: int = 8):
    sales    = data["weekly_sales"].values
    n        = min(16, len(sales))
    baseline = np.mean(sales[-n:])
    std      = np.std(sales[-n:]) if n > 1 else baseline * 0.1
    trend    = (np.mean(sales[-4:]) - np.mean(sales[-8:-4])) / 4 if len(sales) >= 8 else 0
    last     = pd.Timestamp(data["date"].iloc[-1])
    future   = [last + timedelta(weeks=i + 1) for i in range(weeks_ahead)]
    fcst     = [max(0.0, baseline + trend * (i + 1)) for i in range(weeks_ahead)]
    upper    = [v + 1.5 * std for v in fcst]
    lower    = [max(0.0, v - 1.5 * std) for v in fcst]
    return future, fcst, upper, lower


# ── Stockout summary ─────────────────────────────────────────────────────────
@st.cache_data
def stockout_summary(df: pd.DataFrame) -> pd.DataFrame:
    c_hist   = last_date - timedelta(weeks=16)
    c_recent = last_date - timedelta(weeks=4)

    hist_avg   = df[df["date"] > c_hist].groupby("store_id")["weekly_sales"].mean().rename("hist_avg")
    recent_avg = df[df["date"] > c_recent].groupby("store_id")["weekly_sales"].mean().rename("recent_avg")

    info = (
        df[["store_id", "store_type", "store_size", "region"]]
        .drop_duplicates()
        .set_index("store_id")
    )
    s = info.join(hist_avg).join(recent_avg)

    s["stock_wks"]   = s["store_type"].map(STOCK_WEEKS).fillna(2.5)
    s["stock_est"]   = s["hist_avg"] * s["stock_wks"]
    s["weeks_cover"] = s["stock_est"] / s["recent_avg"].clip(1)
    s["demand_chg"]  = (s["recent_avg"] - s["hist_avg"]) / s["hist_avg"].clip(1) * 100

    def status(w):
        if w < 1.5: return "Critical"
        if w < 2.8: return "Low stock"
        return "OK"

    s["status"] = s["weeks_cover"].apply(status)
    return s.reset_index().sort_values("weeks_cover")

stockout = stockout_summary(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 NovaMart")
    st.markdown("**Demand Planner** · Sprint 1")
    st.divider()

    selected_store = st.selectbox("Store", all_stores, format_func=lambda x: f"Store {x:02d}")
    selected_dept  = st.selectbox("Department", all_depts, format_func=lambda x: f"Dept {x:02d}")

    store_row = df[df["store_id"] == selected_store].iloc[0]
    st.divider()
    st.markdown(
        f"**Type:** {store_row['store_type']} &nbsp;·&nbsp; "
        f"**Region:** {store_row['region']}\n\n"
        f"**Size:** {store_row['store_size']:,} sq ft"
    )
    st.divider()

    crit_n = (stockout["status"] == "Critical").sum()
    low_n  = (stockout["status"] == "Low stock").sum()
    ok_n   = (stockout["status"] == "OK").sum()
    st.markdown(f"""
**Stockout summary — all stores**

🔴 Critical: **{crit_n}**
🟡 Low stock: **{low_n}**
🟢 OK: **{ok_n}**
""")
    st.divider()
    st.caption(f"Data through {last_date.strftime('%d %b %Y')}")


# ── Filtered view ─────────────────────────────────────────────────────────────
view = df[
    (df["store_id"] == selected_store) &
    (df["department"] == selected_dept)
].sort_values("date").copy()

future_dates, fcst_vals, upper_vals, lower_vals = generate_forecast(view)

next_wk_fcst = fcst_vals[0]
units_order  = int(next_wk_fcst / AVG_UNIT_PRICE * 1.1)
hours_staff  = max(20, int(next_wk_fcst / SALES_PER_STAFF_HR))

store_stockout_row = stockout[stockout["store_id"] == selected_store].iloc[0]
this_woc = store_stockout_row["weeks_cover"]
this_status = store_stockout_row["status"]


# ── Page header ───────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"## Store {selected_store:02d} · Dept {selected_dept:02d}")
    st.caption(
        f"Type {store_row['store_type']} · {store_row['store_size']:,} sq ft · "
        f"{store_row['region']} region"
    )
with h2:
    badge_css = {"Critical": "badge-critical", "Low stock": "badge-low", "OK": "badge-ok"}
    badge_icon = {"Critical": "🔴", "Low stock": "🟡", "OK": "🟢"}
    st.markdown(
        f"<div style='text-align:right; padding-top:12px'>"
        f"<span class='{badge_css[this_status]}'>{badge_icon[this_status]} "
        f"{this_status} — {this_woc:.1f} wks cover</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
avg_w = view["weekly_sales"].mean()
yoy_delta = None
if len(view) >= 52:
    prev_yr = view.iloc[-52:-26]["weekly_sales"].mean()
    this_yr = view.iloc[-26:]["weekly_sales"].mean()
    yoy_delta = f"{(this_yr - prev_yr) / prev_yr * 100:+.1f}% vs prior year"

k1.metric("Next-week forecast",   f"${next_wk_fcst:,.0f}",  yoy_delta)
k2.metric("Units to order",       f"{units_order:,}",        "incl. 10% safety buffer")
k3.metric("Staff hours needed",   f"{hours_staff} hrs")
k4.metric("Weeks of stock cover", f"{this_woc:.1f}",
          delta=f"{this_woc - store_stockout_row['stock_wks']:.1f} vs buffer target",
          delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROW A — Forecast  |  Sales Drivers
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="section-header">📈 Sales Forecast</p>', unsafe_allow_html=True)

    hist = view.tail(26)
    hols = hist[hist["is_holiday"] == 1]

    fig = go.Figure()

    xs = [pd.Timestamp(d) for d in future_dates]
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=upper_vals + lower_vals[::-1],
        fill="toself",
        fillcolor="rgba(0,184,148,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast range",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["weekly_sales"],
        mode="lines",
        name="Actual sales",
        line=dict(color="#00b894", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=[hist["date"].iloc[-1]] + xs,
        y=[hist["weekly_sales"].iloc[-1]] + fcst_vals,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#00b894", width=2, dash="dash"),
        marker=dict(size=5, color="#00b894"),
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=upper_vals, mode="lines",
        line=dict(color="rgba(0,184,148,0.3)", width=1, dash="dot"),
        name="Upper bound", hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=lower_vals, mode="lines",
        line=dict(color="rgba(0,184,148,0.3)", width=1, dash="dot"),
        name="Lower bound", hoverinfo="skip", showlegend=False,
    ))

    if not hols.empty:
        fig.add_trace(go.Scatter(
            x=hols["date"], y=hols["weekly_sales"],
            mode="markers+text",
            name="Holiday",
            text=hols["holiday_name"],
            textposition="top center",
            textfont=dict(size=9, color="#fdcb6e"),
            marker=dict(color="#fdcb6e", size=11, symbol="star"),
        ))

    fig.add_vline(
        x=last_date, line_dash="dot", line_color="rgba(150,150,150,0.6)",
        annotation_text="Today", annotation_position="top left",
        annotation_font_color="rgba(150,150,150,0.9)",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", title=""),
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(128,128,128,0.15)", title="Weekly Sales"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, x=0, font_size=11),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Last 26 weeks actuals + 8-week rolling-average forecast · {len(view)} total data points")


with col2:
    st.markdown('<p class="section-header">📊 Sales Drivers</p>', unsafe_allow_html=True)

    hol_avg  = view[view["is_holiday"] == 1]["weekly_sales"].mean()
    non_hol  = view[view["is_holiday"] == 0]["weekly_sales"].mean()
    hol_eff  = ((hol_avg - non_hol) / non_hol * 100) if non_hol > 0 and not np.isnan(hol_avg) else 0

    season_avg = view.groupby("season")["weekly_sales"].mean()
    overall    = view["weekly_sales"].mean()
    fall_eff   = ((season_avg.get("Fall", overall) - overall) / overall * 100) if overall > 0 else 0

    md_corr = view["total_markdown"].corr(view["weekly_sales"])
    md_eff  = abs(md_corr) * 40 if not np.isnan(md_corr) else 0

    type_avg = df.groupby("store_type")["weekly_sales"].mean()
    c_avg    = type_avg.get("C", 1)
    this_avg = type_avg.get(store_row["store_type"], c_avg)
    size_eff = (this_avg - c_avg) / c_avg * 100 if c_avg > 0 else 0

    drivers = pd.DataFrame({
        "Driver": ["Holiday lift", "Store size effect", "Fall season", "Markdown impact"],
        "Effect": [hol_eff, max(0, size_eff), fall_eff, md_eff],
        "Color":  [
            "#fdcb6e" if hol_eff > 0 else "#636e72",
            "#74b9ff" if size_eff > 0 else "#636e72",
            "#a29bfe",
            "#fd79a8" if md_eff > 0 else "#636e72",
        ],
    }).sort_values("Effect")

    fig2 = go.Figure(go.Bar(
        x=drivers["Effect"],
        y=drivers["Driver"],
        orientation="h",
        marker_color=drivers["Color"].tolist(),
        text=[f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%" for v in drivers["Effect"]],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig2.update_layout(
        height=200,
        margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(
            range=[0, max(drivers["Effect"].max() * 1.4, 10)],
            gridcolor="rgba(128,128,128,0.15)",
            title="Relative effect (%)",
        ),
        yaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if not season_avg.empty:
        s_df = season_avg.reset_index().rename(columns={"season": "Season", "weekly_sales": "Avg/wk"})
        s_df["vs overall"] = ((s_df["Avg/wk"] - overall) / overall * 100).map(lambda x: f"{x:+.1f}%")
        s_df["Avg/wk"]     = s_df["Avg/wk"].map("${:,.0f}".format)
        st.dataframe(s_df, hide_index=True, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROW B — Planning  |  Stockout alerts
# ══════════════════════════════════════════════════════════════════════════════
col3, col4 = st.columns([1, 1], gap="large")


with col3:
    st.markdown('<p class="section-header">📋 Planning</p>', unsafe_allow_html=True)

    if "confirmed_plans" not in st.session_state:
        st.session_state.confirmed_plans = {}

    plan_key = f"{selected_store}_{selected_dept}"
    confirmed = st.session_state.confirmed_plans.get(plan_key, False)

    if confirmed:
        st.success(f"✅ Plan confirmed — Store {selected_store:02d} / Dept {selected_dept:02d}")
        if st.button("↩ Revise plan", use_container_width=True):
            st.session_state.confirmed_plans[plan_key] = False
            st.rerun()
    else:
        st.info("Review the 4-week plan below, then confirm.")

    plan_rows = []
    for i in range(4):
        d, f, u, l = future_dates[i], fcst_vals[i], upper_vals[i], lower_vals[i]
        plan_rows.append({
            "Week":        pd.Timestamp(d).strftime("%d %b"),
            "Forecast":    f"${f:,.0f}",
            "Range":       f"${l:,.0f}–${u:,.0f}",
            "Order (u)":   f"{int(f / AVG_UNIT_PRICE * 1.1):,}",
            "Staff (hrs)": f"{max(20, int(f / SALES_PER_STAFF_HR))}",
        })

    st.dataframe(pd.DataFrame(plan_rows), hide_index=True, use_container_width=True)

    if not confirmed:
        if st.button("✅ Confirm plan", type="primary", use_container_width=True):
            st.session_state.confirmed_plans[plan_key] = True
            st.rerun()

    st.caption(f"Assumptions: ${AVG_UNIT_PRICE}/unit · ${SALES_PER_STAFF_HR:,} sales/staff hr · +10% safety buffer on orders")


with col4:
    st.markdown('<p class="section-header">🔔 Stockout Alerts</p>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("🔴 Critical",  crit_n, "Immediate action")
    s2.metric("🟡 Low stock", low_n,  "Order this week")
    s3.metric("🟢 OK",        ok_n,   "On track")

    tab_crit, tab_low, tab_ok = st.tabs([
        f"🔴 Critical ({crit_n})",
        f"🟡 Low stock ({low_n})",
        f"🟢 OK ({ok_n})",
    ])

    def render_store_expander(row, df_all):
        sid    = int(row["store_id"])
        status = row["status"]
        woc    = row["weeks_cover"]
        dchg   = row.get("demand_chg", 0)

        badge_html = {
            "Critical": "<span class='badge-critical'>🔴 CRITICAL</span>",
            "Low stock": "<span class='badge-low'>🟡 LOW STOCK</span>",
            "OK":        "<span class='badge-ok'>🟢 OK</span>",
        }

        label = (
            f"Store {sid:02d} · Type {row['store_type']} · {row['region']} "
            f"— {woc:.1f} wks cover · demand {dchg:+.1f}%"
        )

        with st.expander(label, expanded=(status == "Critical")):
            st.markdown(badge_html[status], unsafe_allow_html=True)

            ma, mb, mc = st.columns(3)
            ma.metric("Weeks of cover",  f"{woc:.1f}")
            mb.metric("Recent demand chg", f"{dchg:+.1f}%",
                      delta_color="inverse" if dchg > 0 else "normal")
            mc.metric("Avg weekly (recent)", f"${row['recent_avg']:,.0f}")

            c_r = last_date - timedelta(weeks=4)
            dept_b = (
                df_all[(df_all["store_id"] == sid) & (df_all["date"] > c_r)]
                .groupby("department")["weekly_sales"].mean()
                .reset_index()
                .rename(columns={"department": "Dept", "weekly_sales": "Avg wkly ($)"})
                .sort_values("Avg wkly ($)", ascending=False)
                .head(8)
            )
            dept_b["Est. stock ($)"] = (dept_b["Avg wkly ($)"] * row["stock_wks"]).map("${:,.0f}".format)
            dept_b["Avg wkly ($)"]   = dept_b["Avg wkly ($)"].map("${:,.0f}".format)
            st.caption("**Top departments by sales (last 4 weeks)**")
            st.dataframe(dept_b, hide_index=True, use_container_width=True)

            trend_data = (
                df_all[df_all["store_id"] == sid]
                .groupby("date")["weekly_sales"].sum()
                .reset_index().sort_values("date").tail(16)
            )
            line_color = "#ff4b4b" if status == "Critical" else "#ffa500" if status == "Low stock" else "#00b894"
            fig_t = go.Figure(go.Scatter(
                x=trend_data["date"], y=trend_data["weekly_sales"],
                mode="lines", fill="tozeroy",
                fillcolor=f"{line_color}18",
                line=dict(color=line_color, width=2),
            ))
            fig_t.update_layout(
                height=110,
                margin=dict(l=0, r=0, t=6, b=0),
                xaxis=dict(showticklabels=True, gridcolor="rgba(128,128,128,0.1)"),
                yaxis=dict(tickformat="$,.0f", gridcolor="rgba(128,128,128,0.1)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.caption("**Store-level weekly sales — last 16 weeks**")
            st.plotly_chart(fig_t, use_container_width=True, key=f"trend_chart_{sid}")

            if status in ("Critical", "Low stock"):
                rk = f"reorder_{sid}"
                vk = f"reviewed_{sid}"
                if rk not in st.session_state: st.session_state[rk] = False
                if vk not in st.session_state: st.session_state[vk] = False

                b1, b2 = st.columns(2)
                if b1.button("📦 Initiate reorder", key=f"btn_r_{sid}", use_container_width=True,
                             type="primary" if not st.session_state[rk] else "secondary"):
                    st.session_state[rk] = True
                if b2.button("✓ Mark reviewed", key=f"btn_v_{sid}", use_container_width=True):
                    st.session_state[vk] = True

                if st.session_state[rk]:
                    st.success(f"✅ Reorder flagged for Store {sid:02d} — notify procurement team.")
                if st.session_state[vk]:
                    st.info(f"Store {sid:02d} marked as reviewed.")

    with tab_crit:
        crit_stores = stockout[stockout["status"] == "Critical"]
        if crit_stores.empty:
            st.success("No stores are in critical status right now.")
        else:
            with st.container(height=420, border=False):
                for _, row in crit_stores.iterrows():
                    render_store_expander(row, df)

    with tab_low:
        low_stores = stockout[stockout["status"] == "Low stock"]
        if low_stores.empty:
            st.success("No stores are on low stock right now.")
        else:
            with st.container(height=420, border=False):
                for _, row in low_stores.iterrows():
                    render_store_expander(row, df)

    with tab_ok:
        ok_stores = stockout[stockout["status"] == "OK"]
        if ok_stores.empty:
            st.info("No stores are currently OK.")
        else:
            with st.container(height=420, border=False):
                for _, row in ok_stores.sort_values("weeks_cover", ascending=False).iterrows():
                    render_store_expander(row, df)

    st.caption(
        f"⚠️ Stock buffer: Type A = 3.5 wks · Type B = 2.8 wks · Type C = 2.0 wks of historical avg sales. "
        "Weeks cover = estimated stock ÷ recent 4-week demand. Not live inventory."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "NovaMart Demand Planner · Sprint 1 Prototype · "
    "Data: Jan 2022 – Dec 2024 · Forecasts are indicative only · Not for operational use"
)
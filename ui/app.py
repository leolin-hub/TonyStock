"""
ui/app.py
Streamlit dashboard for StockScreener.

Layout:
  Sidebar  — model threshold + stock search
  Main     — top stocks table (by win_prob) + detail chart for selected stock
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="StockScreener", layout="wide")
st.title("📊 StockScreener — 模型選股")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("篩選條件")

    min_win_prob = st.slider(
        "模型預測門檻（下週漲 20% 機率 ≥）",
        min_value=0, max_value=100, value=50, step=5,
        format="%d%%",
    )

    top_n = st.slider("顯示前幾名", min_value=10, max_value=200, value=50, step=10)

    st.divider()
    st.subheader("個股查詢")
    search_symbol = st.text_input("股票代號", placeholder="例：2330").strip().upper()


# ── Top stocks table ──────────────────────────────────────────────────────────

with st.spinner("載入資料..."):
    try:
        params = {
            "min_win_prob": min_win_prob / 100,
            "top_n": top_n,
        }
        if search_symbol:
            params["symbols"] = search_symbol

        resp = requests.get(f"{API_BASE}/screen", params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ 無法連線到 API，請先執行 `uvicorn api.main:app`")
        st.stop()
    except Exception as e:
        st.error(f"API 錯誤：{e}")
        st.stop()

if not results:
    st.warning("沒有符合條件的股票，試著降低門檻。")
    st.stop()

df = pd.DataFrame(results)
df["win_prob_pct"] = (df["win_prob"] * 100).round(1).astype(str) + "%"
df["latest_score_display"] = df["latest_score"].apply(
    lambda s: "⭐" * int(s) if pd.notna(s) and s else "—"
)

st.subheader(f"符合條件：{len(df)} 支股票（排序依模型預測機率）")

display_df = df[[
    "symbol", "win_prob_pct", "latest_score_display", "latest_week",
]].rename(columns={
    "symbol":               "股票代號",
    "win_prob_pct":         "模型預測（漲20%機率）",
    "latest_score_display": "最新籌碼分數",
    "latest_week":          "資料週",
})

event = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)


# ── Detail chart ──────────────────────────────────────────────────────────────

# 決定要看哪支股票：表格點選 > sidebar 搜尋
selected = None
if event.selection and event.selection.get("rows"):
    selected = df.iloc[event.selection["rows"][0]]["symbol"]
elif search_symbol:
    selected = search_symbol

if selected:
    st.divider()
    st.subheader(f"{selected} — 週度走勢")

    with st.spinner(f"載入 {selected}..."):
        try:
            detail_resp = requests.get(f"{API_BASE}/stock/{selected}", timeout=10)
            detail_resp.raise_for_status()
            detail = detail_resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.error(f"找不到股票代號 {selected}")
            else:
                st.error(f"API 錯誤：{e}")
            st.stop()
        except Exception as e:
            st.error(f"無法載入 {selected}：{e}")
            st.stop()

    detail_df = pd.DataFrame(detail)
    detail_df["week_start"] = pd.to_datetime(detail_df["week_start"])
    detail_df = detail_df.sort_values("week_start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 籌碼分數走勢")
        score_long = detail_df.melt(
            id_vars="week_start",
            value_vars=["score_foreign", "score_trust", "score_dealer", "score_total"],
            var_name="type", value_name="score",
        ).dropna()
        score_long["type"] = score_long["type"].map({
            "score_foreign": "外資",
            "score_trust":   "投信",
            "score_dealer":  "自營商",
            "score_total":   "三大合計",
        })
        chart = (
            alt.Chart(score_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_start:T", title="週"),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[1, 5]), title="Score (1–5)"),
                color=alt.Color("type:N", title="類型"),
                tooltip=["week_start:T", "type:N", "score:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("##### 收盤價 & 模型預測機率")
        price_df = detail_df.dropna(subset=["close"])

        if price_df.empty:
            st.info("無收盤價資料")
        else:
            base = alt.Chart(price_df).encode(x=alt.X("week_start:T", title="週"))

            price_line = base.mark_area(line=True, opacity=0.2).encode(
                y=alt.Y("close:Q", title="收盤價", axis=alt.Axis(titleColor="#1f77b4")),
                tooltip=["week_start:T", "close:Q"],
            )

            prob_df = detail_df.dropna(subset=["win_prob"])
            if not prob_df.empty:
                prob_line = (
                    alt.Chart(prob_df)
                    .mark_line(color="orange", strokeDash=[4, 2])
                    .encode(
                        x=alt.X("week_start:T"),
                        y=alt.Y(
                            "win_prob:Q",
                            title="漲20%預測機率",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(titleColor="orange", format=".0%"),
                        ),
                        tooltip=["week_start:T",
                                 alt.Tooltip("win_prob:Q", format=".1%", title="win_prob")],
                    )
                )
                chart = alt.layer(price_line, prob_line).resolve_scale(y="independent")
            else:
                chart = price_line

            st.altair_chart(chart.properties(height=300), use_container_width=True)

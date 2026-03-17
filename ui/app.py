"""
ui/app.py
Streamlit dashboard for StockScreener.

Layout:
  Sidebar  — screening params
  Main     — ranked results table + weekly score/price/win_prob chart for selected stock
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="StockScreener", layout="wide")
st.title("📊 StockScreener — 籌碼面選股")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("篩選條件")

    win_rate_threshold = st.slider(
        "最低歷史勝率", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        format="%.0f%%",
        help="只顯示歷史勝率 >= 此門檻的股票",
    )

    score_threshold = st.select_slider(
        "籌碼訊號門檻（score_total >=）",
        options=[1, 2, 3, 4, 5], value=4,
        help="幾分以上算一次買進訊號",
    )

    n_weeks = st.slider(
        "觀察週數（訊號後 N 週）",
        min_value=1, max_value=12, value=4,
        help="訊號出現後幾週看漲跌結果",
    )

    return_threshold_pct = st.slider(
        "獲利門檻",
        min_value=1, max_value=20, value=5, step=1,
        format="%d%%",
        help="N 週後漲幾 % 算贏",
    )
    return_threshold = return_threshold_pct / 100

    symbols_input = st.text_input(
        "指定股票（逗號分隔，可留空）",
        placeholder="2330, 2454",
    )

    run = st.button("🔍 開始篩選", use_container_width=True)


# ── Screen ────────────────────────────────────────────────────────────────────

if run:
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()] or None

    with st.spinner("查詢中..."):
        try:
            resp = requests.post(f"{API_BASE}/screen", json={
                "win_rate_threshold": win_rate_threshold,
                "score_threshold":    score_threshold,
                "n_weeks":            n_weeks,
                "return_threshold":   return_threshold,
                "symbols":            symbols,
            }, timeout=10)
            resp.raise_for_status()
            results = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ 無法連線到 API，請先執行 `uvicorn api.main:app`")
            st.stop()
        except Exception as e:
            st.error(f"API 錯誤：{e}")
            st.stop()

    if not results:
        st.warning("沒有符合條件的股票，試著降低勝率門檻。")
        st.stop()

    df = pd.DataFrame(results)
    df["win_rate_pct"]      = (df["win_rate"] * 100).round(1).astype(str) + "%"
    df["win_prob_pct"]      = df["latest_win_prob"].apply(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"
    )
    df["latest_score_display"] = df["latest_score"].apply(
        lambda s: "⭐" * int(s) if pd.notna(s) and s else "—"
    )

    st.subheader(f"符合條件：{len(df)} 支股票")

    display_df = df[[
        "symbol", "win_rate_pct", "total_signals", "wins",
        "latest_score_display", "win_prob_pct", "latest_week",
    ]].rename(columns={
        "symbol":               "股票代號",
        "win_rate_pct":         "歷史勝率",
        "total_signals":        "訊號次數",
        "wins":                 "獲勝次數",
        "latest_score_display": "最新籌碼",
        "win_prob_pct":         "模型預測(下週漲5%)",
        "latest_week":          "最新週",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Detail chart ──────────────────────────────────────────────────────────

    st.divider()
    selected = st.selectbox("查看個股週度走勢", df["symbol"].tolist())

    if selected:
        with st.spinner(f"載入 {selected} 歷史資料..."):
            try:
                detail_resp = requests.get(f"{API_BASE}/stock/{selected}", timeout=10)
                detail_resp.raise_for_status()
                detail = detail_resp.json()
            except Exception as e:
                st.error(f"無法載入 {selected}：{e}")
                st.stop()

        detail_df = pd.DataFrame(detail)
        detail_df["week_start"] = pd.to_datetime(detail_df["week_start"])
        detail_df = detail_df.sort_values("week_start")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### {selected} — 籌碼分數走勢")
            score_long = detail_df.melt(
                id_vars="week_start",
                value_vars=["score_foreign", "score_trust", "score_dealer", "score_total"],
                var_name="類型", value_name="分數",
            ).dropna()
            label_map = {
                "score_foreign": "外資",
                "score_trust":   "投信",
                "score_dealer":  "自營商",
                "score_total":   "三大合計",
            }
            score_long["類型"] = score_long["類型"].map(label_map)

            chart = (
                alt.Chart(score_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("week_start:T", title="週"),
                    y=alt.Y("分數:Q", scale=alt.Scale(domain=[1, 5]), title="Score (1–5)"),
                    color=alt.Color("類型:N"),
                    tooltip=["week_start:T", "類型:N", "分數:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.markdown(f"#### {selected} — 收盤價 & 模型預測機率")
            price_df = detail_df.dropna(subset=["close"])

            if price_df.empty:
                st.info("無收盤價資料")
            else:
                base = alt.Chart(price_df).encode(
                    x=alt.X("week_start:T", title="週")
                )

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
                                title="模型預測機率",
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

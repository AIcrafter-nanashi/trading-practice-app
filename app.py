from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import ssl
import urllib.request
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


INITIAL_CAPITAL = 100000.0
MIN_START_INDEX = 30

TICKERS = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
    "BNB/USDT": "BNBUSDT",
}

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


INTERVAL_MAP = {"1日足": "1d", "1時間足": "1h", "15分足": "15m"}
INTERVAL_LIMIT = {"1d": 1000, "1h": 1000, "15m": 1000}


@dataclass(frozen=True)
class AppConfig:
    ticker: str
    interval: str
    bar_limit: int


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "screen": "top",
        "ticker": "BTC/USDT",
        "interval": "1d",
        "bar_limit": 50,
        "data": None,
        "current_index": MIN_START_INDEX,
        "start_index": MIN_START_INDEX,
        "end_index": None,
        "initial_capital": INITIAL_CAPITAL,
        "user_capital": INITIAL_CAPITAL,
        "user_position": None,
        "user_trades": [],
        "session_finished": False,
        "quiz_questions": [],
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_answers": [],
        "quiz_answered": False,
        "quiz_history": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_market_data(ticker: str, interval: str, bars: int) -> pd.DataFrame:
    symbol = TICKERS.get(ticker, ticker)
    limit = min(max(bars + MIN_START_INDEX + 50, 200), 1000)
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval={interval}&limit={limit}"
    )
    try:
        with urllib.request.urlopen(url, context=_SSL_CTX, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbv", "tqv", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = add_indicators(df)
    return df.tail(max(bars + MIN_START_INDEX + 20, 100)).copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ma5"] = SMAIndicator(close=result["close"], window=5).sma_indicator()
    result["ma20"] = SMAIndicator(close=result["close"], window=20).sma_indicator()
    bb = BollingerBands(close=result["close"], window=20, window_dev=2.0)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_middle"] = bb.bollinger_mavg()
    result["bb_lower"] = bb.bollinger_lband()
    atr = AverageTrueRange(
        high=result["high"], low=result["low"], close=result["close"], window=14,
    )
    result["atr"] = atr.average_true_range()
    return result.dropna().copy()


def reset_session(config: AppConfig) -> None:
    data = load_market_data(config.ticker, config.interval, config.bar_limit)
    if data.empty or len(data) <= MIN_START_INDEX + 5:
        st.error("十分なマーケットデータを取得できませんでした。銘柄や足種を変更してください。")
        return
    start_index = MIN_START_INDEX
    end_index = min(len(data) - 1, start_index + config.bar_limit)
    st.session_state.update({
        "screen": "practice",
        "ticker": config.ticker,
        "interval": config.interval,
        "bar_limit": config.bar_limit,
        "data": data,
        "current_index": start_index,
        "start_index": start_index,
        "end_index": end_index,
        "initial_capital": INITIAL_CAPITAL,
        "user_capital": INITIAL_CAPITAL,
        "user_position": None,
        "user_trades": [],
        "session_finished": False,
    })


def render_chart(df: pd.DataFrame, current_index: int) -> None:
    visible = df.iloc[: current_index + 1]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=visible.index, open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=visible.index, y=visible["ma5"], mode="lines",
                             name="MA5（短期）", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=visible.index, y=visible["ma20"], mode="lines",
                             name="MA20（長期）", line=dict(color="royalblue", width=1.5)))
    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=28, b=10),
        xaxis_rangeslider_visible=False, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_status_panel() -> None:
    user_pos = st.session_state.user_position
    col1, col2 = st.columns(2)
    col1.metric("資金", f"{st.session_state.user_capital:,.0f}円")
    col2.metric(
        "進行",
        f"{st.session_state.current_index - st.session_state.start_index} / {st.session_state.bar_limit}本",
    )
    if user_pos:
        side = "🔼 Buy" if user_pos["side"] == "long" else "🔽 Sell"
        st.info(
            f"{side} | エントリー: {user_pos['entry_price']:.4f} | "
            f"SL: {user_pos['sl']:.4f} | TP: {user_pos['tp']:.4f}"
        )


def render_controls() -> None:
    has_pos = st.session_state.user_position is not None
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Next ▶", use_container_width=True):
        advance_one_bar(); st.rerun()
    if c2.button("🟢 Buy", disabled=has_pos, use_container_width=True):
        open_user_position("long"); st.rerun()
    if c3.button("🔴 Sell", disabled=has_pos, use_container_width=True):
        open_user_position("short"); st.rerun()
    if c4.button("Close", disabled=not has_pos, use_container_width=True):
        close_user_position("manual"); st.rerun()
    if c5.button("Finish", use_container_width=True):
        finish_session(); st.rerun()


def advance_one_bar() -> None:
    if st.session_state.session_finished:
        return
    next_index = st.session_state.current_index + 1
    if next_index >= len(st.session_state.data) or next_index > st.session_state.end_index:
        finish_session()
        return
    st.session_state.current_index = next_index
    update_position_extremes("user_position", next_index)
    check_exit("user_position", "user_trades", "user_capital", next_index)
    if st.session_state.current_index >= st.session_state.end_index:
        finish_session()


def open_user_position(side: str) -> None:
    if st.session_state.user_position is not None:
        return
    position = create_position(side, st.session_state.user_capital, st.session_state.current_index, 2.0, 4.0)
    if position:
        st.session_state.user_position = position


def close_user_position(reason: str) -> None:
    if st.session_state.user_position is None:
        return
    close_position("user_position", "user_trades", "user_capital", st.session_state.current_index, reason)


def create_position(
    side: str, capital: float, index: int, sl_multiplier: float, tp_multiplier: float,
) -> dict[str, Any] | None:
    row = st.session_state.data.iloc[index]
    atr = float(row["atr"])
    if pd.isna(atr) or atr <= 0:
        return None
    entry = float(row["close"])
    sl_width = atr * sl_multiplier
    tp_width = atr * tp_multiplier
    if sl_width == 0:
        return None
    if side == "long":
        sl, tp = entry - sl_width, entry + tp_width
    else:
        sl, tp = entry + sl_width, entry - tp_width
    size = max(capital, 1000.0) * 0.01 / sl_width
    return {
        "side": side,
        "entry_index": index,
        "entry_time": st.session_state.data.index[index],
        "entry_price": entry,
        "size": size,
        "sl": sl,
        "tp": tp,
        "mfe": 0.0,
        "mae": 0.0,
    }


def update_position_extremes(position_key: str, index: int) -> None:
    position = st.session_state[position_key]
    if position is None:
        return
    row = st.session_state.data.iloc[index]
    high, low = float(row["high"]), float(row["low"])
    entry = float(position["entry_price"])
    if position["side"] == "long":
        position["mfe"] = max(position["mfe"], high - entry)
        position["mae"] = min(position["mae"], low - entry)
    else:
        position["mfe"] = max(position["mfe"], entry - low)
        position["mae"] = min(position["mae"], entry - high)
    st.session_state[position_key] = position


def check_exit(position_key: str, trades_key: str, capital_key: str, index: int) -> None:
    position = st.session_state[position_key]
    if position is None:
        return
    row = st.session_state.data.iloc[index]
    high, low = float(row["high"]), float(row["low"])
    if position["side"] == "long":
        if low <= position["sl"]:
            close_position(position_key, trades_key, capital_key, index, "sl", position["sl"])
        elif high >= position["tp"]:
            close_position(position_key, trades_key, capital_key, index, "tp", position["tp"])
    else:
        if high >= position["sl"]:
            close_position(position_key, trades_key, capital_key, index, "sl", position["sl"])
        elif low <= position["tp"]:
            close_position(position_key, trades_key, capital_key, index, "tp", position["tp"])


def close_position(
    position_key: str, trades_key: str, capital_key: str,
    index: int, reason: str, exit_price: float | None = None,
) -> None:
    position = st.session_state[position_key]
    if position is None:
        return
    if exit_price is None:
        exit_price = float(st.session_state.data.iloc[index]["close"])
    direction = 1 if position["side"] == "long" else -1
    pnl = (exit_price - position["entry_price"]) * direction * position["size"]
    new_capital = max(1000.0, float(st.session_state[capital_key]) + pnl)
    trade = {
        **position,
        "exit_index": index,
        "exit_time": st.session_state.data.index[index],
        "exit_price": exit_price,
        "reason": reason,
        "pnl": pnl,
        "capital_after": new_capital,
    }
    st.session_state[trades_key] = [*st.session_state[trades_key], trade]
    st.session_state[capital_key] = new_capital
    st.session_state[position_key] = None


def finish_session() -> None:
    if st.session_state.user_position is not None:
        close_user_position("finish")
    st.session_state.session_finished = True
    st.session_state.screen = "result"


def calculate_metrics(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, float]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "final_capital": initial_capital}
    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    equity = [initial_capital]
    for pnl in pnls:
        equity.append(max(1000.0, equity[-1] + pnl))
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float(gross_profit > 0),
        "max_drawdown": max_dd,
        "final_capital": equity[-1],
    }


def render_pattern_guide() -> None:
    import numpy as np

    rng = np.random.default_rng(42)

    def _base_layout():
        return dict(
            height=150, margin=dict(l=2, r=2, t=2, b=2),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            xaxis_rangeslider_visible=False, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#131722",
            showlegend=False,
        )

    def _candles(o, h, l, c):
        return go.Candlestick(x=list(range(len(o))), open=o, high=h, low=l, close=c,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False)

    def _line(y, color="orange", width=1.5, dash="solid"):
        return go.Scatter(x=list(range(len(y))), y=list(y), mode="lines",
            line=dict(color=color, width=width, dash=dash), showlegend=False)

    def _fig(*traces):
        f = go.Figure(list(traces))
        f.update_layout(**_base_layout())
        return f

    def _ohlc(price, spread=0.3):
        n = len(price)
        h = price + rng.uniform(0.1, spread, n)
        l = price - rng.uniform(0.1, spread, n)
        o = price + rng.uniform(-spread/2, spread/2, n)
        return o, h, l, price

    # ── イラスト用合成データ ───────────────────────────────────────────────────

    def golden_cross():
        n = 18
        # MA5: 下から始まり中盤でMA20を上抜け
        ma5  = np.concatenate([np.linspace(98.0, 99.2, 9), np.linspace(99.2, 101.8, 9)])
        ma20 = np.full(n, 100.0)
        p = ma5 + rng.uniform(-0.25, 0.25, n)
        return _fig(_candles(*_ohlc(p, 0.35)), _line(ma5, "orange"), _line(ma20, "royalblue"))

    def death_cross():
        n = 18
        # MA5: 上から始まり中盤でMA20を下抜け
        ma5  = np.concatenate([np.linspace(102.0, 100.8, 9), np.linspace(100.8, 98.2, 9)])
        ma20 = np.full(n, 100.0)
        p = ma5 + rng.uniform(-0.25, 0.25, n)
        return _fig(_candles(*_ohlc(p, 0.35)), _line(ma5, "orange"), _line(ma20, "royalblue"))

    def gc_skip():
        n = 18
        # MA5: ゴールデンクロスが発生するが価格はBB上限付近
        ma5  = np.concatenate([np.linspace(98.5, 99.5, 9), np.linspace(99.5, 101.5, 9)])
        ma20 = np.full(n, 100.0)
        p = ma5 + rng.uniform(-0.25, 0.25, n)
        mid = np.full(n, 100.0)
        upper = mid + 2.5
        lower = mid - 2.5
        return _fig(_candles(*_ohlc(p, 0.35)), _line(ma5, "orange"), _line(ma20, "royalblue"),
                    _line(upper, "#888", dash="dot"), _line(lower, "#888", dash="dot"))

    def bb_lower():
        n = 20
        p = np.concatenate([np.linspace(100, 96.5, 12), np.linspace(96.5, 99, 8)])
        mid = np.full(n, 100.0)
        return _fig(_candles(*_ohlc(p, 0.4)),
                    _line(mid + 3.5, "#888", dash="dot"), _line(mid, "#555"),
                    _line(mid - 3.5, "#4fc3f7", width=2, dash="dot"))

    def bb_upper():
        n = 20
        p = np.concatenate([np.linspace(100, 103.5, 12), np.linspace(103.5, 101, 8)])
        mid = np.full(n, 100.0)
        return _fig(_candles(*_ohlc(p, 0.4)),
                    _line(mid + 3.5, "#ef9a9a", width=2, dash="dot"), _line(mid, "#555"),
                    _line(mid - 3.5, "#888", dash="dot"))

    def double_bottom():
        p = np.array([103,102,101,99,97.5,99,101,102,101,99,97.5,99,101,103,104,105], dtype=float)
        return _fig(_candles(*_ohlc(p, 0.3)))

    def double_top():
        p = np.array([97,98,99,101,102.5,101,99,98,99,101,102.5,101,99,97,96,95], dtype=float)
        return _fig(_candles(*_ohlc(p, 0.3)))

    def hammer():
        o = [102.0, 101.5, 101.0, 100.5, 100.3]
        c = [101.5, 101.0, 100.5, 100.3, 100.7]
        h = [102.4, 101.8, 101.3, 100.8, 100.8]
        l = [101.2, 100.8, 100.3, 100.0,  98.5]
        return _fig(_candles(o, h, l, c))

    def shooting_star():
        o = [98.0, 98.5, 99.0, 99.5, 99.7]
        c = [98.5, 99.0, 99.5, 99.7, 99.3]
        h = [98.7, 99.2, 99.8, 100.0, 101.8]
        l = [97.8, 98.3, 98.8, 99.2,  99.1]
        return _fig(_candles(o, h, l, c))

    def bull_engulf():
        o = [101.5, 101.0, 100.5, 100.0,  99.5]
        c = [101.0, 100.5, 100.0,  99.5, 101.8]
        h = [101.8, 101.3, 100.8, 100.3, 102.0]
        l = [100.8, 100.3,  99.8,  99.2,  99.3]
        return _fig(_candles(o, h, l, c))

    def bear_engulf():
        o = [99.0, 99.5, 100.0, 100.5, 102.0]
        c = [99.5, 100.0, 100.5, 102.0,  99.2]
        h = [99.8, 100.2, 100.8, 102.3, 102.2]
        l = [98.8,  99.3,  99.8, 100.3,  99.0]
        return _fig(_candles(o, h, l, c))

    def bull_flag():
        pole = np.linspace(97, 106, 7)
        flag = np.linspace(106, 104, 6)
        brk  = np.linspace(104, 109, 5)
        p = np.concatenate([pole, flag, brk])
        return _fig(_candles(*_ohlc(p, 0.3)))

    def bear_flag():
        pole = np.linspace(107, 98, 7)
        flag = np.linspace(98, 100, 6)
        brk  = np.linspace(100, 95, 5)
        p = np.concatenate([pole, flag, brk])
        return _fig(_candles(*_ohlc(p, 0.3)))

    def head_shoulders():
        p = np.array([99,100,101,102,101,100,101,104,101,100,101,102,101,100,99,97.5], dtype=float)
        return _fig(_candles(*_ohlc(p, 0.25)))

    def inv_head_shoulders():
        p = np.array([101,100,99,98,99,100,99,96,99,100,99,98,99,100,101,103], dtype=float)
        return _fig(_candles(*_ohlc(p, 0.25)))

    def rising_wedge():
        n = 14
        highs  = np.linspace(100, 105, n) + np.linspace(0.8, 0.0, n)
        lows   = np.linspace(99,  104, n) + np.linspace(0.0, 0.8, n)
        highs[-1] -= 2.5; lows[-1] -= 2.5
        closes = (highs + lows) / 2
        opens  = closes + rng.uniform(-0.1, 0.1, n)
        return _fig(_candles(opens, highs, lows, closes))

    def falling_wedge():
        n = 14
        highs = np.linspace(106, 101, n) + np.linspace(0.0, 0.8, n)
        lows  = np.linspace(105, 100, n) + np.linspace(0.8, 0.0, n)
        highs[-1] += 2.5; lows[-1] += 2.5
        closes = (highs + lows) / 2
        opens  = closes + rng.uniform(-0.1, 0.1, n)
        return _fig(_candles(opens, highs, lows, closes))

    def asc_triangle():
        n = 14
        highs = np.full(n, 103.5) + rng.uniform(-0.1, 0.1, n)
        lows  = np.linspace(100, 103.0, n) + rng.uniform(-0.1, 0.1, n)
        highs[-1] = 105.0
        closes = (highs + lows) / 2
        opens  = closes + rng.uniform(-0.1, 0.1, n)
        return _fig(_candles(opens, highs, lows, closes))

    def desc_triangle():
        n = 14
        lows  = np.full(n, 97.0) + rng.uniform(-0.1, 0.1, n)
        highs = np.linspace(103, 97.5, n) + rng.uniform(-0.1, 0.1, n)
        lows[-1] = 95.0
        closes = (highs + lows) / 2
        opens  = closes + rng.uniform(-0.1, 0.1, n)
        return _fig(_candles(opens, highs, lows, closes))

    # ── パターン定義 ────────────────────────────────────────────────────────────

    TABS = [
        ("📈 MAクロス", [
            ("📈 ゴールデンクロス", "🟢 Buy", golden_cross, "MA5（橙）がMA20（青）を上抜け"),
            ("📉 デッドクロス", "🔴 Sell", death_cross, "MA5（橙）がMA20（青）を下抜け"),
            ("⚠️ GC＋BB上限（スルー）", "⚪ スルー", gc_skip, "GCだがBB上限に到達→矛盾シグナル"),
        ]),
        ("📊 BB", [
            ("🔵 BB下限タッチ", "🟢 Buy", bb_lower, "価格がBB下限（青点線）以下→逆張り買い"),
            ("🔴 BB上限タッチ", "🔴 Sell", bb_upper, "価格がBB上限（赤点線）以上→逆張り売り"),
        ]),
        ("🕯️ ローソク足", [
            ("🔨 ハンマー足", "🟢 Buy", hammer, "下ヒゲが実体の2倍以上。下落後の反転"),
            ("⭐ 流れ星", "🔴 Sell", shooting_star, "上ヒゲが実体の2倍以上。上昇後の天井"),
            ("🟢 陽の包み足", "🟢 Buy", bull_engulf, "前の陰線を完全に包む大陽線"),
            ("🔴 陰の包み足", "🔴 Sell", bear_engulf, "前の陽線を完全に包む大陰線"),
        ]),
        ("📐 チャート形状", [
            ("🔵 W型（ダブルボトム）", "🟢 Buy", double_bottom, "同水準の安値2回→転換上昇"),
            ("🔴 M型（ダブルトップ）", "🔴 Sell", double_top, "同水準の高値2回→転換下落"),
            ("👤 逆三尊", "🟢 Buy", inv_head_shoulders, "3点底→ネックライン上抜け"),
            ("👤 三尊天井", "🔴 Sell", head_shoulders, "3点天井→ネックライン下抜け"),
            ("🚩 ブルフラッグ", "🟢 Buy", bull_flag, "急騰→浅い調整→高値ブレイク"),
            ("🚩 ベアフラッグ", "🔴 Sell", bear_flag, "急落→浅い戻り→安値ブレイク"),
            ("⬇️ フォーリングウェッジ", "🟢 Buy", falling_wedge, "収束する下降チャネル→上辺ブレイク"),
            ("⬆️ ライジングウェッジ", "🔴 Sell", rising_wedge, "収束する上昇チャネル→下辺ブレイク"),
            ("📐 上昇三角形", "🟢 Buy", asc_triangle, "水平抵抗＋切り上がる安値→上抜け"),
            ("📐 下降三角形", "🔴 Sell", desc_triangle, "水平支持＋切り下がる高値→下抜け"),
        ]),
    ]

    with st.expander("📖 チャートパターン図鑑（クイズ前に確認）", expanded=False):
        st.caption("🟠 オレンジ = MA5（短期）　🔵 青 = MA20（長期）　点線 = BBバンド")
        tabs = st.tabs([t[0] for t in TABS])
        for tab, (_, patterns) in zip(tabs, TABS):
            with tab:
                cols = st.columns(3)
                for i, (name, judgment, fig_fn, desc) in enumerate(patterns):
                    bg = "#1a3a2a" if "Buy" in judgment else "#3a1a1a" if "Sell" in judgment else "#2a2a2a"
                    border = "#4caf50" if "Buy" in judgment else "#ef5350" if "Sell" in judgment else "#888"
                    with cols[i % 3]:
                        st.markdown(
                            f"<div style='background:{bg};border-left:3px solid {border};"
                            f"border-radius:6px;padding:6px 10px;margin-bottom:2px;'>"
                            f"<b style='font-size:0.85rem;'>{name}</b> {judgment}<br>"
                            f"<span style='font-size:0.75rem;color:#bbb;'>{desc}</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(fig_fn(), use_container_width=True, key=f"pat_{name}")


def render_glossary() -> None:
    with st.expander("📖 用語集", expanded=False):
        st.markdown("""
| 用語 | 意味 |
|---|---|
| **PF**（プロフィットファクター） | 総利益 ÷ 総損失。**1.0以上が黒字** |
| **勝率** | 全取引のうち利益が出た割合 |
| **SL**（ストップロス） | 損切りライン。触れると自動決済 |
| **TP**（テイクプロフィット） | 利確ライン。触れると自動決済 |
| **ATR** | 直近の平均的な値動き幅。SL/TPの基準 |
| **MFE** | エントリー後の含み益の最大値 |
| **MAE** | エントリー後の含み損の最大値 |
| **ゴールデンクロス** | MA5（短期）がMA20（長期）を上抜け → 買いシグナル |
| **デッドクロス** | MA5（短期）がMA20（長期）を下抜け → 売りシグナル |
""")


# ── Quiz Mode ─────────────────────────────────────────────────────────────────

_QUIZ_CONTEXT = 35


def _extract_signals(df: pd.DataFrame) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for i in range(max(_QUIZ_CONTEXT, MIN_START_INDEX), len(df) - 5):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        ma20_slope = (curr["ma20"] - df.iloc[i - 5]["ma20"]) / (df.iloc[i - 5]["ma20"] + 1e-9)
        cross_size = abs(curr["ma5"] - curr["ma20"]) / (curr["ma20"] + 1e-9)
        golden = prev["ma5"] < prev["ma20"] and curr["ma5"] > curr["ma20"]
        death  = prev["ma5"] > prev["ma20"] and curr["ma5"] < curr["ma20"]

        if golden:
            if curr["close"] >= curr["bb_upper"] * 0.99:
                signals.append({"i": i, "type": "golden_but_overbought", "answer": "skip",
                    "label": "⚠️ ゴールデンクロス＋BB上限（矛盾）",
                    "exp": (
                        f"MA5（{curr['ma5']:.2f}）がMA20を上抜けたが、価格がBB上限（{curr['bb_upper']:.2f}）に達している。\n\n"
                        "順張りシグナルと買われすぎが**同時に発生**。BB上限タッチは「行きすぎ」の警告→**スルー**が安全。"
                    )})
            elif ma20_slope < -0.04:
                signals.append({"i": i, "type": "golden_but_downtrend", "answer": "skip",
                    "label": "⚠️ ゴールデンクロス＋MA20急下降中（ダマシ）",
                    "exp": (
                        f"MA5がMA20を上抜けたが、**MA20が急角度で下向き**（傾き{ma20_slope*100:+.2f}%）。\n\n"
                        "長期トレンドが下降中のゴールデンクロスは『一時的な反発』にすぎないことが多い。→**スルー**推奨。"
                    )})
            elif cross_size < 0.0008:
                signals.append({"i": i, "type": "weak_golden", "answer": "skip",
                    "label": "⚠️ ゴールデンクロス（クロス幅が極小）",
                    "exp": (
                        f"クロスしたが**MA5とMA20の差がほぼゼロ**（差: {cross_size*100:.3f}%）。\n\n"
                        "力のないクロスはすぐ逆クロスする『ダマシ』になりやすい→**スルー**。"
                    )})
            else:
                signals.append({"i": i, "type": "ma_golden", "answer": "buy",
                    "label": "📈 MAゴールデンクロス",
                    "exp": (
                        f"MA5（{curr['ma5']:.2f}）がMA20（{curr['ma20']:.2f}）を**上抜け**。\n"
                        f"MA20は上向き（傾き{ma20_slope*100:+.2f}%）。短期モメンタムが長期トレンドと一致→**買い**。"
                    )})

        elif death:
            if curr["close"] <= curr["bb_lower"] * 1.01:
                signals.append({"i": i, "type": "death_but_oversold", "answer": "skip",
                    "label": "⚠️ デッドクロス＋BB下限（矛盾）",
                    "exp": (
                        f"MA5がMA20を下抜けたが、価格がBB下限（{curr['bb_lower']:.2f}）付近。\n\n"
                        "下降シグナルと売られすぎが**同時**。BB下限は反発の警告→**スルー**が安全。"
                    )})
            elif ma20_slope > 0.04:
                signals.append({"i": i, "type": "death_but_uptrend", "answer": "skip",
                    "label": "⚠️ デッドクロス＋MA20急上昇中（逆張りリスク大）",
                    "exp": (
                        f"デッドクロスだが**MA20が急角度で上向き**（傾き{ma20_slope*100:+.2f}%）。\n\n"
                        "強い上昇トレンドの途中でのデッドクロスは一時的な押し目のことが多い。→**スルー**推奨。"
                    )})
            elif cross_size < 0.0008:
                signals.append({"i": i, "type": "weak_death", "answer": "skip",
                    "label": "⚠️ デッドクロス（クロス幅が極小）",
                    "exp": (
                        f"デッドクロスだが**MA5とMA20の差がほぼゼロ**（差: {cross_size*100:.3f}%）。\n\n"
                        "ダマシのクロスの可能性→**スルー**推奨。"
                    )})
            else:
                signals.append({"i": i, "type": "ma_death", "answer": "sell",
                    "label": "📉 MAデッドクロス",
                    "exp": (
                        f"MA5（{curr['ma5']:.2f}）がMA20（{curr['ma20']:.2f}）を**下抜け**。\n"
                        f"MA20は下向き（傾き{ma20_slope*100:+.2f}%）。下降モメンタムが長期と一致→**売り**。"
                    )})

        elif curr["close"] <= curr["bb_lower"]:
            if ma20_slope < -0.04:
                signals.append({"i": i, "type": "bb_lower_but_downtrend", "answer": "skip",
                    "label": "⚠️ BB下限タッチ＋強い下降トレンド",
                    "exp": (
                        f"BB下限（{curr['bb_lower']:.2f}）に触れているが、**MA20が急角度で下落中**（傾き{ma20_slope*100:.2f}%）。\n\n"
                        "強いトレンド相場では逆張りは機能しにくい→**スルー**。"
                    )})
            else:
                signals.append({"i": i, "type": "bb_lower", "answer": "buy",
                    "label": "🔵 BB下限タッチ（逆張り買い）",
                    "exp": (
                        f"終値（{curr['close']:.2f}）がBB下限（{curr['bb_lower']:.2f}）以下。\n"
                        f"MA20は比較的フラット（傾き{ma20_slope*100:+.2f}%）→レンジ相場で平均回帰が期待できる。**逆張り買い**。"
                    )})

        elif curr["close"] >= curr["bb_upper"]:
            if ma20_slope > 0.04:
                signals.append({"i": i, "type": "bb_upper_but_uptrend", "answer": "skip",
                    "label": "⚠️ BB上限タッチ＋強い上昇トレンド",
                    "exp": (
                        f"BB上限（{curr['bb_upper']:.2f}）に触れているが、**MA20が急角度で上昇中**（傾き{ma20_slope*100:.2f}%）。\n\n"
                        "強い上昇トレンドではBB上限後も上昇継続することが多い→**スルー**。"
                    )})
            else:
                signals.append({"i": i, "type": "bb_upper", "answer": "sell",
                    "label": "🔴 BB上限タッチ（逆張り売り）",
                    "exp": (
                        f"終値（{curr['close']:.2f}）がBB上限（{curr['bb_upper']:.2f}）以上。\n"
                        f"MA20は比較的フラット（傾き{ma20_slope*100:+.2f}%）→レンジ相場で平均回帰が期待できる。**逆張り売り**。"
                    )})

    return signals


def _local_minima(arr: "np.ndarray", window: int = 3) -> list[int]:
    import numpy as np
    return [j for j in range(window, len(arr) - window)
            if arr[j] == min(arr[j - window: j + window + 1])]


def _local_maxima(arr: "np.ndarray", window: int = 3) -> list[int]:
    import numpy as np
    return [j for j in range(window, len(arr) - window)
            if arr[j] == max(arr[j - window: j + window + 1])]


def _extract_price_action(df: pd.DataFrame) -> list[dict[str, Any]]:
    import numpy as np
    patterns: list[dict[str, Any]] = []
    lb = 25

    for i in range(max(_QUIZ_CONTEXT, lb + 5, MIN_START_INDEX), len(df) - 5):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        w = df.iloc[i - lb: i + 1]

        body    = abs(curr["close"] - curr["open"])
        total   = curr["high"] - curr["low"]
        if total < 1e-9:
            continue
        lo_wick = min(curr["open"], curr["close"]) - curr["low"]
        hi_wick = curr["high"] - max(curr["open"], curr["close"])

        lows_arr = w["low"].values
        mins_idx = _local_minima(lows_arr, window=3)
        if len(mins_idx) >= 2:
            j1, j2 = mins_idx[-2], mins_idx[-1]
            v1, v2 = lows_arr[j1], lows_arr[j2]
            if j2 - j1 >= 5 and abs(v1 - v2) / (v1 + 1e-9) < 0.025:
                neckline = float(w["high"].iloc[j1: j2 + 1].max())
                if curr["close"] >= neckline * 0.98:
                    patterns.append({"i": i, "type": "double_bottom", "answer": "buy",
                        "label": "🔵 W型（ダブルボトム）",
                        "exp": (
                            f"同水準の安値が**2回**形成され（{v1:.2f} / {v2:.2f}）、ネックライン（{neckline:.2f}）を上抜けた。\n\n"
                            "W型はトレンド転換の代表パターン。下降→上昇への切り替わりシグナル→**買い**。"
                        )})
                    continue

        highs_arr = w["high"].values
        maxs_idx  = _local_maxima(highs_arr, window=3)
        if len(maxs_idx) >= 2:
            j1, j2 = maxs_idx[-2], maxs_idx[-1]
            h1, h2 = highs_arr[j1], highs_arr[j2]
            if j2 - j1 >= 5 and abs(h1 - h2) / (h1 + 1e-9) < 0.025:
                neckline = float(w["low"].iloc[j1: j2 + 1].min())
                if curr["close"] <= neckline * 1.02:
                    patterns.append({"i": i, "type": "double_top", "answer": "sell",
                        "label": "🔴 M型（ダブルトップ）",
                        "exp": (
                            f"同水準の高値が**2回**形成され（{h1:.2f} / {h2:.2f}）、ネックライン（{neckline:.2f}）を下抜けた。\n\n"
                            "M型はトレンド転換の代表パターン。上昇→下降への切り替わりシグナル→**売り**。"
                        )})
                    continue

        is_hammer = (lo_wick >= body * 2.0 and hi_wick <= body * 0.5 and body / total <= 0.35)
        recent_trend = (w["close"].iloc[-1] - w["close"].iloc[0]) / (w["close"].iloc[0] + 1e-9)
        if is_hammer and recent_trend < -0.02:
            patterns.append({"i": i, "type": "hammer", "answer": "buy",
                "label": "🔨 ハンマー足（反転シグナル）",
                "exp": (
                    "下ヒゲが実体の**2倍以上**の長さ（ハンマー足）。\n\n"
                    "下落中に出現→「一度は大きく売られたが買い戻された」ことを示す。反転の兆候→**買い**候補。"
                )})
            continue

        is_star = (hi_wick >= body * 2.0 and lo_wick <= body * 0.5 and body / total <= 0.35)
        if is_star and recent_trend > 0.02:
            patterns.append({"i": i, "type": "shooting_star", "answer": "sell",
                "label": "⭐ 流れ星（反転シグナル）",
                "exp": (
                    "上ヒゲが実体の**2倍以上**の長さ（流れ星/シューティングスター）。\n\n"
                    "上昇中に出現→「一度は大きく買われたが売り戻された」ことを示す。天井の兆候→**売り**候補。"
                )})
            continue

        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        is_bull_engulf = (
            prev["close"] < prev["open"] and curr["close"] > curr["open"] and
            curr["open"] <= prev["close"] and curr["close"] >= prev["open"] and
            curr_body >= prev_body * 0.9
        )
        if is_bull_engulf and recent_trend < -0.01:
            patterns.append({"i": i, "type": "bull_engulf", "answer": "buy",
                "label": "🟢 陽の包み足（強気転換）",
                "exp": (
                    "前足（陰線）を**完全に包む大陽線**（陽の包み足）。\n\n"
                    "売り勢力を買い勢力が完全に飲み込んだ形。下落後に出ると強力な反転シグナル→**買い**。"
                )})
            continue

        is_bear_engulf = (
            prev["close"] > prev["open"] and curr["close"] < curr["open"] and
            curr["open"] >= prev["close"] and curr["close"] <= prev["open"] and
            curr_body >= prev_body * 0.9
        )
        if is_bear_engulf and recent_trend > 0.01:
            patterns.append({"i": i, "type": "bear_engulf", "answer": "sell",
                "label": "🔴 陰の包み足（弱気転換）",
                "exp": (
                    "前足（陽線）を**完全に包む大陰線**（陰の包み足）。\n\n"
                    "買い勢力を売り勢力が完全に飲み込んだ形。上昇後に出ると強力な反転シグナル→**売り**。"
                )})
            continue

        if is_hammer and recent_trend > 0.03:
            patterns.append({"i": i, "type": "hammer_in_uptrend", "answer": "skip",
                "label": "⚠️ ハンマー足（上昇中・信頼性低）",
                "exp": (
                    "ハンマー足の形だが、**すでに上昇トレンドの途中**で出現。\n\n"
                    "ハンマーは下落後の底で出て初めて意味を持つ。→**スルー**。"
                )})

        if i >= 30:
            w30 = df.iloc[i - 30: i + 1]
            pole = w30.iloc[:12]
            flag_w = w30.iloc[12:]
            pole_move = (float(pole["close"].iloc[-1]) - float(pole["close"].iloc[0])) / (float(pole["close"].iloc[0]) + 1e-9)
            flag_move = (float(flag_w["close"].iloc[-1]) - float(flag_w["close"].iloc[0])) / (float(flag_w["close"].iloc[0]) + 1e-9)
            flag_high = float(flag_w["high"].iloc[:-1].max()) if len(flag_w) > 1 else float(flag_w["high"].max())
            flag_low  = float(flag_w["low"].iloc[:-1].min())  if len(flag_w) > 1 else float(flag_w["low"].min())
            flag_range = (float(flag_w["high"].max()) - float(flag_w["low"].min())) / (float(flag_w["close"].mean()) + 1e-9)
            if pole_move > 0.04 and -0.025 <= flag_move <= 0.005 and flag_range < 0.06 and curr["close"] > flag_high:
                patterns.append({"i": i, "type": "bull_flag", "answer": "buy",
                    "label": "🚩 ブルフラッグ（継続買い）",
                    "exp": (
                        f"強い上昇ポール後に小さな調整フラッグを作り、直近高値（{flag_high:.2f}）を上抜け。\n\n"
                        "急騰後の浅い調整からのブレイク→トレンド継続パターン。**買い**。"
                    )})
                continue
            if flag_range < 0.06 and pole_move < -0.04 and -0.005 <= flag_move <= 0.025 and curr["close"] < flag_low:
                patterns.append({"i": i, "type": "bear_flag", "answer": "sell",
                    "label": "🚩 ベアフラッグ（継続売り）",
                    "exp": (
                        f"強い下落ポール後に小さな戻りフラッグを作り、直近安値（{flag_low:.2f}）を下抜け。\n\n"
                        "急落後の浅い戻りからのブレイク→下落継続パターン。**売り**。"
                    )})
                continue
            if pole_move > 0.04 and flag_move < -0.025:
                patterns.append({"i": i, "type": "bull_flag_deep_retracement", "answer": "skip",
                    "label": "⚠️ ブルフラッグ（調整深すぎ）",
                    "exp": (
                        "急騰後の調整幅が大きすぎ（ポール高さの50%超の可能性）。\n\n"
                        "正規のフラッグパターンではない→**スルー**が安全。"
                    )})
                continue

        if i >= 25:
            w25 = df.iloc[i - 25: i + 1]
            x25 = np.arange(len(w25), dtype=float)
            high_slope25, high_int25 = np.polyfit(x25, w25["high"].values.astype(float), 1)
            low_slope25,  low_int25  = np.polyfit(x25, w25["low"].values.astype(float),  1)
            start_width = high_int25 - low_int25
            end_width   = (high_slope25 * x25[-1] + high_int25) - (low_slope25 * x25[-1] + low_int25)
            converging  = start_width > 0 and end_width > 0 and end_width < start_width * 0.75
            wedge_low   = float(w25["low"].min())
            wedge_high  = float(w25["high"].max())
            if converging and high_slope25 > 0 and low_slope25 > high_slope25 and curr["close"] < wedge_low * 1.005:
                patterns.append({"i": i, "type": "rising_wedge", "answer": "sell",
                    "label": "⬆️ ライジングウェッジ（反転売り）",
                    "exp": (
                        "高値・安値ともに切り上がるが安値の上昇角度が急でラインが収束→勢い衰退。\n\n"
                        "上昇ウェッジは弱気パターン。下辺ブレイクで**売り**。"
                    )})
                continue
            if converging and high_slope25 < 0 and low_slope25 < high_slope25 and curr["close"] > wedge_high * 0.995:
                patterns.append({"i": i, "type": "falling_wedge", "answer": "buy",
                    "label": "⬇️ フォーリングウェッジ（反転買い）",
                    "exp": (
                        "高値・安値ともに切り下がるが高値の下落角度が急でラインが収束→下落勢い衰退。\n\n"
                        "下降ウェッジは強気反転パターン。上辺ブレイクで**買い**。"
                    )})
                continue

        if i >= 30:
            w30b = df.iloc[i - 30: i + 1]
            x30  = np.arange(len(w30b), dtype=float)
            highs30 = w30b["high"].values.astype(float)
            lows30  = w30b["low"].values.astype(float)
            high_std30 = np.std(highs30[-15:]) / (np.mean(highs30[-15:]) + 1e-9)
            low_std30  = np.std(lows30[-15:])  / (np.mean(lows30[-15:])  + 1e-9)
            low_slope30  = float(np.polyfit(x30, lows30,  1)[0])
            high_slope30 = float(np.polyfit(x30, highs30, 1)[0])
            resistance30 = float(np.max(highs30[:-1]))
            support30    = float(np.min(lows30[:-1]))
            if high_std30 < 0.015 and low_slope30 > 0 and curr["close"] > resistance30 * 0.995:
                patterns.append({"i": i, "type": "ascending_triangle", "answer": "buy",
                    "label": "📐 上昇三角形（ブレイクアウト）",
                    "exp": (
                        f"上値抵抗（{resistance30:.2f}）が水平で安値は切り上がり→買い圧力が蓄積。\n\n"
                        "水平ラインを上抜けブレイク→**買い**。"
                    )})
                continue
            if low_std30 < 0.015 and high_slope30 < 0 and curr["close"] < support30 * 1.005:
                patterns.append({"i": i, "type": "descending_triangle", "answer": "sell",
                    "label": "📐 下降三角形（ブレイクダウン）",
                    "exp": (
                        f"下値支持（{support30:.2f}）が水平で高値は切り下がり→売り圧力が蓄積。\n\n"
                        "水平ラインを下抜けブレイク→**売り**。"
                    )})
                continue

        if i >= 50:
            w50  = df.iloc[i - 50: i + 1]
            h50  = w50["high"].values.astype(float)
            l50  = w50["low"].values.astype(float)
            hs_maxs = _local_maxima(h50, window=3)
            hs_mins = _local_minima(l50, window=3)
            if len(hs_maxs) >= 3:
                p1, p2, p3 = hs_maxs[-3], hs_maxs[-2], hs_maxs[-1]
                h1, h2, h3 = h50[p1], h50[p2], h50[p3]
                if (p2 - p1 >= 5 and p3 - p2 >= 5 and
                        abs(h1 - h3) / (h2 + 1e-9) < 0.05 and
                        h2 > h1 * 1.02 and h2 > h3 * 1.02):
                    neckline_hs = float(w50["low"].values[p1:p3 + 1].min())
                    if curr["close"] < neckline_hs * 1.01:
                        patterns.append({"i": i, "type": "head_shoulders", "answer": "sell",
                            "label": "👤 ヘッドアンドショルダー（三尊天井）",
                            "exp": (
                                f"左肩・頭・右肩の3点天井を形成後、ネックライン（{neckline_hs:.2f}）を下抜け。\n\n"
                                "強い天井反転シグナル→**売り**。"
                            )})
                        continue
            if len(hs_mins) >= 3:
                p1, p2, p3 = hs_mins[-3], hs_mins[-2], hs_mins[-1]
                l1, l2, l3 = l50[p1], l50[p2], l50[p3]
                if (p2 - p1 >= 5 and p3 - p2 >= 5 and
                        abs(l1 - l3) / (abs(l2) + 1e-9) < 0.05 and
                        l2 < l1 * 0.98 and l2 < l3 * 0.98):
                    neckline_ihs = float(w50["high"].values[p1:p3 + 1].max())
                    if curr["close"] > neckline_ihs * 0.99:
                        patterns.append({"i": i, "type": "inverse_head_shoulders", "answer": "buy",
                            "label": "👤 逆三尊（三尊底）",
                            "exp": (
                                f"左肩・頭・右肩の3点底を形成後、ネックライン（{neckline_ihs:.2f}）を上抜け。\n\n"
                                "強い底反転シグナル→**買い**。"
                            )})
                        continue

    return patterns


def _extract_no_signals(df: pd.DataFrame, taken: set[int], n: int) -> list[dict[str, Any]]:
    import random
    candidates = []
    for i in range(max(_QUIZ_CONTEXT, MIN_START_INDEX), len(df) - 5):
        if i in taken:
            continue
        curr = df.iloc[i]
        spread = abs(curr["ma5"] - curr["ma20"]) / (curr["ma20"] + 1e-9)
        if curr["bb_lower"] < curr["close"] < curr["bb_upper"] and spread < 0.003:
            candidates.append({
                "i": i, "type": "no_signal", "answer": "skip",
                "label": "⚪ シグナルなし",
                "exp": (
                    "MAのクロスもなく、価格もBBバンド内に収まっています。\n\n"
                    "明確な方向性シグナルなし → **スルー（見送り）**が正解。\n"
                    "エントリーしない判断もトレード技術のひとつです。"
                ),
            })
    random.shuffle(candidates)
    return candidates[:n]


def _weighted_sample(pool: list[dict[str, Any]], history: dict[str, Any], n: int) -> list[dict[str, Any]]:
    import numpy as np
    if n <= 0 or not pool:
        return []
    candidates = list(pool)
    weights = np.array([
        2.0 if (h := history.get(item["type"], {})).get("total", 0) >= 3
               and h.get("correct", 0) / h["total"] < 0.5
        else 1.0
        for item in candidates
    ], dtype=float)
    weights /= weights.sum()
    k = min(n, len(candidates))
    indices = np.random.choice(len(candidates), size=k, replace=False, p=weights)
    return [candidates[i] for i in indices]


def _build_quiz_questions(ticker: str, interval: str, n: int) -> list[dict[str, Any]]:
    import random
    df = load_market_data(ticker, interval, 500)
    if df.empty or len(df) < _QUIZ_CONTEXT + 20:
        return []

    signals = _extract_signals(df) + _extract_price_action(df)
    seen: set[int] = set()
    deduped = []
    for s in signals:
        if s["i"] not in seen:
            seen.add(s["i"])
            deduped.append(s)
    signals = deduped
    taken = {s["i"] for s in signals}

    by_answer: dict[str, list] = {"buy": [], "sell": [], "skip": []}
    for s in signals:
        by_answer[s["answer"]].append(s)
    no_sig = _extract_no_signals(df, taken, 200)
    by_answer["skip"].extend(no_sig)

    n_buy  = max(1, round(n * 0.30))
    n_sell = max(1, round(n * 0.30))
    n_skip = max(1, n - n_buy - n_sell)
    history = st.session_state.get("quiz_history", {})

    selected: list[dict[str, Any]] = []
    for pool, count in [(by_answer["buy"], n_buy), (by_answer["sell"], n_sell), (by_answer["skip"], n_skip)]:
        selected.extend(_weighted_sample(pool, history, count))
    random.shuffle(selected)

    result = []
    for q in selected[:n]:
        start = max(0, q["i"] - _QUIZ_CONTEXT)
        future_end = min(q["i"] + 26, len(df))
        result.append({
            **q,
            "df":        df.iloc[start: q["i"] + 1].copy(),
            "df_future": df.iloc[q["i"]: future_end].copy(),
        })
    return result


def start_quiz(ticker: str, interval: str, n: int) -> bool:
    # 前回の回答を履歴に反映してからクリア（順序重要）
    prev_answers = list(st.session_state.get("quiz_answers", []))
    history = st.session_state.setdefault("quiz_history", {})
    for a in prev_answers:
        t = a["type"]
        if t not in history:
            history[t] = {"label": a.get("label", t), "total": 0, "correct": 0}
        history[t]["label"] = a.get("label", t)
        history[t]["total"] += 1
        if a["is_correct"]:
            history[t]["correct"] += 1

    questions = _build_quiz_questions(ticker, interval, n)
    if not questions:
        return False
    st.session_state.update({
        "screen": "quiz",
        "quiz_questions": questions,
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_answers": [],
        "quiz_answered": False,
    })
    return True


def _answer_label(key: str) -> str:
    return {"buy": "🟢 Buy（買い）", "sell": "🔴 Sell（売り）", "skip": "⚪ スルー"}.get(key, key)


def render_quiz_screen() -> None:
    questions = st.session_state.quiz_questions
    idx = st.session_state.quiz_index
    total = len(questions)

    if idx >= total:
        st.session_state.screen = "quiz_result"
        st.rerun()
        return

    q = questions[idx]
    answered = st.session_state.quiz_answered

    prog_col, score_col = st.columns([3, 1])
    prog_col.progress(idx / total, text=f"問題 {idx + 1} / {total}")
    score_col.metric("スコア", f"{st.session_state.quiz_score} / {idx}")

    df_slice: pd.DataFrame = q["df"]
    df_future: pd.DataFrame = q.get("df_future", pd.DataFrame())
    show_future = answered and not df_future.empty

    df_combined = pd.concat([df_slice, df_future.iloc[1:]]) if show_future else df_slice

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_slice.index, open=df_slice["open"], high=df_slice["high"],
        low=df_slice["low"], close=df_slice["close"], name="過去",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    if show_future and len(df_future) > 1:
        df_fut_plot = df_future.iloc[1:]
        fig.add_trace(go.Candlestick(
            x=df_fut_plot.index, open=df_fut_plot["open"], high=df_fut_plot["high"],
            low=df_fut_plot["low"], close=df_fut_plot["close"], name="その後",
            increasing_line_color="#80cbc4", decreasing_line_color="#ef9a9a", opacity=0.75,
        ))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined["ma5"], mode="lines",
                             name="MA5（短期）", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined["ma20"], mode="lines",
                             name="MA20（長期）", line=dict(color="royalblue", width=1.5)))
    fig.add_shape(
        type="line",
        x0=str(df_slice.index[-1]), x1=str(df_slice.index[-1]),
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="yellow", width=2, dash="dash"),
    )
    fig.update_layout(height=460, xaxis_rangeslider_visible=False,
                      template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if show_future and len(df_future) > 1:
        entry_price = float(df_slice.iloc[-1]["close"])
        final_price = float(df_future.iloc[-1]["close"])
        move_pct = (final_price - entry_price) / entry_price * 100
        max_high = float(df_future["high"].max())
        min_low  = float(df_future["low"].min())
        up_pct   = (max_high - entry_price) / entry_price * 100
        down_pct = (min_low  - entry_price) / entry_price * 100
        n_bars   = len(df_future) - 1
        direction = "📈 上昇" if move_pct > 0 else "📉 下落"
        st.caption(
            f"**その後{n_bars}本の動き** ｜ 最終: {direction} {move_pct:+.2f}%　"
            f"｜ 最大上昇: +{up_pct:.2f}%　｜ 最大下落: {down_pct:.2f}%"
        )

    if not answered:
        st.markdown("#### 🤔 黄色い点線の足、あなたならどうする？")
        st.caption("🟠 オレンジ = MA5（短期）　🔵 青 = MA20（長期）")
        c1, c2, c3 = st.columns(3)
        user_answer: str | None = None
        if c1.button("🟢 Buy（買い）", use_container_width=True, type="primary"):
            user_answer = "buy"
        if c2.button("🔴 Sell（売り）", use_container_width=True):
            user_answer = "sell"
        if c3.button("⚪ スルー", use_container_width=True):
            user_answer = "skip"
        if user_answer is not None and not st.session_state.quiz_answered:
            correct = q["answer"]
            is_correct = user_answer == correct
            st.session_state.quiz_answers.append({
                "type": q["type"], "label": q["label"],
                "user": user_answer, "correct": correct, "is_correct": is_correct,
            })
            if is_correct:
                st.session_state.quiz_score += 1
            st.session_state.quiz_answered = True
            st.rerun()
    else:
        last = st.session_state.quiz_answers[-1]
        if last["is_correct"]:
            st.markdown(
                f"<div style='background:#1a3a2a;border-left:4px solid #4caf50;border-radius:8px;"
                f"padding:0.8rem 1rem;font-size:1.1rem;'>✅ <b>正解！</b>　模範解答: {_answer_label(last['correct'])}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#3a1a1a;border-left:4px solid #ef5350;border-radius:8px;"
                f"padding:0.8rem 1rem;font-size:1.1rem;'>❌ <b>不正解</b>　"
                f"あなた: {_answer_label(last['user'])} ／ 模範解答: {_answer_label(last['correct'])}</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;"
            f"padding:1rem 1.2rem;'><b>{q['label']}</b><br><br>"
            f"<span style='font-size:0.92rem;color:#ccc;'>{q['exp'].replace(chr(10), '<br>')}</span></div>",
            unsafe_allow_html=True,
        )
        if last["correct"] == "skip" and last["user"] != "skip":
            st.caption("⚠️ 模範解答はスルーですが、実際の値動きを見て「自分の判断の方が正しかった」と感じる場合もあります。")
        st.write("")
        label = "🏁 結果を見る" if idx + 1 >= total else "次の問題へ →"
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.quiz_index += 1
            st.session_state.quiz_answered = False
            if idx + 1 >= total:
                st.session_state.screen = "quiz_result"
            st.rerun()


def render_quiz_result_screen() -> None:
    answers = st.session_state.quiz_answers
    score = st.session_state.quiz_score
    total = len(answers)
    pct = score / total * 100 if total > 0 else 0

    if pct >= 80:
        grade, grade_color, grade_bg = "🥇 Gold", "#ffd700", "#2a2000"
        msg = "素晴らしい！シグナル認識の精度が高い。"
    elif pct >= 60:
        grade, grade_color, grade_bg = "🥈 Silver", "#c0c0c0", "#1a1a1a"
        msg = "良いペース。苦手パターンをもう少し意識しよう。"
    elif pct >= 40:
        grade, grade_color, grade_bg = "🥉 Bronze", "#cd7f32", "#1a1000"
        msg = "基礎は掴めてきた。解説をよく読んで繰り返し練習しよう。"
    else:
        grade, grade_color, grade_bg = "💪 練習あるのみ", "#888", "#111"
        msg = "まだ始まったばかり。スルー判断を意識するだけで正解率が上がります。"

    st.markdown(
        f"<div style='background:{grade_bg};border:1px solid {grade_color}33;"
        f"border-radius:14px;padding:1.5rem 2rem;text-align:center;margin-bottom:1.5rem;'>"
        f"<div style='font-size:3rem;'>{grade}</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{grade_color};'>{pct:.0f}%</div>"
        f"<div style='color:#aaa;margin-top:0.3rem;'>{score} / {total} 問正解　— {msg}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("スコア", f"{score} / {total}")
    c2.metric("正解率", f"{pct:.1f}%")
    c3.metric("評価", grade)

    st.divider()
    st.subheader("パターン別成績")
    type_stats: dict[str, dict[str, Any]] = {}
    for a in answers:
        t = a["type"]
        if t not in type_stats:
            type_stats[t] = {"label": a["label"], "correct": 0, "total": 0}
        type_stats[t]["total"] += 1
        if a["is_correct"]:
            type_stats[t]["correct"] += 1

    if type_stats:
        chart_rows = sorted(
            [{"label": s["label"], "rate": s["correct"] / s["total"] * 100, "total": s["total"]}
             for s in type_stats.values()],
            key=lambda r: r["rate"],
        )
        bar_fig = go.Figure(go.Bar(
            x=[r["rate"] for r in chart_rows],
            y=[r["label"] for r in chart_rows],
            orientation="h",
            text=[f"{r['rate']:.0f}% ({r['total']}問)" for r in chart_rows],
            textposition="auto",
            marker=dict(color=[r["rate"] for r in chart_rows],
                        colorscale="RdYlGn", cmin=0, cmax=100),
        ))
        bar_fig.update_layout(
            height=max(300, 40 * len(chart_rows)),
            xaxis=dict(title="正答率 (%)", range=[0, 100]),
            margin=dict(l=10, r=10, t=20, b=10),
            template="plotly_dark",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("📋 全問見直し", expanded=False):
        for i, a in enumerate(answers):
            icon = "✅" if a["is_correct"] else "❌"
            st.markdown(
                f"**Q{i+1}** {icon} ｜ {a['label']} ｜ "
                f"あなた: {_answer_label(a['user'])} ／ 正解: {_answer_label(a['correct'])}"
            )

    ca, cb = st.columns(2)
    if ca.button("🔄 もう一度チャレンジ", use_container_width=True):
        st.session_state.screen = "top"
        st.rerun()
    if cb.button("🎮 練習モードへ", use_container_width=True):
        st.session_state.screen = "top"
        st.rerun()


# ── Result Screen (bar replay) ─────────────────────────────────────────────────

def render_result_screen() -> None:
    st.title("結果")
    metrics = calculate_metrics(st.session_state.user_trades, st.session_state.initial_capital)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "最終資金",
        f"{metrics['final_capital']:,.0f}円",
        f"{metrics['final_capital'] - st.session_state.initial_capital:+,.0f}円",
    )
    col2.metric("勝率", f"{metrics['win_rate']:.1f}%", f"{metrics['trades']}回取引")
    col3.metric("PF", f"{metrics['profit_factor']:.2f}", "1.0以上が黒字")

    st.divider()
    render_glossary()

    if st.button("← トップへ戻る", use_container_width=True):
        st.session_state.screen = "top"
        st.rerun()


# ── Top Screen ─────────────────────────────────────────────────────────────────

_TOP_CSS = """
<style>
.hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.2; margin-bottom: 0.2rem; }
.hero-sub   { font-size: 1.05rem; color: #888; margin-bottom: 2rem; }
.mode-card  {
    background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.mode-card h3 { margin: 0 0 0.4rem 0; font-size: 1.1rem; }
.mode-card p  { margin: 0; font-size: 0.85rem; color: #aaa; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none; border-radius: 8px; font-size: 1.05rem;
    font-weight: 700; padding: 0.65rem;
}
</style>
"""


def render_top_screen() -> None:
    st.markdown(_TOP_CSS, unsafe_allow_html=True)

    st.markdown('<p class="hero-title">📈 トレード練習アプリ</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">チャートの読み方を学んで、AIと同じ目線で相場を見よう。</p>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="mode-card"><h3>📝 クイズモード（おすすめ）</h3><p>チャートを見て「買い・売り・スルー」を判断。AIが採点＆解説する一問一答。</p></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        ticker = col1.selectbox("銘柄", list(TICKERS.keys()), index=0, label_visibility="collapsed")
        interval_label = col2.selectbox("足種", list(INTERVAL_MAP.keys()), index=0, label_visibility="collapsed")
        interval = INTERVAL_MAP[interval_label]
        n = col3.selectbox("問題数", [10, 20, 30, 50], index=1, label_visibility="collapsed")

        if st.button("📝 クイズを始める", use_container_width=True, type="primary"):
            with st.spinner("問題を生成中..."):
                ok = start_quiz(ticker, interval, n)
            if ok:
                st.rerun()
            else:
                st.error("データ取得に失敗しました。しばらく待ってから再試行してください。")

        st.write("")
        st.markdown('<div class="mode-card"><h3>🎮 バーリプレイ</h3><p>チャートを1本ずつ進めながら実際に売買を体験する練習モード。</p></div>', unsafe_allow_html=True)
        bar_limit = st.slider("練習本数", min_value=20, max_value=200, value=50, step=10, label_visibility="collapsed")
        if st.button("🎮 バーリプレイを始める", use_container_width=True):
            config = AppConfig(ticker, interval, bar_limit)
            with st.spinner("データ取得中..."):
                reset_session(config)
            st.rerun()

    with right:
        st.markdown("""
<div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;padding:1.4rem 1.6rem;">
<h4 style="margin:0 0 1rem 0;">🎓 このアプリで学べること</h4>
<p style="font-size:0.9rem;color:#ccc;margin:0.5rem 0;">
✅ ゴールデンクロス・デッドクロスの見極め方<br>
✅ BB（ボリンジャーバンド）逆張りの判断基準<br>
✅ W底・M天井などチャートパターン15種<br>
✅ 「エントリーしない」判断の重要性<br>
✅ 苦手パターンを自動で多く出題（適応学習）
</p>
<hr style="border-color:#2d2d4e;margin:1rem 0;">
<h4 style="margin:0 0 0.5rem 0;">📊 対応銘柄</h4>
<p style="font-size:0.85rem;color:#888;margin:0;">
BTC / ETH / SOL / XRP / BNB<br>
1日足・1時間足・15分足
</p>
</div>
""", unsafe_allow_html=True)

    st.write("")
    render_pattern_guide()


# ── Practice Screen ────────────────────────────────────────────────────────────

def render_practice_screen() -> None:
    st.title(f"{st.session_state.ticker} 練習モード")
    data = st.session_state.data
    if data is None or data.empty:
        st.warning("データがありません。")
        st.session_state.screen = "top"
        return
    render_status_panel()
    render_chart(data, st.session_state.current_index)
    render_controls()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="トレード練習アプリ",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()

    screen = st.session_state.screen
    if screen == "top":
        render_top_screen()
    elif screen == "practice":
        render_practice_screen()
    elif screen == "result":
        render_result_screen()
    elif screen == "quiz":
        render_quiz_screen()
    elif screen == "quiz_result":
        render_quiz_result_screen()
    else:
        render_top_screen()


if __name__ == "__main__":
    main()

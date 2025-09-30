# main.py
import os
import time
import threading
import csv
from datetime import datetime
from collections import deque

import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_file

# --- CONFIG (يمكن تغييرها عبر متغيرات البيئة في Render dashboard) ---
SYMBOLS = os.getenv("SYMBOLS", "ETHUSDT,BTCUSDT,BNBUSDT,SOLUSDT,ADAUSDT").split(",")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))   # فريم دقيقة => poll كل 60s
EMA_SHORT = int(os.getenv("EMA_SHORT", "20"))
EMA_LONG = int(os.getenv("EMA_LONG", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.2"))  # حجم > avg_volume * multiplier
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.01"))  # 1% stop loss
TRADE_LOG = os.getenv("TRADE_LOG", "trades.csv")
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10.0"))  # رصيد وهمي ابتدائي
KL_LIMIT = int(os.getenv("KL_LIMIT", "200"))

# Endpoint for public klines
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# --- Global state ---
balance_lock = threading.Lock()
balance = INITIAL_BALANCE
current_trade = None  # dict with keys: symbol, entry_price, qty, stop_price, entry_time
trades = []  # list of trade dicts (history); also append to CSV
stats = {"trades": 0, "wins": 0, "losses": 0, "profit_usd": 0.0}

app = Flask(__name__)

# --- Utilities: fetch klines and indicators ---
def fetch_klines(symbol, interval="1m", limit=KL_LIMIT):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","count","taker_base","taker_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    # RSI standard
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Trade logging ---
def append_trade_csv(tr):
    header = ["time","symbol","side","entry","exit","profit_usd","balance_after","reason"]
    exists = False
    try:
        exists = open(TRADE_LOG, "r")
        exists.close()
        exists = True
    except Exception:
        exists = False
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            tr["time"], tr["symbol"], tr["side"], f"{tr['entry']:.8f}",
            f"{tr['exit']:.8f}", f"{tr['profit']:.8f}", f"{tr['balance_after']:.8f}", tr["reason"]
        ])

# --- Enter / Exit simulation (virtual wallet) ---
def enter_trade(symbol, entry_price):
    global current_trade, balance
    with balance_lock:
        if current_trade is not None:
            return False
        qty = balance / entry_price  # use full balance spot
        stop_price = entry_price * (1 - STOP_LOSS_PCT)
        current_trade = {
            "symbol": symbol,
            "entry_price": entry_price,
            "qty": qty,
            "stop_price": stop_price,
            "entry_time": datetime.utcnow().isoformat(),
            "side": "LONG"
        }
        print(f"[ENTER] {symbol} @ {entry_price:.6f} qty={qty:.8f} bal={balance:.8f}")
        return True

def exit_trade(exit_price, reason="X"):
    global current_trade, balance, stats, trades
    with balance_lock:
        if current_trade is None:
            return False
        proceeds = current_trade["qty"] * exit_price
        cost = current_trade["qty"] * current_trade["entry_price"]
        profit = proceeds - cost
        balance = proceeds  # compound: new balance equals proceeds
        tr = {
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": current_trade["symbol"],
            "side": current_trade["side"],
            "entry": current_trade["entry_price"],
            "exit": exit_price,
            "profit": profit,
            "balance_after": balance,
            "reason": reason
        }
        trades.append(tr)
        append_trade_csv(tr)
        stats["trades"] += 1
        if profit >= 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["profit_usd"] = round(balance - INITIAL_BALANCE, 8)
        print(f"[EXIT] {tr['symbol']} exit {exit_price:.6f} profit={profit:.8f} newbal={balance:.8f} reason={reason}")
        current_trade = None
        return True

# --- Decision logic per symbol ---
def evaluate_symbol(symbol):
    """
    Fetch klines, compute EMA/RSI/volume filter and return signals:
      - 'enter' -> price to enter (last closed price)
      - 'exit'  -> price to exit (last closed price) with reason 'X' or 'SL'
      - None
    """
    global current_trade
    try:
        df = fetch_klines(symbol)
    except Exception as e:
        print(f"fetch error {symbol}: {e}")
        return None, None

    # Use closed candle (index -2)
    if len(df) < max(EMA_LONG, RSI_PERIOD) + 5:
        return None, None

    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)

    ema_short = compute_ema(closes, EMA_SHORT)
    ema_long = compute_ema(closes, EMA_LONG)
    rsi = compute_rsi(closes, RSI_PERIOD)

    # last closed candle indices
    last_idx = -2
    prev_idx = -3

    # compute cross
    s_now = ema_short.iloc[last_idx]
    s_prev = ema_short.iloc[prev_idx]
    l_now = ema_long.iloc[last_idx]
    l_prev = ema_long.iloc[prev_idx]

    cross_up = (s_prev <= l_prev) and (s_now > l_now)
    cross_down = (s_prev >= l_prev) and (s_now < l_now)

    last_close = closes.iloc[last_idx]
    prev_close = closes.iloc[prev_idx]

    # volume filter: compare last closed candle volume with its 20-period avg
    avg_vol = volumes.rolling(window=20, min_periods=1).mean().iloc[last_idx]
    vol_ok = volumes.iloc[last_idx] > (avg_vol * VOLUME_MULTIPLIER)

    # RSI check (avoid extreme)
    rsi_now = rsi.iloc[last_idx]
    rsi_ok = (rsi_now > 25) and (rsi_now < 75)

    # Compose entry rule: cross_up + volume + rsi
    if current_trade is None:
        if cross_up and vol_ok and rsi_ok:
            return "enter", float(last_close)
        else:
            return None, None
    else:
        # if there is an open trade on this symbol: check SL or cross_down to exit
        if current_trade["symbol"] == symbol:
            # stop loss
            if last_close <= current_trade["stop_price"]:
                return "exit_sl", float(last_close)
            # exit on cross down
            if cross_down:
                return "exit_x", float(last_close)
    return None, None

# --- Main worker loop ---
def worker_loop():
    global current_trade
    print("Worker started. Symbols:", SYMBOLS)
    while True:
        try:
            for sym in SYMBOLS:
                decision, price = evaluate_symbol(sym)
                if decision == "enter":
                    # enter only if no current trade
                    with balance_lock:
                        if current_trade is None:
                            enter_trade(sym, price)
                elif decision == "exit_sl":
                    exit_trade(price, reason="SL")
                elif decision == "exit_x":
                    exit_trade(price, reason="X")
                # if current trade is on another symbol we ignore other signals (sequential)
            # sleep then poll again
        except Exception as e:
            print("Worker error:", e)
        time.sleep(POLL_SECONDS)

# --- Flask endpoints for status & recent trades ---
@app.route("/status")
def status():
    with balance_lock:
        bal = balance
        ct = current_trade.copy() if current_trade else None
        s = dict(stats)
    recent = list(reversed(trades[-20:]))
    return jsonify({
        "balance": round(bal,8),
        "current_trade": ct,
        "stats": s,
        "recent_trades": recent,
        "config": {
            "symbols": SYMBOLS,
            "poll_seconds": POLL_SECONDS,
            "ema_short": EMA_SHORT,
            "ema_long": EMA_LONG,
            "rsi_period": RSI_PERIOD,
            "volume_multiplier": VOLUME_MULTIPLIER,
            "stop_loss_pct": STOP_LOSS_PCT
        }
    })

@app.route("/download_trades")
def download_trades():
    # return CSV file
    try:
        return send_file(TRADE_LOG, as_attachment=True)
    except Exception:
        return jsonify({"error":"no trades yet"}), 404

# --- Start background worker thread on app start ---
def start_background():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    # start worker when run directly
    start_background()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
else:
    # when imported by gunicorn, start worker
    start_background()

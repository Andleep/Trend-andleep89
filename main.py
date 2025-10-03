# main.py
# TradeBot Smart — safer backtester + simple web UI (Flask)
# Designed to replace broken version: fixes compounding & sizing bugs, adds TP/Trailing/limits.

import os, time, math, csv, requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- CONFIG (ENV friendly) ----------------
SYMBOLS = os.getenv("SYMBOLS", "ETHUSDT").split(",")
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10.0"))

# Strategy indicators
EMA_FAST = int(os.getenv("EMA_FAST", "8"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.0"))
KL_LIMIT = int(os.getenv("KL_LIMIT", "1000"))
BINANCE_KLINES = os.getenv("BINANCE_KLINES", "https://api.binance.com/api/v3/klines")

# Money management (safe defaults)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))       # نخاطر 1% من الرصيد كحد أقصى
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.03")) # نفتح مركز بقيمة 3% من الرصيد افتراضياً
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))   # لا نستخدم اكثر من 25% من الرصيد للمركز بأكمله
MAX_POSITION_VALUE = float(os.getenv("MAX_POSITION_VALUE", "5000")) # حماية قصوى لقيمة المركز بالدولار
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.01"))         # 1% stop loss by default
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))     # 2% take profit default
TRAILING_PCT = float(os.getenv("TRAILING_PCT", "0.01"))           # 1% trailing stop (optional)
MIN_TIME_BETWEEN_TRADES = int(os.getenv("MIN_TIME_BETWEEN_TRADES", "60")) # seconds
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "50"))
MAX_CONSECUTIVE_LOSSES_BEFORE_COOLDOWN = int(os.getenv("MAX_CONSECUTIVE_LOSSES_BEFORE_COOLDOWN", "10"))
COOLDOWN_SECONDS_AFTER_LOSS_RUN = int(os.getenv("COOLDOWN_SECONDS_AFTER_LOSS_RUN", "3600"))

TRADE_LOG = "trades.csv"

# ---------------- Utilities ----------------
def fetch_klines(symbol, interval="1m", limit=1000, startTime=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if startTime is not None:
        params["startTime"] = int(startTime)
    headers = {"User-Agent": "TradeBot/1.0"}
    r = requests.get(BINANCE_KLINES, params=params, timeout=20, headers=headers)
    r.raise_for_status()
    data = r.json()
    out=[]
    for k in data:
        out.append({"time": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
    return out

# Simple indicators (pure python)
def ema_list(values, span):
    if not values: return []
    alpha = 2.0/(span+1)
    out = [values[0]]
    for v in values[1:]:
        out.append((v - out[-1]) * alpha + out[-1])
    return out

def sma(values, period):
    out=[]
    s=0.0
    for i,v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i-period]
            out.append(s/period)
        elif i==period-1:
            out.append(s/period)
    return out

def rsi_list(values, period=14):
    n=len(values)
    if n < period+1: return [50.0]*n
    deltas=[values[i]-values[i-1] for i in range(1,n)]
    ups=[d if d>0 else 0 for d in deltas]
    downs=[-d if d<0 else 0 for d in deltas]
    up_avg=sum(ups[:period]) / period
    down_avg=sum(downs[:period]) / period if sum(downs[:period])!=0 else 1e-9
    out=[50.0]*(period+1)
    for u,d in zip(ups[period:], downs[period:]):
        up_avg = (up_avg*(period-1) + u)/period
        down_avg = (down_avg*(period-1) + d)/period
        rs = up_avg/(down_avg+1e-12)
        out.append(100 - (100/(1+rs)))
    if len(out)<n:
        out = [out[0]]*(n - len(out)) + out
    return out

def macd_list(values, fast=12, slow=26, signal=9):
    ef = ema_list(values, fast)
    es = ema_list(values, slow)
    # align shorter
    L = min(len(ef), len(es))
    macd = [ef[-L + i] - es[-L + i] for i in range(L)]
    sig = ema_list(macd, signal) if macd else []
    pad = len(values) - len(macd)
    macd_full = [0.0]*pad + macd
    sig_full = [0.0]*pad + sig
    return macd_full, sig_full

# ---------------- Backtest engine (safe compounding & sizing) ----------------
def run_backtest(candles, initial_balance=INITIAL_BALANCE,
                 risk_per_trade=RISK_PER_TRADE, stop_loss_pct=STOP_LOSS_PCT,
                 take_profit_pct=TAKE_PROFIT_PCT, trailing_pct=TRAILING_PCT):
    """
    Input candles: list of dicts {time,open,high,low,close,volume} sorted ascending by time
    Returns stats dict + trades list
    """
    if not candles:
        return {"error":"no candles"}, []

    closes=[c["close"] for c in candles]
    highs=[c["high"] for c in candles]
    lows=[c["low"] for c in candles]
    vols=[c["volume"] for c in candles]
    times=[c["time"] for c in candles]

    ema_fast = ema_list(closes, EMA_FAST)
    ema_slow = ema_list(closes, EMA_SLOW)
    rsi_vals = rsi_list(closes, RSI_PERIOD)
    macd_vals, macd_sig = macd_list(closes)

    # avg vol 20
    avg_vol20 = []
    for i in range(len(vols)):
        w = vols[max(0,i-20):i+1]
        avg_vol20.append(sum(w)/len(w) if w else vols[i])

    balance = float(initial_balance)
    position = None  # dict with entry, qty, stop, tp, trailing_active, max_price_reached, position_value
    trades=[]
    wins=losses=0
    last_trade_time = None
    trades_today = 0
    consecutive_losses = 0
    cooldown_until = 0

    # clamp position size
    POS_PCT = max(1e-6, min(POSITION_SIZE_PCT, MAX_POSITION_PCT))

    for i in range(len(closes)):
        # minimal data
        if i < max(EMA_SLOW, RSI_PERIOD) + 2:
            continue

        now_time = times[i]
        price = closes[i]
        prev = i - 1

        # prevent trading during cooldown due to loss run
        if time.time() < cooldown_until:
            continue

        # signals
        cross_up = (ema_fast[prev] <= ema_slow[prev]) and (ema_fast[i] > ema_slow[i])
        cross_down = (ema_fast[prev] >= ema_slow[prev]) and (ema_fast[i] < ema_slow[i])
        vol_ok = vols[i] > (avg_vol20[i] * VOLUME_MULTIPLIER)
        rsi_ok = (rsi_vals[prev] > 30 and rsi_vals[prev] < 70)
        macd_ok = False
        try:
            macd_ok = macd_vals[i] > macd_sig[i]
        except Exception:
            macd_ok = False

        # ensemble: require >=2 of signals
        score = sum([1 if cross_up else 0, 1 if vol_ok else 0, 1 if macd_ok else 0, 1 if rsi_ok else 0])
        enter_allowed = (score >= 2)

        # reset trades_today at UTC midnight (approx)
        if last_trade_time:
            last_dt = datetime.utcfromtimestamp(last_trade_time/1000)
            now_dt = datetime.utcfromtimestamp(now_time/1000)
            if now_dt.date() != last_dt.date():
                trades_today = 0

        # ENTRY
        if position is None and enter_allowed:
            # limit trades per day
            if trades_today >= MAX_TRADES_PER_DAY:
                continue
            # enforce min time between trades
            if last_trade_time and (now_time - last_trade_time) < (MIN_TIME_BETWEEN_TRADES*1000):
                continue

            # position sizing (safe)
            desired_pos_value = balance * POS_PCT
            desired_pos_value = min(desired_pos_value, balance * MAX_POSITION_PCT, MAX_POSITION_VALUE)
            # ensure risk_per_trade cap
            est_risk = desired_pos_value * stop_loss_pct
            max_risk_allowed = balance * risk_per_trade
            if est_risk > max_risk_allowed and stop_loss_pct > 0:
                desired_pos_value = max_risk_allowed / stop_loss_pct
                desired_pos_value = min(desired_pos_value, balance)

            # final guard
            if desired_pos_value < 1e-8 or price <= 0:
                continue

            qty = desired_pos_value / price
            qty = max(qty, 0.0)

            stop_price = price * (1 - stop_loss_pct)
            take_price = price * (1 + take_profit_pct)
            position = {
                "entry": price, "qty": qty, "stop": stop_price, "tp": take_price,
                "position_value": desired_pos_value,
                "entry_time": now_time,
                "trailing_active": trailing_pct>0,
                "max_price": price
            }
            last_trade_time = now_time
            trades_today += 1
            # continue to next candle (we don't exit same candle)
            continue

        # If there is an open position -> check exits
        if position is not None:
            # update max price for trailing
            if price > position["max_price"]:
                position["max_price"] = price

            # check stop loss breach (by low)
            if lows[i] <= position["stop"]:
                exit_price = position["stop"]
                proceeds = position["qty"] * exit_price
                cost = position.get("position_value", position["qty"] * position["entry"])
                profit = proceeds - cost
                balance += profit
                trades.append({"time": now_time, "entry": position["entry"], "exit": exit_price,
                               "profit": profit, "balance_after": balance, "reason":"SL"})
                if profit >= 0: wins += 1; consecutive_losses = 0
                else: losses += 1; consecutive_losses += 1
                position = None
                # if many consecutive losses -> cooldown
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES_BEFORE_COOLDOWN:
                    cooldown_until = time.time() + COOLDOWN_SECONDS_AFTER_LOSS_RUN
                continue

            # check take profit
            if highs[i] >= position["tp"]:
                exit_price = position["tp"]
                proceeds = position["qty"] * exit_price
                cost = position.get("position_value", position["qty"] * position["entry"])
                profit = proceeds - cost
                balance += profit
                trades.append({"time": now_time, "entry": position["entry"], "exit": exit_price,
                               "profit": profit, "balance_after": balance, "reason":"TP"})
                if profit >= 0: wins += 1; consecutive_losses = 0
                else: losses += 1; consecutive_losses += 1
                position = None
                continue

            # trailing stop: if active and price dropped from max by trailing_pct
            if position.get("trailing_active") and position.get("max_price"):
                trail_level = position["max_price"] * (1 - trailing_pct)
                if lows[i] <= trail_level:
                    exit_price = trail_level
                    proceeds = position["qty"] * exit_price
                    cost = position.get("position_value", position["qty"] * position["entry"])
                    profit = proceeds - cost
                    balance += profit
                    trades.append({"time": now_time, "entry": position["entry"], "exit": exit_price,
                                   "profit": profit, "balance_after": balance, "reason":"TRAIL"})
                    if profit >= 0: wins += 1; consecutive_losses = 0
                    else: losses += 1; consecutive_losses += 1
                    position = None
                    continue

            # optional exit on ema cross down as safety
            if (ema_fast[prev] >= ema_slow[prev]) and (ema_fast[i] < ema_slow[i]):
                exit_price = price
                proceeds = position["qty"] * exit_price
                cost = position.get("position_value", position["qty"] * position["entry"])
                profit = proceeds - cost
                balance += profit
                trades.append({"time": now_time, "entry": position["entry"], "exit": exit_price,
                               "profit": profit, "balance_after": balance, "reason":"X"})
                if profit >= 0: wins += 1; consecutive_losses = 0
                else: losses += 1; consecutive_losses += 1
                position = None
                continue

    stats = {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 8),
        "profit_usd": round(balance - initial_balance, 8),
        "trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": round((wins / (wins+losses) * 100) if (wins+losses)>0 else 0, 2)
    }
    return stats, trades

# ---------------- Flask endpoints ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({"balance": INITIAL_BALANCE, "symbols": SYMBOLS})

@app.route("/api/candles")
def api_candles():
    symbol = request.args.get("symbol", SYMBOLS[0])
    interval = request.args.get("interval", "1m")
    limit = int(request.args.get("limit", 500))
    try:
        candles = fetch_klines(symbol, interval=interval, limit=limit)
        return jsonify({"symbol": symbol, "candles": candles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    # CSV upload handling
    if "csv" in request.files:
        f = request.files["csv"]
        text = f.read().decode("utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        candles = []
        for i, ln in enumerate(lines):
            if i==0 and ("time" in ln.lower() and "open" in ln.lower()): continue
            parts = ln.split(",")
            if len(parts) < 6: continue
            t = parts[0].strip()
            try:
                if t.isdigit() and len(t) > 10: t = int(t)
                else: t = int(datetime.fromisoformat(t).timestamp()*1000)
            except Exception:
                t = int(datetime.utcnow().timestamp()*1000)
            candles.append({"time": int(t), "open": float(parts[1]), "high": float(parts[2]), "low": float(parts[3]), "close": float(parts[4]), "volume": float(parts[5])})
        initial = float(data.get("initial_balance", INITIAL_BALANCE))
        risk = float(data.get("risk_per_trade", RISK_PER_TRADE))
        stop = float(data.get("stop_loss_pct", STOP_LOSS_PCT))
        tp = float(data.get("take_profit_pct", TAKE_PROFIT_PCT))
        stats, trades = run_backtest(candles, initial_balance=initial, risk_per_trade=risk, stop_loss_pct=stop, take_profit_pct=tp)
        return jsonify({"stats": stats, "trades": trades})

    # API mode: fetch candles from binance for months
    symbol = data.get("symbol", SYMBOLS[0])
    months = int(data.get("months", 1))
    interval = data.get("interval", "1m")
    initial = float(data.get("initial_balance", INITIAL_BALANCE))
    risk = float(data.get("risk_per_trade", RISK_PER_TRADE))
    stop = float(data.get("stop_loss_pct", STOP_LOSS_PCT))
    tp = float(data.get("take_profit_pct", TAKE_PROFIT_PCT))

    end = datetime.utcnow()
    start = end - timedelta(days=30*months)
    all_candles=[]
    start_ms = int(start.timestamp()*1000)
    while True:
        try:
            part = fetch_klines(symbol, interval=interval, limit=1000, startTime=start_ms)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        if not part: break
        all_candles.extend(part)
        if len(part) < 1000: break
        start_ms = part[-1]["time"] + 1
        time.sleep(0.12)

    if not all_candles:
        return jsonify({"error":"no candles retrieved"}), 500

    stats, trades = run_backtest(all_candles, initial_balance=initial, risk_per_trade=risk, stop_loss_pct=stop, take_profit_pct=tp)
    # save CSV
    try:
        with open(TRADE_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time","entry","exit","profit","balance_after","reason"])
            for t in trades:
                writer.writerow([datetime.utcfromtimestamp(t["time"]/1000).strftime("%Y-%m-%d %H:%M:%S"), t["entry"], t["exit"], t["profit"], t["balance_after"], t["reason"]])
    except Exception:
        pass
    return jsonify({"stats": stats, "trades": trades})

@app.route("/download_trades")
def download_trades():
    if os.path.exists(TRADE_LOG):
        return send_file(TRADE_LOG, as_attachment=True)
    return jsonify({"error":"no trades file"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")))

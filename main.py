#!/usr/bin/env python3
# ROMEOPTP LIQUIDITY MANIPULATION ENGINE - Bybit Edition
# TRANSFORMED FROM SIRTS v10 TO PURE PRICE-BASED MANIPULATION ANALYSIS
# Requirements: requests, pandas, numpy
# BOT_TOKEN and CHAT_ID must be set as environment variables: "BOT_TOKEN", "CHAT_ID"

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== SYMBOL SANITIZATION =====
def sanitize_symbol(symbol: str) -> str:
    """Ensure symbol only contains legal Bybit characters and is upper-case.
       Allow letters, numbers, - _ . and max length 20."""
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30
COOLDOWN_TIME_DEFAULT = 1800
COOLDOWN_TIME_SUCCESS = 15 * 60
COOLDOWN_TIME_FAIL    = 45 * 60

VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60

API_CALL_DELAY = 0.06  # slightly higher for Bybit

TIMEFRAMES = ["15m", "30m", "1h", "4h"]

# ===== ROMEOPTP CORE PARAMETERS =====
class RomeoptpLiquidityEngine:
    def __init__(self):
        # Core Romeoptp Parameters
        self.CRT_TOLERANCE = 0.0005  # 0.05% tolerance for equal highs/lows
        self.SWEEP_CONFIRMATION_PCT = 0.001  # 0.1% break for sweep confirmation
        self.RECLAIM_CONFIRMATION = 0.0003  # 0.03% reclaim threshold
        self.LOOKBACK_SWINGS = 30  # For turtle soup detection
        self.MIN_COMPRESSION_BARS = 5  # Minimum bars for compression detection
        self.FVG_TOLERANCE = 0.002  # 0.2% for FVG detection

# ===== DISABLED FILTERS =====
# ALL trend, sentiment, and momentum filters removed as per specification
ENABLE_TREND_ALIGNMENT_FILTER = False
ENABLE_MARKET_CONTEXT_FILTER = False
ENABLE_INTELLIGENT_SENTIMENT = False
ENABLE_SR_FILTER = False
ENABLE_MOMENTUM_FILTER = False
ENABLE_BTC_DOMINANCE_FILTER = False
STRICT_TF_AGREE = False

# ===== BYBIT PUBLIC ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
BYBIT_PRICE = "https://api.bybit.com/v5/market/tickers"

LOG_CSV = "./romeoptp_signals_bybit.csv"

# ===== NEW SAFEGUARDS =====
MAX_OPEN_TRADES = 200
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

# ===== RISK & CONFIDENCE =====
BASE_RISK = 0.05   # AGGRESSIVE: 5% per trade default
MAX_RISK  = 0.06
MIN_RISK  = 0.01

# ===== STATE =====
last_trade_time      = {}
open_trades          = []
signals_sent_total   = 0
signals_hit_total    = 0
signals_fail_total   = 0
signals_breakeven    = 0
total_checked_signals= 0
skipped_signals      = 0
last_heartbeat       = time.time()
last_summary         = time.time()
volatility_pause_until= 0
last_trade_result = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== REMOVED FILTER FUNCTIONS =====
# All trend_alignment_ok, market_context_ok, intelligent_sentiment_check functions removed
# All EMA, trend, sentiment logic completely eliminated

# ===== HELPERS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=5, retries=1):
    """Fetch JSON with light retry/backoff and logging."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== BYBIT / SYMBOL FUNCTIONS =====
def get_top_symbols(n=70):
    """Get top n USDT pairs by quote volume using Bybit tickers."""
    params = {"category": "linear"}
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=5, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT"]
    rows = j["result"]["list"]
    usdt = []
    for d in rows:
        s = d.get("symbol","")
        if not s.upper().endswith("USDT"):
            continue
        try:
            vol = float(d.get("volume24h", 0))
            last = float(d.get("lastPrice", 0)) or 0
            quote_vol = vol * (last or 1.0)
            usdt.append((s.upper(), quote_vol))
        except Exception:
            continue
    usdt.sort(key=lambda x: x[1], reverse=True)
    syms = [sanitize_symbol(s[0]) for s in usdt[:n]]
    if not syms:
        return ["BTCUSDT","ETHUSDT"]
    return syms

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=5, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                vol = float(d.get("volume24h", 0))
                last = float(d.get("lastPrice", 0)) or 0
                return vol * (last or 1.0)
            except:
                return 0.0
    return 0.0

def interval_to_bybit(interval):
    """Map "15m","30m","1h","4h" to Bybit kline interval values."""
    m = {"1m":"1", "3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","2h":"120","4h":"240","1d":"D"}
    return m.get(interval, interval)

def get_klines(symbol, interval="15m", limit=200):
    """Fetch klines from Bybit public API and return pandas DF with open/high/low/close/volume."""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    iv = interval_to_bybit(interval)
    params = {
        "category": "linear",
        "symbol": symbol, 
        "interval": iv, 
        "limit": limit
    }
    j = safe_get_json(BYBIT_KLINES, params=params, timeout=6, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    data = j["result"]["list"]
    if not isinstance(data, list):
        return None
    try:
        df = pd.DataFrame(data, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_PRICE, params=params, timeout=5, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                return float(d.get("lastPrice", 0))
            except:
                return None
    return None

# ===== ROMEOPTP CORE LOGIC MODULES =====

def detect_crt_range(df, lookback=50):
    """
    🔥 CRT ENGINE: Identify engineered equal highs and equal lows
    Returns: range_high, range_low, quality_score
    """
    engine = RomeoptpLiquidityEngine()
    
    if len(df) < lookback:
        return None, None, 0
        
    high = df['high'].tail(lookback)
    low = df['low'].tail(lookback)
    
    # Find equal highs within tolerance
    high_max = high.max()
    high_min = high_max * (1 - engine.CRT_TOLERANCE)
    equal_highs = high[high >= high_min]
    
    # Find equal lows within tolerance  
    low_min = low.min()
    low_max = low_min * (1 + engine.CRT_TOLERANCE)
    equal_lows = low[low <= low_max]
    
    # Quality score based on number of touches
    high_touches = len(equal_highs)
    low_touches = len(equal_lows)
    quality_score = min((high_touches + low_touches) / (lookback * 0.1), 1.0)
    
    print(f"      🔍 CRT Analysis: {high_touches} high touches, {low_touches} low touches")
    
    if high_touches >= 2 and low_touches >= 2:
        return high_max, low_min, quality_score
    else:
        return None, None, 0

def detect_crt_sweep(df, range_high, range_low):
    """
    🔥 CRT SWEEP DETECTION: Wick must break range, body must close back inside
    Returns: "swept_high", "swept_low", or None
    """
    engine = RomeoptpLiquidityEngine()
    
    if len(df) < 3 or range_high is None or range_low is None:
        return None
        
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_high = current['high']
    current_low = current['low']
    current_close = current['close']
    current_open = current['open']
    
    prev_high = prev['high']
    prev_low = prev['low']
    
    # Check for high sweep
    high_break = current_high > range_high * (1 + engine.SWEEP_CONFIRMATION_PCT)
    high_reclaim = current_close < range_high and current_open < range_high
    
    # Check for low sweep  
    low_break = current_low < range_low * (1 - engine.SWEEP_CONFIRMATION_PCT)
    low_reclaim = current_close > range_low and current_open > range_low
    
    if high_break and high_reclaim:
        return "swept_high"
    elif low_break and low_reclaim:
        return "swept_low"
    
    return None

def detect_compression(df, period=20):
    """
    🔥 COMPRESSION DETECTION: Pre-manipulation tightening
    Higher lows + lower highs pattern
    """
    if len(df) < period:
        return False, 0.0
    
    # Calculate ATR for volatility measurement
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = []
    for i in range(1, len(df)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr.append(max(tr1, tr2, tr3))
    
    if not tr:
        return False, 0.0
        
    current_atr = np.mean(tr[-5:]) if len(tr) >= 5 else tr[-1]
    historical_atr = np.mean(tr) if tr else current_atr
    
    # Compression: current volatility significantly lower than historical
    compression_ratio = current_atr / historical_atr if historical_atr > 0 else 1.0
    is_compressed = compression_ratio < 0.6  # 40%+ reduction in volatility
    
    return is_compressed, compression_ratio

def detect_crt_reclaim(df, range_high, range_low, sweep_direction):
    """
    🔥 CRT RECLAIM: After sweep → candle closes inside → reclaim confirmed
    """
    engine = RomeoptpLiquidityEngine()
    
    if len(df) < 2 or range_high is None or range_low is None:
        return False
        
    current = df.iloc[-1]
    current_close = current['close']
    
    if sweep_direction == "swept_high":
        # For high sweep reclaim, price should close back below range high
        reclaim = current_close < range_high * (1 - engine.RECLAIM_CONFIRMATION)
        return reclaim
    elif sweep_direction == "swept_low":
        # For low sweep reclaim, price should close back above range low  
        reclaim = current_close > range_low * (1 + engine.RECLAIM_CONFIRMATION)
        return reclaim
    
    return False

def detect_turtle_soup(df, lookback=30):
    """
    🔥 TURTLE SOUP ENGINE: Identify previous swing high/low breaks with reclaim
    Returns: "bullish", "bearish", or None
    """
    if len(df) < lookback + 5:
        return None
        
    # Find previous swing high and low
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    swing_high = highs.max()
    swing_low = lows.min()
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_high = current['high']
    current_low = current['low'] 
    current_close = current['close']
    
    prev_high = prev['high']
    prev_low = prev['low']
    
    # Bullish Turtle Soup: Break below swing low then reclaim
    low_break = current_low < swing_low * 0.998
    low_reclaim = current_close > swing_low * 1.002
    
    # Bearish Turtle Soup: Break above swing high then reclaim  
    high_break = current_high > swing_high * 1.002
    high_reclaim = current_close < swing_high * 0.998
    
    if low_break and low_reclaim:
        return "bullish"
    elif high_break and high_reclaim:
        return "bearish"
    
    return None

def detect_bos_mss(df, sweep_direction):
    """
    🔥 MARKET STRUCTURE ENGINE: Break of Structure / Market Structure Shift
    After sweep → displacement candle opposite direction
    Returns: 'bullish_shift', 'bearish_shift', or None
    """
    if len(df) < 5:
        return None
        
    current = df.iloc[-1]
    prev_1 = df.iloc[-2]
    prev_2 = df.iloc[-3]
    
    if sweep_direction == "swept_high":
        # For high sweep, look for bearish MSS (lower highs)
        lower_high = current['high'] < prev_1['high'] and prev_1['high'] < prev_2['high']
        lower_low = current['low'] < prev_1['low'] 
        if lower_high and lower_low:
            return 'bearish_shift'
            
    elif sweep_direction == "swept_low":
        # For low sweep, look for bullish MSS (higher lows)
        higher_low = current['low'] > prev_1['low'] and prev_1['low'] > prev_2['low']
        higher_high = current['high'] > prev_1['high']
        if higher_low and higher_high:
            return 'bullish_shift'
    
    return None

def detect_order_block(df, direction):
    """
    🔥 ORDER BLOCK ENGINE: 
    For longs: last down candle before up displacement
    For shorts: last up candle before down displacement
    Returns: OB zone (high, low)
    """
    if len(df) < 10:
        return None, None
        
    if direction == "BUY":
        # Look for the last bearish candle before a bullish move
        for i in range(-5, -len(df), -1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1] if i < -1 else None
            
            if next_candle is None:
                continue
                
            # Bearish candle (close < open) followed by bullish candle
            is_bearish = current['close'] < current['open']
            next_bullish = next_candle['close'] > next_candle['open']
            
            if is_bearish and next_bullish:
                return current['high'], current['low']
                
    elif direction == "SELL":
        # Look for the last bullish candle before a bearish move
        for i in range(-5, -len(df), -1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1] if i < -1 else None
            
            if next_candle is None:
                continue
                
            # Bullish candle (close > open) followed by bearish candle
            is_bullish = current['close'] > current['open']
            next_bearish = next_candle['close'] < next_candle['open']
            
            if is_bullish and next_bearish:
                return current['high'], current['low']
    
    return None, None

def detect_fvg(df):
    """
    🔥 FAIR VALUE GAP ENGINE: 3-candle imbalance detection
    Returns: bullish_fvgs[], bearish_fvgs[]
    """
    engine = RomeoptpLiquidityEngine()
    
    if len(df) < 3:
        return [], []
        
    bullish_fvgs = []
    bearish_fvgs = []
    
    for i in range(len(df) - 2):
        candle1 = df.iloc[i]
        candle2 = df.iloc[i+1] 
        candle3 = df.iloc[i+2]
        
        # Bullish FVG: candle1 low > candle3 high
        if candle1['low'] > candle3['high']:
            gap_high = candle1['low']
            gap_low = candle3['high']
            if (gap_high - gap_low) / gap_low > engine.FVG_TOLERANCE:
                bullish_fvgs.append((gap_low, gap_high))
        
        # Bearish FVG: candle1 high < candle3 low  
        elif candle1['high'] < candle3['low']:
            gap_low = candle1['high']
            gap_high = candle3['low']
            if (gap_high - gap_low) / gap_low > engine.FVG_TOLERANCE:
                bearish_fvgs.append((gap_low, gap_high))
    
    return bullish_fvgs, bearish_fvgs

def next_liquidity_target(df, current_price, direction):
    """
    🔥 NEXT LIQUIDITY TARGET ENGINE: Find next equal highs/lows or nearby FVG
    Returns: target_price
    """
    if len(df) < 50:
        return current_price * 1.02 if direction == "BUY" else current_price * 0.98
    
    lookback = min(100, len(df))
    
    if direction == "BUY":
        # For longs, look for next equal highs (liquidity above)
        highs = df['high'].tail(lookback)
        # Find significant highs (top 20%)
        significant_highs = highs.nlargest(int(lookback * 0.2))
        if len(significant_highs) > 0:
            target = significant_highs.max()
            # Ensure target is above current price
            if target > current_price:
                return target
        # Fallback: 2% above current
        return current_price * 1.02
        
    else:  # SELL
        # For shorts, look for next equal lows (liquidity below)
        lows = df['low'].tail(lookback)
        # Find significant lows (bottom 20%)
        significant_lows = lows.nsmallest(int(lookback * 0.2))
        if len(significant_lows) > 0:
            target = significant_lows.min()
            # Ensure target is below current price
            if target < current_price:
                return target
        # Fallback: 2% below current
        return current_price * 0.98

def btc_volatility_spike(window=20, threshold=2.0):
    """
    🔥 BTC VOLATILITY SPIKE DETECTION
    Check if BTC has abnormal volatility that might affect all markets
    Returns: True if volatility spike detected, False otherwise
    """
    try:
        # Get BTC data
        btc_df = get_klines("BTCUSDT", "5m", window * 2)
        if btc_df is None or len(btc_df) < window:
            return False
        
        # Calculate returns and volatility
        prices = btc_df['close'].values
        if len(prices) < window:
            return False
            
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(abs(ret))  # Use absolute returns for volatility
        
        if len(returns) < window:
            return False
            
        # Current volatility (last window)
        current_vol = np.mean(returns[-window:])
        # Historical volatility (previous window)
        historical_vol = np.mean(returns[-window*2:-window]) if len(returns) >= window*2 else current_vol
        
        # Check for spike
        if historical_vol > 0 and current_vol > historical_vol * threshold:
            print(f"⚠️ BTC Volatility Spike: {current_vol:.4f} vs {historical_vol:.4f} (threshold: {threshold}x)")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ BTC volatility check error: {e}")
        return False

# ===== MISSING FUNCTIONS ADDED =====
def process_romeoptp_signal(signal):
    """
    🔥 PROCESS ROMEOPTP SIGNAL: Execute trade based on Romeoptp signal
    """
    global signals_sent_total, open_trades, last_trade_time
    
    symbol = signal['symbol']
    direction = signal['direction']
    entry_price = signal['entry']
    confidence = signal['confidence']
    
    print(f"🎯 PROCESSING ROMEOPTP SIGNAL: {symbol} {direction} at {entry_price}")
    
    # Calculate position sizing
    risk_pct = min(max(BASE_RISK * (confidence / 100), MIN_RISK), MAX_RISK)
    margin_usd = CAPITAL * risk_pct
    exposure_usd = margin_usd * LEVERAGE
    
    # Calculate TP/SL levels
    if direction == "BUY":
        sl_price = entry_price * (1 - 0.01)  # 1% SL
        tp1 = entry_price * (1 + 0.005)      # 0.5% TP1
        tp2 = entry_price * (1 + 0.01)       # 1% TP2  
        tp3 = entry_price * (1 + 0.015)      # 1.5% TP3
    else:  # SELL
        sl_price = entry_price * (1 + 0.01)  # 1% SL
        tp1 = entry_price * (1 - 0.005)      # 0.5% TP1
        tp2 = entry_price * (1 - 0.01)       # 1% TP2
        tp3 = entry_price * (1 - 0.015)      # 1.5% TP3
    
    # Create trade object
    trade = {
        "s": symbol,
        "side": direction,
        "entry": entry_price,
        "sl": sl_price,
        "tp1": tp1,
        "tp2": tp2, 
        "tp3": tp3,
        "entry_tf": signal['timeframe'],
        "margin": margin_usd,
        "exposure": exposure_usd,
        "risk_pct": risk_pct,
        "confidence_pct": confidence,
        "st": "open",
        "opened_at": time.time()
    }
    
    # Add to open trades
    open_trades.append(trade)
    
    # Update cooldown
    last_trade_time[symbol] = time.time() + COOLDOWN_TIME_SUCCESS
    
    # Update stats
    signals_sent_total += 1
    STATS["by_side"][direction]["sent"] += 1
    STATS["by_tf"][signal['timeframe']]["sent"] += 1
    
    # Send Telegram alert
    message = (
        f"🎯 ROMEOPTP SIGNAL\n"
        f"Symbol: {symbol} {direction}\n"
        f"Entry: {entry_price:.4f}\n" 
        f"TP: {tp1:.4f} / {tp2:.4f} / {tp3:.4f}\n"
        f"SL: {sl_price:.4f}\n"
        f"Risk: {risk_pct*100:.1f}% | Confidence: {confidence:.1f}%\n"
        f"TF: {signal['timeframe']} | Margin: ${margin_usd:.2f}"
    )
    send_message(message)
    
    # Log to CSV
    log_signal([
        datetime.utcnow().isoformat(), symbol, direction, entry_price, 
        tp1, tp2, tp3, sl_price, signal['timeframe'], 
        "N/A", margin_usd, exposure_usd, risk_pct*100, confidence, "open", "Romeoptp"
    ])
    
    return True

def check_trades():
    """
    🔥 CHECK OPEN TRADES FOR TP/SL
    """
    global open_trades, signals_hit_total, signals_fail_total, signals_breakeven
    
    current_time = time.time()
    still_open = []
    
    for trade in open_trades:
        symbol = trade["s"]
        current_price = get_price(symbol)
        
        if current_price is None:
            still_open.append(trade)
            continue
            
        direction = trade["side"]
        entry = trade["entry"]
        sl = trade["sl"]
        tp1, tp2, tp3 = trade["tp1"], trade["tp2"], trade["tp3"]
        
        # Check for TP/SL
        if direction == "BUY":
            if current_price <= sl:
                # Stop loss hit
                trade["st"] = "closed"
                trade["close_reason"] = "SL"
                trade["closed_at"] = current_time
                signals_fail_total += 1
                STATS["by_side"][direction]["fail"] += 1
                STATS["by_tf"][trade["entry_tf"]]["fail"] += 1
                send_message(f"❌ SL HIT: {symbol} BUY | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price >= tp3:
                # All TPs hit
                trade["st"] = "closed" 
                trade["close_reason"] = "TP3"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"🎉 TP3 HIT: {symbol} BUY | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price >= tp2:
                # TP2 hit
                trade["st"] = "closed"
                trade["close_reason"] = "TP2"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"✅ TP2 HIT: {symbol} BUY | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price >= tp1:
                # TP1 hit
                trade["st"] = "closed"
                trade["close_reason"] = "TP1"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"✅ TP1 HIT: {symbol} BUY | Entry: {entry:.4f} | Current: {current_price:.4f}")
            else:
                still_open.append(trade)
                
        else:  # SELL
            if current_price >= sl:
                # Stop loss hit
                trade["st"] = "closed"
                trade["close_reason"] = "SL"
                trade["closed_at"] = current_time
                signals_fail_total += 1
                STATS["by_side"][direction]["fail"] += 1
                STATS["by_tf"][trade["entry_tf"]]["fail"] += 1
                send_message(f"❌ SL HIT: {symbol} SELL | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price <= tp3:
                # All TPs hit
                trade["st"] = "closed"
                trade["close_reason"] = "TP3"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"🎉 TP3 HIT: {symbol} SELL | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price <= tp2:
                # TP2 hit
                trade["st"] = "closed"
                trade["close_reason"] = "TP2"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"✅ TP2 HIT: {symbol} SELL | Entry: {entry:.4f} | Current: {current_price:.4f}")
            elif current_price <= tp1:
                # TP1 hit
                trade["st"] = "closed"
                trade["close_reason"] = "TP1"
                trade["closed_at"] = current_time
                signals_hit_total += 1
                STATS["by_side"][direction]["hit"] += 1
                STATS["by_tf"][trade["entry_tf"]]["hit"] += 1
                send_message(f"✅ TP1 HIT: {symbol} SELL | Entry: {entry:.4f} | Current: {current_price:.4f}")
            else:
                still_open.append(trade)
    
    # Log closed trades
    for trade in open_trades:
        if trade.get("st") == "closed" and not trade.get("logged"):
            log_trade_close(trade)
            trade["logged"] = True
    
    # Update open trades list
    open_trades = [t for t in still_open if t.get("st") == "open"]

def heartbeat():
    """Send periodic heartbeat"""
    open_count = len([t for t in open_trades if t.get("st") == "open"])
    message = (
        f"❤️ ROMEOPTP HEARTBEAT\n"
        f"Cycles: {cycle_count}\n"
        f"Open Trades: {open_count}\n"
        f"Signals: {signals_sent_total} sent, {signals_hit_total} hits\n"
        f"Active: {len(SYMBOLS)} symbols"
    )
    send_message(message)

def summary():
    """Send daily summary"""
    hit_rate = (signals_hit_total / signals_sent_total * 100) if signals_sent_total > 0 else 0
    message = (
        f"📊 ROMEOPTP DAILY SUMMARY\n"
        f"Signals: {signals_sent_total} sent\n"
        f"Hits: {signals_hit_total} | Fails: {signals_fail_total}\n"
        f"Hit Rate: {hit_rate:.1f}%\n"
        f"Total Scans: {total_checked_signals}\n"
        f"Skipped: {skipped_signals}"
    )
    send_message(message)

# ===== ROMEOPTP SIGNAL GENERATION WITH DETAILED LOGGING =====
def generate_signal(symbol):
    """
    🔥 REWRITTEN SIGNAL GENERATION: Pure Romeoptp liquidity manipulation flow
    WITH ENHANCED DEBUGGING
    """
    global total_checked_signals, skipped_signals, signals_sent_total
    
    total_checked_signals += 1
    print(f"\n🎯 === SCANNING {symbol} ===")
    
    # Get data for multiple timeframes
    best_signal = None
    best_confidence = 0
    best_tf = None
    
    for tf in TIMEFRAMES:
        print(f"\n📊 Timeframe: {tf}")
        df = get_klines(symbol, tf, 100)
        if df is None or len(df) < 50:
            print(f"   ❌ SKIP: No data or insufficient bars")
            continue
        
        # Check dataframe structure
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            print(f"   ❌ SKIP: Missing required columns. Available: {list(df.columns)}")
            continue
            
        current_price = df['close'].iloc[-1]
        print(f"   💰 Current Price: {current_price}")
        print(f"   📈 Data Points: {len(df)} bars")
            
        # 1. IDENTIFY CRT RANGE
        range_high, range_low, range_quality = detect_crt_range(df)
        print(f"   📈 CRT Range - High: {range_high}, Low: {range_low}, Quality: {range_quality}")
        
        if range_quality <= 0.3:
            print(f"   ❌ SKIP: Range quality too low ({range_quality} <= 0.3)")
            continue
            
        print(f"   ✅ Range quality OK: {range_quality}")

        # 2. DETECT SWEEP
        sweep_direction = detect_crt_sweep(df, range_high, range_low)
        print(f"   🔄 Sweep Detection: {sweep_direction}")
        
        if not sweep_direction:
            print(f"   ❌ SKIP: No sweep detected")
            continue
            
        print(f"   ✅ Sweep detected: {sweep_direction}")

        # 3. CONFIRM RECLAIM
        reclaim_confirmed = detect_crt_reclaim(df, range_high, range_low, sweep_direction)
        print(f"   🔁 Reclaim Confirmation: {reclaim_confirmed}")
        
        if not reclaim_confirmed:
            print(f"   ❌ SKIP: No reclaim confirmed")
            continue
            
        print(f"   ✅ Reclaim confirmed")

        # 4. CONFIRM BOS/MSS
        mss_confirmation = detect_bos_mss(df, sweep_direction)
        print(f"   🏗️  BOS/MSS Confirmation: {mss_confirmation}")
        
        # 5. CHECK TURTLE SOUP (alternative signal)
        turtle_signal = detect_turtle_soup(df)
        print(f"   🐢 Turtle Soup: {turtle_signal}")
        
        # ROMEOPTP SIGNAL LOGIC: turtle_soup OR (crt_sweep AND mss)
        valid_signal = False
        direction = None
        
        if turtle_signal == "bullish" or (sweep_direction == "swept_low" and mss_confirmation == "bullish_shift"):
            direction = "BUY"
            valid_signal = True
            print(f"   🟢 BULLISH SETUP: Turtle={turtle_signal}, Sweep={sweep_direction}, MSS={mss_confirmation}")
        elif turtle_signal == "bearish" or (sweep_direction == "swept_high" and mss_confirmation == "bearish_shift"):
            direction = "SELL"
            valid_signal = True
            print(f"   🔴 BEARISH SETUP: Turtle={turtle_signal}, Sweep={sweep_direction}, MSS={mss_confirmation}")
        else:
            print(f"   ❌ SKIP: No valid signal combination")
            print(f"      Turtle: {turtle_signal}, Sweep: {sweep_direction}, MSS: {mss_confirmation}")
        
        if valid_signal and direction:
            # Calculate confidence based on multiple confirmations
            confidence = range_quality * 0.4
            if mss_confirmation: 
                confidence += 0.3
                print(f"   📊 Confidence +0.3 for MSS")
            if turtle_signal: 
                confidence += 0.3
                print(f"   📊 Confidence +0.3 for Turtle Soup")
            
            final_confidence = confidence * 100
            print(f"   🎯 FINAL CONFIDENCE: {final_confidence:.1f}%")
            
            if final_confidence > best_confidence:
                best_signal = {
                    'symbol': symbol,
                    'direction': direction,
                    'timeframe': tf,
                    'entry': float(df['close'].iloc[-1]),
                    'range_high': range_high,
                    'range_low': range_low,
                    'sweep_direction': sweep_direction,
                    'mss_confirmation': mss_confirmation,
                    'turtle_signal': turtle_signal,
                    'confidence': final_confidence
                }
                best_confidence = final_confidence
                best_tf = tf
                print(f"   💾 NEW BEST SIGNAL: {direction} at {best_signal['entry']}")

    if best_signal and best_confidence >= 40:
        print(f"\n🎉 ✅ SIGNAL GENERATED: {symbol} {best_signal['direction']} | Confidence: {best_confidence:.1f}%")
        return process_romeoptp_signal(best_signal)
    elif best_signal:
        print(f"\n❌ SKIP: Confidence too low ({best_confidence:.1f}% < 40%)")
    else:
        print(f"\n❌ NO SIGNAL: No valid setups found across all timeframes")
    
    return False

# ===== ENHANCED ANALYZE_SYMBOL WITH LOGGING =====
def analyze_symbol(symbol):
    """
    🔥 PURE ROMEOPTP ANALYSIS: Only uses Romeoptp liquidity manipulation detection
    WITH DETAILED LOGGING
    """
    global total_checked_signals, skipped_signals, last_trade_time, volatility_pause_until
    
    total_checked_signals += 1
    now = time.time()
    
    if time.time() < volatility_pause_until:
        print(f"⏸️  VOLATILITY PAUSE: Skipping {symbol}")
        skipped_signals += 1
        return False

    if not symbol or symbol in SYMBOL_BLACKLIST:
        print(f"🚫 BLACKLIST: Skipping {symbol}")
        skipped_signals += 1
        return False

    # Basic volume check only (no complex volume analysis)
    try:
        vol24 = get_24h_quote_volume(symbol)
        if vol24 < 1_500_000.0:
            print(f"📉 LOW VOLUME: {symbol} - ${vol24:,.0f} (need $1.5M)")
            skipped_signals += 1
            return False
        else:
            print(f"📈 VOLUME OK: {symbol} - ${vol24:,.0f}")
    except Exception as e:
        print(f"⚠️ Volume check error for {symbol}: {e}")
        skipped_signals += 1
        return False

    # Cooldown check
    if last_trade_time.get(symbol, 0) > now:
        cooldown_left = last_trade_time[symbol] - now
        print(f"⏰ COOLDOWN: {symbol} - {cooldown_left:.0f}s remaining")
        skipped_signals += 1
        return False

    # Global open-trade limits
    open_trade_count = len([t for t in open_trades if t.get("st") == "open"])
    if open_trade_count >= MAX_OPEN_TRADES:
        print(f"📊 MAX TRADES: {open_trade_count}/{MAX_OPEN_TRADES} - Skipping {symbol}")
        skipped_signals += 1
        return False
    else:
        print(f"📊 TRADES: {open_trade_count}/{MAX_OPEN_TRADES} open")

    # 🔥 USE ONLY ROMEOPTP SIGNAL GENERATION - NO LEGACY SCORING
    print(f"🎯 ROMEOPTP SCANNING {symbol}...")
    try:
        return generate_signal(symbol)
    except Exception as e:
        print(f"❌ ERROR in generate_signal for {symbol}: {e}")
        import traceback
        print(f"🔧 Stack trace: {traceback.format_exc()}")
        return False

# ===== LOGGING FUNCTIONS =====
def init_csv():
    """Initialize the CSV log file with headers"""
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc", "symbol", "side", "entry", "tp1", "tp2", "tp3", "sl",
                "tf", "units", "margin_usd", "exposure_usd", "risk_pct", "confidence_pct", "status", "breakdown"
            ])
        print(f"✅ CSV log initialized: {LOG_CSV}")

def log_signal(row):
    """Log a signal to CSV"""
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

def log_trade_close(trade):
    """Log trade closure to CSV"""
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), trade["s"], trade["side"], trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                trade.get("risk_pct")*100 if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), "Romeoptp_Closed"
            ])
    except Exception as e:
        print("log_trade_close error:", e)

# ===== STARTUP =====
init_csv()
send_message("🚀 ROMEOPTP LIQUIDITY MANIPULATION ENGINE DEPLOYED\n"
             "🎯 Pure Price-Based Analysis | No EMA/Trend Filters\n"
             "🔧 CRT + Turtle Soup + BOS/MSS + Order Blocks\n"
             "⚡ Liquidity Targeting & Sweep Detection Active\n"
             "📊 Detailed Logging Enabled - Monitoring 70 Symbols")

try:
    SYMBOLS = get_top_symbols(70)
    print(f"🚀 ROMEOPTP ENGINE STARTED")
    print(f"📊 Monitoring {len(SYMBOLS)} symbols")
    print(f"🎯 Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"🔧 Pure Price Manipulation Analysis")
    print(f"📈 Detailed logging enabled")
    print(f"{'='*60}")
    
    # ADD THIS DELAY TO SHOW DEPLOYMENT IS WORKING
    print("⏳ Starting main loop in 3 seconds...")
    time.sleep(3)
    
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.")

# ===== MAIN LOOP =====
cycle_count = 0
while True:
    try:
        cycle_count += 1
        print(f"\n🔄 CYCLE {cycle_count} STARTED at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        
        # Check for BTC volatility spikes
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")

        # Scan all symbols for Romeoptp setups
        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(SYMBOLS)}] 🎯 ROMEOPTP SCANNING {sym}")
            print(f"{'='*60}")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
            time.sleep(API_CALL_DELAY)

        # Check open trades for TP/SL
        check_trades()

        # Periodic maintenance
        now = time.time()
        if now - last_heartbeat > 43200:  # 12 hours
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:    # 24 hours
            summary()
            last_summary = now

        print(f"\n✅ Cycle {cycle_count} completed at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        print(f"⏳ Waiting {CHECK_INTERVAL} seconds for next cycle...")
        time.sleep(CHECK_INTERVAL)
        
    except Exception as e:
        print(f"❌ Main loop error: {e}")
        print("🔄 Restarting main loop in 10 seconds...")
        time.sleep(10)
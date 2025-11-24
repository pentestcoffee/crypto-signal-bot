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

# ===== ROMEOPTP SIGNAL GENERATION =====
def generate_signal(symbol):
    """
    🔥 REWRITTEN SIGNAL GENERATION: Pure Romeoptp liquidity manipulation flow
    Signal Flow:
    1. Identify CRT range
    2. Detect sweep of range high/low  
    3. Confirm reclaim inside the range
    4. Confirm BOS/MSS in direction of reversal
    5. Identify order block or FVG as entry zone
    6. SL = sweep wick high/low
    7. TP = next liquidity target
    """
    global total_checked_signals, skipped_signals, signals_sent_total
    
    total_checked_signals += 1
    
    # Get data for multiple timeframes
    best_signal = None
    best_confidence = 0
    
    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf, 100)
        if df is None or len(df) < 50:
            continue
            
        # 1. IDENTIFY CRT RANGE
        range_high, range_low, range_quality = detect_crt_range(df)
        
        if range_quality > 0.3:  # Minimum range quality
            # 2. DETECT SWEEP
            sweep_direction = detect_crt_sweep(df, range_high, range_low)
            
            if sweep_direction:
                # 3. CONFIRM RECLAIM
                reclaim_confirmed = detect_crt_reclaim(df, range_high, range_low, sweep_direction)
                
                if reclaim_confirmed:
                    # 4. CONFIRM BOS/MSS
                    mss_confirmation = detect_bos_mss(df, sweep_direction)
                    
                    # 5. CHECK TURTLE SOUP (alternative signal)
                    turtle_signal = detect_turtle_soup(df)
                    
                    # ROMEOPTP SIGNAL LOGIC: turtle_soup OR (crt_sweep AND mss)
                    valid_signal = False
                    direction = None
                    
                    if turtle_signal == "bullish" or (sweep_direction == "swept_low" and mss_confirmation == "bullish_shift"):
                        direction = "BUY"
                        valid_signal = True
                    elif turtle_signal == "bearish" or (sweep_direction == "swept_high" and mss_confirmation == "bearish_shift"):
                        direction = "SELL" 
                        valid_signal = True
                    
                    if valid_signal and direction:
                        # Calculate confidence based on multiple confirmations
                        confidence = range_quality * 0.4
                        if mss_confirmation: confidence += 0.3
                        if turtle_signal: confidence += 0.3
                        
                        if confidence > best_confidence:
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
                                'confidence': confidence * 100
                            }
                            best_confidence = confidence
    
    if best_signal and best_confidence >= 60:  # Minimum 60% confidence
        return process_romeoptp_signal(best_signal)
    
    return False

def process_romeoptp_signal(signal):
    """
    Process validated Romeoptp signal with proper risk management
    """
    global signals_sent_total, STATS, recent_signals
    
    symbol = signal['symbol']
    direction = signal['direction']
    entry = signal['entry']
    confidence = signal['confidence']
    
    # Get additional data for precise calculations
    df = get_klines(symbol, signal['timeframe'], 100)
    if df is None:
        return False
    
    # 5. IDENTIFY ORDER BLOCK OR FVG AS ENTRY ZONE
    ob_high, ob_low = detect_order_block(df, direction)
    if ob_high and ob_low:
        # Use order block midpoint as refined entry
        entry = (ob_high + ob_low) / 2
    
    # 6. SL = SWEEP WICK HIGH/LOW
    if direction == "BUY":
        stop_loss = signal['range_low'] * 0.998  # Slightly below range low
    else:
        stop_loss = signal['range_high'] * 1.002  # Slightly above range high
    
    # 7. TP = NEXT LIQUIDITY TARGET
    take_profit = next_liquidity_target(df, entry, direction)
    
    # Calculate position sizing
    units, margin, exposure, risk_used = pos_size_units(entry, stop_loss, confidence)
    
    if units <= 0:
        return False
    
    # Check for duplicate signals
    sig_key = (symbol, direction, round(entry, 6))
    current_time = time.time()
    if sig_key in recent_signals:
        time_since_last = current_time - recent_signals[sig_key]
        if time_since_last < RECENT_SIGNAL_SIGNATURE_EXPIRE:
            print(f"Skipping {symbol}: duplicate recent signal")
            skipped_signals += 1
            return False
    
    recent_signals[sig_key] = current_time
    
    # Prepare signal message
    compression_detected, compression_ratio = detect_compression(df)
    
    message = (f"🎯 ROMEOPTP LIQUIDITY SIGNAL\n"
               f"✅ {direction} {symbol} | {signal['timeframe']}\n"
               f"💵 Entry: {entry:.4f}\n"
               f"🎯 TP: {take_profit:.4f}\n"
               f"🛑 SL: {stop_loss:.4f}\n"
               f"💰 Units: {units:.4f} | Exposure: ${exposure:.2f}\n"
               f"⚡ Confidence: {confidence:.1f}%\n"
               f"🔍 Sweep: {signal['sweep_direction']}\n"
               f"📊 MSS: {signal['mss_confirmation']}\n"
               f"🐢 Turtle: {signal['turtle_signal']}\n"
               f"📉 Compression: {'Yes' if compression_detected else 'No'} (Ratio: {compression_ratio:.2f})")
    
    send_message(message)
    
    # Create trade object
    trade_obj = {
        "s": symbol,
        "side": direction,
        "entry": entry,
        "tp1": take_profit,
        "sl": stop_loss,
        "st": "open",
        "units": units,
        "margin": margin,
        "exposure": exposure,
        "risk_pct": risk_used,
        "confidence_pct": confidence,
        "tp1_taken": False,
        "placed_at": time.time(),
        "entry_tf": signal['timeframe'],
        "romeoptp_data": signal
    }
    
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][direction]["sent"] += 1
    STATS["by_tf"][signal['timeframe']]["sent"] += 1
    
    log_signal([
        datetime.utcnow().isoformat(), symbol, direction, entry,
        take_profit, None, None, stop_loss, signal['timeframe'], 
        units, margin, exposure, risk_used*100, confidence, "open", 
        f"Romeoptp_{signal['sweep_direction']}"
    ])
    
    print(f"✅ ROMEOPTP Signal: {symbol} {direction} at {entry}. Confidence: {confidence:.1f}%")
    return True

# ===== RETAINED UTILITY FUNCTIONS =====
def get_atr(symbol, period=14):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs:
        return None
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8,2.8,3.8), conf_multiplier=1.0):
    atr = get_atr(symbol)
    if atr is None:
        return None
    atr = max(min(atr, entry * 0.05), entry * 0.0001)
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)
    if side == "BUY":
        sl  = round(entry - atr * adj_sl_multiplier, 8)
        tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 8)
    else:
        sl  = round(entry + atr * adj_sl_multiplier, 8)
        tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 8)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct):
    conf = max(0.0, min(100.0, confidence_pct))
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
    risk_percent = max(risk_percent, BASE_RISK)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    min_sl = max(entry * MIN_SL_DISTANCE_PCT, 1e-8)
    if sl_dist < min_sl:
        return 0.0, 0.0, 0.0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * MAX_EXPOSURE_PCT
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    margin_req = exposure / LEVERAGE
    if margin_req < MIN_MARGIN_USD:
        return 0.0, 0.0, 0.0, risk_percent
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent  # ✅ FIXED
    
def btc_volatility_spike():
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown"
            ])

def log_signal(row):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
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

# ===== TRADE CHECK (TP/SL) =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven, STATS, last_trade_time, last_trade_result
    
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
            
        p = get_price(t["s"])
        if p is None:
            continue
            
        side = t["side"]

        if side == "BUY":
            # Check for TP hit
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["st"] = "closed"
                send_message(f"🎯 {t['s']} TP Hit {p:.4f} — Trade closed.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
                
            # Check for SL hit
            if p <= t["sl"]:
                t["st"] = "fail"
                signals_fail_total += 1
                STATS["by_side"]["BUY"]["fail"] += 1
                STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                send_message(f"❌ {t['s']} SL Hit {p:.4f}")
                last_trade_result[t["s"]] = "loss"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                log_trade_close(t)
                
        else:  # SELL
            # Check for TP hit
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["st"] = "closed"
                send_message(f"🎯 {t['s']} TP Hit {p:.4f} — Trade closed.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
                
            # Check for SL hit
            if p >= t["sl"]:
                t["st"] = "fail"
                signals_fail_total += 1
                STATS["by_side"]["SELL"]["fail"] += 1
                STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                send_message(f"❌ {t['s']} SL Hit {p:.4f}")
                last_trade_result[t["s"]] = "loss"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                log_trade_close(t)

    # Cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ===== VOLUME CHECK (SIMPLIFIED) =====
def volume_ok(symbol):
    """Basic volume check - ensure symbol has sufficient liquidity"""
    vol24 = get_24h_quote_volume(symbol)
    return vol24 >= 1_500_000.0  # $1.5M minimum 24h volume

# ===== SYMBOL ANALYSIS =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, last_trade_time, volatility_pause_until
    
    total_checked_signals += 1
    now = time.time()
    
    if time.time() < volatility_pause_until:
        return False

    if not symbol or symbol in SYMBOL_BLACKLIST:
        skipped_signals += 1
        return False

    # Basic volume check
    if not volume_ok(symbol):
        skipped_signals += 1
        return False

    # Cooldown check
    if last_trade_time.get(symbol, 0) > now:
        print(f"Cooldown active for {symbol}, skipping")
        skipped_signals += 1
        return False

    # Global open-trade limits
    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max open trades reached ({MAX_OPEN_TRADES}).")
        skipped_signals += 1
        return False

    # Generate Romeoptp signal
    return generate_signal(symbol)

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 Romeoptp Engine Active {datetime.utcnow().strftime('%H:%M UTC')}")
    print("💓 Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    acc   = (hits / total * 100) if total > 0 else 0.0
    
    message = (f"📊 ROMEOPTP DAILY SUMMARY\n"
               f"Signals Sent: {total}\n"
               f"Signals Checked: {total_checked_signals}\n"
               f"Signals Skipped: {skipped_signals}\n"
               f"✅ Hits: {hits}\n"
               f"❌ Fails: {fails}\n"
               f"🎯 Accuracy: {acc:.1f}%\n"
               f"🔧 Pure Price Manipulation Analysis")
    
    send_message(message)
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")

# ===== STARTUP =====
init_csv()
send_message("🚀 ROMEOPTP LIQUIDITY MANIPULATION ENGINE DEPLOYED\n"
             "🎯 Pure Price-Based Analysis | No EMA/Trend Filters\n"
             "🔧 CRT + Turtle Soup + BOS/MSS + Order Blocks\n"
             "⚡ Liquidity Targeting & Sweep Detection Active")

try:
    SYMBOLS = get_top_symbols(70)
    print(f"Monitoring {len(SYMBOLS)} symbols.")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.")

# ===== MAIN LOOP =====
while True:
    try:
        # Check for BTC volatility spikes
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")

        # Scan all symbols for Romeoptp setups
        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} for liquidity setups...")
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

        print(f"🔁 Cycle completed at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        time.sleep(CHECK_INTERVAL)
        
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIVE ROMEOPT 6-STEP SCANNER (Enhanced + Elite Features)
- Fully live early signals
- RomeOPT 6-step logic
- TP/SL tracking with ATR or OB
- Dynamic TP/SL updates (market-structure-based)
- Telegram alerts
- Async SQLite logging
- Filters: Score >=5, Displacement +2, Sweep+2 OR Zone+1, avoid counter-trend
- Improved Order Block detection
- Adaptive Market Regime detection
- HTF + Sweep scoring threshold
- Elite multi-timeframe confirmation (15m,1h,4h)
"""

import os, time, asyncio, logging, datetime
import aiosqlite
import httpx
import ccxt.async_support as ccxt
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from collections import defaultdict, deque

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "changeme")
DB_PATH = "/app/data/signals.db"

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))
TOP_N = int(os.getenv("TOP_N", 60))

# ==== CHANGE 1: BYBIT TIMEFRAMES ====
TIMEFRAMES = ["30m", "1h", "2h", "3h", "4h"]  # Changed from ["1m","3m","5m","15m","30m"]
MIN_SCORE = 5
CRITICAL_FACTORS_MIN = 2  # HTF Alignment + Liquidity Sweep minimum

# ==== ADDED: BYBIT TP/SL CONFIG ====
MIN_TP_DISTANCE_RATIO = 0.15  # Minimum TP1 distance as % of risk (15%)
MIN_RR_RATIO = 1.2  # Minimum risk/reward ratio 1:1.2

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("romeopt_bot")
db_lock = asyncio.Lock()
db_conn = None
exchange = None  # Global exchange object

# ---------------- TELEGRAM ----------------
def escape_html(msg: str) -> str:
    if not msg: return "-"
    return str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

async def tg(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    safe_msg = escape_html(msg)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": safe_msg, "parse_mode":"HTML"})
        except Exception as e:
            log.warning(f"Telegram send failed: {e}")

# ---------------- DATABASE ----------------
async def init_db():
    global db_conn
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        db_conn = await aiosqlite.connect(DB_PATH)
        await db_conn.execute("PRAGMA journal_mode=WAL;")
        await db_conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Check if table exists and add rr_ratio column if needed
        cursor = await db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
        table_exists = await cursor.fetchone()
        
        if not table_exists:
            await db_conn.execute("""
                CREATE TABLE signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    entry REAL,
                    sl REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    timestamp TEXT,
                    status TEXT,
                    reason TEXT,
                    score INTEGER,
                    tp1_hit INTEGER DEFAULT 0,
                    tp2_hit INTEGER DEFAULT 0,
                    tp3_hit INTEGER DEFAULT 0,
                    latest_ob TEXT,
                    rr_ratio REAL
                );
            """)
        else:
            # Check if rr_ratio column exists
            cursor = await db_conn.execute("PRAGMA table_info(signals)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            if 'rr_ratio' not in column_names:
                await db_conn.execute("ALTER TABLE signals ADD COLUMN rr_ratio REAL")
        
        await db_conn.commit()
        log.info("Database initialized successfully")
    except Exception as e:
        log.error(f"Database initialization failed: {e}")
        raise

# ---------------- OHLCV ----------------
async def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit=200):
    try:
        return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        log.debug("fetch_ohlcv failed for %s %s: %s", symbol, timeframe, e)
        return None

# ---------------- INDICATORS ----------------
def atr(df: pd.DataFrame, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.DataFrame({
        "h-l": high - low,
        "h-pc": (high - close.shift(1)).abs(),
        "l-pc": (low - close.shift(1)).abs()
    }).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# ---------------- MARKET REGIME ----------------
async def detect_market_regime(df: pd.DataFrame):
    ma_htf = df["close"].rolling(50).mean().iloc[-1]
    price = df["close"].iloc[-1]
    recent_high = df["high"].iloc[-20:].max()
    recent_low = df["low"].iloc[-20:].min()
    range_pct = (recent_high - recent_low) / max(1e-8, recent_low)
    if price > ma_htf and range_pct > 0.02:
        return "BULL"
    elif price < ma_htf and range_pct > 0.02:
        return "BEAR"
    else:
        return "RANGE"

# ---------------- MULTI-TIMEFRAME ELITE CONFIRM ----------------
async def elite_tf_alignment(exchange, symbol: str, side: str):
    # ==== CHANGE 2: UPDATED TIMEFRAMES FOR BYBIT ====
    tfs = ["30m", "1h", "2h"]  # Changed from ["15m","1h","4h"] to match Bybit focus
    for tf in tfs:
        ohlcv = await fetch_ohlcv(exchange, symbol, tf, 50)
        if not ohlcv: return False
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
        trend = df["close"].iloc[-1] - df["close"].iloc[-5]
        trend_side = "BUY" if trend>0 else "SELL"
        if trend_side != side:
            return False
    return True

# ---------------- ROMEOPT 6-STEP SIGNAL ----------------
async def generate_signal_romeopt(exchange, df: pd.DataFrame, symbol: str, tf: str):
    if df is None or len(df) < 20: return None
    last = df.iloc[-1]
    prev5 = df.iloc[-6:-1]
    score = 0
    reasons = []

    # Step 1: Liquidity Sweep
    sweep_high = last["high"] > prev5["high"].max()
    sweep_low = last["low"] < prev5["low"].min()
    has_sweep = sweep_high or sweep_low
    liquidity_sweep = 2 if has_sweep else 0
    score += liquidity_sweep
    reasons.append(f"Liquidity Sweep +{liquidity_sweep}")

    # Step 2: Displacement
    displacement = abs(last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-8)
    has_disp = displacement > 0.6
    if has_disp:
        score += 2; reasons.append("Displacement +2")
    else:
        reasons.append("Displacement +0")

    # Step 3 & 4: Order Block & Zone
    ob_zone = None
    for i in range(len(df)-5, len(df)-1):
        candle, prev_candle = df.iloc[i], df.iloc[i-1]
        if candle["close"]>candle["open"] and prev_candle["close"]<prev_candle["open"]:
            ob_zone={"type":"bullish","low":min(candle["low"], prev_candle["low"]),"high":candle["close"]}; break
        elif candle["close"]<candle["open"] and prev_candle["close"]>prev_candle["open"]:
            ob_zone={"type":"bearish","low":candle["close"],"high":max(candle["high"], prev_candle["high"])}; break

    if ob_zone:
        ob_type = ob_zone["type"]
        if ob_type=="bullish" and last["close"] <= ob_zone["high"]: score+=1; reasons.append("Zone Approach +1")
        elif ob_type=="bearish" and last["close"] >= ob_zone["low"]: score+=1; reasons.append("Zone Approach +1")
        else: reasons.append("Zone Approach +0")
    else:
        reasons.append("Zone Approach +0"); ob_type=None

    # Step 5: HTF Alignment
    # ==== CHANGE 3: UPDATED HTF MAPPING FOR BYBIT TIMEFRAMES ====
    tf_map = {
        "30m": "1h",
        "1h": "2h", 
        "2h": "3h",
        "3h": "4h",
        "4h": "6h"
    }
    htf = tf_map.get(tf, "1h")
    ohlcv_htf = await fetch_ohlcv(exchange, symbol, htf, 50)
    htf_alignment = 0
    if ohlcv_htf:
        df_htf = pd.DataFrame(ohlcv_htf, columns=["ts","open","high","low","close","vol"])
        trend = df_htf["close"].iloc[-1] - df_htf["close"].iloc[-5]
        htf_dir = "bullish" if trend>0 else "bearish"
        if ob_type and htf_dir==ob_type:
            score+=1; htf_alignment=1; reasons.append(f"HTF ({htf}) Alignment +1")
        else:
            reasons.append(f"HTF ({htf}) Alignment +0")
    else:
        reasons.append("HTF Alignment ?")

    # Step 6: Momentum
    momentum_ratio = abs(last["close"]-last["open"])/(last["high"]-last["low"]+1e-8)
    if ob_type=="bullish" and momentum_ratio>0.5 and last["close"]>last["open"]:
        score+=1; reasons.append("Momentum +1")
    elif ob_type=="bearish" and momentum_ratio>0.5 and last["close"]<last["open"]:
        score+=1; reasons.append("Momentum +1")
    else:
        reasons.append("Momentum +0")

    if not ob_type: return None
    side = "BUY" if ob_type=="bullish" else "SELL"
    entry = float(last["close"])

    # ---------------- CRITICAL FILTERS ----------------
    critical_score = htf_alignment + liquidity_sweep
    if critical_score < CRITICAL_FACTORS_MIN: return None
    if score < MIN_SCORE: return None
    if not has_disp: return None
    
    # ---------------- NEW: HTF ALIGNMENT MANDATORY FILTER ----------------
    if htf_alignment != 1:  # MUST HAVE HTF Alignment = 1
        return None

    market_regime = await detect_market_regime(df)
    if (market_regime=="BULL" and side=="SELL") or (market_regime=="BEAR" and side=="BUY"): return None

    trend_ma = df["close"].rolling(20).mean().iloc[-1]
    if (side=="BUY" and last["close"]<trend_ma) or (side=="SELL" and last["close"]>trend_ma): return None

    # ---------------- ELITE MTF CONFIRMATION ----------------
    if not await elite_tf_alignment(exchange, symbol, side):
        return None
    reasons.append("Elite MTF Alignment ✅")

    sig = {"symbol":symbol,"side":side,"entry":entry,"score":score,"reason":"RomeOPT 6-Step",
           "reason_list":reasons,"htf_alignment":htf_alignment,"liquidity_sweep":liquidity_sweep}
    sig = update_tp_sl_live(sig, df)
    
    # ==== CHANGE 4: ENHANCED TP/SL VALIDATION ====
    if sig and "sl" in sig and "tp1" in sig:
        risk = abs(sig["entry"] - sig["sl"])
        tp1_distance = abs(sig["tp1"] - sig["entry"])
        
        # Reject if TP1 is less than minimum ratio of risk
        if tp1_distance < risk * MIN_TP_DISTANCE_RATIO:
            return None
        
        # Calculate risk/reward ratio
        rr_ratio = tp1_distance / risk if risk > 0 else 0
        sig["rr_ratio"] = rr_ratio
        
        # Reject if risk/reward ratio is too low
        if rr_ratio < MIN_RR_RATIO:
            return None
    
    return sig

# ---------------- TP/SL HELPERS ----------------
def romeopt_tp_sl(entry, side, atr_val, ob_zone, df):
    """
    OPTIMIZED TP/SL using market structure + ATR
    - Enhanced for Bybit volatility
    - Proper risk/reward ratios
    - Maintains all your elite logic
    """
    recent_high = df['high'].iloc[-10:].max()
    recent_low = df['low'].iloc[-10:].min()

    if side == "BUY":
        # Calculate SL using conservative approach
        sl_ob = ob_zone["low"] - (atr_val * 0.3)
        sl_structure = recent_low - (atr_val * 0.3)
        sl = min(sl_ob, sl_structure)
        
        risk = entry - sl
        
        # Ensure minimum meaningful risk for Bybit
        min_risk = atr_val * 0.8  # Increased for Bybit volatility
        if risk < min_risk:
            risk = min_risk
            sl = entry - risk
        
        # Calculate TP levels with proper spacing
        base_tp1 = entry + (risk * 1.2)  # 1.2:1 RR
        base_tp2 = entry + (risk * 2.0)  # 2.0:1 RR  
        base_tp3 = entry + (risk * 3.0)  # 3.0:1 RR
        
        # Get market structure levels
        nearest_resistance = df['high'].tail(20).max()
        major_resistance = df['high'].tail(50).max()
        
        # Choose better profit level (calculated vs structure)
        tp1 = min(base_tp1, nearest_resistance) if nearest_resistance > entry else base_tp1
        tp2 = min(base_tp2, major_resistance) if major_resistance > tp1 else base_tp2
        tp3 = base_tp3
        
        # Ensure proper ordering with meaningful distances
        min_tp_gap = risk * 0.4  # Increased for Bybit
        
        tp1 = max(tp1, entry + (risk * 0.8))  # At least 0.8R profit
        tp2 = max(tp2, tp1 + min_tp_gap)
        tp3 = max(tp3, tp2 + min_tp_gap)
        
    else:  # SELL
        # Calculate SL using conservative approach
        sl_ob = ob_zone["high"] + (atr_val * 0.3)
        sl_structure = recent_high + (atr_val * 0.3)
        sl = max(sl_ob, sl_structure)
        
        risk = sl - entry
        
        # Ensure minimum meaningful risk for Bybit
        min_risk = atr_val * 0.8
        if risk < min_risk:
            risk = min_risk
            sl = entry + risk
        
        # Calculate TP levels with proper spacing
        base_tp1 = entry - (risk * 1.2)
        base_tp2 = entry - (risk * 2.0)
        base_tp3 = entry - (risk * 3.0)
        
        # Get market structure levels
        nearest_support = df['low'].tail(20).min()
        major_support = df['low'].tail(50).min()
        
        # Choose better profit level
        tp1 = max(base_tp1, nearest_support) if nearest_support < entry else base_tp1
        tp2 = max(base_tp2, major_support) if major_support < tp1 else base_tp2
        tp3 = base_tp3
        
        # Ensure proper ordering with meaningful distances
        min_tp_gap = risk * 0.4
        
        tp1 = min(tp1, entry - (risk * 0.8))
        tp2 = min(tp2, tp1 - min_tp_gap)
        tp3 = min(tp3, tp2 - min_tp_gap)

    return sl, tp1, tp2, tp3

def find_latest_ob(df: pd.DataFrame):
    for i in range(len(df)-5, len(df)-1):
        candle, prev_candle = df.iloc[i], df.iloc[i-1]
        if candle["close"]>candle["open"] and prev_candle["close"]<prev_candle["open"]:
            return {"type":"bullish","low":min(candle["low"], prev_candle["low"]),"high":candle["close"]}
        elif candle["close"]<candle["open"] and prev_candle["close"]>prev_candle["open"]:
            return {"type":"bearish","low":candle["close"],"high":max(candle["high"], prev_candle["high"])}
    return None

def update_tp_sl_live(sig: dict, df: pd.DataFrame):
    latest_ob = find_latest_ob(df)
    if not latest_ob: return sig
    atr_val = float(atr(df,14).iloc[-1])
    entry = sig["entry"]
    side = sig["side"]
    sl,tp1,tp2,tp3 = romeopt_tp_sl(entry, side, atr_val, latest_ob, df)
    sig["sl"]=sl; sig["tp1"]=tp1; sig["tp2"]=tp2; sig["tp3"]=tp3
    sig["latest_ob"]=latest_ob
    return sig

# ---------------- SL CLUSTER ----------------
recent_sl = defaultdict(lambda: deque())
def record_sl_hit(symbol: str, lookback_minutes=30):
    now = time.time(); dq = recent_sl[symbol]; dq.append(now)
    cutoff = now - lookback_minutes*60
    while dq and dq[0]<cutoff: dq.popleft()
def deprioritized(symbol: str, threshold=3, lookback=30):
    dq = recent_sl[symbol]; now=time.time(); cutoff=now-lookback*60
    while dq and dq[0]<cutoff: dq.popleft()
    return len(dq)>=threshold

# ---------------- LOG SIGNAL ----------------
async def log_signal(sig):
    async with db_lock:
        await db_conn.execute("""
            INSERT INTO signals (symbol,side,entry,sl,tp1,tp2,tp3,timestamp,status,reason,score,latest_ob,rr_ratio)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (sig["symbol"],sig["side"],sig["entry"],sig.get("sl"),sig.get("tp1"),sig.get("tp2"),sig.get("tp3"),
              datetime.datetime.utcnow().isoformat(),"OPEN",sig["reason"],sig["score"],str(sig.get("latest_ob","")),
              sig.get("rr_ratio", 0)))
        await db_conn.commit()

# ---------------- MONITOR SIGNALS ----------------
async def monitor_signals():
    while True:
        try:
            async with db_lock:
                async with db_conn.execute("SELECT id,symbol,side,entry,sl,tp1,tp2,tp3,tp1_hit,tp2_hit,tp3_hit,status FROM signals WHERE status='OPEN'") as cursor:
                    async for row in cursor:
                        sig_id, symbol, side, entry, sl, tp1, tp2, tp3, tp1_hit, tp2_hit, tp3_hit, status = row
                        try:
                            ticker = await exchange.fetch_ticker(symbol)
                            last_price = ticker.get("last")
                            if last_price is None: continue
                        except:
                            continue

                        ohlcv = await fetch_ohlcv(exchange, symbol, "30m", 50)
                        if ohlcv:
                            df_live = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
                            for c in ["open","high","low","close","vol"]: df_live[c]=pd.to_numeric(df_live[c],errors="coerce")
                            sig = {"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3}
                            sig = update_tp_sl_live(sig, df_live)
                            sl,tp1,tp2,tp3 = sig["sl"], sig["tp1"], sig["tp2"], sig["tp3"]

                        hits=[]; sl_hit=False
                        if side=="BUY":
                            if not tp1_hit and last_price>=tp1: hits.append("TP1"); tp1_hit=1
                            if not tp2_hit and last_price>=tp2: hits.append("TP2"); tp2_hit=1
                            if not tp3_hit and last_price>=tp3: hits.append("TP3"); tp3_hit=1
                            if last_price<=sl: hits.append("SL"); status="CLOSED"; sl_hit=True
                        else:
                            if not tp1_hit and last_price<=tp1: hits.append("TP1"); tp1_hit=1
                            if not tp2_hit and last_price<=tp2: hits.append("TP2"); tp2_hit=1
                            if not tp3_hit and last_price<=tp3: hits.append("TP3"); tp3_hit=1
                            if last_price>=sl: hits.append("SL"); status="CLOSED"; sl_hit=True

                        if hits:
                            await tg(f"🎯 {symbol} {side} update\nEntry:{entry}\nLast:{last_price}\nHits:{','.join(hits)}\nSL:{sl}\nTP1:{tp1} TP2:{tp2} TP3:{tp3}")

                        if sl_hit: record_sl_hit(symbol)
                        await db_conn.execute("UPDATE signals SET tp1_hit=?,tp2_hit=?,tp3_hit=?,status=? WHERE id=?",
                                             (tp1_hit,tp2_hit,tp3_hit,status,sig_id))
                await db_conn.commit()
        except Exception as e: 
            log.error(f"Monitor error: {e}")
        await asyncio.sleep(SCAN_INTERVAL)

# ---------------- SCAN LOOP ----------------
last_signal_time = {}
async def scan_loop(exchange):
    while True:
        t0=time.time()
        try:
            tickers = await exchange.fetch_tickers()
            top = sorted([(s,v.get("quoteVolume",0)) for s,v in tickers.items() if s.endswith("USDT")], key=lambda x:x[1], reverse=True)[:TOP_N]
            signals_found = 0
            for symbol,_ in top:
                if deprioritized(symbol): continue
                for tf in TIMEFRAMES:
                    key=f"{symbol}:{tf}"
                    if key in last_signal_time and time.time()-last_signal_time[key]<60: continue
                    ohlcv = await fetch_ohlcv(exchange,symbol,tf,200)
                    if not ohlcv: continue
                    df=pd.DataFrame(ohlcv,columns=["ts","open","high","low","close","vol"])
                    for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
                    sig = await generate_signal_romeopt(exchange,df,symbol,tf)
                    if sig:
                        htf_flag = sig.get("htf_alignment", "N/A")
                        sweep_flag = sig.get("liquidity_sweep", "N/A")
                        rr_ratio = sig.get("rr_ratio", 0)
                        await tg(f"🏆 {sig['symbol']} ({tf}) {sig['side']}\nEntry:{sig['entry']}\nSL:{sig.get('sl')}\nTP1:{sig.get('tp1')} TP2:{sig.get('tp2')} TP3:{sig.get('tp3')}\nScore:{sig['score']}\nHTF:{htf_flag} Sweep:{sweep_flag} RR:{rr_ratio:.2f}:1\nBreakdown:{', '.join(sig['reason_list'])}")
                        await log_signal(sig)
                        last_signal_time[key]=time.time()
                        signals_found+=1
            log.info(f"📊 Scan complete: {signals_found} RomeOPT signals found")
        except Exception as e: 
            log.error(f"Scan error: {e}")
        elapsed=time.time()-t0
        await asyncio.sleep(max(1,SCAN_INTERVAL-elapsed))

# ---------------- FASTAPI ----------------
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "RomeOPT Bybit Scanner", "timeframes": TIMEFRAMES}

@app.post("/webhook")
async def webhook(request: Request):
    token = request.headers.get("X-Auth","")
    if token!=WEBHOOK_SECRET: 
        raise HTTPException(status_code=403, detail="Invalid secret")
    data = await request.json()
    log.info("Webhook received: %s", data)
    return {"ok":True}

# ---------------- MAIN ----------------
async def main():
    try:
        # Initialize database
        await init_db()
        
        # Initialize Bybit exchange
        global exchange
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Test connection
        await exchange.load_markets()
        log.info("✅ Connected to Bybit successfully")
        
        await tg("🏆 ROMEOPT 6-Step Scanner Started - Bybit Edition\n📊 Timeframes: 30m, 1h, 2h, 3h, 4h")
        
        # Run both coroutines
        scan_task = asyncio.create_task(scan_loop(exchange))
        monitor_task = asyncio.create_task(monitor_signals())
        
        # Wait for both tasks
        await asyncio.gather(scan_task, monitor_task)
        
    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
    except Exception as e:
        log.error(f"Fatal error in main: {e}")
        await tg(f"❌ Bot crashed: {e}")
        raise
    finally:
        # Cleanup
        log.info("Cleaning up resources...")
        if db_conn:
            await db_conn.close()
        if exchange:
            await exchange.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run HTTP server")
    args = parser.parse_args()
    
    if args.http:
        uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nBot stopped by user")
            exit(0)
        except Exception as e:
            print(f"Fatal error: {e}")
            exit(1)
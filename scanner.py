"""
NSE Momentum Swing Scanner v8 -> Telegram
Strategy (chosen by backtest): new 50-day high on 2x volume, top 10% 6-month strength,
above 200 SMA, only when Nifty is above its 200 SMA.
Exit: stop loss 2 x ATR below entry, then sell when a day closes below the 20 EMA.
Prices : Angel One SmartAPI (NSE).  M.Cap : AMFI list (amfi_mcap.pdf).
MODE=live : during market hours.  MODE=eod : after market close.
Educational tool, not investment advice.
"""
import os, re, json, math, time, html, hashlib, calendar
import datetime as dt
import numpy as np
import pandas as pd
import requests
import feedparser
import pyotp

try:
    from SmartApi import SmartConnect
except ImportError:
    from smartapi import SmartConnect

# ===================== SETTINGS =====================
CAPITAL        = float(os.getenv("CAPITAL") or 100000)  # trading capital in Rs
RISK_PCT       = 0.01    # risk 1% of capital per trade
MAX_ALLOC      = 0.20    # max 20% of capital in one stock
MAX_POSITIONS  = 5       # max stocks held at once
MAX_ALERTS     = 5       # max buy ideas per message
ATR_STOP_MULT  = 2.0     # stop loss = entry - 2 x ATR
MAX_HOLD_DAYS  = 60      # time exit (trading days)
MIN_MCAP_CR    = 5000    # minimum market cap, Rs crore
MIN_TURNOVER   = 5e7     # minimum avg daily traded value: Rs 5 crore
HIGH_LOOKBACK  = 50      # new N-day high
VOL_MULT       = 2.0     # volume must be > 2x the 20-day average
MIN_CLV        = 0.7     # close in the top 30% of the day's range
RS_LOOKBACK    = 126     # 6-month performance for strength rank
MIN_RS_RANK    = 0.90    # top 10% of all scanned stocks
MARKET_FILTER  = os.getenv("MARKET_FILTER", "on")   # "on" = no buys when Nifty < 200 SMA
LIVE_SKIP_MIN  = 30      # live mode ignores the first 30 minutes after open
MODE           = os.getenv("MODE", "eod").lower()   # "live" or "eod"

HIST_DAYS       = 2000
FETCH_PAUSE     = 0.4
REFRESH_PER_DAY = 40
MAX_FETCH_LIVE  = 300

AMFI_FILE  = "amfi_mcap.pdf"
CACHE_FILE = "history.pkl.gz"
PARSER_VER = 3
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
NIFTY, VIX = "NIFTY50", "INDIAVIX"
NIFTY_TOKEN, VIX_TOKEN = "99926000", "99926017"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

POS_WORDS = ["upgrade", "order win", "bags order", "wins order", "record profit", "profit rises",
             "profit jumps", "beats estimate", "buyback", "target raised", "raises target",
             "stake buy", "expansion", "bonus", "outperform"]
NEG_WORDS = ["fraud", "raid", "probe", "sebi notice", "sebi order", "penalty", "downgrade",
             "default", "resigns", "resignation", "profit falls", "loss widens", "plunge",
             "crash", "ban", "lawsuit", "investigation", "pledge", "misses estimate",
             "target cut", "cuts target", "underperform", "recall"]


# ===================== TIME =====================
def now_ist():
    return dt.datetime.now(IST)


def minutes_since_open(now):
    return (now.hour * 60 + now.minute) - (9 * 60 + 15)


# ===================== SAVED DATA (CACHE) =====================
def load_cache():
    try:
        c = pd.read_pickle(CACHE_FILE)
        if isinstance(c, dict):
            for k in ("bars", "fetched", "amfi", "tokens", "alerts", "market"):
                c.setdefault(k, {})
            return c
    except Exception as e:
        print("No saved data yet:", e)
    return {"bars": {}, "fetched": {}, "amfi": {}, "tokens": {}, "alerts": {}, "market": {}}


def save_cache(c):
    pd.to_pickle(c, CACHE_FILE)


# ===================== AMFI MARKET CAP LIST =====================
ISIN_RE = re.compile(r"(IN[A-Z0-9]{9}\d)")
NUM_RE = re.compile(r"^\d[\d,]*(\.\d+)?$")
CAT_RE = re.compile(r"(Large|Mid|Small)\s*Cap", re.I)
SYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-\.]*$")
RANK_RE = re.compile(r"^\s*(\d{1,5})\s")


def clean_name(name):
    name = re.sub(r"[,\.\s]+$", "", str(name))
    return re.sub(r"\s+(Ltd|Limited|Ltd\.)$", "", name, flags=re.I).strip()


def to_num(s):
    return float(s.replace(",", ""))


def join_split_numbers(tokens):
    out = []
    for t in tokens:
        if out and t[:1].isdigit() and re.fullmatch(r"\d[\d,]*,|\d[\d,]*,\d{1,2}", out[-1]):
            out[-1] += t
        else:
            out.append(t)
    return out


def parse_cells(cells):
    cells = [re.sub(r"\s+", " ", str(c or "")).strip() for c in cells]
    idx = next((i for i, c in enumerate(cells) if ISIN_RE.fullmatch(c.replace(" ", ""))), None)
    if idx is None:
        return None
    rest = [c.replace(" ", "").strip("*") if NUM_RE.match(c.replace(" ", "")) else c.strip("*").strip()
            for c in cells[idx + 1:]]
    nums = [c for c in rest if NUM_RE.match(c)]
    if not nums:
        return None
    if len(rest) >= 4 and rest[2] and SYM_RE.match(rest[2]) and not NUM_RE.match(rest[2]):
        sym = rest[2]
    else:
        syms = [c for c in rest if c and c != "-" and not NUM_RE.match(c) and SYM_RE.match(c)]
        if not syms:
            return None
        sym = syms[1] if len(syms) >= 2 else syms[0]
    name = cells[idx - 1] if idx >= 1 and cells[idx - 1] else sym
    rank = int(cells[0]) if idx >= 2 and cells[0].isdigit() else None
    return sym, name, to_num(nums[-1]), rank


def parse_line(line):
    mi = ISIN_RE.search(line)
    if not mi or not CAT_RE.search(line[mi.end():]):
        return None
    after = CAT_RE.split(line[mi.end():])[0]
    tokens = join_split_numbers([t.strip("*") for t in after.split()])
    nums = [t for t in tokens if NUM_RE.match(t)]
    syms = [t for t in tokens if t and t != "-" and not NUM_RE.match(t) and SYM_RE.match(t)]
    if not nums or not syms:
        return None
    sym = syms[1] if len(syms) >= 2 else syms[0]
    before = line[:mi.start()]
    rm = RANK_RE.match(before)
    rank = int(rm.group(1)) if rm else None
    name = re.sub(r"^\s*\d+\s+", "", before).strip() or sym
    return sym, name, to_num(nums[-1]), rank


def parse_amfi(path):
    import pdfplumber
    found = {"text_3": {}, "text_1": {}, "table_lines": {}, "table_text": {}}
    period = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if period is None:
                m = re.search(r"six months ended\s*(\d{1,2}[\s\-]*[A-Za-z]+[\s\-,]*\d{4})", text, re.I)
                if m:
                    period = m.group(1)
            for key, tol in (("text_3", 3), ("text_1", 1)):
                try:
                    t = page.extract_text(x_tolerance=tol) or ""
                except Exception:
                    t = ""
                for line in t.splitlines():
                    p = parse_line(line)
                    if p:
                        found[key][p[0]] = p
            table_methods = {
                "table_lines": lambda: page.extract_tables(),
                "table_text": lambda: page.extract_tables({"vertical_strategy": "text",
                                                           "horizontal_strategy": "text"}),
            }
            for key, get_tables in table_methods.items():
                try:
                    tables = get_tables() or []
                except Exception:
                    tables = []
                for table in tables:
                    for cells in table:
                        p = parse_cells(cells)
                        if p:
                            found[key][p[0]] = p
    print("AMFI rows per method:", {k: len(v) for k, v in found.items()})
    merged = {}
    for rows in found.values():
        for sym, name, val, rank in rows.values():
            m = merged.get(sym)
            if m is None:
                merged[sym] = {"name": name, "val": val, "rank": rank}
            else:
                m["val"] = max(m["val"], val)
                if m["rank"] is None:
                    m["rank"] = rank
    ranked = sorted((m["rank"], s) for s, m in merged.items() if m["rank"])
    for m in merged.values():
        m["mcap"] = m["val"]
    if len(ranked) > 500:
        ser = pd.Series([merged[s]["val"] for _, s in ranked], index=[r for r, _ in ranked])
        smooth = ser.rolling(15, center=True, min_periods=5).median()
        for (_, s), v in zip(ranked, smooth.values):
            if not np.isnan(v):
                merged[s]["mcap"] = max(float(v), merged[s]["val"]) if merged[s]["val"] <= v * 1.5 else float(v)
    big = sum(1 for m in merged.values() if m["mcap"] >= MIN_MCAP_CR)
    print(f"Companies above limit after repair: {big}")
    if big > 2500:
        for m in merged.values():
            m["mcap"] = m["val"]
    rows = {s: {"name": clean_name(m["name"]) or s, "mcap": m["mcap"]} for s, m in merged.items()}
    return rows, period


def get_universe(cache):
    with open(AMFI_FILE, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    a = cache["amfi"]
    if a.get("md5") != md5 or a.get("ver") != PARSER_VER:
        print("Reading AMFI file (takes a few minutes the first time)...")
        rows, period = parse_amfi(AMFI_FILE)
        a.clear()
        a.update({"md5": md5, "ver": PARSER_VER, "rows": rows, "period": period})
    universe = {s: v for s, v in a["rows"].items() if v["mcap"] >= MIN_MCAP_CR}
    return universe, a.get("period")


def amfi_age_days(period):
    d = pd.to_datetime(period, errors="coerce", dayfirst=True) if period else None
    if d is None or pd.isna(d):
        return None
    return (pd.Timestamp(now_ist().date()) - d.normalize()).days


# ===================== ANGEL ONE =====================
def angel_login():
    api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"].strip())
    totp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"].replace(" ", "").strip()).now()
    res = api.generateSession(os.environ["ANGEL_CLIENT_ID"].strip(),
                              os.environ["ANGEL_PIN"].strip(), totp)
    if not res or not res.get("status"):
        raise RuntimeError((res or {}).get("message", "no response"))
    return api


def get_tokens(cache):
    t = cache["tokens"]
    today = str(now_ist().date())
    if t.get("date") == today and t.get("map"):
        return t
    try:
        data = requests.get(SCRIP_MASTER_URL, timeout=90).json()
        m, nifty, vix = {}, None, None
        for r in data:
            if r.get("exch_seg") != "NSE":
                continue
            s = str(r.get("symbol", "")).strip()
            if s.endswith("-EQ"):
                m[s[:-3]] = str(r["token"])
            elif r.get("instrumenttype") == "AMXIDX":
                if s.lower() == "nifty 50":
                    nifty = str(r["token"])
                elif s.lower() == "india vix":
                    vix = str(r["token"])
        if m:
            t.clear()
            t.update({"date": today, "map": m,
                      "nifty": nifty or NIFTY_TOKEN, "vix": vix or VIX_TOKEN})
    except Exception as e:
        print("Instrument list download failed:", e)
    return t


def fetch_candles(api, token, days=HIST_DAYS):
    end = now_ist()
    start = end - dt.timedelta(days=days)
    params = {"exchange": "NSE", "symboltoken": str(token), "interval": "ONE_DAY",
              "fromdate": start.strftime("%Y-%m-%d 09:15"),
              "todate": end.strftime("%Y-%m-%d 15:30")}
    for attempt in range(3):
        try:
            res = api.getCandleData(params)
            if res and res.get("status"):
                rows = res.get("data") or []
                if not rows:
                    return None
                df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10])
                return df.set_index("Date").astype(float).sort_index()
            print("Candle error", token, (res or {}).get("message"))
        except Exception as e:
            print("Candle exception", token, e)
        time.sleep(3 * (attempt + 1))
    return None


def fetch_quotes(api, tokens):
    out = {}
    for i in range(0, len(tokens), 50):
        chunk = tokens[i:i + 50]
        for attempt in range(2):
            try:
                res = api.getMarketData("FULL", {"NSE": chunk})
                for q in ((res or {}).get("data") or {}).get("fetched", []):
                    out[str(q.get("symbolToken"))] = q
                break
            except Exception as e:
                print("Quote error:", e)
                time.sleep(3)
        time.sleep(1.1)
    return out


def quote_bar(q):
    if not q:
        return None, None
    t = pd.to_datetime(q.get("exchFeedTime") or q.get("exchTradeTime"), dayfirst=True, errors="coerce")
    day = None if t is None or pd.isna(t) else t.normalize()
    try:
        bar = {"Open": float(q["open"]), "High": float(q["high"]), "Low": float(q["low"]),
               "Close": float(q["ltp"]), "Volume": float(q.get("tradeVolume") or 0)}
    except (KeyError, TypeError, ValueError):
        return day, None
    return day, bar


def with_today(hist, q, today):
    base = None if hist is None else hist[hist.index < today]
    day, bar = quote_bar(q)
    if bar is None or day is None or day != today or bar["Open"] <= 0:
        return base
    row = pd.DataFrame([bar], index=pd.DatetimeIndex([today]))
    return row if base is None or base.empty else pd.concat([base, row])


# ===================== INDICATORS & SIGNAL =====================
def add_indicators(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["VOL_AVG"] = v.rolling(20).mean().shift(1)
    df["HH50"] = h.rolling(HIGH_LOOKBACK).max().shift(1)
    df["CLV"] = (c - l) / (h - l).replace(0, np.nan)
    df["EXT"] = c / df["EMA20"] - 1
    return df


def add_signals(df, nifty_close, rs_rank=None):
    c = df["Close"]
    n = nifty_close.reindex(df.index).ffill()
    df["RS"] = (c / c.shift(63)) / (n / n.shift(63))
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG"]
    df["RSR"] = rs_rank.reindex(df.index) if rs_rank is not None else np.nan
    liquid = (c * df["VOL_AVG"]) > MIN_TURNOVER
    df["SIGNAL"] = (liquid & (c > df["SMA200"]) & (c > df["HH50"]) &
                    (df["VOL_RATIO"] > VOL_MULT) & (df["RSR"] >= MIN_RS_RANK) &
                    (df["CLV"] >= MIN_CLV))
    return df


def strength_ranks(data, symbols):
    closes = pd.DataFrame({s: data[s]["Close"] for s in symbols if s in data}).sort_index()
    return (closes / closes.shift(RS_LOOKBACK) - 1).rank(axis=1, pct=True)


# ===================== MARKET & NEWS =====================
def market_state(nifty, vix):
    n = add_indicators(nifty.copy()).iloc[-1]
    vix_now = float(vix["Close"].iloc[-1]) if vix is not None and len(vix) else float("nan")
    return bool(n.Close > n.SMA200), n, vix_now


DATE_IN_TITLE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                           r"[a-z]*,?\s+(\d{4})", re.I)


def title_too_old(title, days):
    m = DATE_IN_TITLE.search(title)
    if not m:
        return False
    d = pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}", errors="coerce")
    if d is None or pd.isna(d):
        return False
    return (pd.Timestamp(now_ist().date()) - d).days > days + 1


def news(query, days=3, n=10):
    url = ("https://news.google.com/rss/search?q=" +
           requests.utils.quote(f"{query} when:{days}d") + "&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        entries = feedparser.parse(resp.content).entries
    except Exception as e:
        print("News error:", e)
        return 0, []
    cutoff = time.time() - days * 86400
    titles = []
    for e in entries:
        pp = getattr(e, "published_parsed", None)
        title = getattr(e, "title", "")
        if not pp or calendar.timegm(pp) < cutoff or title_too_old(title, days):
            continue
        titles.append(title)
        if len(titles) >= n:
            break
    score = 0
    for t in titles:
        low = t.lower()
        score += sum(w in low for w in POS_WORDS) - sum(w in low for w in NEG_WORDS)
    return score, titles


# ===================== SCANNING =====================
def alerts_today(cache, today):
    a = cache["alerts"]
    if a.get("date") != str(today.date()):
        a.clear()
        a.update({"date": str(today.date()), "sent": []})
    return a


def find_hits(universe, data, nifty_close, today, ranks, live_frac=None):
    hits = []
    for s, info in universe.items():
        df = data.get(s)
        if df is None or len(df) < 220 or df.index[-1] != today or s not in ranks:
            continue
        df = df.copy()
        df["Volume"] = df["Volume"].astype(float)
        if live_frac:   # project today's partial volume to a full day
            df.iloc[-1, df.columns.get_loc("Volume")] /= live_frac
        df = add_signals(add_indicators(df), nifty_close, ranks[s])
        row = df.iloc[-1]
        if row["SIGNAL"]:
            hits.append((s, info, row))
    hits.sort(key=lambda x: x[2]["RSR"], reverse=True)
    return hits


def trade_plan(num, sym, r, mcap, score, titles):
    entry = float(r.Close)
    atr = float(r.ATR)
    stop = entry - ATR_STOP_MULT * atr
    risk = entry - stop
    if risk <= 0:
        return None
    qty = math.floor(min(CAPITAL * RISK_PCT / risk, CAPITAL * MAX_ALLOC / entry))
    if qty < 1:
        return None
    pct = lambda x: (x / entry - 1) * 100
    ref2, ref4 = entry + 2 * risk, entry + 4 * risk
    top = max(1, round((1 - float(r.RSR)) * 100))
    return [
        f"\n<b>{num}. {sym}</b> – Momentum: new {HIGH_LOOKBACK}-day high",
        f"M.Cap ≈ ₹{mcap:,.0f} Cr | Top {top}% by 6-month strength | Vol {r.VOL_RATIO:.1f}x",
        f"Price ₹{entry:.2f} | ATR ₹{atr:.2f} | Buy up to ₹{entry + 0.5 * atr:.2f} (don't chase)",
        f"🛑 Stop loss ₹{stop:.2f} ({pct(stop):.1f}%) – place as SL/GTT right after buying",
        f"🚪 Exit rule: sell when a day CLOSES below the 20 EMA (now ₹{r.EMA20:.2f}). No fixed target.",
        f"📏 Reference only: +2R ₹{ref2:.2f} (+{pct(ref2):.1f}%), +4R ₹{ref4:.2f} (+{pct(ref4):.1f}%)",
        f"Qty {qty} (≈ ₹{qty * entry:,.0f} invested, risk ≈ ₹{qty * risk:,.0f})",
        f"News score {score:+d}" + (f": {html.escape(titles[0])}" if titles else ": no recent news"),
    ]


def pick_setups(hits, limit):
    lines, alerted, filtered = [], [], []
    for s, info, r in hits:
        if len(alerted) >= limit:
            break
        score, titles = news(f'"{info["name"]}" share')
        if score <= -2:
            head = html.escape(titles[0]) if titles else ""
            lines.append(f"\n⛔ {s} skipped – negative news: {head}")
            filtered.append(s)
            continue
        plan = trade_plan(len(alerted) + 1, s, r, info["mcap"], score, titles)
        if plan is None:
            filtered.append(s)
            continue
        alerted.append(s)
        lines += plan
    return lines, alerted, filtered


# ===================== POSITIONS =====================
def clean(sym):
    return str(sym).upper().replace(".NS", "").strip()


def load_positions():
    try:
        with open("positions.json") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def check_positions(data, positions):
    out = []
    for p in positions:
        s = clean(p["symbol"])
        df = data.get(s)
        if df is None or len(df) < 50:
            out.append(f"⚠️ {s}: no data")
            continue
        r = add_indicators(df.copy()).iloc[-1]
        entry, stop = float(p["entry"]), float(p["stop"])
        risk = entry - stop if entry > stop else entry * 0.05
        R = (r.Close - entry) / risk
        held = None
        d = pd.to_datetime(p.get("date"), errors="coerce") if p.get("date") else None
        if d is not None and not pd.isna(d):
            held = int((df.index > d).sum())
        if r.Low <= stop:
            action = "🔴 STOP LOSS hit – exit"
        elif r.Close < r.EMA20:
            action = f"🔴 EXIT – closed below 20 EMA (₹{r.EMA20:.2f}). Sell tomorrow morning."
        elif held is not None and held >= MAX_HOLD_DAYS:
            action = f"⏰ Held {held} days – time exit, sell"
        else:
            action = f"✅ Hold – exit if a day closes below 20 EMA (now ₹{r.EMA20:.2f})"
        out.append(f"<b>{s}</b> ₹{r.Close:.2f} ({R:+.1f}R) – {action}")
    return out


def live_position_alerts(data, positions, today, alerts, now):
    out = []
    late = minutes_since_open(now) >= 345     # after 3:00 PM
    for p in positions:
        s = clean(p["symbol"])
        df = data.get(s)
        if df is None or len(df) < 50 or df.index[-1] != today:
            continue
        r = add_indicators(df.copy()).iloc[-1]
        stop = float(p["stop"])
        if r.Low <= stop and f"{s}:SL" not in alerts["sent"]:
            out.append(f"🔴 <b>{s}</b> touched stop loss ₹{stop:.2f} (now ₹{r.Close:.2f})")
            alerts["sent"].append(f"{s}:SL")
        if late and r.Close < r.EMA20 and f"{s}:EMA" not in alerts["sent"]:
            out.append(f"⚠️ <b>{s}</b> is trading below its 20 EMA (₹{r.EMA20:.2f}), now ₹{r.Close:.2f}. "
                       f"If it closes there, the exit rule says sell.")
            alerts["sent"].append(f"{s}:EMA")
    return out


# ===================== TELEGRAM =====================
def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(text)
        return
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 3900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    for chunk in chunks:
        r = requests.post(f"https://api.telegram.org/bot{token.strip()}/sendMessage",
                          data={"chat_id": chat.strip(), "text": chunk, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        if not r.ok:
            print("Telegram error:", r.text)


def notify_error(msg, manual):
    print("ERROR:", msg)
    if MODE == "eod" or manual:
        send_telegram("❌ <b>Swing scanner problem</b>\n" + html.escape(msg))


# ===================== LIVE MODE =====================
def run_live(universe, data, nifty, today, can_buy, ranks, alerts, positions, now):
    frac = min(max(minutes_since_open(now) / 375, 0.1), 1.0)
    done = {k.split(":")[0] for k in alerts["sent"] if k.endswith((":BUY", ":X"))}
    setup_lines = []
    if can_buy:
        hits = [h for h in find_hits(universe, data, nifty["Close"], today, ranks, frac)
                if h[0] not in done]
        setup_lines, alerted, filtered = pick_setups(hits, MAX_ALERTS)
        alerts["sent"] += [f"{s}:BUY" for s in alerted] + [f"{s}:X" for s in filtered]
    pos_lines = live_position_alerts(data, positions, today, alerts, now)
    if not setup_lines and not pos_lines:
        print("Nothing new this run.")
        return
    lines = [f"⚡ <b>LIVE – {now:%d %b, %I:%M %p} IST</b>",
             f"Nifty {nifty['Close'].iloc[-1]:,.0f} | You hold {len(positions)}/{MAX_POSITIONS}"]
    if setup_lines:
        lines.append("\n🟢 <b>NEW MOMENTUM SIGNALS (today's candle not closed yet)</b>")
        lines += setup_lines
        lines.append("\n<i>Live signals can fail by 3:30 PM. Safer: buy after 3 PM if still "
                     "valid, or wait for the EOD report.</i>")
    if pos_lines:
        lines.append("\n💼 <b>YOUR POSITIONS</b>")
        lines += pos_lines
    send_telegram("\n".join(lines))


# ===================== EOD MODE =====================
def run_eod(universe, data, nifty, today, can_buy, market_ok, n, vix_now, ranks,
            alerts, positions, pending, period, turned):
    live_alerted = {k.split(":")[0] for k in alerts["sent"] if k.endswith(":BUY")}
    hits = find_hits(universe, data, nifty["Close"], today, ranks)

    status = "✅ OK to buy" if market_ok else "⛔ No new buys"
    lines = [f"📊 <b>EOD Momentum Scan – {today:%d %b %Y}</b>",
             f"Market: <b>{status}</b> | Nifty {n.Close:,.0f} vs 200SMA {n.SMA200:,.0f} | VIX {vix_now:.1f}",
             f"Scanned {len(universe)} stocks (M.Cap ≥ ₹{MIN_MCAP_CR:,} Cr) | {len(hits)} momentum signals",
             f"💼 You hold {len(positions)}/{MAX_POSITIONS} positions"]
    if turned:
        lines.append(turned)
    if pending:
        lines.append(f"⏳ Price history still loading for {pending} stocks.")
    age = amfi_age_days(period)
    if age is not None and age > 195:
        lines.append(f"📅 Your AMFI file is for the period ending {period}. "
                     f"A newer list should be out – please replace {AMFI_FILE}.")

    _, mkt_titles = news("Sensex Nifty stock market", days=2, n=3)
    if mkt_titles:
        lines.append("\n📰 <b>Market headlines</b>")
        lines += [f"• {html.escape(t)}" for t in mkt_titles]

    if live_alerted:
        hit_set = {s for s, _, _ in hits}
        held = sorted(live_alerted & hit_set)
        failed = sorted(live_alerted - hit_set)
        lines.append("\n⚡ <b>Today's live alerts at close</b>")
        if held:
            lines.append("✅ Still valid: " + ", ".join(held))
        if failed:
            lines.append("❌ Failed by close (avoid; if bought, watch your stop): " + ", ".join(failed))

    lines.append("\n🟢 <b>BUY SIGNALS</b>")
    if not can_buy:
        lines.append("Nifty is below its 200-day average. The strategy doesn't buy in this market "
                     "(in the backtest these trades lost money). Your positions are still watched.")
    elif not hits:
        lines.append("No momentum signals today.")
    else:
        if len(positions) >= MAX_POSITIONS:
            lines.append("Your 5 slots are full – only buy if you exit something first.")
        setup_lines, alerted, _ = pick_setups(hits, MAX_ALERTS)
        lines += setup_lines
        if not alerted:
            lines.append("No signals left after filters.")

    lines.append("\n💼 <b>OPEN POSITIONS</b>")
    lines += check_positions(data, positions) or ["None listed in positions.json"]
    lines.append("\n<i>Check the chart and results date before buying. Not investment advice.</i>")
    send_telegram("\n".join(lines))


# ===================== MAIN =====================
def run(cache, now, manual):
    try:
        universe, period = get_universe(cache)
    except FileNotFoundError:
        notify_error(f"{AMFI_FILE} was not found in your repository. Please upload it.", manual)
        return
    except Exception as e:
        notify_error(f"Could not read {AMFI_FILE}: {e}", manual)
        return
    if not universe:
        notify_error(f"No companies above ₹{MIN_MCAP_CR} Cr were found in {AMFI_FILE}.", manual)
        return

    try:
        api = angel_login()
    except Exception as e:
        notify_error(f"Angel One login failed: {e}. Check the four ANGEL_ secrets.", manual)
        return
    tok = get_tokens(cache)
    if not tok.get("map"):
        notify_error("Could not download Angel One's instrument list.", manual)
        return

    positions = load_positions()
    symbols = [s for s in dict.fromkeys(list(universe) + [clean(p["symbol"]) for p in positions])
               if s in tok["map"]]
    print(f"{len(universe)} companies from AMFI, {len(symbols)} found on NSE")

    quotes = fetch_quotes(api, [tok["nifty"], tok["vix"]] + [tok["map"][s] for s in symbols])
    nq = quotes.get(tok["nifty"])
    days = [d for d in (quote_bar(q)[0] for q in quotes.values()) if d is not None]
    nday = quote_bar(nq)[0] or (max(set(days), key=days.count) if days else None)
    today = pd.Timestamp(now.date())
    if nday is None or nday != today:
        if not manual:
            print("Market not open today (holiday?). Exiting.")
            return
        if nday is not None:
            today = nday

    nifty_hist = fetch_candles(api, tok["nifty"])
    if nifty_hist is None:
        nifty_hist = cache["bars"].get(NIFTY)
    if nifty_hist is None:
        notify_error("Could not get Nifty data from Angel One.", manual)
        return
    nifty = with_today(nifty_hist, nq, today)
    vix_hist = fetch_candles(api, tok["vix"])
    if vix_hist is None:
        vix_hist = cache["bars"].get(VIX)
    vix = with_today(vix_hist, quotes.get(tok["vix"]), today) if vix_hist is not None else None
    prev_session = nifty.index[nifty.index < today].max()

    need = []
    for s in symbols:
        h = cache["bars"].get(s)
        if h is None or h.empty or h.index.max() < prev_session:
            need.append(s)
            continue
        _, bar = quote_bar(quotes.get(tok["map"][s]))
        last = h[h.index < today]["Close"]
        if bar and len(last) and bar["Open"] > 0:
            ratio = bar["Open"] / last.iloc[-1]
            if ratio < 0.6 or ratio > 1.6:
                need.append(s)
    if MODE == "eod":
        others = sorted((s for s in symbols if s not in need),
                        key=lambda s: cache["fetched"].get(s, ""))
        need += others[:REFRESH_PER_DAY]
    limit = MAX_FETCH_LIVE if MODE == "live" else len(need)
    print(f"Downloading history for {min(limit, len(need))} stocks...")
    fails = 0
    for s in need[:limit]:
        df = fetch_candles(api, tok["map"][s])
        time.sleep(FETCH_PAUSE)
        if df is None or df.empty:
            fails += 1
            if fails >= 10:
                print("Too many download failures in a row; will continue next run.")
                break
            continue
        fails = 0
        cache["bars"][s] = df[df.index < today]
        cache["fetched"][s] = str(now.date())
    pending = sum(1 for s in symbols if cache["bars"].get(s) is None
                  or cache["bars"][s].index.max() < prev_session)

    data = {}
    for s in symbols:
        f = with_today(cache["bars"].get(s), quotes.get(tok["map"][s]), today)
        if f is not None and len(f):
            data[s] = f
    day_complete = today < pd.Timestamp(now.date()) or minutes_since_open(now) >= 380
    if day_complete:
        for s, f in data.items():
            cache["bars"][s] = f
        cache["bars"][NIFTY] = nifty
        if vix is not None:
            cache["bars"][VIX] = vix
    else:
        cache["bars"][NIFTY] = nifty[nifty.index < today]
        if vix is not None:
            cache["bars"][VIX] = vix[vix.index < today]

    market_ok, n, vix_now = market_state(nifty, vix)
    can_buy = market_ok or MARKET_FILTER == "off"
    ranks = strength_ranks(data, [s for s in universe if s in data])
    alerts = alerts_today(cache, today)

    if MODE == "live":
        run_live(universe, data, nifty, today, can_buy, ranks, alerts, positions, now)
    else:
        turned = ""
        prev = cache["market"].get("ok")
        if prev is not None and prev != market_ok:
            turned = ("🔔 <b>Market turned OK for buying</b> (Nifty back above its 200-day average)"
                      if market_ok else
                      "🔔 <b>Market turned weak</b> (Nifty fell below its 200-day average) – no new buys")
        cache["market"]["ok"] = market_ok
        run_eod(universe, data, nifty, today, can_buy, market_ok, n, vix_now, ranks,
                alerts, positions, pending, period, turned)


def main():
    now = now_ist()
    manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if MODE == "live" and not manual:
        mins = minutes_since_open(now)
        if now.weekday() >= 5 or not (LIVE_SKIP_MIN <= mins <= 375):
            print("Outside live market window. Exiting.")
            return
    cache = load_cache()
    try:
        run(cache, now, manual)
    finally:
        save_cache(cache)


if __name__ == "__main__":
    main()

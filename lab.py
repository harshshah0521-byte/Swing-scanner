"""Strategy lab: compares many entry/exit ideas on the saved price history."""
import numpy as np
import pandas as pd
from scanner import NIFTY, VIX, load_cache, add_indicators, add_signals

COST = 0.003                       # ~0.3% round trip costs
SPLIT = pd.Timestamp("2025-01-01") # results shown separately before/after this date
EXIT_NAMES = ["fixed_2R", "trail_ema20", "chandelier_3atr"]


def entries(df):
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
    liquid = (c * df["VOL_AVG"]) > 5e7
    template = ((c > df["SMA50"]) & (df["SMA50"] > df["SMA150"]) & (df["SMA150"] > df["SMA200"]) &
                (df["SMA200"] > df["SMA200"].shift(21)) & (c >= 1.3 * df["LO252"]) &
                (c >= 0.75 * df["HI252"]) & (df["RSR"] >= 0.8))
    dip = (l.rolling(3).min() <= df["EMA20"] * 1.01) & (c > df["EMA20"]) & (c > o) & (c > h.shift(1))
    return {
        "pullback": df["PULLBACK"],
        "breakout": df["BREAKOUT"],
        "leader_breakout": template & liquid & (c > df["HH20"]) & (df["VOL_RATIO"] > 1.5) & (df["CLV"] >= 0.6),
        "leader_pullback": template & liquid & dip & (df["VOL_RATIO"] > 0.8),
        "momentum_50d": (liquid & (c > df["SMA200"]) & (c > df["HH50"]) & (df["VOL_RATIO"] > 2) &
                         (df["RSR"] >= 0.9) & (df["CLV"] >= 0.7)),
    }


def simulate(arr, sig_idx, exit_kind):
    o, h, l, c, atr, ema20 = arr
    n = len(c)
    trades, next_free = [], 0
    for i in sig_idx:
        if i < next_free or i >= n - 1:
            continue
        entry = o[i + 1]
        stop = c[i] - 2 * atr[i]
        if not np.isfinite(entry) or not np.isfinite(stop) or entry <= stop:
            continue
        risk = entry - stop
        exit_px, k_exit = None, None
        if exit_kind == "fixed_2R":
            target, last = entry + 2 * risk, min(i + 20, n - 1)
            for k in range(i + 1, last + 1):
                if l[k] <= stop:
                    exit_px, k_exit = min(o[k], stop), k
                    break
                if h[k] >= target:
                    exit_px, k_exit = max(o[k], target), k
                    break
        elif exit_kind == "trail_ema20":
            last = min(i + 60, n - 1)
            for k in range(i + 1, last + 1):
                if l[k] <= stop:
                    exit_px, k_exit = min(o[k], stop), k
                    break
                if k > i + 1 and c[k] < ema20[k]:
                    exit_px, k_exit = c[k], k
                    break
        else:   # chandelier: trailing stop 3 ATR below the highest close
            last, trail, hc = min(i + 60, n - 1), stop, entry
            for k in range(i + 1, last + 1):
                if l[k] <= trail:
                    exit_px, k_exit = min(o[k], trail), k
                    break
                hc = max(hc, c[k])
                trail = max(trail, hc - 3 * atr[k])
        if exit_px is None:
            k_exit = last
            exit_px = c[last]
        trades.append((i + 1, k_exit, (exit_px - entry - COST * entry) / risk))
        next_free = k_exit + 1
    return trades


def main():
    bars = load_cache()["bars"]
    if NIFTY not in bars:
        print("No saved price history yet. Run 'EOD Scan' first.")
        return
    nifty = add_indicators(bars[NIFTY].copy())
    mkt_ok = nifty["Close"] > nifty["SMA200"]
    stocks = {s: d for s, d in bars.items()
              if s not in (NIFTY, VIX) and d is not None and len(d) >= 300}
    closes = pd.DataFrame({s: d["Close"] for s, d in stocks.items()}).sort_index()
    rsr = (closes / closes.shift(126) - 1).rank(axis=1, pct=True)   # 6-month strength rank
    print(f"Testing {len(stocks)} stocks, {nifty.index[0]:%b %Y} to {nifty.index[-1]:%b %Y}\n")

    results = {}
    for s, d in stocks.items():
        df = add_signals(add_indicators(d.copy()), nifty["Close"])
        c = df["Close"]
        df["SMA50"] = c.rolling(50).mean()
        df["SMA150"] = c.rolling(150).mean()
        df["HI252"] = df["High"].rolling(252, min_periods=200).max()
        df["LO252"] = df["Low"].rolling(252, min_periods=200).min()
        df["HH50"] = df["High"].rolling(50).max().shift(1)
        df["RSR"] = rsr[s].reindex(df.index)
        ok = mkt_ok.reindex(df.index).ffill().fillna(False).to_numpy(dtype=bool)
        arr = tuple(df[k].to_numpy() for k in ["Open", "High", "Low", "Close", "ATR", "EMA20"])
        dates = df.index
        for ename, sig in entries(df).items():
            sig = sig.fillna(False).to_numpy(dtype=bool)
            sig[:200] = False
            for mname, mask in (("any", None), ("nifty>200", ok)):
                idx = np.flatnonzero(sig if mask is None else (sig & mask))
                if not len(idx):
                    continue
                for xname in EXIT_NAMES:
                    for ei, ki, R in simulate(arr, idx, xname):
                        results.setdefault((ename, xname, mname), []).append(
                            (dates[ei], R, ki - ei + 1))

    rows = []
    for (e, x, m), lst in results.items():
        t = pd.DataFrame(lst, columns=["date", "R", "days"])
        if len(t) < 30:
            continue
        win = t["R"] > 0
        loss = -t["R"][~win].sum()
        pf = t["R"][win].sum() / loss if loss else np.inf
        tr, te = t[t["date"] < SPLIT]["R"], t[t["date"] >= SPLIT]["R"]
        rows.append({"entry": e, "exit": x, "market": m, "trades": len(t),
                     "win%": round(win.mean() * 100, 1), "avgR": round(t["R"].mean(), 2),
                     "PF": round(pf, 2), "days": round(t["days"].mean(), 1),
                     "R_22-24": round(tr.mean(), 2) if len(tr) else np.nan,
                     "R_25-26": round(te.mean(), 2) if len(te) else np.nan})
    summ = pd.DataFrame(rows)
    summ["score"] = summ[["R_22-24", "R_25-26"]].min(axis=1)
    summ = summ.sort_values("score", ascending=False)
    pd.set_option("display.width", 250)
    print("RESULTS (sorted by 'score' = the weaker of the two periods; higher is better)\n")
    print(summ.to_string(index=False))
    summ.to_csv("lab_results.csv", index=False)

    print("\nTOP 3 BY YEAR")
    for _, r in summ.head(3).iterrows():
        t = pd.DataFrame(results[(r["entry"], r["exit"], r["market"])], columns=["date", "R", "days"])
        print(f"\n{r['entry']} + {r['exit']} + {r['market']}")
        print(t.groupby(pd.to_datetime(t["date"]).dt.year)["R"].agg(["count", "mean", "sum"]).round(2).to_string())


if __name__ == "__main__":
    main()

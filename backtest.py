"""Backtest: tests the scanner rules on the price history saved by the scanner."""
import numpy as np
import pandas as pd
from scanner import (NIFTY, VIX, ATR_STOP_MULT, T1_ATR_MULT,
                     load_cache, add_indicators, add_signals)

COST = 0.003                               # ~0.3% round trip: STT, brokerage, charges
TARGET_R = T1_ATR_MULT / ATR_STOP_MULT     # exits at Target 1
MAX_HOLD = 20                              # exit after 20 trading days if nothing hit


def regime_series(nifty):
    n = add_indicators(nifty.copy())
    labels = np.select([(n["Close"] > n["EMA50"]) & (n["Close"] > n["SMA200"]),
                        n["Close"] > n["SMA200"]], ["BULLISH", "CAUTION"], default="BEARISH")
    return pd.Series(labels, index=n.index)


def stats(r, label):
    if r.empty:
        print(f"{label:<34} no trades")
        return
    win = r["R"] > 0
    losses = -r["R"][~win].sum()
    pf = r["R"][win].sum() / losses if losses else float("inf")
    print(f"{label:<34} trades {len(r):>5} | win {win.mean():5.1%} | "
          f"avg R {r['R'].mean():+.2f} | profit factor {pf:.2f}")


def main():
    cache = load_cache()
    bars = cache["bars"]
    if NIFTY not in bars or len(bars) < 50:
        print("No saved price history yet. Run 'EOD Scan' once first, then try again.")
        return
    nifty = bars[NIFTY]
    regime = regime_series(nifty)
    trades, used = [], 0

    for s, df0 in bars.items():
        if s in (NIFTY, VIX) or df0 is None or len(df0) < 300:
            continue
        used += 1
        df = add_signals(add_indicators(df0.copy()), nifty["Close"])
        mkt = regime.reindex(df.index).ffill().fillna("BEARISH").to_numpy()
        sig, brk = df["SIGNAL"].to_numpy(), df["BREAKOUT"].to_numpy()
        o, h, l, c = (df[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
        atr = df["ATR"].to_numpy()

        i = 200
        while i < len(df) - 1:
            if not sig[i]:
                i += 1
                continue
            entry = o[i + 1]                       # buy next day's open
            stop = c[i] - ATR_STOP_MULT * atr[i]
            if entry <= stop:
                i += 1
                continue
            risk = entry - stop
            target = entry + TARGET_R * risk
            exit_px = None
            for k in range(i + 1, min(i + 1 + MAX_HOLD, len(df))):
                if l[k] <= stop:                   # stop checked first = cautious
                    exit_px = min(o[k], stop)
                    break
                if h[k] >= target:
                    exit_px = max(o[k], target)
                    break
            if exit_px is None:
                exit_px = c[k]
            trades.append({"symbol": s, "entry_date": df.index[i + 1].date(),
                           "setup": "Breakout" if brk[i] else "Pullback", "market": mkt[i],
                           "R": round((exit_px - entry - COST * entry) / risk, 2),
                           "return_%": round((exit_px / entry - 1 - COST) * 100, 2)})
            i = k + 1

    print(f"\nData: {nifty.index[0]:%b %Y} to {nifty.index[-1]:%b %Y} | {used} stocks tested")
    res = pd.DataFrame(trades)
    if res.empty:
        print("No trades")
        return
    print("\n===== OVERALL =====")
    stats(res, "All signals")
    print("\n===== BY MARKET CONDITION =====")
    stats(res[res["market"] == "BULLISH"], "Nifty above 50 EMA & 200 SMA")
    stats(res[res["market"] == "CAUTION"], "Nifty above 200 SMA only")
    stats(res[res["market"] == "BEARISH"], "Nifty below 200 SMA")
    print("\n===== BY SETUP =====")
    stats(res[res["setup"] == "Breakout"], "Base breakout")
    stats(res[res["setup"] == "Pullback"], "Pullback to 20 EMA")
    print("\n===== BY YEAR =====")
    print(res.groupby(pd.to_datetime(res["entry_date"]).dt.year)["R"]
          .agg(["count", "mean", "sum"]).round(2))
    res.to_csv("backtest_trades.csv", index=False)


if __name__ == "__main__":
    main()

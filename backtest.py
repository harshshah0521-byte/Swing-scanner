"""Backtest of the momentum strategy on the price history saved by the scanner."""
import numpy as np
import pandas as pd
from scanner import (NIFTY, VIX, ATR_STOP_MULT, MAX_HOLD_DAYS, RS_LOOKBACK,
                     load_cache, add_indicators, add_signals)

COST = 0.003   # ~0.3% round trip: STT, brokerage, charges


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
    ranks = (closes / closes.shift(RS_LOOKBACK) - 1).rank(axis=1, pct=True)

    trades = []
    for s, d in stocks.items():
        df = add_signals(add_indicators(d.copy()), nifty["Close"], ranks[s])
        ok = mkt_ok.reindex(df.index).ffill().fillna(False).to_numpy(dtype=bool)
        sig = np.array(df["SIGNAL"].fillna(False), dtype=bool) & ok
        sig[:200] = False
        o, h, l, c, atr, ema = (df[k].to_numpy() for k in ["Open", "High", "Low", "Close", "ATR", "EMA20"])
        n, next_free = len(df), 0
        for i in np.flatnonzero(sig):
            if i < next_free or i >= n - 1:
                continue
            entry = o[i + 1]
            stop = c[i] - ATR_STOP_MULT * atr[i]
            if not np.isfinite(entry) or entry <= stop:
                continue
            risk = entry - stop
            exit_px, k_exit = None, None
            last = min(i + MAX_HOLD_DAYS, n - 1)
            for k in range(i + 1, last + 1):
                if l[k] <= stop:
                    exit_px, k_exit = min(o[k], stop), k
                    break
                if k > i + 1 and c[k] < ema[k]:
                    exit_px, k_exit = c[k], k
                    break
            if exit_px is None:
                exit_px, k_exit = c[last], last
            trades.append({"symbol": s, "entry_date": df.index[i + 1].date(),
                           "exit_date": df.index[k_exit].date(), "days": k_exit - i,
                           "R": round((exit_px - entry - COST * entry) / risk, 2),
                           "return_%": round((exit_px / entry - 1 - COST) * 100, 2)})
            next_free = k_exit + 1

    res = pd.DataFrame(trades)
    if res.empty:
        print("No trades")
        return
    win = res["R"] > 0
    loss_sum = -res["R"][~win].sum()
    print(f"\nData: {nifty.index[0]:%b %Y} to {nifty.index[-1]:%b %Y} | {len(stocks)} stocks\n")
    print("===== OVERALL =====")
    print(f"Trades: {len(res)} | Win rate: {win.mean():.1%} | Avg R: {res['R'].mean():+.2f} | "
          f"Profit factor: {res['R'][win].sum() / loss_sum:.2f}")
    print(f"Avg return per trade: {res['return_%'].mean():+.2f}% | "
          f"Avg winner: {res.loc[win, 'return_%'].mean():+.2f}% | "
          f"Avg loser: {res.loc[~win, 'return_%'].mean():+.2f}% | Avg days held: {res['days'].mean():.1f}")
    print(f"Biggest winner: {res['return_%'].max():+.1f}% | Biggest loser: {res['return_%'].min():+.1f}%")
    print("\n===== BY YEAR =====")
    yr = res.groupby(pd.to_datetime(res["entry_date"]).dt.year)
    print(pd.DataFrame({"trades": yr.size(),
                        "win%": (yr["R"].apply(lambda r: (r > 0).mean() * 100)).round(1),
                        "avg R": yr["R"].mean().round(2),
                        "avg return %": yr["return_%"].mean().round(2)}).to_string())
    res.to_csv("backtest_trades.csv", index=False)


if __name__ == "__main__":
    main()

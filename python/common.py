import numpy as np
import pandas as pd

SPREAD = 0.30
MIN_RISK_ATR = 1.0
MAX_RISK_ATR = 4.0
RRS = (1.5, 2.0, 3.0)


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def context(df, bars_per_year):
    # extra indicators on a frame already processed by smc_engine.prepare
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["rsi14"] = rsi(c, 14)
    med = df["atr"].rolling(bars_per_year, min_periods=bars_per_year // 2).median()
    df["rel_vol"] = df["atr"] / med
    return df


def pivots(df, n=3):
    # confirmed pivots: known only n bars after the pivot bar (no lookahead)
    h, l = df["high"].values, df["low"].values
    size = len(df)
    hmax = df["high"].rolling(2 * n + 1, center=True).max().values
    lmin = df["low"].rolling(2 * n + 1, center=True).min().values
    ph = np.full(size, np.nan); phk = np.full(size, -1)
    pl = np.full(size, np.nan); plk = np.full(size, -1)
    for k in range(n, size - n):
        if h[k] == hmax[k]:
            ph[k + n] = h[k]; phk[k + n] = k
        if l[k] == lmin[k]:
            pl[k + n] = l[k]; plk[k + n] = k
    return ph, phk, pl, plk


class Orders:
    def __init__(self, df):
        self.df = df
        self.c = df["close"].values
        self.atr = df["atr"].values
        self.rows = []
        self.last = {}

    def add(self, i, family, setup, d, mode, limit, stop, cool=6, **extra):
        c, atr = self.c[i], self.atr[i]
        if np.isnan(atr) or atr <= 0:
            return
        key = (setup, d)
        if i - self.last.get(key, -999) < cool:
            return
        ref = limit if mode == "limit" else c
        if mode == "limit":
            if (d == 1 and not (stop < limit < c)) or (d == -1 and not (c < limit < stop)):
                return
        else:
            if (d == 1 and not stop < c) or (d == -1 and not stop > c):
                return
        risk = abs(ref - stop)
        if risk < MIN_RISK_ATR * atr:
            stop = ref - d * MIN_RISK_ATR * atr
            risk = MIN_RISK_ATR * atr
        if risk > MAX_RISK_ATR * atr:
            return
        self.last[key] = i
        self.rows.append(dict(bar=i, family=family, setup=setup, dir=d, mode=mode,
                              limit=limit if mode == "limit" else np.nan,
                              stop=stop, ref=ref, **extra))

    def frame(self):
        return pd.DataFrame(self.rows)


def add_features(df, o, tf_min):
    i = o["bar"].values.astype(int)
    d = o["dir"].values
    g = lambda k: df[k].values[i]
    atr, c = g("atr"), g("close")
    o["time"] = df.index[i] + pd.Timedelta(minutes=tf_min)
    o["atr"] = atr
    o["risk_atr"] = np.abs(o["ref"].values - o["stop"].values) / atr
    o["entry_dist"] = np.abs(c - o["ref"].values) / atr
    hr = df.index.hour.values[i]
    o["hour"] = hr
    o["dow"] = df.index.dayofweek.values[i]
    o["session"] = np.where(hr < 7, 0, np.where(hr < 12, 1, np.where(hr < 21, 2, 3)))
    tr = np.sign(g("ema50") - g("ema200"))
    o["trend"] = tr
    o["with_trend"] = (tr == d).astype(int)
    o["ema_dist"] = (c - g("ema200")) / atr * d
    o["ema_slope"] = g("ema_slope") * d
    o["pd_pos"] = (c - g("rng_lo")) / (g("rng_hi") - g("rng_lo") + 1e-9)
    o["d_pdh"] = (c - g("pdh")) / atr
    o["d_pdl"] = (c - g("pdl")) / atr
    o["rel_vol"] = g("rel_vol")
    r14 = g("rsi14")
    o["rsi_dir"] = np.where(d == 1, r14, 100 - r14)
    o["ret20"] = (c - df["close"].shift(20).values[i]) / atr * d
    o["bar_range"] = (g("high") - g("low")) / atr
    o["bar_body"] = np.abs(c - g("open")) / atr
    return o


def label(df, o, expire, hold):
    h, l, op, c = (df[k].values for k in ("high", "low", "open", "close"))
    n, m = len(df), len(o)
    bar = o["bar"].values.astype(int)
    dr = o["dir"].values.astype(int)
    mkt = (o["mode"].values == "mkt")
    lim = o["limit"].values
    stp = o["stop"].values
    filled = np.full(m, np.nan)
    Rc = {rr: np.full(m, np.nan) for rr in RRS}
    INF = 10 ** 9
    for t in range(m):
        i, d = bar[t], dr[t]
        if i + 2 + expire + hold > n:
            continue
        if mkt[t]:
            jf = i + 1
            entry = op[jf] + SPREAD if d == 1 else op[jf]
            if (d == 1 and stp[t] >= entry) or (d == -1 and stp[t] <= entry):
                filled[t] = 0
                continue
            start, sl_fill = jf, False
        else:
            lj, hj = l[i + 1:i + 1 + expire], h[i + 1:i + 1 + expire]
            fill = (lj <= lim[t] - SPREAD) if d == 1 else (hj >= lim[t])
            if not fill.any():
                filled[t] = 0
                continue
            jf = i + 1 + int(fill.argmax())
            entry = lim[t]
            sl_fill = (l[jf] <= stp[t]) if d == 1 else (h[jf] + SPREAD >= stp[t])
            start = jf + 1
        filled[t] = 1
        R = abs(entry - stp[t])
        sh_, sl_ = h[start:start + hold], l[start:start + hold]
        hit_sl = (sl_ <= stp[t]) if d == 1 else (sh_ + SPREAD >= stp[t])
        t_sl = int(hit_sl.argmax()) if hit_sl.any() else INF
        cx = c[start + hold - 1]
        mtm = (cx - entry) / R if d == 1 else (entry - cx - SPREAD) / R
        for rr in RRS:
            if sl_fill:
                r = -1.0
            else:
                tp = entry + d * rr * R
                hit_tp = (sh_ >= tp) if d == 1 else (sl_ + SPREAD <= tp)
                t_tp = int(hit_tp.argmax()) if hit_tp.any() else INF
                if t_tp < t_sl:
                    r = float(rr)
                elif t_sl < INF:
                    r = -1.0
                else:
                    r = float(np.clip(mtm, -1.0, rr))
            Rc[rr][t] = r
    o = o.copy()
    o["filled"] = filled
    for rr in RRS:
        o[f"R_{rr}"] = Rc[rr]
    return o


# END OF FILE

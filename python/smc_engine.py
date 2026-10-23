import numpy as np
import pandas as pd

SPREAD = 0.30          # gold spread in price units (data is bid)
RRS = (1.5, 2.0, 3.0)
EXPIRE = 48            # pending limit order lives 48 bars (12h on M15)
MAX_HOLD = 96          # max bars to hold after fill (24h)
SWING_N = 3
COOLDOWN = 6
MIN_RISK_ATR = 1.5     # minimum stop distance in ATR
MAX_RISK_ATR = 4.0
BAR_MIN = 15


def atr_series(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def prepare(df):
    df = df.copy()
    df["atr"] = atr_series(df)
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema_slope"] = (df["ema200"] - df["ema200"].shift(20)) / df["atr"]
    df["rng_hi"] = df["high"].rolling(100).max()
    df["rng_lo"] = df["low"].rolling(100).min()

    hour = df.index.hour
    d = df.index.normalize()
    td = (df.index + pd.Timedelta(hours=2)).normalize()  # trading day rolls at 22:00 UTC
    day = df.groupby(td).agg(high=("high", "max"), low=("low", "min"))
    df["pdh"] = day["high"].shift(1).reindex(td).values
    df["pdl"] = day["low"].shift(1).reindex(td).values

    for name, (a, b) in {"asia": (0, 7), "london": (7, 12)}.items():
        m = (hour >= a) & (hour < b)
        hi = df["high"].where(m).groupby(d).transform("max")
        lo = df["low"].where(m).groupby(d).transform("min")
        done = hour >= b
        df[f"{name}_h"] = hi.where(done)
        df[f"{name}_l"] = lo.where(done)
    return df


def _ny_window(ts):
    """Silver Bullet windows in New York time: 03-04, 10-11, 14-15."""
    ny = ts.tz_localize("UTC").tz_convert("America/New_York")
    m = np.asarray(ny.hour * 60 + ny.minute)
    w = np.zeros(len(ts), dtype=int)
    w[(m >= 180) & (m < 240)] = 1
    w[(m >= 600) & (m < 660)] = 2
    w[(m >= 840) & (m < 900)] = 3
    return w


def detect(df):
    """Creates PENDING LIMIT ORDERS at bar close (no lookahead).
    Each row = one order: limit price, stop, setup, direction, features."""
    n = len(df)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = df["atr"].values
    ema200 = df["ema200"].values
    slope = df["ema_slope"].values
    rng_hi, rng_lo = df["rng_hi"].values, df["rng_lo"].values
    hour = df.index.hour.values
    dow = df.index.dayofweek.values
    win = _ny_window(df.index)
    lv = {k: df[k].values for k in ("pdh", "pdl", "asia_h", "asia_l", "london_h", "london_l")}

    sh, sl = [], []
    last_sh = last_sl = None
    sh_broken = sl_broken = True
    trend = 0
    zones, zid = [], 0
    sweeps = []
    bos = None
    leg = None
    eq = []
    swept = set()
    last_sig = {}
    out = []

    def sw_recent(d, i0, i1):
        return any(s["dir"] == d and i0 <= s["i"] <= i1 for s in sweeps)

    def emit(i, setup, d, limit, stop, z=None):
        key = (setup, d)
        if i - last_sig.get(key, -99) < COOLDOWN:
            return
        # limit must be on the pullback side of the current close
        if (d == 1 and not (stop < limit < c[i])) or (d == -1 and not (c[i] < limit < stop)):
            return
        risk = abs(limit - stop)
        if risk < MIN_RISK_ATR * atr[i]:      # widen tight stops (spread cost)
            stop = limit - d * MIN_RISK_ATR * atr[i]
            risk = MIN_RISK_ATR * atr[i]
        if risk > MAX_RISK_ATR * atr[i]:
            return
        last_sig[key] = i
        out.append(dict(
            time=df.index[i] + pd.Timedelta(minutes=BAR_MIN), bar=i, setup=setup, dir=d,
            limit=limit, stop=stop, close=c[i], atr=atr[i],
            risk_atr=risk / atr[i], entry_dist=abs(c[i] - limit) / atr[i],
            hour=hour[i], dow=dow[i],
            session=0 if hour[i] < 7 else 1 if hour[i] < 12 else 2 if hour[i] < 21 else 3,
            trend=trend, with_trend=int(trend == d),
            ema_dist=(c[i] - ema200[i]) / atr[i] * d, ema_slope=slope[i] * d,
            pd_pos=(c[i] - rng_lo[i]) / (rng_hi[i] - rng_lo[i] + 1e-9),
            d_pdh=(c[i] - lv["pdh"][i]) / atr[i], d_pdl=(c[i] - lv["pdl"][i]) / atr[i],
            zone_size=(z["hi"] - z["lo"]) / atr[i] if z else np.nan,
            zone_disp=z["disp"] if z else np.nan,
            bars_since_sweep=min(99, i - sweeps[-1]["i"]) if sweeps else 99,
            sweep_same=int(bool(sweeps) and sweeps[-1]["dir"] == d),
            bars_since_bos=min(99, i - bos["i"]) if bos else 99,
            bos_same=int(bool(bos) and bos["dir"] == d),
            bar_range=(h[i] - l[i]) / atr[i], bar_body=abs(c[i] - o[i]) / atr[i],
        ))

    def zone_order(i, setup, z):
        d = z["dir"]
        mid = (z["lo"] + z["hi"]) / 2
        stop = z["lo"] - 0.1 * atr[i] if d == 1 else z["hi"] + 0.1 * atr[i]
        emit(i, setup, d, mid, stop, z)

    for i in range(120, n):
        if np.isnan(atr[i]) or np.isnan(ema200[i]) or np.isnan(rng_hi[i]):
            continue
        N = SWING_N

        # --- A) confirmed swings (confirmed N bars later = no lookahead)
        k = i - N
        if h[k] == h[k - N:i + 1].max():
            if sh and abs(h[k] - sh[-1][0]) <= 0.12 * atr[i] and k - sh[-1][1] <= 60:
                eq.append(dict(name="eqh", p=max(h[k], sh[-1][0]), sw=False, i=i))
            sh.append((h[k], k)); last_sh = sh[-1]; sh_broken = False
            # OTE: leg top confirmed -> pending limit at 0.705 retracement
            if leg and leg["dir"] == 1 and trend == 1 and not leg["used"] and h[k] >= leg["b"] - 1e-9:
                rng = leg["b"] - leg["a"]
                if rng >= 1.5 * atr[i]:
                    leg["used"] = True
                    emit(i, "OTE", 1, leg["b"] - 0.705 * rng, leg["a"] - 0.1 * atr[i])
        if l[k] == l[k - N:i + 1].min():
            if sl and abs(l[k] - sl[-1][0]) <= 0.12 * atr[i] and k - sl[-1][1] <= 60:
                eq.append(dict(name="eql", p=min(l[k], sl[-1][0]), sw=False, i=i))
            sl.append((l[k], k)); last_sl = sl[-1]; sl_broken = False
            if leg and leg["dir"] == -1 and trend == -1 and not leg["used"] and l[k] <= leg["b"] + 1e-9:
                rng = leg["a"] - leg["b"]
                if rng >= 1.5 * atr[i]:
                    leg["used"] = True
                    emit(i, "OTE", -1, leg["b"] + 0.705 * rng, leg["a"] + 0.1 * atr[i])
        if len(eq) > 30:
            eq = eq[-30:]
        if len(sh) > 50:
            sh = sh[-50:]
        if len(sl) > 50:
            sl = sl[-50:]

        # --- B) structure break (BOS / CHoCH) + order block
        for d in (1, -1):
            if d == 1:
                ok = last_sh is not None and not sh_broken and c[i] > last_sh[0]
            else:
                ok = last_sl is not None and not sl_broken and c[i] < last_sl[0]
            if not ok:
                continue
            kind = "CHOCH" if trend == -d else "BOS"
            trend = d
            if d == 1:
                sh_broken = True
            else:
                sl_broken = True
            ob = None
            for j in range(i - 1, max(i - 11, 0), -1):
                if (d == 1 and c[j] < o[j]) or (d == -1 and c[j] > o[j]):
                    ob = (l[j], h[j]); break
            zid += 1
            if ob:
                z = dict(id=zid, kind="OB", dir=d, lo=ob[0], hi=ob[1], born=i,
                         disp=abs(c[i] - o[i]) / atr[i])
                zones.append(z)
                sp = sw_recent(d, i - 30, i)
                zone_order(i, "SWEEP_CHOCH_OB" if (kind == "CHOCH" and sp) else "BOS_OB", z)
            bos = dict(dir=d, i=i, kind=kind)
            if d == 1:
                leg = dict(dir=1, a=last_sl[0] if last_sl else l[i], b=h[i], used=False)
            else:
                leg = dict(dir=-1, a=last_sh[0] if last_sh else h[i], b=l[i], used=False)

        # --- C) displacement FVG
        body1 = abs(c[i - 1] - o[i - 1])
        if body1 >= atr[i]:
            for d in (1, -1):
                if d == 1 and c[i - 1] > o[i - 1] and l[i] > h[i - 2]:
                    lo_, hi_ = h[i - 2], l[i]
                elif d == -1 and c[i - 1] < o[i - 1] and h[i] < l[i - 2]:
                    lo_, hi_ = h[i], l[i - 2]
                else:
                    continue
                if hi_ - lo_ < 0.1 * atr[i]:
                    continue
                zid += 1
                z = dict(id=zid, kind="FVG", dir=d, lo=lo_, hi=hi_, born=i, disp=body1 / atr[i])
                zones.append(z)
                sp = sw_recent(d, i - 15, i)
                bs = bool(bos) and bos["dir"] == d and i - bos["i"] <= 40
                if sp:
                    zone_order(i, "SWEEP_DISP_FVG", z)
                elif bs:
                    zone_order(i, "BOS_FVG", z)
                if win[i] != 0:
                    zone_order(i, "SILVER_BULLET", z)

        # --- D) liquidity sweeps (+ Judas Swing): limit at the swept level (retest)
        cands = [(k_, lv[k_][i]) for k_ in lv if not np.isnan(lv[k_][i])]
        cands += [(e["name"], e["p"]) for e in eq if not e["sw"] and i - e["i"] < 200]
        for name, p in cands:
            key = (name, round(float(p), 2))
            if key in swept:
                continue
            if h[i] > p and c[i] < p and o[i] < p:
                d, ext = -1, h[i]
            elif l[i] < p and c[i] > p and o[i] > p:
                d, ext = 1, l[i]
            else:
                continue
            swept.add(key)
            for e in eq:
                if e["name"] == name and e["p"] == p:
                    e["sw"] = True
            sweeps.append(dict(dir=d, i=i, ext=ext, name=name))
            sweeps = sweeps[-10:]
            stop = ext - 0.1 * atr[i] if d == 1 else ext + 0.1 * atr[i]
            emit(i, "SWEEP_RECLAIM", d, p, stop)
            if name.startswith("asia") and 7 <= hour[i] < 10:
                emit(i, "JUDAS_SWING", d, p, stop)
        if len(swept) > 500:
            swept = set(list(swept)[-200:])

        # --- E) zone lifecycle: broken zone with displacement flips into Breaker / IFVG
        for z in zones[:]:
            if z["born"] >= i:
                continue
            if i - z["born"] > 150:
                zones.remove(z); continue
            d = z["dir"]
            broken = c[i] < z["lo"] if d == 1 else c[i] > z["hi"]
            if broken:
                zones.remove(z)
                if abs(c[i] - o[i]) >= 0.8 * atr[i]:
                    zid += 1
                    nz = dict(id=zid, kind="BREAKER" if z["kind"] == "OB" else "IFVG", dir=-d,
                              lo=z["lo"], hi=z["hi"], born=i, disp=abs(c[i] - o[i]) / atr[i])
                    zone_order(i, nz["kind"], nz)
        if len(zones) > 40:
            zones = zones[-40:]

        # --- F) leg bookkeeping
        if leg:
            if (leg["dir"] == 1 and c[i] < leg["a"]) or (leg["dir"] == -1 and c[i] > leg["a"]):
                leg = None
            else:
                leg["b"] = max(leg["b"], h[i]) if leg["dir"] == 1 else min(leg["b"], l[i])

    return pd.DataFrame(out)


def label_signals(df, sig):
    """Simulates the pending limit order.
    Long:  fills when bid low <= limit - spread (ask touches limit); SL/TP on bid.
    Short: fills when bid high >= limit; SL when ask (high+spread) >= stop; TP when ask

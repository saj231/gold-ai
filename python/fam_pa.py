import numpy as np
from common import pivots


def run(O, df):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr, e20, e50, e200 = (df[k].values for k in ("atr", "ema20", "ema50", "ema200"))
    n = len(df)
    ph, phk, pl, plk = pivots(df, 3)
    pend = []          # pending patterns
    PH, PL = [], []    # confirmed pivots (bar, price)
    unb_h = unb_l = None
    for i in range(60, n):
        a = atr[i]
        if np.isnan(a) or np.isnan(e200[i]):
            continue
        rng, body = h[i] - l[i], abs(c[i] - o[i])
        lo5, hi5 = l[i - 5:i].min(), h[i - 5:i].max()
        # --- pin bars at a local extreme
        if rng >= 0.8 * a and body <= 0.3 * rng:
            if min(o[i], c[i]) - l[i] >= 0.6 * rng and l[i] <= lo5:
                O.add(i, "PA", "PIN_BAR", 1, "mkt", None, l[i] - 0.1 * a)
            if h[i] - max(o[i], c[i]) >= 0.6 * rng and h[i] >= hi5:
                O.add(i, "PA", "PIN_BAR", -1, "mkt", None, h[i] + 0.1 * a)
        # --- engulfing at a local extreme
        if body >= 0.8 * a:
            if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1] and l[i] <= lo5:
                O.add(i, "PA", "ENGULFING", 1, "mkt", None, l[i] - 0.1 * a)
            if c[i - 1] > o[i - 1] and c[i] < o[i] and c[i] <= o[i - 1] and o[i] >= c[i - 1] and h[i] >= hi5:
                O.add(i, "PA", "ENGULFING", -1, "mkt", None, h[i] + 0.1 * a)
        # --- inside bar breakout (mother bar >= 1 ATR)
        if h[i - 1] < h[i - 2] and l[i - 1] > l[i - 2] and (h[i - 2] - l[i - 2]) >= a:
            if c[i] > h[i - 2]:
                O.add(i, "PA", "INSIDE_BREAK", 1, "mkt", None, l[i - 2])
            elif c[i] < l[i - 2]:
                O.add(i, "PA", "INSIDE_BREAK", -1, "mkt", None, h[i - 2])
        # --- trend pullback to EMA20 with rejection candle
        if e50[i] > e200[i] and l[i] <= e20[i] < c[i] and c[i] > o[i] and body >= 0.3 * a:
            O.add(i, "PA", "EMA_PULLBACK", 1, "mkt", None, min(l[i], l[i - 1]) - 0.1 * a)
        if e50[i] < e200[i] and h[i] >= e20[i] > c[i] and c[i] < o[i] and body >= 0.3 * a:
            O.add(i, "PA", "EMA_PULLBACK", -1, "mkt", None, max(h[i], h[i - 1]) + 0.1 * a)
        # --- pivots: double top / bottom, break & retest
        if not np.isnan(ph[i]):
            if PH and abs(ph[i] - PH[-1][1]) <= 0.25 * a and 8 <= phk[i] - PH[-1][0] <= 50:
                neck = l[PH[-1][0]:phk[i] + 1].min()
                pend.append(dict(kind="DBL_TOP", d=-1, level=neck, stop=max(ph[i], PH[-1][1]) + 0.1 * a, exp=i + 30))
            PH.append((phk[i], ph[i])); unb_h = PH[-1][1]
        if not np.isnan(pl[i]):
            if PL and abs(pl[i] - PL[-1][1]) <= 0.25 * a and 8 <= plk[i] - PL[-1][0] <= 50:
                neck = h[PL[-1][0]:plk[i] + 1].max()
                pend.append(dict(kind="DBL_BOTTOM", d=1, level=neck, stop=min(pl[i], PL[-1][1]) - 0.1 * a, exp=i + 30))
            PL.append((plk[i], pl[i])); unb_l = PL[-1][1]
        if unb_h is not None and c[i] > unb_h:
            pend.append(dict(kind="RETEST", d=1, level=unb_h, born=i, exp=i + 12)); unb_h = None
        if unb_l is not None and c[i] < unb_l:
            pend.append(dict(kind="RETEST", d=-1, level=unb_l, born=i, exp=i + 12)); unb_l = None
        keep = []
        for p in pend:
            if i > p["exp"]:
                continue
            if p["kind"] == "DBL_TOP" and c[i] < p["level"]:
                O.add(i, "PA", "DOUBLE_TOP", -1, "mkt", None, p["stop"], cool=10); continue
            if p["kind"] == "DBL_BOTTOM" and c[i] > p["level"]:
                O.add(i, "PA", "DOUBLE_BOTTOM", 1, "mkt", None, p["stop"], cool=10); continue
            if p["kind"] == "RETEST" and i > p["born"]:
                L = p["level"]
                if p["d"] == 1 and l[i] <= L + 0.2 * a and c[i] > L and c[i] > o[i]:
                    O.add(i, "PA", "BREAK_RETEST", 1, "mkt", None, min(l[i], L) - 0.3 * a, cool=6); continue
                if p["d"] == -1 and h[i] >= L - 0.2 * a and c[i] < L and c[i] < o[i]:
                    O.add(i, "PA", "BREAK_RETEST", -1, "mkt", None, max(h[i], L) + 0.3 * a, cool=6); continue
            keep.append(p)
        pend = keep[-30:]


# END OF FILE

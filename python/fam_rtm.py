import numpy as np
from common import pivots


def run(O, df):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = df["atr"].values
    n = len(df)
    ph, phk, pl, plk = pivots(df, 3)
    PH, PL = [], []
    qm_done = set()
    for i in range(60, n):
        a = atr[i]
        if np.isnan(a):
            continue
        # --- RBR / DBR / RBD / DBD base patterns (zone known at leg-out close)
        out = c[i] - o[i]
        if abs(out) >= a:
            d = 1 if out > 0 else -1
            for nb in (1, 2, 3):
                bs = slice(i - nb, i)
                if (np.abs(c[bs] - o[bs]) > 0.5 * a).any() or ((h[bs] - l[bs]) > 1.2 * a).any():
                    break
                li = c[i - nb - 1] - o[i - nb - 1]       # leg-in candle
                if abs(li) < a:
                    continue
                lo_, hi_ = l[bs].min(), h[bs].max()
                if hi_ - lo_ > 1.2 * a:
                    continue
                di = 1 if li > 0 else -1
                name = ("RBR" if di == 1 else "DBR") if d == 1 else ("RBD" if di == 1 else "DBD")
                ftr = int((d == 1 and c[i] > h[i - 10:i - nb].max()) or (d == -1 and c[i] < l[i - 10:i - nb].min()))
                if d == 1 and c[i] > hi_ + 0.5 * a:
                    O.add(i, "RTM", "RTM_" + name, 1, "limit", hi_, lo_ - 0.1 * a,
                          zone_size=(hi_ - lo_) / a, zone_disp=abs(out) / a, base_n=nb, ftr=ftr)
                elif d == -1 and c[i] < lo_ - 0.5 * a:
                    O.add(i, "RTM", "RTM_" + name, -1, "limit", lo_, hi_ + 0.1 * a,
                          zone_size=(hi_ - lo_) / a, zone_disp=abs(out) / a, base_n=nb, ftr=ftr)
                break
        # --- Quasimodo (QM): higher-high then break of the last higher-low
        if not np.isnan(ph[i]):
            PH.append((phk[i], ph[i]))
        if not np.isnan(pl[i]):
            PL.append((plk[i], pl[i]))
        if len(PH) >= 2 and len(PL) >= 2:
            (k1, H1), (k2, H2) = PH[-2], PH[-1]
            (j1, L1), (j2, L2) = PL[-2], PL[-1]
            if H2 > H1 and L2 > L1 and j2 > k1 and k2 > j2 and c[i] < L2 and ("b", j2) not in qm_done:
                qm_done.add(("b", j2))
                O.add(i, "RTM", "RTM_QM", -1, "limit", H1, H2 + 0.1 * a, cool=3)
            if L2 < L1 and H2 < H1 and k2 > j1 and j2 > k2 and c[i] > H2 and ("u", k2) not in qm_done:
                qm_done.add(("u", k2))
                O.add(i, "RTM", "RTM_QM", 1, "limit", L1, L2 - 0.1 * a, cool=3)
        if len(PH) > 10:
            PH = PH[-10:]
        if len(PL) > 10:
            PL = PL[-10:]
        if len(qm_done) > 200:
            qm_done = set(list(qm_done)[-50:])


# END OF FILE

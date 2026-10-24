import numpy as np
import pandas as pd


def run(O, df):
    c, h, l = df["close"], df["high"], df["low"]
    atr = df["atr"]
    S = {}
    for N in (20, 55):
        S[f"donchian{N}"] = (c > h.shift(1).rolling(N).max(), c < l.shift(1).rolling(N).min())
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    up, lo = ma + 2 * sd, ma - 2 * sd
    pct = ((up - lo) / ma).rolling(200).rank(pct=True).shift(1)
    S["bb_squeeze_break"] = ((pct < 0.2) & (c > up), (pct < 0.2) & (c < lo))
    r = df["rsi14"]
    e200 = df["ema200"]
    S["rsi_momentum"] = ((c > e200) & (r > 60) & (r.shift(1) <= 60),
                         (c < e200) & (r < 40) & (r.shift(1) >= 40))
    pdh, pdl = df["pdh"], df["pdl"]
    S["prev_day_break"] = ((c > pdh) & (c.shift(1) <= pdh.shift(1)),
                           (c < pdl) & (c.shift(1) >= pdl.shift(1)))
    hr = pd.Series(df.index.hour, index=df.index)
    inl = (hr >= 7) & (hr < 10)
    S["asia_range_break"] = (inl & (c > df["asia_h"]), inl & (c < df["asia_l"]))
    cv, av = c.values, atr.values
    for name, (lg, sh) in S.items():
        for d, sg in ((1, lg), (-1, sh)):
            for i in np.flatnonzero(sg.fillna(False).values):
                if i < 250 or np.isnan(av[i]):
                    continue
                O.add(i, "TREND", name, d, "mkt", None, cv[i] - d * 1.5 * av[i], cool=12)


# END OF FILE

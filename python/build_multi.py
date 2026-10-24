import sys
import numpy as np
import pandas as pd
import smc_engine as se
import common as cm
import fam_trend, fam_pa, fam_rtm

TFS = {"H1": ("1h", 60, 24, 48), "H4": ("4h", 240, 12, 30)}
ICT_EXTRA = ["zone_size", "zone_disp", "bars_since_sweep", "sweep_same", "bars_since_bos", "bos_same"]


def build(m15, tf):
    rule, mins, expire, hold = TFS[tf]
    df = m15.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    df = se.prepare(df)
    df = cm.context(df, int(365 * 24 * 60 / mins * 0.7))
    parts = []
    # ICT / SMC family (limit orders)
    se.BAR_MIN, se.MIN_RISK_ATR, se.MAX_RISK_ATR = mins, cm.MIN_RISK_ATR, cm.MAX_RISK_ATR
    ict = se.detect(df)
    if len(ict):
        x = pd.DataFrame(dict(bar=ict["bar"], family="ICT", setup=ict["setup"], dir=ict["dir"],
                              mode="limit", limit=ict["limit"], stop=ict["stop"], ref=ict["limit"]))
        for k in ICT_EXTRA:
            x[k] = ict[k].values
        parts.append(x)
    for name, fam in (("TREND", fam_trend), ("PA", fam_pa), ("RTM", fam_rtm)):
        O = cm.Orders(df)
        fam.run(O, df)
        f = O.frame()
        print(f"  {tf} {name}: {len(f)} orders")
        parts.append(f)
    o = pd.concat(parts, ignore_index=True)
    o = cm.add_features(df, o, mins)
    o = cm.label(df, o, expire, hold)
    o = o.dropna(subset=["filled"]).reset_index(drop=True)
    o["tf"] = tf
    return o


if __name__ == "__main__":
    src, prefix = sys.argv[1], sys.argv[2]
    m15 = pd.read_parquet(src)
    for tf in TFS:
        o = build(m15, tf)
        o.to_parquet(f"{prefix}_{tf}.parquet", compression="zstd")
        f = o[o.filled == 1]
        print(f"{tf}: orders {len(o)} filled {len(f)}")
        g = f.groupby(["family", "setup"])
        rep = pd.DataFrame({"n": g.size(), "R1.5": g["R_1.5"].mean(),
                            "R2": g["R_2.0"].mean(), "R3": g["R_3.0"].mean()})
        print(rep.round(3).sort_values("R2", ascending=False).to_string())
        print("ALL:", {rr: round(f[f"R_{rr}"].mean(), 3) for rr in cm.RRS})


# END OF FILE

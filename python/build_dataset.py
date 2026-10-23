import sys
import pandas as pd
from smc_engine import prepare, detect, label_signals, RRS

src, out = sys.argv[1], sys.argv[2]          # m15.parquet  dataset.parquet
m15 = pd.read_parquet(src)
df = prepare(m15)
sig = detect(df)
sig = label_signals(df, sig)
sig = sig.dropna(subset=["filled"]).reset_index(drop=True)
sig.to_parquet(out, compression="zstd")

print("orders:", len(sig), "| range:", sig["time"].min(), "->", sig["time"].max())
g = sig.groupby("setup")
rep = pd.DataFrame({"orders": g.size(), "fill_rate": g["filled"].mean()})
f = sig[sig["filled"] == 1]
gf = f.groupby("setup")
rep["filled"] = gf.size()
for rr in RRS:
    rep[f"win{rr}"] = gf[f"y_{rr}"].mean()
for rr in RRS:
    rep[f"R{rr}"] = gf[f"R_{rr}"].mean()
print(rep.round(3).sort_values(f"R{RRS[1]}", ascending=False).to_string())
print()
print("ALL filled:", len(f), {f"R{rr}": round(f[f"R_{rr}"].mean(), 3) for rr in RRS})

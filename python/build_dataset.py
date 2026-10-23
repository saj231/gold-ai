import sys
import pandas as pd
from smc_engine import prepare, detect, label_signals, RRS

src, out = sys.argv[1], sys.argv[2]          # m15.parquet  dataset.parquet
m15 = pd.read_parquet(src)
df = prepare(m15)
sig = detect(df)
sig = label_signals(df, sig)
sig = sig.dropna(subset=[f"y_{RRS[0]}"]).reset_index(drop=True)
sig.to_parquet(out, compression="zstd")

print("signals:", len(sig), "| range:", sig["time"].min(), "->", sig["time"].max())
summary = sig.groupby("setup").agg(
    n=("bar", "count"),
    win_1_5=("y_1.5", "mean"),
    win_2=("y_2.0", "mean"),
    win_3=("y_3.0", "mean"),
)
# expectancy in R (ignoring timeouts nuance): win*RR - (1-win)
summary["exp_R_2"] = summary["win_2"] * 2 - (1 - summary["win_2"])
print(summary.round(3).sort_values("exp_R_2", ascending=False))

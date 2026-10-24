import sys, os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from train_multi import NUM, TOP_Q, RR, TRAIN_END, VALID_END

src, outdir = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
df = df[df["filled"] == 1].copy()
df["time"] = pd.to_datetime(df["time"])
fams, sets = sorted(df["family"].unique()), sorted(df["setup"].unique())
df["fam_code"] = df["family"].map({k: i for i, k in enumerate(fams)})
df["setup_code"] = df["setup"].map({k: i for i, k in enumerate(sets)})
for c in NUM:
    if c not in df:
        df[c] = np.nan
F = NUM + ["fam_code", "setup_code"]
y, R = (df[f"R_{RR}"] > 0).astype(int), f"R_{RR}"
tr = df[df.time < TRAIN_END]
va = df[(df.time >= TRAIN_END) & (df.time < VALID_END)]
te = df[df.time >= VALID_END]
p = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=10.0, verbose=-1, seed=7)
m = lgb.train(p, lgb.Dataset(tr[F], y.loc[tr.index], categorical_feature=["fam_code", "setup_code"]), 250)
thr = float(np.quantile(m.predict(va[F]), 1 - TOP_Q))
sel = te[m.predict(te[F]) >= thr]
print(f"threshold={thr:.4f} | test all {te[R].mean():+.3f} | test selected {sel[R].mean():+.3f} (n={len(sel)})")
os.makedirs(outdir, exist_ok=True)
m.save_model(os.path.join(outdir, "h1_model.txt"))
json.dump(dict(version=1, threshold=thr, features=F, families=fams, setups=sets,
               rr=RR, trained_until=TRAIN_END), open(os.path.join(outdir, "meta.json"), "w"), indent=1)
print("saved model to", outdir)


# END OF FILE

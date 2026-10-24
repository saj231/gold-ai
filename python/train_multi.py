import sys, json
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

NUM = ["dir", "risk_atr", "entry_dist", "hour", "dow", "session", "trend", "with_trend",
       "ema_dist", "ema_slope", "pd_pos", "d_pdh", "d_pdl", "rel_vol", "rsi_dir", "ret20",
       "bar_range", "bar_body", "zone_size", "zone_disp", "bars_since_sweep", "sweep_same",
       "bars_since_bos", "bos_same", "base_n", "ftr"]
TOP_Q = 0.20            # fixed in advance: trade only the top 20% by predicted probability
RR = 2.0                # fixed in advance
TRAIN_END, VALID_END = "2024-01-01", "2025-01-01"


def fit(X, y):
    if HAVE_LGB:
        p = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=10.0, verbose=-1, seed=7)
        m = lgb.train(p, lgb.Dataset(X, y, categorical_feature=["fam_code", "setup_code"]), 250)
        return lambda Z: m.predict(Z)
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150,
                                       min_samples_leaf=100, l2_regularization=10.0,
                                       random_state=7)
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def boot_ci(sel_r, all_r, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    if len(sel_r) < 30:
        return (np.nan, np.nan)
    d = [rng.choice(sel_r, len(sel_r)).mean() - rng.choice(all_r, len(all_r)).mean() for _ in range(B)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def run_tf(path, tag):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)
    df = df[df["filled"] == 1].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["fam_code"] = df["family"].astype("category").cat.codes
    df["setup_code"] = df["setup"].astype("category").cat.codes
    F = NUM + ["fam_code", "setup_code"]
    for c in NUM:
        if c not in df:
            df[c] = np.nan
    y, R = (df[f"R_{RR}"] > 0).astype(int), f"R_{RR}"
    tr = df[df.time < TRAIN_END]; va = df[(df.time >= TRAIN_END) & (df.time < VALID_END)]
    te = df[df.time >= VALID_END]
    print(f"\n######## {tag}: train {len(tr)} | valid {len(va)} | test {len(te)}")
    pred = fit(tr[F], y.loc[tr.index])
    thr = float(np.quantile(pred(va[F]), 1 - TOP_Q))
    pt = pred(te[F]); sel = pt >= thr
    print(f"TEST 2025-26 (RR {RR}, top {int(TOP_Q*100)}% by model):")
    print(f"  all orders        : n={len(te):5d} meanR={te[R].mean():+.3f}")
    s = te[sel]
    lo, hi = boot_ci(s[R].values, te[R].values)
    print(f"  model selected    : n={len(s):5d} meanR={s[R].mean():+.3f}  lift={s[R].mean()-te[R].mean():+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
    for side, nm in ((1, "long "), (-1, "short")):
        a, b = te[te.dir == side], s[s.dir == side]
        if len(b) >= 20:
            print(f"    {nm}: all {a[R].mean():+.3f} (n={len(a)}) | selected {b[R].mean():+.3f} (n={len(b)})")
    g = s.groupby(["family", "setup"])[R].agg(["count", "mean"]).round(3)
    print(g[g["count"] >= 15].sort_values("mean", ascending=False).to_string())

    wf = []
    for yr in range(2016, 2027):
        a = df[df.time < f"{yr-1}-01-01"]
        v = df[(df.time >= f"{yr-1}-01-01") & (df.time < f"{yr}-01-01")]
        t = df[(df.time >= f"{yr}-01-01") & (df.time < f"{yr+1}-01-01")]
        if len(t) < 100 or len(v) < 100:
            continue
        pr = fit(a[F], y.loc[a.index])
        th = float(np.quantile(pr(v[F]), 1 - TOP_Q))
        st = t[pr(t[F]) >= th]
        if len(st) < 20:
            continue
        wf.append(dict(year=yr, n=len(st), sel=round(float(st[R].mean()), 3),
                       base=round(float(t[R].mean()), 3),
                       lift=round(float(st[R].mean() - t[R].mean()), 3)))
    w = pd.DataFrame(wf)
    print("\nwalk-forward (top 20% each year):")
    print(w.to_string(index=False))
    print(f"years selected>0: {(w.sel > 0).sum()}/{len(w)} | years lift>0: {(w.lift > 0).sum()}/{len(w)} "
          f"| mean selected {w.sel.mean():+.3f} | mean lift {w.lift.mean():+.3f}")
    return dict(tag=tag, test_all=float(te[R].mean()), test_sel=float(s[R].mean()),
                lift_ci=[lo, hi], wf=wf)


if __name__ == "__main__":
    out = []
    for tf in sys.argv[2:]:
        out.append(run_tf(f"{sys.argv[1]}_{tf}.parquet", tf))
    json.dump(out, open(f"{sys.argv[1]}_report.json", "w"), indent=1, default=str)


# END OF FILE

import sys, json
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

SETUPS = ["BOS_FVG", "BOS_OB", "BREAKER", "IFVG", "JUDAS_SWING", "OTE",
          "SILVER_BULLET", "SWEEP_CHOCH_OB", "SWEEP_DISP_FVG", "SWEEP_RECLAIM"]
NUM = ["dir", "risk_atr", "entry_dist", "hour", "dow", "session", "trend", "with_trend",
       "ema_dist", "ema_slope", "pd_pos", "d_pdh", "d_pdl", "zone_size", "zone_disp",
       "bars_since_sweep", "sweep_same", "bars_since_bos", "bos_same", "bar_range", "bar_body"]
FEATURES = NUM + ["setup_code"]
RRS = [1.5, 2.0, 3.0]
TRAIN_END, VALID_END = "2024-01-01", "2025-01-01"
MIN_TRADES = 150


def fit(X, y, Xv=None, yv=None):
    if HAVE_LGB:
        p = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=10.0, verbose=-1, seed=7)
        dtr = lgb.Dataset(X, y, categorical_feature=["setup_code"])
        if Xv is not None:
            dv = lgb.Dataset(Xv, yv, reference=dtr)
            m = lgb.train(p, dtr, 600, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m = lgb.train(p, dtr, 300)
        return m, (lambda Z: m.predict(Z))
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                       min_samples_leaf=200, l2_regularization=10.0,
                                       random_state=7)
    m.fit(X, y)
    return m, (lambda Z: m.predict_proba(Z)[:, 1])


def best_threshold(p, r):
    best = (None, -9)
    for thr in np.arange(0.25, 0.75, 0.01):
        sel = p >= thr
        if sel.sum() >= MIN_TRADES and r[sel].mean() > best[1]:
            best = (float(thr), float(r[sel].mean()))
    return best


def stat(r, sel):
    return dict(n=int(sel.sum()), meanR=round(float(r[sel].mean()), 3) if sel.sum() else None,
                win=round(float((r[sel] > 0).mean()), 3) if sel.sum() else None)


def main(src, out_prefix):
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_pickle(src)
    df = df[df["filled"] == 1].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["setup_code"] = df["setup"].map({s: i for i, s in enumerate(SETUPS)}).astype(int)
    tr = df[df.time < TRAIN_END]
    va = df[(df.time >= TRAIN_END) & (df.time < VALID_END)]
    te = df[df.time >= VALID_END]
    print(f"filled orders: train {len(tr)} | valid(2024) {len(va)} | test(2025-26) {len(te)}")
    report = {"split": dict(train=len(tr), valid=len(va), test=len(te)), "rr": {}}
    print(f"\nBASELINE (all orders, no model) on TEST: " +
          ", ".join(f"R{rr}={te[f'R_{rr}'].mean():.3f}" for rr in RRS))

    for rr in RRS:
        y, R = f"y_{rr}", f"R_{rr}"
        m, pred = fit(tr[FEATURES], tr[y], va[FEATURES], va[y])
        pv, pt = pred(va[FEATURES]), pred(te[FEATURES])
        thr, vR = best_threshold(pv, va[R].values)
        print(f"\n=== RR {rr} ===  threshold(valid)={thr}  valid meanR={vR if thr else None}")
        if thr is None:
            report["rr"][str(rr)] = {"threshold": None}
            print("no threshold with enough trades on validation")
            continue
        sel_t = pt >= thr
        res = stat(te[R].values, sel_t)
        print("TEST (untouched 2025-2026):", res, "| baseline all:", round(te[R].mean(), 3))
        by = te[sel_t].groupby("setup")[R].agg(["count", "mean"]).round(3)
        print(by.to_string())

        # walk-forward: expanding window, threshold from the previous year
        wf = []
        for yr in range(2016, 2027):
            a = df[df.time < f"{yr-1}-01-01"]
            v = df[(df.time >= f"{yr-1}-01-01") & (df.time < f"{yr}-01-01")]
            t = df[(df.time >= f"{yr}-01-01") & (df.time < f"{yr+1}-01-01")]
            if len(t) < 200:
                continue
            mm, pr = fit(a[FEATURES], a[y])
            th, _ = best_threshold(pr(v[FEATURES]), v[R].values)
            if th is None:
                continue
            st = stat(t[R].values, pr(t[FEATURES]) >= th)
            wf.append(dict(year=yr, thr=round(th, 2), **st, base=round(float(t[R].mean()), 3)))
        wfd = pd.DataFrame(wf)
        print("\nwalk-forward by year:")
        print(wfd.to_string(index=False))
        pos = int((wfd["meanR"].fillna(-9) > 0).sum()) if len(wfd) else 0
        print(f"profitable years: {pos}/{len(wfd)}")
        report["rr"][str(rr)] = dict(threshold=thr, valid_meanR=vR, test=res,
                                     walk_forward=wf)

        if HAVE_LGB:
            final, _ = fit(df[FEATURES], df[y])           # final model: all data
            json.dump(dict(rr=rr, threshold=thr, features=FEATURES, setups=SETUPS,
                           model=final.dump_model()),
                      open(f"{out_prefix}_rr{rr}.json", "w"), default=str)
    json.dump(report, open(f"{out_prefix}_report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


# END OF FILE

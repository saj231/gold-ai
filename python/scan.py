import os, sys, json, urllib.request, urllib.parse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import smc_engine as se
import common as cm
import fam_trend, fam_pa, fam_rtm
import tracker
from build_multi import ICT_EXTRA

MINS = 60
SYMBOL = "XAU/USD"
MD = os.environ.get("MODEL_DIR", "../model")
FAM_FA = {"ICT": "ICT/SMC", "PA": "پرایس اکشن", "RTM": "RTM", "TREND": "ترند/مومنتوم"}


def http_json(url, data=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    h = {"User-Agent": "gold-scanner"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=40) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt else None


def fetch_bars(key, n=5000):
    q = urllib.parse.urlencode(dict(symbol=SYMBOL, interval="1h", outputsize=n,
                                    timezone="UTC", order="ASC", apikey=key))
    js = http_json("https://api.twelvedata.com/time_series?" + q)
    if not js or "values" not in js:
        raise RuntimeError(f"TwelveData error: {js}")
    df = pd.DataFrame(js["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close"]]


def compute_orders(raw):
    df = se.prepare(raw)
    df = cm.context(df, int(365 * 24 * 60 / MINS * 0.7))
    se.BAR_MIN, se.MIN_RISK_ATR, se.MAX_RISK_ATR = MINS, cm.MIN_RISK_ATR, cm.MAX_RISK_ATR
    parts = []
    ict = se.detect(df)
    if len(ict):
        x = pd.DataFrame(dict(bar=ict["bar"], family="ICT", setup=ict["setup"], dir=ict["dir"],
                              mode="limit", limit=ict["limit"], stop=ict["stop"], ref=ict["limit"]))
        for k in ICT_EXTRA:
            x[k] = ict[k].values
        parts.append(x)
    for fam in (fam_trend, fam_pa, fam_rtm):
        O = cm.Orders(df)
        fam.run(O, df)
        parts.append(O.frame())
    parts = [p for p in parts if len(p)]
    if not parts:
        return df, pd.DataFrame()
    o = pd.concat(parts, ignore_index=True)
    return df, cm.add_features(df, o, MINS)


def make_X(o, meta):
    X = pd.DataFrame(index=o.index)
    fm = {k: i for i, k in enumerate(meta["families"])}
    sm = {k: i for i, k in enumerate(meta["setups"])}
    for c in meta["features"]:
        if c == "fam_code":
            X[c] = o["family"].map(fm).fillna(-1)
        elif c == "setup_code":
            X[c] = o["setup"].map(sm).fillna(-1)
        else:
            X[c] = o[c] if c in o else np.nan
    return X.astype(float)


def message(r, thr, tag=""):
    d = int(r["dir"])
    ref, stop = float(r["ref"]), float(r["stop"])
    R = abs(ref - stop)
    tp = lambda k: ref + d * k * R
    side = "🟢 خرید (BUY)" if d == 1 else "🔴 فروش (SELL)"
    if r["mode"] == "limit":
        entry = f"لیمیت روی {ref:.2f} (تا ۲۴ ساعت معتبر)"
    else:
        entry = f"بازار، نزدیک {ref:.2f} (باز شدن کندل بعد)"
    t = pd.Timestamp(r["time"]).strftime("%Y-%m-%d %H:%M")
    return (f"{tag}🔔 ستاپ جدید | XAUUSD H1\n{side}\n"
            f"سبک: {FAM_FA.get(r['family'], r['family'])} | {r['setup']}\n"
            f"ورود: {entry}\nاستاپ: {stop:.2f} (ریسک {R:.2f}$)\n"
            f"تارگت: 1.5R {tp(1.5):.2f} | 2R {tp(2):.2f} | 3R {tp(3):.2f}\n"
            f"امتیاز مدل: {r['score']:.2f} (آستانه {thr:.2f})\n"
            f"کندل بسته‌شده تا: {t} UTC\n"
            f"⚠️ هشدار تحلیلی برای بررسی دستی، نه سیگنال تضمین‌شده. حتماً مدیریت ریسک.")


def telegram(token, chat, text):
    http_json(f"https://api.telegram.org/bot{token}/sendMessage",
              data=dict(chat_id=chat, text=text), headers={"Content-Type": "application/json"})


def supabase_new(url, key, rows):
    # insert; return only rows that were actually new (duplicates ignored)
    hdr = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json",
           "Prefer": "resolution=ignore-duplicates,return=representation"}
    q = "/rest/v1/signals?on_conflict=symbol,bar_time,setup,direction"
    res = http_json(url.rstrip("/") + q, data=rows, headers=hdr)
    return res or []


def select(o, meta, booster, last_bars):
    X = make_X(o, meta)
    o = o.copy()
    o["score"] = booster.predict(X[meta["features"]])
    o = o[o["bar"] >= last_bars]
    return o.sort_values("score", ascending=False)


def group_picks(picks):
    # several setups on the same bar/direction share entry and stop: send one message
    out = []
    for _, g in picks.groupby(["time", "dir"], sort=False):
        top = g.iloc[0].copy()
        top["setup"] = " + ".join(g["setup"].tolist()[:3])
        out.append(top)
    return sorted(out, key=lambda r: -r["score"])


def main():
    import lightgbm as lgb
    mode = os.environ.get("MODE", "scan")
    tok, chat = os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    su, sk = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    meta = json.load(open(os.path.join(MD, "meta.json")))
    booster = lgb.Booster(model_file=os.path.join(MD, "h1_model.txt"))
    thr = meta["threshold"]
    now = datetime.utcnow()
    raw = fetch_bars(os.environ["TWELVEDATA_KEY"])
    closed = raw[raw.index + pd.Timedelta(minutes=MINS) <= pd.Timestamp(now)]
    last_close = closed.index[-1] + pd.Timedelta(minutes=MINS)
    age_h = (pd.Timestamp(now) - last_close).total_seconds() / 3600
    print("last closed bar ends", last_close, f"age {age_h:.1f}h, bars {len(closed)}")
    if mode == "test":
        telegram(tok, chat, "✅ اسکنر طلا به تلگرام وصل است. چند نمونه آزمایشی از ۳ روز اخیر:")
    elif age_h > 2.5:
        print("market closed / stale data, skipping")
        return
    df, o = compute_orders(closed)
    n = len(df)
    look = 72 if mode == "test" else (3 if su else 1)
    if len(o) == 0:
        print("no orders"); return
    s = select(o, meta, booster, n - look)
    picks = s if mode == "test" else s[s["score"] >= thr]
    print(f"orders in window {len(s)}, above threshold {(s['score'] >= thr).sum()}")
    if mode != "test" and su and len(picks):
        rows = [dict(bar_time=pd.Timestamp(r.time).isoformat() + "Z", symbol="XAUUSD",
                     setup=r.setup, direction="BUY" if r.dir == 1 else "SELL",
                     entry=float(r.ref), stop=float(r.stop),
                     tp1=float(r.ref + r.dir * 1.5 * abs(r.ref - r.stop)),
                     tp2=float(r.ref + r.dir * 2.0 * abs(r.ref - r.stop)),
                     tp3=float(r.ref + r.dir * 3.0 * abs(r.ref - r.stop)),
                     probability=float(r.score), model_version=meta.get("version", 1),
                     mode=r.mode, family=r.family)
                for r in picks.itertuples()]
        try:
            new = supabase_new(su, sk, rows)
            keys = {(x["bar_time"][:16], x["setup"], x["direction"]) for x in new}
            picks = picks[[(pd.Timestamp(r.time).strftime("%Y-%m-%dT%H:%M"), r.setup,
                            "BUY" if r.dir == 1 else "SELL") in keys for r in picks.itertuples()]]
        except Exception as e:
            print("supabase error:", e)
            picks = picks[picks["bar"] >= n - 1]
    groups = group_picks(picks) if len(picks) else []
    for r in groups[:3]:
        telegram(tok, chat, message(r, thr, "🧪 نمونه آزمایشی\n" if mode == "test" else ""))
    if len(groups) > 3:
        telegram(tok, chat, f"ℹ️ {len(groups) - 3} ستاپ دیگر هم با امتیاز کمتر پیدا شد.")
    if mode != "test" and su:
        try:
            for r in tracker.update(su, sk, closed)[:5]:
                telegram(tok, chat, tracker.close_message(r))
            if now.weekday() == 4 and now.hour == 20:
                telegram(tok, chat, tracker.weekly_report(su, sk))
        except Exception as e:
            print("tracker error:", e)
    if mode != "test" and now.hour == 6 and now.weekday() < 5:
        telegram(tok, chat, "✅ اسکنر طلا فعال است.")


if __name__ == "__main__":
    main()


# END OF FILE

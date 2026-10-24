import json, urllib.request, urllib.parse
import numpy as np
import pandas as pd

SPREAD = 0.30
EXPIRE = 24      # pending limit lives 24 H1 bars
HOLD = 48        # max hold after fill (H1 bars)
RRS = (1.5, 2.0, 3.0)
HOUR = pd.Timedelta(hours=1)
INF = 10 ** 9


def resolve(bars, bar_time, d, mode, ref, stop):
    # replays one signal over closed H1 bars; same rules as the backtest labeler
    idx = bars.index
    p0 = int(idx.searchsorted(pd.Timestamp(bar_time)))
    n = len(bars)
    if p0 >= n:
        return dict(status="open")
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    if mode == "mkt":
        jf = p0
        entry = o[jf] + SPREAD if d == 1 else o[jf]
        if (d == 1 and stop >= entry) or (d == -1 and stop <= entry):
            return dict(status="expired")
        start, sl_fill = jf, False
    else:
        end = min(p0 + EXPIRE, n)
        fill = (l[p0:end] <= ref - SPREAD) if d == 1 else (h[p0:end] >= ref)
        if not fill.any():
            return dict(status="expired") if n >= p0 + EXPIRE else dict(status="open")
        jf = p0 + int(fill.argmax())
        entry = ref
        sl_fill = (l[jf] <= stop) if d == 1 else (h[jf] + SPREAD >= stop)
        start = jf + 1
    R = abs(entry - stop)
    out = dict(status="filled", fill_time=idx[jf], fill_price=float(entry))
    if sl_fill:
        out.update(status="closed", closed_time=idx[jf] + HOUR, r_1_5=-1.0, r_2_0=-1.0, r_3_0=-1.0)
        return out
    sh_, sl_ = h[start:start + HOLD], l[start:start + HOLD]
    hit_sl = (sl_ <= stop) if d == 1 else (sh_ + SPREAD >= stop)
    t_sl = int(hit_sl.argmax()) if hit_sl.any() else INF
    res, done = {}, []
    for rr in RRS:
        tp = entry + d * rr * R
        hit_tp = (sh_ >= tp) if d == 1 else (sl_ + SPREAD <= tp)
        t_tp = int(hit_tp.argmax()) if hit_tp.any() else INF
        if t_tp < t_sl:
            res[rr] = float(rr); done.append(t_tp)
        elif t_sl < INF:
            res[rr] = -1.0; done.append(t_sl)
        elif len(sh_) >= HOLD:
            cx = c[start + HOLD - 1]
            m = (cx - entry) / R if d == 1 else (entry - cx - SPREAD) / R
            res[rr] = float(np.clip(m, -1.0, rr)); done.append(HOLD - 1)
    for rr, key in zip(RRS, ("r_1_5", "r_2_0", "r_3_0")):
        if rr in res:
            out[key] = res[rr]
    if len(res) == 3:
        out["status"] = "closed"
        out["closed_time"] = idx[start + max(done)] + HOUR
    return out


def sb(url, key, path, method="GET", data=None):
    hdr = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json",
           "User-Agent": "gold-scanner"}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=body, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=40) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt else None


def _naive(ts):
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def update(url, key, bars):
    # replay open/filled signals; write back changes; return newly closed ones
    rows = sb(url, key, "/rest/v1/signals?status=in.(open,filled)&select=*&order=bar_time.asc&limit=500") or []
    closed_now = []
    for r in rows:
        d = 1 if r["direction"] == "BUY" else -1
        res = resolve(bars, _naive(r["bar_time"]), d, r.get("mode") or "mkt",
                      float(r["entry"]), float(r["stop"]))
        patch = {k: (v.isoformat() + "Z" if isinstance(v, pd.Timestamp) else v) for k, v in res.items()}
        if patch.get("status") == r["status"] and not any(k.startswith("r_") for k in patch):
            continue
        sb(url, key, f"/rest/v1/signals?id=eq.{r['id']}", "PATCH", patch)
        if patch.get("status") == "closed":
            closed_now.append({**r, **patch})
    return closed_now


def close_message(r):
    side = "🟢 BUY" if r["direction"] == "BUY" else "🔴 SELL"
    f = lambda v: "—" if v is None else f"{v:+.2f}R"
    return (f"📌 نتیجه ستاپ بسته شد\n{side} | {r['setup']}\n"
            f"ورود: {float(r['fill_price']):.2f} | استاپ: {float(r['stop']):.2f}\n"
            f"نتیجه با تارگت 1.5R: {f(r.get('r_1_5'))} | 2R: {f(r.get('r_2_0'))} | 3R: {f(r.get('r_3_0'))}")


def weekly_report(url, key):
    rows = sb(url, key, "/rest/v1/signals?status=eq.closed&select=*&order=bar_time.asc&limit=2000") or []
    if not rows:
        return "📊 گزارش هفتگی: هنوز سیگنال بسته‌شده‌ای نداریم."
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["bar_time"], utc=True)
    wk = df[df["t"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]

    def stat(x, tag):
        if not len(x):
            return f"{tag}: بدون سیگنال"
        s = f"{tag}: {len(x)} سیگنال"
        for k, nm in (("r_1_5", "1.5R"), ("r_2_0", "2R"), ("r_3_0", "3R")):
            v = x[k].astype(float)
            s += f"\n  {nm}: برد {100 * (v > 0).mean():.0f}٪ | میانگین {v.mean():+.2f}R"
        return s

    msg = "📊 گزارش اسکنر طلا\n" + stat(wk, "این هفته") + "\n" + stat(df, "کل از ابتدا")
    for side, nm in (("BUY", "خرید"), ("SELL", "فروش")):
        x = df[df["direction"] == side]
        if len(x) >= 5:
            msg += f"\n{nm} (کل، 2R): {len(x)} مورد، میانگین {x['r_2_0'].astype(float).mean():+.2f}R"
    if len(df) < 100:
        msg += f"\n⚠️ هنوز فقط {len(df)} سیگنال داریم؛ برای قضاوت حداقل ۱۰۰ تا ۱۵۰ لازمه."
    return msg


# END OF FILE
